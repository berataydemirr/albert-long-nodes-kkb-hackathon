"""Tests for tool-loop context compaction.

The invariants matter more than the savings: a compactor that shrinks the
history but breaks assistant/tool message pairing turns every later API
call into a 400, which is a far worse failure than a large context.
"""

import json

from src.agent.context_compactor import (
    COMPACTED_MARKER,
    compact_messages,
    estimate_tokens,
)


def _big_tool_message(tool_call_id: str, rows: int = 200) -> dict:
    payload = {
        "columns": ["Period", "Deger"],
        "rows": [{"Period": f"2021-{i % 12 + 1:02d}", "Deger": float(i)} for i in range(rows)],
        "row_count": rows,
    }
    return {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(payload, ensure_ascii=False)}


def _history(tool_messages: int, rows: int = 200) -> list[dict]:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "kullanici sorusu"},
    ]
    for i in range(tool_messages):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"call_{i}", "function": {"name": "lakehouse_query", "arguments": "{}"}}],
        })
        messages.append(_big_tool_message(f"call_{i}", rows=rows))
    return messages


def test_small_history_is_left_completely_untouched():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "kisa soru"},
    ]

    compacted, stats = compact_messages(messages)

    assert compacted == messages
    assert stats["ran"] is False
    assert stats["compacted_messages"] == 0


def test_large_history_is_compacted_and_shrinks():
    messages = _history(tool_messages=8)

    compacted, stats = compact_messages(messages)

    assert stats["ran"] is True
    assert stats["compacted_messages"] > 0
    assert stats["after_tokens"] < stats["before_tokens"]
    assert estimate_tokens(compacted) < estimate_tokens(messages)


def test_no_message_is_ever_dropped():
    """Message count must be identical - a missing 'tool' message orphans
    its tool_call_id and the next API call fails with a 400."""
    messages = _history(tool_messages=8)

    compacted, _ = compact_messages(messages)

    assert len(compacted) == len(messages)
    assert [m["role"] for m in compacted] == [m["role"] for m in messages]


def test_tool_call_ids_survive_compaction():
    messages = _history(tool_messages=8)

    compacted, _ = compact_messages(messages)

    original_ids = [m.get("tool_call_id") for m in messages if m["role"] == "tool"]
    compacted_ids = [m.get("tool_call_id") for m in compacted if m["role"] == "tool"]
    assert original_ids == compacted_ids


def test_system_prompt_and_user_question_are_never_rewritten():
    messages = _history(tool_messages=8)

    compacted, _ = compact_messages(messages)

    assert compacted[0] == messages[0]
    assert compacted[1] == messages[1]


def test_most_recent_tool_outputs_are_preserved_in_full():
    """The model is still reasoning over the latest results; compacting
    those would remove the data it is about to answer from."""
    messages = _history(tool_messages=8)

    compacted, _ = compact_messages(messages, keep_recent=3)

    tool_messages = [m for m in compacted if m["role"] == "tool"]
    for message in tool_messages[-3:]:
        assert not message["content"].startswith(COMPACTED_MARKER)


def test_compacted_output_keeps_the_shape_of_the_result():
    """A digest that loses the row count is useless - the model needs to
    know it already has this data to avoid re-running the same query."""
    messages = _history(tool_messages=8)

    compacted, _ = compact_messages(messages)

    digested = [m for m in compacted if m["role"] == "tool" and m["content"].startswith(COMPACTED_MARKER)]
    assert digested, "expected at least one digested tool message"
    assert "row_count" in digested[0]["content"]


def test_input_list_is_not_mutated():
    messages = _history(tool_messages=8)
    snapshot = [dict(m) for m in messages]

    compact_messages(messages)

    assert messages == snapshot


def test_compaction_is_idempotent():
    """Running twice must not digest a digest into noise."""
    messages = _history(tool_messages=8)

    once, _ = compact_messages(messages)
    twice, stats = compact_messages(once)

    assert stats["compacted_messages"] == 0
    assert twice == once


def test_non_json_tool_output_is_still_compacted():
    messages = _history(tool_messages=0)
    for i in range(8):
        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"call_{i}", "function": {"name": "web_url_extract", "arguments": "{}"}}],
        })
        messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "plain text " * 2000})

    compacted, stats = compact_messages(messages)

    assert stats["ran"] is True
    assert stats["after_tokens"] < stats["before_tokens"]


def test_tool_call_arguments_count_toward_the_estimate():
    """Arguments live outside 'content' but are still sent every round -
    the array-shaped analysis calls this project used to make were almost
    entirely argument text, so missing them would hide the real cost."""
    without = [{"role": "assistant", "content": None}]
    with_args = [{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "causality_analysis", "arguments": "x" * 4000}}],
    }]

    assert estimate_tokens(with_args) > estimate_tokens(without)


def test_a_pass_that_saves_almost_nothing_keeps_the_full_history():
    """Seen live: two tool results digested to save 1.7% of the history.
    Below MIN_USEFUL_SAVING the original messages must come back intact."""
    from src.agent.context_compactor import compact_messages
    big_question = {"role": "user", "content": "q" * 20_000}
    tools = [{"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps({"rows": [{"v": i}] * 30})}
             for i in range(5)]
    messages = [big_question, *tools]
    result, stats = compact_messages(messages, budget_tokens=100, keep_recent=3)
    assert result is messages
    assert not stats["ran"]
