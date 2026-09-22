"""_collect_stream turns a streamed chat completion back into the
non-streamed shape ask() reads, forwarding text fragments as they come."""

from types import SimpleNamespace as NS

from src.agent.orchestrator import _collect_stream


def _chunk(content=None, tool_calls=None):
    return NS(usage=None, choices=[NS(delta=NS(content=content, tool_calls=tool_calls))])


def _tc(index, id=None, name=None, arguments=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=arguments))


def test_text_fragments_are_forwarded_and_joined():
    seen = []
    response = _collect_stream(iter([_chunk("Faiz "), _chunk("yüzde "), _chunk("50.")]), seen.append)
    assert seen == ["Faiz ", "yüzde ", "50."]
    message = response.choices[0].message
    assert message.content == "Faiz yüzde 50."
    assert message.tool_calls is None


def test_tool_call_arguments_split_across_chunks_are_reassembled():
    response = _collect_stream(iter([
        _chunk(tool_calls=[_tc(0, id="call_a", name="lakehouse_query", arguments='{"sql": ')]),
        _chunk(tool_calls=[_tc(0, arguments='"SELECT 1"}'), _tc(1, id="call_b", name="web_search", arguments="{}")]),
    ]), lambda _: None)
    calls = response.choices[0].message.tool_calls
    assert [(c.id, c.function.name, c.function.arguments) for c in calls] == [
        ("call_a", "lakehouse_query", '{"sql": "SELECT 1"}'),
        ("call_b", "web_search", "{}"),
    ]
    assert response.choices[0].message.model_dump(exclude_none=True)["role"] == "assistant"


def test_a_failing_callback_does_not_break_the_completion():
    def boom(_):
        raise RuntimeError("ui gone")
    assert _collect_stream(iter([_chunk("ok")]), boom).choices[0].message.content == "ok"


def test_a_stream_that_outlives_the_run_deadline_is_cut_off():
    import time
    import openai
    import pytest
    with pytest.raises(openai.APITimeoutError):
        _collect_stream(iter([_chunk("a"), _chunk("b")]), lambda _: None, deadline=time.monotonic() - 1)
