"""Integration tests for the live Seri_Kod hallucination guard wired into
orchestrator.ask() (see src/agent/hallucination_guard.py for the pure
check itself).

WHY THIS EXISTS: benchmarks/agent_benchmark.py already measures this
failure mode offline as `hallucination_rate`, but that only reports the
problem after a manual run - it does not stop a fabricated series code
from reaching a real user. These tests drive the actual ask() loop with a
scripted fake model client so the correction round and the caveat
fallback are proven against the real control flow, not just the helper
functions in isolation.
"""

import logging

from src.agent import orchestrator


class _FakeMessage:
    """Stands in for openai's ChatCompletionMessage: only the two fields
    orchestrator.ask() reads (content, tool_calls) plus the model_dump()
    it calls before appending the message to history."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


class _FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]
        self.usage = None


class _ScriptedCompletions:
    """Returns one scripted message per call, in order. Raises with a
    clear message (rather than an opaque IndexError) if the loop asks for
    more calls than the test scripted - that mismatch IS the test result
    for the "must not loop forever" cases below."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.call_count = 0
        self.calls = []  # kwargs of every create() call, for inspecting what the model was actually told

    def create(self, **kwargs):
        self.call_count += 1
        self.calls.append(kwargs)
        if not self._messages:
            raise AssertionError(
                f"model was called a {self.call_count}th time; "
                f"the hallucination-correction round must be bounded to one retry"
            )
        return _FakeResponse(self._messages.pop(0))


class _ScriptedClient:
    def __init__(self, messages):
        completions = _ScriptedCompletions(messages)
        self.chat = type("Chat", (), {"completions": completions})()


def _install(monkeypatch, messages, valid_codes):
    client = _ScriptedClient(messages)
    monkeypatch.setattr(orchestrator, "get_client", lambda: client)
    monkeypatch.setattr(orchestrator, "known_series_codes", lambda: valid_codes)
    return client


def test_a_real_series_code_ships_untouched_with_a_single_model_call(monkeypatch):
    client = _install(
        monkeypatch,
        messages=[_FakeMessage(content="TP.BKR.TRY.18'e göre faiz %5 düştü.")],
        valid_codes={"TP.BKR.TRY.18"},
    )

    answer = orchestrator.ask("test sorusu")

    assert answer == "TP.BKR.TRY.18'e göre faiz %5 düştü."
    assert client.chat.completions.call_count == 1


def test_an_invented_code_triggers_exactly_one_correction_round(monkeypatch):
    client = _install(
        monkeypatch,
        messages=[
            _FakeMessage(content="TP.UYDURULMUS.KOD'a göre faiz %5 düştü."),
            _FakeMessage(content="Özür dilerim, gerçek kod TP.BKR.TRY.18 imiş; ona göre faiz %5 düştü."),
        ],
        valid_codes={"TP.BKR.TRY.18"},
    )

    answer = orchestrator.ask("test sorusu")

    assert answer == "Özür dilerim, gerçek kod TP.BKR.TRY.18 imiş; ona göre faiz %5 düştü."
    assert "UYDURULMUS" not in answer
    assert client.chat.completions.call_count == 2


def test_the_correction_message_names_the_invented_code(monkeypatch):
    """The model needs to know WHICH code was wrong to fix it - a generic
    "you made a mistake" would be far less likely to converge."""
    client = _install(
        monkeypatch,
        messages=[
            _FakeMessage(content="TP.UYDURULMUS.KOD'a göre faiz düştü."),
            _FakeMessage(content="TP.BKR.TRY.18'e göre faiz düştü."),
        ],
        valid_codes={"TP.BKR.TRY.18"},
    )

    orchestrator.ask("test sorusu")

    assert client.chat.completions.call_count == 2
    # The second create() call is the correction round - its message list
    # must carry a user message naming the exact invented code, not a
    # vague "something was wrong".
    second_call_messages = client.chat.completions.calls[1]["messages"]
    correction_texts = [m["content"] for m in second_call_messages if m.get("role") == "user"]
    assert any("TP.UYDURULMUS.KOD" in text for text in correction_texts)


def test_a_persistent_hallucination_ships_with_a_visible_caveat_after_one_retry(monkeypatch):
    client = _install(
        monkeypatch,
        messages=[
            _FakeMessage(content="TP.UYDURULMUS.KOD'a göre faiz %5 düştü."),
            _FakeMessage(content="TP.UYDURULMUS.KOD'a göre yine faiz %5 düştü."),
        ],
        valid_codes={"TP.BKR.TRY.18"},
    )

    answer = orchestrator.ask("test sorusu")

    assert answer.startswith(orchestrator.HALLUCINATION_CAVEAT_PREFIX.format(codes="TP.UYDURULMUS.KOD"))
    assert "TP.UYDURULMUS.KOD" in answer
    # Bounded: exactly the original call plus the one correction attempt,
    # never a third "let's try again" call.
    assert client.chat.completions.call_count == 2


def test_two_different_invented_codes_are_both_named_in_the_caveat(monkeypatch):
    client = _install(
        monkeypatch,
        messages=[
            _FakeMessage(content="TP.BIRINCI.HAYALI ve TP.IKINCI.HAYALI serilerine göre..."),
            _FakeMessage(content="TP.BIRINCI.HAYALI ve TP.IKINCI.HAYALI serilerine göre yine..."),
        ],
        valid_codes={"TP.BKR.TRY.18"},
    )

    answer = orchestrator.ask("test sorusu")

    assert "TP.BIRINCI.HAYALI" in answer
    assert "TP.IKINCI.HAYALI" in answer
    assert client.chat.completions.call_count == 2


def test_no_catalog_available_means_no_correction_round_is_spent(monkeypatch):
    """An empty/missing catalog must never masquerade as "everything is
    hallucinated" - that would burn a correction round on every single
    answer whenever the gold database is not built."""
    client = _install(
        monkeypatch,
        messages=[_FakeMessage(content="TP.HERHANGI.BIRSEY'e göre faiz düştü.")],
        valid_codes=set(),
    )

    answer = orchestrator.ask("test sorusu")

    assert answer == "TP.HERHANGI.BIRSEY'e göre faiz düştü."
    assert client.chat.completions.call_count == 1


def test_the_forced_closure_answer_also_gets_a_caveat_when_it_hallucinates(monkeypatch):
    """When the round budget runs out, ask() forces one more tool-free
    call to produce SOME answer rather than a raw error (see the "round
    budget exhausted" branch). That answer is model-generated prose too,
    so it needs the same check - there is no retry budget left there, so
    it must go straight to the caveat instead of looping again."""
    tool_call = _FakeToolCall("lakehouse_schema", "{}")
    client = _install(
        monkeypatch,
        messages=[
            _FakeMessage(content=None, tool_calls=[tool_call]),
            _FakeMessage(content="TP.UYDURULMUS.KOD'a göre faiz düştü."),
        ],
        valid_codes={"TP.BKR.TRY.18"},
    )
    monkeypatch.setattr(orchestrator, "execute_tool", lambda name, args: "{}")

    answer = orchestrator.ask("test sorusu", max_rounds=1)

    assert answer.startswith(orchestrator.HALLUCINATION_CAVEAT_PREFIX.format(codes="TP.UYDURULMUS.KOD"))
    assert client.chat.completions.call_count == 2


def test_an_answer_with_no_code_shaped_text_is_never_touched(monkeypatch):
    client = _install(
        monkeypatch,
        messages=[_FakeMessage(content="Bu soruyla ilgili lakehouse'da veri bulunamadı.")],
        valid_codes={"TP.BKR.TRY.18"},
    )

    answer = orchestrator.ask("test sorusu")

    assert answer == "Bu soruyla ilgili lakehouse'da veri bulunamadı."
    assert client.chat.completions.call_count == 1


def test_a_code_fetched_live_this_run_is_not_treated_as_hallucinated(monkeypatch):
    """Real bug found live: fetch_evds_series can pull a series code that
    was never built into the static gold_fact_long catalog (it exists in
    EVDS but was only fetched on demand this run) - find_hallucinated_codes
    used to check ONLY the static catalog, so citing that freshly fetched
    code in the final answer got wrongly flagged as "muhtemelen uydurulmuş"
    even though the model had just gotten it from a real tool result."""
    tool_call = _FakeToolCall(
        "fetch_evds_series",
        '{"datagroup": "bie_kt210a", "seri_kod": "TP.BKR.TRY.KTF10", '
        '"start_period": "2024-09", "end_period": "2026-08"}',
    )
    client = _install(
        monkeypatch,
        messages=[
            _FakeMessage(content=None, tool_calls=[tool_call]),
            _FakeMessage(content="TP.BKR.TRY.KTF10'a göre ihtiyaç kredisi faizi %45.2."),
        ],
        valid_codes={"TP.BKR.TRY.18"},  # the live-fetched code is NOT in the static catalog
    )
    monkeypatch.setattr(
        orchestrator, "execute_tool",
        lambda name, args: '{"seri_kod": "TP.BKR.TRY.KTF10", "rows": [{"Period": "2026-06", "Value": 45.2}]}',
    )

    answer = orchestrator.ask("test sorusu")

    assert answer == "TP.BKR.TRY.KTF10'a göre ihtiyaç kredisi faizi %45.2."
    assert not answer.startswith("⚠️")
    assert client.chat.completions.call_count == 2  # tool round + final answer, no correction round spent


# ---------------------------------------------------------------------
# CROSS-TURN case: the code above (test_a_code_fetched_live_this_run_is_
# not_treated_as_hallucinated) only proves same-turn safety, because
# _codes_seen_in_trace(trace) reads THIS ask() call's own trace list -
# trace is a local variable created fresh inside ask() (see
# `trace: list[dict] = []` near the top of the function body) and is
# never persisted anywhere. web_app.py is stateless per request (see
# ask()'s own docstring on `history`): the ONLY thing that survives
# between turns is whatever the CLIENT resends as `history`, which is
# plain question/answer prose plus a short `note` string
# (summarize_turn()'s one-liner). A series code fetched in turn 1 and
# cited again - honestly, from that same note - in turn 2 has NO trace
# entry in turn 2's ask() call at all, so
# _valid_codes_for_check() | _codes_seen_in_trace(trace) does not
# contain it, and find_hallucinated_codes flags it as invented. This
# reproduces that exact scenario end to end, across two real ask() calls.
# ---------------------------------------------------------------------

def test_turn1_fetches_a_live_code_turn2_cites_it_again_from_history(monkeypatch, caplog):
    """THE BUG: reproduces a dynamically-fetched EVDS code (never in the
    static gold_fact_long catalog) being cited again one turn later, using
    only the conversation history a real browser tab would resend - no
    tool call happens in turn 2 at all. Heavy logging throughout so the
    exact point where the code "disappears" from the trust set is visible
    in the test output (run with `pytest -s -log-cli-level=DEBUG` to see
    it live)."""
    caplog.set_level(logging.DEBUG, logger="orchestrator")
    log = logging.getLogger("test_cross_turn_repro")

    LIVE_CODE = "TP.BKR.TRY.1"
    STATIC_CATALOG = {"TP.BKR.TRY.18"}  # LIVE_CODE deliberately absent - it was never gold-built

    # --- Turn 1: model calls fetch_evds_series, then answers citing the code ---
    log.debug("TURN 1: scripting fetch_evds_series(seri_kod=%r) + an answer citing it", LIVE_CODE)
    tool_call = _FakeToolCall(
        "fetch_evds_series",
        f'{{"datagroup": "bie_kt210a", "seri_kod": "{LIVE_CODE}", '
        f'"start_period": "2024-09", "end_period": "2026-08"}}',
    )
    client_1 = _install(
        monkeypatch,
        messages=[
            _FakeMessage(content=None, tool_calls=[tool_call]),
            _FakeMessage(content=f"{LIVE_CODE}'e göre ticari kredi faizi %38.4."),
        ],
        valid_codes=STATIC_CATALOG,
    )
    monkeypatch.setattr(
        orchestrator, "execute_tool",
        lambda name, args: f'{{"seri_kod": "{LIVE_CODE}", "rows": [{{"Period": "2026-06", "Value": 38.4}}]}}',
    )

    turn1_trace: list[dict] = []
    answer_1 = orchestrator.ask("Ticari kredi faizini EVDS'ten çek", return_trace=True, deadline_sec=None)
    # ask() with return_trace=True returns (answer, trace, ...) - unpack defensively
    # since the exact tuple shape has changed before; only answer and trace matter here.
    if isinstance(answer_1, tuple):
        answer_1, turn1_trace, *_rest = answer_1
    log.debug("TURN 1 raw trace entries: %s", turn1_trace)
    log.debug("TURN 1 answer: %r", answer_1)
    assert client_1.chat.completions.call_count == 2, "turn 1 must not itself trigger a false-positive correction round"
    assert LIVE_CODE in answer_1

    # This is EXACTLY what web_app.py does server-side and returns to the
    # browser as `turn_note` - and exactly what the browser resends next
    # turn as history[i]["note"] (see web_app.py's HistoryTurn.note).
    turn1_note = orchestrator.summarize_turn(turn1_trace)
    log.debug("TURN 1 summarize_turn() note that the client will resend next turn: %r", turn1_note)
    assert LIVE_CODE in turn1_note, (
        "sanity check: summarize_turn must have captured the live code in its note, "
        "otherwise this test would not be reproducing the real client round-trip"
    )

    # --- Turn 2: NO tool call at all - the model just answers directly,
    # citing the SAME code, which it legitimately knows about only because
    # turn 1's note is sitting in the conversation history right in front
    # of it (see _history_messages: "[Bu yanıtta kullanılan - {note}]"). ---
    log.debug("TURN 2: model answers directly (NO tool call), citing %r again from memory of turn 1", LIVE_CODE)
    client_2 = _install(
        monkeypatch,
        messages=[_FakeMessage(content=f"Evet, {LIVE_CODE} serisine göre faiz aynı kalmış, hâlâ %38.4.")],
        valid_codes=STATIC_CATALOG,  # same static catalog - LIVE_CODE still not gold-built
    )

    history = [{"question": "Ticari kredi faizini EVDS'ten çek", "answer": answer_1, "note": turn1_note}]
    log.debug("TURN 2 history being resent to ask(): %s", history)

    answer_2 = orchestrator.ask("Aynı seri hâlâ geçerli mi, kontrol eder misin?", history=history)
    log.debug("TURN 2 final answer: %r", answer_2)
    log.debug("TURN 2 model call count: %s (2 calls means a spurious correction round fired)",
              client_2.chat.completions.call_count)

    # THE ASSERTION THAT MATTERS: a code the model only knows about
    # because it legitimately appears in the resent history/note must NOT
    # be treated as invented. If this fails, `answer_2` starts with
    # HALLUCINATION_CAVEAT_PREFIX and/or call_count == 2 (an unnecessary,
    # confusing "Özür dilerim..." self-correction over a REAL code).
    assert client_2.chat.completions.call_count == 1, (
        f"turn 2 spent a correction round on a code ({LIVE_CODE}) that was legitimately "
        f"fetched live in turn 1 - _codes_seen_in_trace(trace) cannot see it because trace "
        f"is reset empty at the top of every ask() call and turn 1's trace is gone; "
        f"_valid_codes_for_check() cannot see it either because it was never gold-built. "
        f"See orchestrator.py's `trace: list[dict] = []` and _codes_seen_in_trace()'s own "
        f"docstring, which already documents the SAME-turn version of this gap."
    )
    assert not answer_2.startswith("⚠️"), f"got a false-positive hallucination caveat: {answer_2!r}"
    assert answer_2 == f"Evet, {LIVE_CODE} serisine göre faiz aynı kalmış, hâlâ %38.4."
