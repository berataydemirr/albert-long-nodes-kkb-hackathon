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
