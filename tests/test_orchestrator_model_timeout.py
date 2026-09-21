"""Tests for _call_model's retry/deadline behavior, and for ask()'s
handling when the model call itself keeps failing (not a tool failure).

WHY THIS EXISTS: a real incident, found live via the actual server logs -
a broad "find every BDDK/TCMB indicator" question hit an MIA endpoint that
was slow/overloaded. _call_model retried 3 times, reusing the SAME
~298s timeout on every attempt because the run's own wall-clock deadline
(AGENT_RUN_DEADLINE_SEC, 300s by default) was never checked once inside
the retry loop - so the whole call sequence overshot the nominal 5-minute
budget by roughly 3x before finally raising. That raise then propagated
straight out of ask() as an unhandled exception (no try/except around
the main loop's model call at all), surfacing to the user as a generic
"Agent yanıt üretemedi. Sunucu günlüklerini kontrol edin." with nothing
actionable in it, even though the SAME retry-exhaustion fallback
machinery already existed and already degrades gracefully for the
round/deadline-limit case.
"""
import time

try:
    import httpx
except ModuleNotFoundError:  # Python 3.14 compatibility fork used by this environment
    import httpx2 as httpx
import openai
import pytest

from src.agent import orchestrator


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(httpx.Request("POST", "https://example.com"))


class _AlwaysFailingCompletions:
    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        # Cost more than one clock tick. time.monotonic() on Windows moves
        # in ~15.6ms steps, so a call that raises in microseconds is timed
        # as exactly 0.0ms - measured here across repeated runs: two
        # trials reported 0.0 and one reported 16.0 for identical work.
        # That, not machine speed, is what made
        # test_ask_never_raises_...'s `llm_ms > 0` assertion flaky. 25ms
        # clears the tick with margin and still costs the suite nothing.
        time.sleep(0.025)
        raise self._exc


class _RetryFakeClient:
    """Mimics the OpenAI SDK's with_options(timeout=...) fluent call -
    returns a client still wired to the SAME completions object, so every
    attempt's calls and requested timeout land on one shared record."""

    def __init__(self, exc: Exception):
        self.completions = _AlwaysFailingCompletions(exc)
        self.chat = type("Chat", (), {"completions": self.completions})()
        self.timeouts_used: list[float] = []

    def with_options(self, timeout=None):
        self.timeouts_used.append(timeout)
        return self


def test_stops_retrying_immediately_once_the_deadline_has_already_passed():
    client = _RetryFakeClient(_timeout_error())
    past_deadline = time.monotonic() - 1.0

    with pytest.raises(TimeoutError, match="deadline"):
        orchestrator._call_model(client, [{"role": "user", "content": "x"}], [],
                                  timeout=30.0, deadline=past_deadline)

    assert client.completions.calls == []  # never even tried a real call


def test_does_not_sleep_into_overtime_when_backoff_would_overshoot_the_deadline():
    exc = _timeout_error()
    client = _RetryFakeClient(exc)
    # Enough budget for exactly one attempt, nowhere near enough for the
    # first backoff wait (MODEL_CALL_BACKOFF_BASE_SEC=2.0s) on top of it.
    near_deadline = time.monotonic() + 0.05

    with pytest.raises(openai.APITimeoutError):
        orchestrator._call_model(client, [{"role": "user", "content": "x"}], [],
                                  timeout=30.0, deadline=near_deadline)

    assert len(client.completions.calls) == 1  # only the one attempt that actually fit


def test_per_attempt_timeout_shrinks_to_the_remaining_deadline_budget():
    """The exact bug found live: the SDK's per-request timeout used to be
    fixed once before the retry loop, so a caller-supplied 240-300s
    timeout was reused unshrunk on every retry regardless of how much of
    the run's own deadline was actually left."""
    client = _RetryFakeClient(_timeout_error())
    deadline = time.monotonic() + 5.0

    with pytest.raises(openai.APITimeoutError):
        orchestrator._call_model(client, [{"role": "user", "content": "x"}], [],
                                  timeout=300.0, deadline=deadline)

    assert client.timeouts_used, "at least one attempt should have been made"
    assert all(t <= 5.0 for t in client.timeouts_used), (
        f"attempt timeout(s) {client.timeouts_used} exceeded the 5s deadline budget, "
        f"even though a 300s timeout was requested"
    )


def test_a_full_read_timeout_is_not_replayed_even_without_a_deadline():
    """A timeout already spent the whole call budget; retrying the same
    large request is the failure amplification this guard prevents."""
    client = _RetryFakeClient(_timeout_error())

    with pytest.raises(openai.APITimeoutError):
        orchestrator._call_model(client, [{"role": "user", "content": "x"}], [], timeout=12.0)

    assert client.timeouts_used == [12.0]


# --- ask()-level: a persistently failing model call must degrade to a
# real, honest answer instead of raising ------------------------------

class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}
        return {k: v for k, v in d.items() if v is not None} if exclude_none else d


def test_ask_never_raises_when_the_model_call_keeps_failing(monkeypatch):
    """The core regression check for the live incident: ask() must return
    a real, honest answer - not propagate a raw exception up to web_app's
    generic 502 handler - when every model call attempt fails."""
    client = _RetryFakeClient(_timeout_error())
    monkeypatch.setattr(orchestrator, "get_client", lambda: client)
    monkeypatch.setattr(orchestrator, "known_series_codes", lambda: {"TP.BKR.TRY.18"})

    # The deadline only has to be short enough to keep the test fast - the
    # fake client fails instantly, so nothing here actually waits. It used
    # to be 2.0s, which made this test flaky for a reason that had nothing
    # to do with what it checks: building the system prompt reads the
    # series catalog out of DuckDB, and on a cold cache that alone could
    # outlast 2s. _call_model then refused to start any attempt at all
    # ("deadline already passed"), so no model call was ever timed, llm_ms
    # came back 0.0, and the assertion below failed on a slow machine
    # while passing on a warm one.
    answer, _, timing = orchestrator.ask(
        "test sorusu", deadline_sec=30.0, return_trace=True, return_timing=True)

    assert "süre içinde yanıt vermedi" in answer.lower()
    assert "check server logs" not in answer.lower()
    assert "günlüklerini kontrol edin" not in answer.lower()  # the old generic message must not appear
    assert timing["summary"]["llm_ms"] > 0
    assert timing["summary"]["llm_errors"] == 1
    assert timing["timeline"][0]["error"], "failed model wait must be visible in the UI timeline"
    assert "başarısız" in timing["timeline"][0]["name"]


def test_ask_with_no_model_error_still_answers_normally(monkeypatch):
    """Sanity check that the new try/except around the main-loop model
    call does not swallow or alter the ordinary success path."""
    class _OneShotCompletions:
        def create(self, **kwargs):
            return type("R", (), {"choices": [type("C", (), {
                "message": _FakeMessage(content="Gerçek cevap.")})()], "usage": None})()

    class _OneShotClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _OneShotCompletions()})()

        def with_options(self, timeout=None):
            return self

    monkeypatch.setattr(orchestrator, "get_client", lambda: _OneShotClient())
    monkeypatch.setattr(orchestrator, "known_series_codes", lambda: {"TP.BKR.TRY.18"})

    answer = orchestrator.ask("test sorusu")

    assert answer == "Gerçek cevap."
