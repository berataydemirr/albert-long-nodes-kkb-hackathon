"""Regression test for a real, measured bug: the OpenAI SDK's own
max_retries silently doubled (or worse) the wall time of every model call,
undermining orchestrator._call_model's own deadline-aware retry/backoff -
that layer already owns retries for this endpoint end to end, so the SDK
underneath it must never retry on its own.

Found live: a round whose deadline-aware timeout should have capped it
at ~563s instead ran ~1127s (almost exactly 2x) before _call_model's own
except block ever saw the exception - openai._base_client's request()
catches httpx timeout exceptions directly and retries internally when
max_retries > 0, with no log line marking that it happened.
"""
from src.agent import mia_client


def test_sdk_level_retries_are_disabled():
    """max_retries=0 means openai._base_client.SyncAPIClient.request()'s
    `for retries_taken in range(max_retries + 1)` loop runs exactly once -
    any failure (timeout, 429, 5xx) raises straight to _call_model instead
    of being silently retried underneath it first."""
    assert mia_client.SDK_MAX_RETRIES == 0


def test_get_client_passes_zero_retries_to_the_sdk(monkeypatch):
    captured = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(mia_client, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(mia_client, "_client", None)
    monkeypatch.setattr("config.settings.MIA_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(mia_client, "MIA_API_KEY", "test-key")

    mia_client.get_client()

    assert captured["max_retries"] == 0
