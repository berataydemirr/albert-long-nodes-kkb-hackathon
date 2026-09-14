"""Tests for the tool-loop guard and the search provider's failure signal.

Both exist because of one measured run: while DuckDuckGo was serving a
CAPTCHA to every request, the agent spent 4 web_search calls across 5
rounds rephrasing a query that could never have succeeded. An empty
result list reads to the model as "your query found nothing", so it
rephrases; nothing in the loop said "stop".
"""

import json

import pytest

from src.agent.orchestrator import MAX_CALLS_PER_TOOL_PER_RUN, _loop_guard, _normalized_args


def _fresh_history() -> dict:
    return {"signatures": set(), "per_tool": {}}


# --- the guard ----------------------------------------------------------

def test_first_call_is_allowed():
    assert _loop_guard("web_search", '{"query": "faiz"}', _fresh_history()) is None


def test_identical_repeat_is_blocked():
    history = _fresh_history()
    _loop_guard("web_search", '{"query": "faiz"}', history)

    blocked = _loop_guard("web_search", '{"query": "faiz"}', history)

    assert blocked is not None
    assert json.loads(blocked)["reason"] == "identical_repeat"


def test_key_order_does_not_disguise_an_identical_call():
    history = _fresh_history()
    _loop_guard("web_search", '{"query": "faiz", "max_results": 5}', history)

    blocked = _loop_guard("web_search", '{"max_results": 5, "query": "faiz"}', history)

    assert blocked is not None, "reordered keys are the same call"


def test_different_arguments_are_allowed():
    history = _fresh_history()
    _loop_guard("web_search", '{"query": "faiz"}', history)

    assert _loop_guard("web_search", '{"query": "kredi"}', history) is None


def test_a_tool_is_cut_off_after_the_per_run_limit():
    """Different arguments every time is still a loop past some point."""
    history = _fresh_history()
    for i in range(MAX_CALLS_PER_TOOL_PER_RUN):
        assert _loop_guard("web_search", f'{{"query": "q{i}"}}', history) is None

    blocked = _loop_guard("web_search", '{"query": "one more"}', history)

    assert blocked is not None
    assert json.loads(blocked)["reason"] == "call_limit_reached"


def test_the_limit_is_per_tool_not_global():
    history = _fresh_history()
    for i in range(MAX_CALLS_PER_TOOL_PER_RUN):
        _loop_guard("web_search", f'{{"query": "q{i}"}}', history)

    assert _loop_guard("lakehouse_query", '{"sql": "SELECT 1"}', history) is None


def test_blocked_result_tells_the_model_what_to_do_instead():
    """A block with no next step just moves the loop somewhere else."""
    history = _fresh_history()
    _loop_guard("web_search", '{"query": "faiz"}', history)

    payload = json.loads(_loop_guard("web_search", '{"query": "faiz"}', history))

    assert payload["blocked"] is True
    assert payload["next_step"].strip()


def test_guard_returns_a_result_rather_than_raising():
    """A raised guard would break the assistant/tool message pairing and
    fail the next API call."""
    history = _fresh_history()
    _loop_guard("web_search", '{"query": "faiz"}', history)

    blocked = _loop_guard("web_search", '{"query": "faiz"}', history)

    assert isinstance(blocked, str)
    json.loads(blocked)


def test_unparseable_arguments_still_deduplicate():
    history = _fresh_history()
    assert _loop_guard("t", "not json", history) is None
    assert _loop_guard("t", "not json", history) is not None


def test_normalized_args_is_stable_across_key_order():
    assert _normalized_args('{"a": 1, "b": 2}') == _normalized_args('{"b": 2, "a": 1}')


# --- the search provider signal -----------------------------------------

def test_bot_challenge_is_detected():
    from src.agent.web_search import _looks_like_a_bot_challenge

    page = "Unfortunately, bots use DuckDuckGo too. Please complete the following challenge"
    assert _looks_like_a_bot_challenge(page)


def test_ordinary_results_page_is_not_flagged():
    from src.agent.web_search import _looks_like_a_bot_challenge

    assert not _looks_like_a_bot_challenge("<div class='result'>TCMB politika faizi</div>")


def test_blocked_provider_reports_itself_as_unavailable(monkeypatch):
    """The distinction that matters: 'provider is down' must not look like
    'your query found nothing', or the model rephrases and retries."""
    import src.agent.web_search as ws

    def blocked(query, max_results):
        raise ws.SearchProviderUnavailable("bot challenge")

    monkeypatch.setattr(ws, "_duckduckgo_search", blocked)
    monkeypatch.setattr(ws, "WEB_SEARCH_PROVIDER", "duckduckgo")

    result = ws.web_search("TCMB faiz")

    assert result["provider_available"] is False
    assert result["result_count"] == 0
    assert "web_search" in result["next_step"]


def test_genuine_zero_results_are_marked_available_but_bounded(monkeypatch):
    import src.agent.web_search as ws

    monkeypatch.setattr(ws, "_duckduckgo_search", lambda q, n: [])
    monkeypatch.setattr(ws, "WEB_SEARCH_PROVIDER", "duckduckgo")

    result = ws.web_search("bulunmayan bir sey")

    assert result["provider_available"] is True
    assert result["next_step"], "a zero-result answer should bound the retries"


def test_successful_search_carries_no_retry_prompt(monkeypatch):
    import src.agent.web_search as ws

    monkeypatch.setattr(ws, "_duckduckgo_search", lambda q, n: [
        {"title": "TCMB", "url": "https://tcmb.gov.tr", "snippet": "faiz", "source": "tcmb.gov.tr"}
    ])
    monkeypatch.setattr(ws, "WEB_SEARCH_PROVIDER", "duckduckgo")

    result = ws.web_search("TCMB faiz")

    assert result["provider_available"] is True
    assert result["result_count"] == 1
    assert "next_step" not in result
