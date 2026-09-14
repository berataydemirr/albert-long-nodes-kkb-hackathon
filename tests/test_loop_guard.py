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


# --- the SearXNG provider ------------------------------------------------

def test_searxng_success_is_parsed_into_the_common_result_shape(monkeypatch):
    """Field names confirmed against a real running instance, not assumed
    from docs: results carry {url, title, content, engine}."""
    import src.agent.web_search as ws

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [
                {"url": "https://www.tcmb.gov.tr/x", "title": "TCMB Faiz",
                 "content": "Politika faizi ...", "engine": "brave"},
            ]}

    monkeypatch.setattr(ws.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(ws, "WEB_SEARCH_PROVIDER", "searxng")

    result = ws.web_search("TCMB faiz")

    assert result["provider_available"] is True
    assert result["result_count"] == 1
    assert result["results"][0]["url"] == "https://www.tcmb.gov.tr/x"
    assert result["results"][0]["engine"] == "brave"


def test_searxng_403_means_json_format_not_enabled(monkeypatch):
    """The JSON API is off by default in SearXNG's own template. A 403 here
    is a config problem, not 'no results' - it must not look retryable by
    rephrasing, same as a DuckDuckGo bot challenge."""
    import src.agent.web_search as ws

    class _Resp:
        status_code = 403

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ws.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(ws, "WEB_SEARCH_PROVIDER", "searxng")

    result = ws.web_search("TCMB faiz")

    assert result["provider_available"] is False
    assert "json" in result["error"].lower()


def test_searxng_unreachable_instance_is_reported_not_a_zero_result(monkeypatch):
    """The likeliest real failure: the container is not running."""
    import src.agent.web_search as ws

    def boom(*a, **k):
        raise ws.requests.ConnectionError("Connection refused")

    monkeypatch.setattr(ws.requests, "get", boom)
    monkeypatch.setattr(ws, "WEB_SEARCH_PROVIDER", "searxng")

    result = ws.web_search("TCMB faiz")

    assert result["provider_available"] is False
    assert "next_step" in result


def test_searxng_non_json_body_is_reported_not_a_zero_result(monkeypatch):
    import src.agent.web_search as ws

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("not JSON")

    monkeypatch.setattr(ws.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(ws, "WEB_SEARCH_PROVIDER", "searxng")

    result = ws.web_search("TCMB faiz")

    assert result["provider_available"] is False


def test_unknown_provider_is_rejected(monkeypatch):
    import src.agent.web_search as ws

    monkeypatch.setattr(ws, "WEB_SEARCH_PROVIDER", "bing")

    with pytest.raises(RuntimeError, match="duckduckgo.*searxng"):
        ws.web_search("TCMB faiz")


def test_tavily_has_no_functional_path_left(monkeypatch):
    """The open-source requirement means no residual path back to a paid
    provider. Comments explaining WHY it was removed are fine and expected
    (that is documentation, not a leftover integration) - what must be gone
    is anything that could still call it: the config value, the request
    function, and the provider branch."""
    import config.settings as settings
    import src.agent.web_search as ws

    assert not hasattr(settings, "TAVILY_API_KEY")
    assert not hasattr(ws, "_tavily_search")
    assert not hasattr(ws, "_TAVILY_SEARCH_URL")

    monkeypatch.setattr(ws, "WEB_SEARCH_PROVIDER", "tavily")
    with pytest.raises(RuntimeError, match="duckduckgo.*searxng"):
        ws.web_search("TCMB faiz")
