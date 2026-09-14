from fastapi.testclient import TestClient

import web_app


def test_browser_ui_is_served():
    response = TestClient(web_app.app).get("/")

    assert response.status_code == 200
    assert "KKB Agent" in response.text
    assert "/api/ask" in response.text
    assert "/api/tools" in response.text
    # Trace drawer sections rendered from the /api/ask timing payload.
    assert "Tool calls & timing" in response.text
    assert "Timeline" in response.text
    assert "Per tool" in response.text
    assert "Lakehouse'a ne sormak istersin?" in response.text


def test_classic_ui_is_still_served():
    response = TestClient(web_app.app).get("/classic")

    assert response.status_code == 200
    assert "KKB Agentic Analytics" in response.text
    assert "Örnek Promptlar" in response.text
    assert "Kullanılabilir Tool'lar" in response.text


def test_tool_registry_endpoint_includes_c4_and_c6():
    response = TestClient(web_app.app).get("/api/tools")

    assert response.status_code == 200
    body = response.json()
    names = {tool["name"] for tool in body["tools"]}
    assert body["count"] == len(body["tools"])
    assert {"anomaly_analysis", "change_detection"} <= names


def test_ask_endpoint_returns_agent_answer(monkeypatch):
    monkeypatch.setattr(web_app, "ask", lambda question, return_trace, return_timing: (
        f"yanıt: {question}",
        [{"round": 1, "name": "web_search", "arguments": '{"query":"faiz"}', "error": None,
          "output_preview": "{}", "start_ms": 12.0, "duration_ms": 340.0}],
        {
            "timeline": [
                {"kind": "llm", "round": 1, "name": "LLM düşünme (round 1)", "start_ms": 0.0, "duration_ms": 12.0, "error": None, "tokens": 40},
                {"kind": "tool", "round": 1, "name": "web_search", "start_ms": 12.0, "duration_ms": 340.0, "error": None},
            ],
            "summary": {"total_ms": 360.0, "llm_ms": 12.0, "tool_ms": 340.0, "overhead_ms": 8.0,
                        "llm_calls": 1, "tool_calls": 1, "rounds": 1,
                        "per_tool": [{"name": "web_search", "calls": 1, "duration_ms": 340.0, "errors": 0}]},
        },
    ))

    response = TestClient(web_app.app).post("/api/ask", json={"question": "  güncel faiz  "})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "yanıt: güncel faiz"
    assert body["route"]["name"] == "web_search"
    assert "web_search" in body["route"]["allowed_tools"]
    assert body["trace"][0]["name"] == "web_search"
    assert body["trace"][0]["duration_ms"] == 340.0
    assert body["timing"]["summary"]["total_ms"] == 360.0
    assert [e["kind"] for e in body["timing"]["timeline"]] == ["llm", "tool"]


def test_ask_endpoint_rejects_blank_questions():
    response = TestClient(web_app.app).post("/api/ask", json={"question": "   "})

    assert response.status_code == 422


def test_overview_endpoint_reports_gold_coverage():
    """The landing page reads its coverage strip from here, so an empty or
    malformed payload silently blanks that strip rather than erroring."""
    response = TestClient(web_app.app).get("/api/overview")

    assert response.status_code == 200
    payload = response.json()
    assert "available" in payload

    if payload["available"]:
        assert payload["series_count"] > 0
        assert payload["row_count"] >= payload["series_count"]
        assert payload["first_period"] <= payload["last_period"]
        assert payload["sources"]
        assert all({"kaynak", "series"} <= set(src) for src in payload["sources"])


def test_overview_never_500s_when_the_database_is_missing(monkeypatch):
    """A fresh clone has no built database. The page must still load, so the
    endpoint reports unavailability instead of failing the request."""
    def boom():
        raise FileNotFoundError("gold database not found")

    monkeypatch.setattr(web_app, "gold_summary", boom)
    response = TestClient(web_app.app).get("/api/overview")

    assert response.status_code == 200
    assert response.json()["available"] is False


def test_page_declares_a_title_and_icon():
    """Both were missing - the browser tab rendered with no name at all."""
    response = TestClient(web_app.app).get("/")

    assert "<title>KKB Agent</title>" in response.text
    assert 'rel="icon"' in response.text


def test_page_uses_the_bogazici_palette_and_typeface():
    response = TestClient(web_app.app).get("/")

    assert "#1e4a8b" in response.text   # PANTONE 7686C
    assert "#8cc8ea" in response.text   # PANTONE 297C
    assert "Sora" in response.text
    assert "Space Mono" in response.text
    # The old dark/amber theme must be fully gone, not merely overridden.
    assert "#e0aa4a" not in response.text
    assert "#0a0a0b" not in response.text
