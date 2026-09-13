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
