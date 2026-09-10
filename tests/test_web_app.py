from fastapi.testclient import TestClient

import web_app


def test_browser_ui_is_served():
    response = TestClient(web_app.app).get("/")

    assert response.status_code == 200
    assert "KKB Agentic Analytics" in response.text
    assert "/api/ask" in response.text


def test_ask_endpoint_returns_agent_answer(monkeypatch):
    monkeypatch.setattr(web_app, "ask", lambda question: f"yanıt: {question}")

    response = TestClient(web_app.app).post("/api/ask", json={"question": "  güncel faiz  "})

    assert response.status_code == 200
    assert response.json() == {"answer": "yanıt: güncel faiz"}


def test_ask_endpoint_rejects_blank_questions():
    response = TestClient(web_app.app).post("/api/ask", json={"question": "   "})

    assert response.status_code == 422
