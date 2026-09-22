import json
import pytest
from fastapi.testclient import TestClient

import web_app


@pytest.fixture(autouse=True)
def _demo_auth_off_by_default(monkeypatch):
    """Real bug found live: this machine's .env has actual DEMO_AUTH_USER/
    PASS set for an active tunnel demo, and since web_app imports them at
    module level, every test in this file started 401ing (TestClient's
    default Host is 'testserver', not 127.0.0.1, so the local-bypass in
    demo_basic_auth doesn't apply to it either) - 13 unrelated tests broke
    at once. Auth-off is the correct default for a test run regardless of
    what happens to be in the developer's own .env; the handful of tests
    that specifically exercise the auth gate override this explicitly."""
    monkeypatch.setattr(web_app, "DEMO_AUTH_USER", None)
    monkeypatch.setattr(web_app, "DEMO_AUTH_PASS", None)


def test_browser_ui_is_served():
    response = TestClient(web_app.app).get("/")

    assert response.status_code == 200
    assert "Albert Long Nodes" in response.text
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
    monkeypatch.setattr(web_app, "ask", lambda question, history, history_digest, return_trace, return_timing: (
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


def test_ask_endpoint_forwards_history_to_the_orchestrator(monkeypatch):
    """The browser tab resends its own (question, answer) transcript on
    every call - the server itself holds no session state - so /api/ask
    must pass whatever `history` it was given straight through to ask()."""
    received = {}

    def fake_ask(question, history, history_digest, return_trace, return_timing):
        received["history"] = history
        received["history_digest"] = history_digest
        return "ok", [], {"timeline": [], "summary": {
            "total_ms": 0.0, "llm_ms": 0.0, "tool_ms": 0.0, "overhead_ms": 0.0,
            "llm_calls": 0, "tool_calls": 0, "rounds": 0, "per_tool": [],
        }}

    monkeypatch.setattr(web_app, "ask", fake_ask)

    response = TestClient(web_app.app).post("/api/ask", json={
        "question": "peki ya geçen çeyrek?",
        "history": [{"question": "konut kredisi faizi nedir?", "answer": "%42.5"}],
        "history_digest": "araçlar: sql_query; seriler: TP.KREDI.KONUT",
    })

    assert response.status_code == 200
    assert received["history"] == [{"question": "konut kredisi faizi nedir?", "answer": "%42.5"}]
    assert received["history_digest"] == "araçlar: sql_query; seriler: TP.KREDI.KONUT"


def test_ask_endpoint_rejects_blank_questions():
    response = TestClient(web_app.app).post("/api/ask", json={"question": "   "})

    assert response.status_code == 422


def test_page_declares_a_title_and_icon():
    """Both were missing - the browser tab rendered with no name at all."""
    response = TestClient(web_app.app).get("/")

    assert "<title>Albert Long Nodes</title>" in response.text
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


def test_page_has_a_dark_theme_token_block():
    """Dark mode needs its own validated tokens, not an automatic invert of
    the light ones - the light accent fails outright as dark-mode text
    (contrast-checked at 2.0-2.15:1 against the dark surface)."""
    response = TestClient(web_app.app).get("/")

    assert 'data-theme="dark"' in response.text
    assert "prefers-color-scheme: dark" in response.text
    # The dark-mode accent must differ from the light one - reusing #1e4a8b
    # as text/line colour on a dark surface is the specific mistake this
    # guards against.
    assert "#5b9bdc" in response.text


def test_classic_page_also_has_a_dark_theme():
    response = TestClient(web_app.app).get("/classic")

    assert 'data-theme="dark"' in response.text
    assert "#5b9bdc" in response.text


def test_upload_endpoint_imports_a_real_csv(tmp_path, monkeypatch):
    import src.agent.data_import as di

    monkeypatch.setattr(di, "LAKEHOUSE_DB_PATH", tmp_path / "web_upload_test.duckdb")

    csv_bytes = b"Il,Deger\nAnkara,1\nIstanbul,2\n"
    response = TestClient(web_app.app).post(
        "/api/upload",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
        data={"dataset_name": "web_upload_test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_name"] == "web_upload_test"
    assert body["row_count"] == 2
    assert body["file_name"] == "test.csv"  # the ORIGINAL name, not the server-side random one


def test_upload_endpoint_rejects_an_unsupported_extension():
    response = TestClient(web_app.app).post(
        "/api/upload",
        files={"file": ("notes.txt", b"just text", "text/plain")},
    )

    assert response.status_code == 422


def test_upload_endpoint_rejects_an_oversized_file(monkeypatch):
    import web_app as wa

    monkeypatch.setattr(wa, "MAX_FILE_BYTES", 10)
    response = TestClient(web_app.app).post(
        "/api/upload",
        files={"file": ("big.csv", b"a,b\n1,2\n3,4\n5,6\n", "text/csv")},
    )

    assert response.status_code == 422


def _make_cover_page_pdf(path):
    """A table on page 1, with a text-only cover page before it - the
    normal shape of a BDDK/TCMB bulletin (title page, table of contents),
    not a hand-picked edge case."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

    doc = SimpleDocTemplate(str(path), pagesize=A4)
    elements = [
        Paragraph("ENFLASYON RAPORU - kapak sayfasi, tablo yok.", getSampleStyleSheet()["Normal"]),
        PageBreak(),
    ]
    t = Table([["Emtia", "Yillik % Degisim"], ["Dogal Gaz (Avrupa)", "72,6"], ["Brent Petrol", "29,9"]])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    elements.append(t)
    doc.build(elements)


def test_upload_endpoint_skips_a_tableless_cover_page_when_no_page_is_given(tmp_path, monkeypatch):
    """Real repro: the frontend's upload form never sends page/table_index
    (static/design/index.html's FormData only carries file/dataset_name),
    so a hardcoded page=0 default meant every report with a non-tabular
    cover page (the normal shape of a corporate bulletin) failed outright
    on the first upload attempt. Page/table_index left unset must now
    auto-select the first real table pdfplumber can find, instead of
    blindly assuming page 0."""
    import src.agent.data_import as di

    monkeypatch.setattr(di, "LAKEHOUSE_DB_PATH", tmp_path / "web_upload_cover_test.duckdb")
    pdf_path = tmp_path / "bulletin.pdf"
    _make_cover_page_pdf(pdf_path)

    response = TestClient(web_app.app).post(
        "/api/upload",
        files={"file": ("bulletin.pdf", pdf_path.read_bytes(), "application/pdf")},
        data={"dataset_name": "cover_test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 2
    assert body["pdf_page_auto_selected"]["page"] == 1, "page 0 (the cover) must be skipped, not imported as if it were data"
    assert body["pdf_page_auto_selected"]["table_index"] == 0


def test_upload_endpoint_skips_a_malformed_table_and_tries_the_next_candidate(tmp_path, monkeypatch):
    """Real repro: on a real 65-page report, the FIRST table list_pdf_tables
    finds (after the cover page fix above) had a blank column header -
    pdfplumber mis-splitting a merged cell, import_pdf_table's own
    pre-existing, correct refusal. Stopping at the first candidate would
    have made 'no page given' fail just as often as the old hardcoded
    page=0 did. Auto-select must cascade to the next candidate instead."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import PageBreak, SimpleDocTemplate, Table, TableStyle
    import src.agent.data_import as di

    monkeypatch.setattr(di, "LAKEHOUSE_DB_PATH", tmp_path / "web_upload_cascade_test.duckdb")
    pdf_path = tmp_path / "bulletin3.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    # Exactly one data row: too short for the merged-header-flattening
    # fallback to even attempt reconstructing the blank cell (it needs a
    # 3rd row to borrow a sub-header from), so this reliably hits the
    # "blank column header" rejection instead of being silently repaired.
    bad = Table([["", "Deger"], ["a", "1"]])
    bad.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    good = Table([["Il", "Deger"], ["Ankara", "1"], ["Istanbul", "2"], ["Izmir", "3"]])
    good.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    doc.build([bad, PageBreak(), good])

    response = TestClient(web_app.app).post(
        "/api/upload",
        files={"file": ("bulletin3.pdf", pdf_path.read_bytes(), "application/pdf")},
        data={"dataset_name": "cascade_test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 3, "must land on the good 3-row table, not the malformed one"
    assert body["pdf_page_auto_selected"]["page"] == 1
    assert len(body["pdf_auto_select_skipped"]) == 1
    assert "blank column header" in body["pdf_auto_select_skipped"][0]["error"]


def test_upload_endpoint_skips_a_trivial_one_column_fragment(tmp_path, monkeypatch):
    """Real repro: on a real report, auto-select's first CLEAN import (no
    header ambiguity) was a 2-row x 1-column fragment - a stray box on the
    page, not real data - while a real 8x28 data table sat a few
    candidates later. A 1-column grid has no header to mis-split, so it
    always passes the blank-header check trivially; a size floor is
    needed on top of "imports without error"."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import PageBreak, SimpleDocTemplate, Table, TableStyle
    import src.agent.data_import as di

    monkeypatch.setattr(di, "LAKEHOUSE_DB_PATH", tmp_path / "web_upload_tiny_test.duckdb")
    pdf_path = tmp_path / "bulletin4.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    tiny = Table([["42"], ["7"]])  # 1 column, no ambiguity, "imports cleanly" but is not data
    tiny.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    real = Table([["Il", "Deger"], ["Ankara", "1"], ["Istanbul", "2"], ["Izmir", "3"]])
    real.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    doc.build([tiny, PageBreak(), real])

    response = TestClient(web_app.app).post(
        "/api/upload",
        files={"file": ("bulletin4.pdf", pdf_path.read_bytes(), "application/pdf")},
        data={"dataset_name": "tiny_skip_test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 3, "must land on the real table, not the 1-column fragment"
    assert body["pdf_page_auto_selected"]["page"] == 1


def test_upload_endpoint_still_honors_an_explicit_page_choice(tmp_path, monkeypatch):
    """The auto-select fallback must never override a caller that DID pick
    a page - only fill in for one that left it unset."""
    import src.agent.data_import as di

    monkeypatch.setattr(di, "LAKEHOUSE_DB_PATH", tmp_path / "web_upload_explicit_test.duckdb")
    pdf_path = tmp_path / "bulletin2.pdf"
    _make_cover_page_pdf(pdf_path)

    response = TestClient(web_app.app).post(
        "/api/upload",
        files={"file": ("bulletin2.pdf", pdf_path.read_bytes(), "application/pdf")},
        data={"dataset_name": "explicit_test", "page": "1", "table_index": "0"},
    )

    assert response.status_code == 200
    assert "pdf_page_auto_selected" not in response.json()


def test_upload_endpoint_defaults_a_dataset_name_from_the_filename(tmp_path, monkeypatch):
    """dataset_name is optional in the form - a caller that only picks a
    file should still get a usable, valid table name back."""
    import src.agent.data_import as di

    monkeypatch.setattr(di, "LAKEHOUSE_DB_PATH", tmp_path / "web_upload_test2.duckdb")

    response = TestClient(web_app.app).post(
        "/api/upload",
        files={"file": ("İl Hastane Verisi!.csv", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code == 200
    name = response.json()["dataset_name"]
    import re
    assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name), f"generated name {name!r} is not a valid table name"


# --- opt-in demo Basic Auth (config.DEMO_AUTH_USER/PASS) -------------------

def test_no_auth_required_when_demo_auth_unset(monkeypatch):
    """Default state (normal local dev, nothing in .env): no credentials
    needed - the middleware must not lock out a plain `uvicorn web_app:app`."""
    monkeypatch.setattr(web_app, "DEMO_AUTH_USER", None)
    monkeypatch.setattr(web_app, "DEMO_AUTH_PASS", None)

    response = TestClient(web_app.app).get("/health")
    assert response.status_code == 200


def test_auth_rejects_missing_credentials_when_enabled(monkeypatch):
    monkeypatch.setattr(web_app, "DEMO_AUTH_USER", "demo")
    monkeypatch.setattr(web_app, "DEMO_AUTH_PASS", "sekret123")

    response = TestClient(web_app.app).get("/health")
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Basic")


def test_auth_skipped_for_direct_localhost_access(monkeypatch):
    """A developer hitting the server directly (Host: 127.0.0.1:8000, the
    real case reported live: the auth prompt showed up in their OWN
    browser too, not just for tunnel visitors) must never see a login
    prompt - only a request whose Host header is the public tunnel
    hostname should be gated."""
    monkeypatch.setattr(web_app, "DEMO_AUTH_USER", "demo")
    monkeypatch.setattr(web_app, "DEMO_AUTH_PASS", "sekret123")

    response = TestClient(web_app.app, base_url="http://127.0.0.1:8000").get("/health")
    assert response.status_code == 200


def test_auth_still_required_for_a_non_local_host_header(monkeypatch):
    """cloudflared connects to the origin from 127.0.0.1 too, so the
    client peer address can't distinguish a tunnel visitor from the
    developer - only the Host header (forwarded unchanged from the
    public request) can. A non-local Host must still be gated."""
    monkeypatch.setattr(web_app, "DEMO_AUTH_USER", "demo")
    monkeypatch.setattr(web_app, "DEMO_AUTH_PASS", "sekret123")

    response = TestClient(web_app.app, base_url="https://some-tunnel.trycloudflare.com").get("/health")
    assert response.status_code == 401


def test_auth_rejects_wrong_credentials_when_enabled(monkeypatch):
    monkeypatch.setattr(web_app, "DEMO_AUTH_USER", "demo")
    monkeypatch.setattr(web_app, "DEMO_AUTH_PASS", "sekret123")

    response = TestClient(web_app.app).get("/health", auth=("demo", "yanlis-sifre"))
    assert response.status_code == 401


def test_auth_accepts_correct_credentials_when_enabled(monkeypatch):
    monkeypatch.setattr(web_app, "DEMO_AUTH_USER", "demo")
    monkeypatch.setattr(web_app, "DEMO_AUTH_PASS", "sekret123")

    response = TestClient(web_app.app).get("/health", auth=("demo", "sekret123"))
    assert response.status_code == 200


def test_auth_covers_api_endpoints_too_not_just_health(monkeypatch):
    """The gate must not be limited to one route - /api/ask (which spends
    real MIA API quota) is exactly what a stranger with the tunnel URL
    must not reach without credentials."""
    monkeypatch.setattr(web_app, "DEMO_AUTH_USER", "demo")
    monkeypatch.setattr(web_app, "DEMO_AUTH_PASS", "sekret123")

    response = TestClient(web_app.app).post("/api/ask", json={"question": "test"})
    assert response.status_code == 401


def test_ask_endpoint_streams_deltas_then_result_as_ndjson(monkeypatch):
    def fake_ask(question, history, history_digest, return_trace, return_timing, on_event=None, stream=False):
        assert stream is True
        on_event({"kind": "llm", "round": 1})
        on_event({"kind": "delta", "round": 1, "text": "Mer"})
        on_event({"kind": "delta", "round": 1, "text": "haba"})
        return "Merhaba", [], {"timeline": [], "summary": {"total_ms": 1.0, "llm_ms": 1.0, "tool_ms": 0.0}}

    monkeypatch.setattr(web_app, "ask", fake_ask)
    response = TestClient(web_app.app).post("/api/ask", json={
        "question": "selam", "stream": True, "request_id": "req_stream_1"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert "".join(line["text"] for line in lines if line["type"] == "delta") == "Merhaba"
    assert lines[-1]["type"] == "result"
    assert lines[-1]["answer"] == "Merhaba"
    # Stage events still go to the poller; deltas never do.
    kinds = [e["kind"] for e in web_app.progress("req_stream_1")["events"]]
    assert kinds == ["llm"]
