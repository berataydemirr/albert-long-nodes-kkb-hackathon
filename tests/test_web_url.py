from io import BytesIO

import pandas as pd
import pytest
from PIL import Image

from src.agent import web_url
from src.agent.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


class FakeResponse:
    def __init__(self, body, content_type="text/plain", status=200, headers=None):
        self.body, self.status_code = body, status
        self.headers = {"Content-Type": content_type, **(headers or {})}

    def iter_content(self, chunk_size):
        yield self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise web_url.requests.HTTPError("bad response")

    def close(self):
        pass


def _public_url(monkeypatch):
    monkeypatch.setattr(web_url, "_validate_public_url", lambda url: url)


def test_extracts_html_and_removes_script(monkeypatch):
    _public_url(monkeypatch)
    monkeypatch.setattr(web_url.requests, "get", lambda *args, **kwargs: FakeResponse(b"<title>Rapor</title><p>Gelir 42</p><script>ignore</script>", "text/html"))
    result = web_url.web_url_extract("https://example.com/report")
    assert result["kind"] == "html"
    assert result["title"] == "Rapor"
    assert "Gelir 42" in result["text"] and "ignore" not in result["text"]
    assert result["content_trust"] == "untrusted_remote_content"


def test_extracts_excel_preview(monkeypatch):
    _public_url(monkeypatch)
    buffer = BytesIO()
    pd.DataFrame({"Dönem": ["2026-01"], "Değer": [42]}).to_excel(buffer, index=False)
    monkeypatch.setattr(web_url.requests, "get", lambda *args, **kwargs: FakeResponse(buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
    result = web_url.web_url_extract("https://example.com/report.xlsx")
    assert result["kind"] == "excel"
    assert result["sheets"][0]["rows"] == [["2026-01", 42]]


def test_extracts_image_text_with_ocr(monkeypatch):
    _public_url(monkeypatch)
    buffer = BytesIO()
    Image.new("RGB", (20, 10), "white").save(buffer, format="PNG")
    monkeypatch.setattr(web_url.requests, "get", lambda *args, **kwargs: FakeResponse(buffer.getvalue(), "image/png"))
    monkeypatch.setattr(web_url.pytesseract, "image_to_string", lambda image, lang: "Kredi 42")
    result = web_url.web_url_extract("https://example.com/chart.png")
    assert (result["kind"], result["ocr_status"], result["text"]) == ("image", "extracted", "Kredi 42")
    assert (result["width"], result["height"]) == (20, 10)


def test_blocks_private_network_target():
    with pytest.raises(ValueError, match="private or reserved"):
        web_url._validate_public_url("http://127.0.0.1/private")


def test_tool_is_registered():
    assert TOOL_FUNCTIONS["web_url_extract"] is web_url.web_url_extract
    assert "web_url_extract" in {schema["function"]["name"] for schema in TOOL_SCHEMAS}
