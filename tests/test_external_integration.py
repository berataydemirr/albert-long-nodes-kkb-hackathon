"""External-data integration scenarios for the demo.

Five things a judge is likely to throw at the agent: a complex Excel from
a URL, a GitHub CSV, Turkish-dated data joined against gold_fact_long,
shuffled data analysed in chronological order, and a broken link.

Deterministic on purpose: the NETWORK is faked at the lowest layer
(socket.getaddrinfo + requests.get inside src/agent/web_url.py), so the
real _download path still runs end to end - GitHub blob->raw rewriting,
the private-address guard, redirect/size handling - but CI never depends
on an external host being up. The Excel scenario serves the repo's REAL
TCMB EVDS export (test_excels/), not a hand-made imitation.

Run with live logs to see exactly where a step fails:
    pytest tests/test_external_integration.py -s --log-cli-level=DEBUG
"""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path

import duckdb
import pytest
import requests

from src.agent import data_import, finance_fetch, orchestrator, tools, web_url
from src.agent.data_cleaning import parse_flexible_date

log = logging.getLogger("test_external_integration")

REAL_EVDS_EXCEL = Path("test_excels/tcmb_kredi_ve_mevduat_faiz_oranlari.xlsx")
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# --- fake network ---------------------------------------------------------

class _FakeResponse:
    def __init__(self, status: int, body: bytes = b"", content_type: str = "text/csv"):
        self.status_code = status
        self._body = body
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error: Not Found")

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass


@pytest.fixture
def net(monkeypatch):
    """routes: exact URL -> _FakeResponse. Anything unrouted is a 404.
    Hostnames ending in .invalid do not resolve; intranet.local resolves
    to a private address (the SSRF guard's case)."""
    routes: dict[str, _FakeResponse] = {}
    requested: list[str] = []

    def fake_get(url, **_kwargs):
        requested.append(url)
        response = routes.get(url, _FakeResponse(404))
        log.debug("fake network: GET %s -> %s (%d bytes)", url, response.status_code, len(response._body))
        return response

    def fake_getaddrinfo(host, port, *_args, **_kwargs):
        if host.endswith(".invalid"):
            log.debug("fake network: DNS %s -> NXDOMAIN", host)
            raise socket.gaierror("Name or service not known")
        ip = "10.0.0.5" if host == "intranet.local" else "140.82.112.4"
        log.debug("fake network: DNS %s -> %s", host, ip)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(web_url.requests, "get", fake_get)
    monkeypatch.setattr(web_url.socket, "getaddrinfo", fake_getaddrinfo)
    return routes, requested


@pytest.fixture
def lake(tmp_path, monkeypatch):
    """A throwaway lakehouse holding a small gold_fact_long, with both the
    importer and lakehouse_query pointed at it - the real one is never
    touched."""
    db = tmp_path / "lakehouse.duckdb"
    monkeypatch.setattr(data_import, "LAKEHOUSE_DB_PATH", db)
    monkeypatch.setattr(tools, "DB_PATH", db)
    con = duckdb.connect(str(db))
    con.execute("""
        CREATE TABLE gold_fact_long AS
        SELECT printf('2024-%02d', m) AS Period, 'TP.DK.USD.A.YTL' AS Seri_Kod,
               30.0 + m AS Deger
        FROM range(1, 13) t(m)
    """)
    con.close()
    log.debug("lake fixture: %s with gold_fact_long 2024-01..2024-12", db)
    return db


# --- 1. complex Excel from a URL (the repo's real TCMB EVDS export) -------

def test_real_evds_excel_from_a_url_imports_as_clean_typed_data(net, lake):
    routes, _ = net
    url = "https://evds2.tcmb.gov.tr/export/faiz.xlsx"
    routes[url] = _FakeResponse(200, REAL_EVDS_EXCEL.read_bytes(), XLSX_TYPE)

    result = finance_fetch.import_url_dataset(url)
    table = result["dataset_name"]
    log.debug("scenario 1: imported %s rows=%s quality=%s descriptions=%s",
              table, result["row_count"], result["data_quality"], result.get("series_descriptions_table"))

    # The export is ~270 weekly data rows followed by a ~600-row "Seri
    # Açıklamaları" footer. Before the footer was split off, it imported
    # as 884 rows with the first value column stored as TEXT.
    assert result["row_count"] == 270
    assert result["data_quality"]["series_description_rows_split"] > 500

    con = duckdb.connect(str(lake), read_only=True)
    try:
        value_type = con.execute(f'SELECT typeof("TP_KTF10") FROM "{table}" LIMIT 1').fetchone()[0]
        weekly_rows = con.execute(f'SELECT count(*) FROM "{table}" WHERE Period = \'2021-07\'').fetchone()[0]
        # The footer names a series dotted (TP.KTF10), the data table's
        # column is underscored (TP_KTF10) - Kolon is the bridge.
        described = dict(con.execute(
            f'SELECT Kolon, Bilgi FROM "{result["series_descriptions_table"]}" WHERE Alan = \'Açıklama\''
        ).fetchall())
        source = con.execute(
            f'SELECT Bilgi FROM "{result["series_descriptions_table"]}" '
            f'WHERE Seri_Kod = \'TP.KBH.USD.30\' AND Alan = \'Veri Kaynağı\''
        ).fetchone()
        no_nan_codes = con.execute(
            f'SELECT count(*) FROM "{result["series_descriptions_table"]}" WHERE Seri_Kod = \'nan\''
        ).fetchone()[0]
    finally:
        con.close()
    log.debug("scenario 1: TP_KTF10 type=%s, July-2021 weekly rows=%s, TP_KTF10 described as %r",
              value_type, weekly_rows, described.get("TP_KTF10"))

    assert value_type == "DOUBLE"
    # Weekly data keeps its real dates; Period is ADDED for joining, so
    # the weeks are not collapsed into one indistinguishable month.
    assert weekly_rows >= 3
    assert "İhtiyaç Kredisi" in described["TP_KTF10"]
    # The per-series metadata blocks survive too, attached to the right code.
    assert source == ("Katılım Bankaları",)
    assert no_nan_codes == 0


# --- 2. GitHub CSV -----------------------------------------------------------

def test_github_page_link_is_fetched_from_raw_and_imports(net, lake):
    """A user pastes the github.com page link (an HTML page); the fetch
    must go to raw.githubusercontent.com instead of importing HTML."""
    routes, requested = net
    raw = "https://raw.githubusercontent.com/datasets/oil/main/data/brent.csv"
    routes[raw] = _FakeResponse(200, b"Date,Price\n2024-01,80.1\n2024-02,82.5\n2024-03,85.0\n",
                                "text/plain; charset=utf-8")

    result = finance_fetch.import_url_dataset("https://github.com/datasets/oil/blob/main/data/brent.csv")
    log.debug("scenario 2: requested=%s result=%s", requested, {k: result[k] for k in ("dataset_name", "row_count", "source_url")})

    assert requested == [raw]
    assert result["row_count"] == 3
    assert result["source_url"] == raw


# --- 3. Turkish-dated external data JOINed with gold_fact_long -------------

def test_turkish_excel_style_dates_join_gold_fact_long_on_period(net, lake):
    """'Oca.24' is Turkish Excel's default month format. It used to parse
    to None, the column stayed raw text, and a Period JOIN against
    gold_fact_long silently matched nothing."""
    routes, _ = net
    months = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
    body = "Tarih;Kredi Hacmi\n" + "\n".join(f"{m}.24;{1000 + i}.{i}50,5" for i, m in enumerate(months))
    url = "https://veri.example.org/kredi.csv"
    routes[url] = _FakeResponse(200, body.encode("utf-8"), "text/csv")

    result = finance_fetch.import_url_dataset(url)
    table = result["dataset_name"]
    log.debug("scenario 3: %s quality=%s", table, result["data_quality"])
    assert "Tarih" in result["data_quality"]["date_columns_normalized"]

    joined = tools.lakehouse_query(f"""
        SELECT e.Tarih AS Period, e."Kredi Hacmi" AS kredi, g.Deger AS usd
        FROM "{table}" e JOIN gold_fact_long g ON g.Period = e.Tarih
        WHERE g.Seri_Kod = 'TP.DK.USD.A.YTL'
        ORDER BY e.Tarih
    """)
    log.debug("scenario 3: joined rows=%s first=%s", joined["row_count"], joined["rows"][:2])

    assert joined["row_count"] == 12
    assert joined["rows"][0]["Period"] == "2024-01"
    assert joined["rows"][0]["usd"] == 31.0
    # Turkish number text ("1.000,5") came through as a real number.
    assert joined["rows"][0]["kredi"] == pytest.approx(1000050.5)


# --- 4. shuffled rows -> chronological order -> analysis ------------------

def test_shuffled_mixed_format_rows_sort_chronologically_and_correlate(net, lake):
    routes, _ = net
    # Deliberately out of order and in six different real-world notations.
    rows = [("Mart 2024", 71), ("2024-01", 67), ("Haz-24", 77), ("15.02.2024", 69),
            ("2024M05", 75), ("Nis.24", 73)]
    body = "Donem,Endeks\n" + "\n".join(f"{d},{v}" for d, v in rows)
    url = "https://veri.example.org/endeks.csv"
    routes[url] = _FakeResponse(200, body.encode("utf-8"))

    table = finance_fetch.import_url_dataset(url)["dataset_name"]
    ordered = tools.lakehouse_query(f'SELECT Donem, Endeks FROM "{table}" ORDER BY Donem')["rows"]
    log.debug("scenario 4: ordered=%s", ordered)

    assert [r["Donem"] for r in ordered] == [f"2024-{m:02d}" for m in range(1, 7)]

    gold = tools.lakehouse_query(
        "SELECT Period, Deger FROM gold_fact_long WHERE Period <= '2024-06' ORDER BY Period")["rows"]
    result = tools.correlation_analysis(
        [r["Donem"] for r in ordered], [r["Endeks"] for r in ordered],
        [r["Period"] for r in gold], [r["Deger"] for r in gold],
    )
    log.debug("scenario 4: correlation=%s", {k: result.get(k) for k in ("pearson_correlation", "shared_periods")})

    # Endeks = 2*USD + 5 month by month, so a correct month alignment is r = 1.
    assert result["pearson_correlation"] == pytest.approx(1.0)


# --- 5. a broken link --------------------------------------------------------

@pytest.mark.parametrize("url,why", [
    ("https://nope.invalid/data.csv", "hostname does not resolve"),
    ("https://veri.example.org/yok.csv", "404"),
    ("https://intranet.local/secret.csv", "private address (SSRF guard)"),
])
def test_a_broken_link_raises_a_readable_error_and_writes_nothing(net, lake, url, why):
    with pytest.raises((ValueError, RuntimeError, requests.HTTPError)) as excinfo:
        finance_fetch.import_url_dataset(url)
    log.debug("scenario 5 (%s): %s: %s", why, type(excinfo.value).__name__, excinfo.value)

    assert str(excinfo.value)
    con = duckdb.connect(str(lake), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert tables == {"gold_fact_long"}


def test_an_html_page_is_refused_instead_of_imported_as_a_table(net, lake):
    routes, _ = net
    url = "https://veri.example.org/sayfa"
    routes[url] = _FakeResponse(200, b"<html><body>not data</body></html>", "text/html")

    with pytest.raises(ValueError, match="CSV/Excel"):
        finance_fetch.import_url_dataset(url)


# --- 5b. the agent itself survives a broken link -------------------------

class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}
        return {k: v for k, v in d.items() if v is not None}


class _Call:
    def __init__(self, name, arguments):
        self.id = "call_1"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _Completions:
    def __init__(self, messages):
        self._messages = list(messages)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        choice = type("C", (), {"message": self._messages.pop(0)})()
        return type("R", (), {"choices": [choice], "usage": None})()


def test_the_agent_reports_a_broken_link_instead_of_crashing(net, lake, monkeypatch):
    """End to end through ask(): the model asks to import a dead URL, the
    tool fails for real (through the fake network), and the run still
    ends with an answer - the failure reaches the model as a tool result
    it can explain, not as an exception out of the loop."""
    bad_url = "https://veri.example.org/yok.csv"
    completions = _Completions([
        _Msg(tool_calls=[_Call("import_url_dataset", json.dumps({"url": bad_url}))]),
        _Msg(content="Bağlantı çalışmıyor (404); veri içe aktarılamadı, lütfen URL'yi kontrol edin."),
    ])
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    monkeypatch.setattr(orchestrator, "get_client", lambda: client)
    monkeypatch.setattr(orchestrator, "_system_prompt", lambda addon="": "SYS")
    monkeypatch.setattr(orchestrator, "known_series_codes", lambda: set())

    answer, trace = orchestrator.ask(f"{bad_url} adresindeki veriyi içe aktar", return_trace=True)
    tool_messages = [m for m in completions.calls[1]["messages"] if m.get("role") == "tool"]
    log.debug("scenario 5b: answer=%r trace=%s tool_msgs=%s", answer, trace, tool_messages)

    assert "404" in answer
    assert trace[0]["name"] == "import_url_dataset" and trace[0]["error"]
    assert tool_messages and "404" in tool_messages[0]["content"]


# --- the date formats behind scenarios 3 and 4 -----------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Oca.24", "2024-01"), ("Şub-24", "2024-02"), ("EKİ-24", "2024-10"), ("Ağu.25", "2025-08"),
    ("Mart-2024", "2024-03"), ("15-03-2024", "2024-03"), ("03.2024", "2024-03"), ("3/2024", "2024-03"),
    ("2024M03", "2024-03"), ("Jan 2024", "2024-01"), ("Dec-23", "2023-12"),
    # formats that already worked keep working
    ("Ocak 2024", "2024-01"), ("15.03.2024", "2024-03"), ("2024 1. Çeyrek", "2024-03"), ("2024-Q1", "2024-03"),
    # never a guess
    ("13.2024", None), ("2024M13", None), ("Kasa 2024", None), ("Toplam", None),
])
def test_external_date_notations(raw, expected):
    assert parse_flexible_date(raw) == expected
