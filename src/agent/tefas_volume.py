"""Fetch TEFAS's official monthly total-volume report into the lakehouse."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import re

import pandas as pd
import requests

from config.settings import BRONZE_DIR, LAKEHOUSE_DB_PATH, SILVER_DIR
from src.agent.db_utils import connect_with_retry

SOURCE_URL = "https://www.tefas.gov.tr/api/statistics/tefas/getFplToplamIslemHacmi"
REPORT_URL = "https://www.tefas.gov.tr/tr/tefas-istatistikleri?tab=volume"
TABLE = "gold_tefas_monthly_volume"
SNAPSHOT = SILVER_DIR / "tefas_volume_2021_01_2025_12.csv"
_PERIOD = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _month_range(start: str, end: str) -> list[str]:
    if not _PERIOD.fullmatch(start) or not _PERIOD.fullmatch(end) or start > end:
        raise ValueError("Dönemler sıralı YYYY-MM biçiminde olmalı.")
    periods = pd.period_range(start, end, freq="M").astype(str).tolist()
    if len(periods) > 120:
        raise ValueError("Tek sorgu en fazla 120 ay içermeli.")
    return periods


def _amount(row: dict, key: str, period: str) -> str:
    value = row.get(key)
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError(f"TEFAS {period}: {key} sayısal değil: {value!r}") from None
    if not number.is_finite() or number < 0 or number.as_tuple().exponent < -2:
        raise ValueError(f"TEFAS {period}: {key} geçersiz para tutarı: {value!r}")
    return str(number)


def _validated_rows(payload: dict, expected: list[str]) -> list[dict]:
    if not isinstance(payload, dict) or payload.get("errorCode") is not None:
        raise ValueError(f"TEFAS rapor hatası: {payload.get('errorMessage') if isinstance(payload, dict) else payload!r}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("TEFAS raporunda data listesi yok.")
    rows = {}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("TEFAS raporunda beklenmeyen satır tipi.")
        period = item.get("donem")
        if period == "Toplam":
            continue  # report footer, not a 61st month
        if period not in expected or period in rows:
            raise ValueError(f"TEFAS beklenmeyen veya yinelenen dönem: {period!r}")
        rows[period] = {
            "Period": period,
            "Toplam_Islem_Hacmi_TL": _amount(item, "toplamIslemHacmi", period),
            "Net_Takas_Tutari_TL": _amount(item, "netTakasTutar", period),
        }
    missing = sorted(set(expected) - set(rows))
    if missing:
        raise ValueError(f"TEFAS eksik aylar: {missing}")
    return [rows[period] for period in expected]


def fetch_tefas_volume(start_period: str, end_period: str) -> dict:
    """One official POST retrieves both TL measures for the whole month range.

    The page itself may show 'Request Rejected'; the report JSON endpoint
    is independently checked for HTTP, schema, month coverage and amounts.
    No existing lakehouse rows are changed until every check succeeds.
    """
    expected = _month_range(start_period, end_period)
    start_year, start_month = start_period.split("-")
    end_year, end_month = end_period.split("-")
    body = {
        "basYil": start_year, "basAy": start_month, "basHafta": "",
        "bitYil": end_year, "bitAy": end_month, "bitHafta": "",
        "paraBirimi": "TL", "dil": "TR",
    }
    response = requests.post(
        SOURCE_URL, json=body, timeout=30,
        headers={"Origin": "https://www.tefas.gov.tr", "Referer": REPORT_URL,
                 "Accept": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    if "json" not in response.headers.get("content-type", "").lower():
        raise ValueError("TEFAS JSON yerine farklı içerik döndürdü.")
    payload = json.loads(response.text, parse_float=Decimal)
    rows = _validated_rows(payload, expected)
    fetched_at = datetime.now(timezone.utc).isoformat()

    # Keep the exact response and parameters for later audit. Decimal is
    # serialized as a string; the gold table stores monetary values as cents.
    raw_path = BRONZE_DIR / "tefas" / f"volume_{start_period}_{end_period}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({"request": body, "response": payload,
                                    "fetched_at_utc": fetched_at}, ensure_ascii=False,
                                   indent=2, default=str), encoding="utf-8")

    if start_period == "2021-01" and end_period == "2025-12":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(SNAPSHOT, index=False)

    con = connect_with_retry(LAKEHOUSE_DB_PATH, read_only=False)
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            Period VARCHAR PRIMARY KEY,
            Toplam_Islem_Hacmi_TL DECIMAL(22,2) NOT NULL,
            Net_Takas_Tutari_TL DECIMAL(22,2) NOT NULL,
            Kaynak_URL VARCHAR NOT NULL,
            Cekim_Zamani_UTC VARCHAR NOT NULL
        )""")
        con.execute(f"DELETE FROM {TABLE} WHERE Period BETWEEN ? AND ?",
                    [start_period, end_period])
        frame = pd.DataFrame(rows)
        frame["Kaynak_URL"] = SOURCE_URL
        frame["Cekim_Zamani_UTC"] = fetched_at
        con.register("tefas_rows", frame)
        con.execute(f"""INSERT INTO {TABLE}
            SELECT Period, CAST(Toplam_Islem_Hacmi_TL AS DECIMAL(22,2)),
                   CAST(Net_Takas_Tutari_TL AS DECIMAL(22,2)), Kaynak_URL,
                   Cekim_Zamani_UTC FROM tefas_rows""")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    # The worked savings comparison has a fixed 2021-01..2025-12 scope.
    # Refresh it whenever its whole TEFAS range is present; partial live
    # fetches remain available in the source table without fabricating gaps.
    con = connect_with_retry(LAKEHOUSE_DB_PATH, read_only=True)
    try:
        covered = con.execute(f"""SELECT COUNT(*) FROM {TABLE}
            WHERE Period BETWEEN '2021-01' AND '2025-12'""").fetchone()[0]
    finally:
        con.close()
    comparison = None
    if covered == 60:
        from src.gold.build_savings_tefas import build_savings_tefas_monthly
        comparison = build_savings_tefas_monthly()
    return {"table": TABLE, "source_url": SOURCE_URL, "raw_path": str(raw_path),
            "start_period": start_period, "end_period": end_period,
            "row_count": len(rows), "missing_periods": [],
            "columns": ["Period", "Toplam_Islem_Hacmi_TL", "Net_Takas_Tutari_TL"],
            "unit": "TL", "frequency": "monthly_flow", "fetched_at_utc": fetched_at,
            "comparison": comparison}


def load_tefas_snapshot() -> int:
    """Rebuild the validated public 60-month report without network access."""
    if not SNAPSHOT.exists():
        return 0
    frame = pd.read_csv(SNAPSHOT, dtype=str)
    required = {"Period", "Toplam_Islem_Hacmi_TL", "Net_Takas_Tutari_TL"}
    if set(frame.columns) != required:
        raise ValueError("TEFAS anlık görüntüsü kolonları uyuşmuyor.")
    expected = _month_range("2021-01", "2025-12")
    if frame["Period"].tolist() != expected:
        raise ValueError("TEFAS anlık görüntüsü 60 ayın tamamını sırayla içermiyor.")
    for row in frame.to_dict("records"):
        for key in ("Toplam_Islem_Hacmi_TL", "Net_Takas_Tutari_TL"):
            _amount(row, key, row["Period"])
    frame["Kaynak_URL"] = SOURCE_URL
    frame["Cekim_Zamani_UTC"] = "public_snapshot"
    con = connect_with_retry(LAKEHOUSE_DB_PATH, read_only=False)
    try:
        con.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            Period VARCHAR PRIMARY KEY,
            Toplam_Islem_Hacmi_TL DECIMAL(22,2) NOT NULL,
            Net_Takas_Tutari_TL DECIMAL(22,2) NOT NULL,
            Kaynak_URL VARCHAR NOT NULL,
            Cekim_Zamani_UTC VARCHAR NOT NULL
        )""")
        con.register("tefas_snapshot_rows", frame)
        con.execute(f"DELETE FROM {TABLE} WHERE Period BETWEEN '2021-01' AND '2025-12'")
        con.execute(f"""INSERT INTO {TABLE}
            SELECT Period, CAST(Toplam_Islem_Hacmi_TL AS DECIMAL(22,2)),
                   CAST(Net_Takas_Tutari_TL AS DECIMAL(22,2)), Kaynak_URL,
                   Cekim_Zamani_UTC FROM tefas_snapshot_rows""")
        return len(frame)
    finally:
        con.close()
