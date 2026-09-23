"""Deposit stocks, monthly changes and CPI in one unambiguous monthly table."""

from datetime import datetime, timezone
import json

from config.settings import BRONZE_DIR, LAKEHOUSE_DB_PATH, SILVER_DIR
from src.agent.db_utils import connect_with_retry

TABLE = "gold_deposit_cpi_monthly"
BASELINE_TABLE = "gold_bddk_deposit_baseline"
BASELINE_SNAPSHOT = SILVER_DIR / "bddk_deposit_baseline_2020_12.csv"
REAL_CHANGE_VIEW = "gold_deposit_real_monthly_change"
REAL_STOCK_VIEW = "gold_deposit_real_monthly_stock"


def fetch_deposit_baseline_202012() -> dict:
    """Store the official previous-month stocks needed for Jan 2021 deltas."""
    from src.agent.bddk_live import fetch_bddk_monthly_table

    report = fetch_bddk_monthly_table(2020, 12, 10)
    if report["unit"] != "milyon TL" or report["period"] != "2020-12" or report["truncated"]:
        raise ValueError("BDDK 2020-12 tablo_10 raporu beklenen kapsamda değil.")
    by_no = {int(row["Column_1"]): row for row in report["rows"]}
    if not {1, 5} <= by_no.keys():
        raise ValueError("BDDK 2020-12 TP/YP ana satırları eksik.")
    tp, yp = by_no[1], by_no[5]
    if "Yurt İçi Yerleşik" not in tp["Column_2"] or "Yurt İçi Yerleşik" not in yp["Column_2"]:
        raise ValueError("BDDK 2020-12 TP/YP yerleşiklik satırları uyuşmuyor.")
    short = sum(int(tp[key]) for key in ("Vadesiz", "1 Aya Kadar", "1-3 Ay Arası"))
    long = sum(int(tp[key]) for key in ("3-6 Ay Arası", "6-12 Ay Arası", "1  Yıl Üzeri"))
    values = (int(tp["Toplam"]), int(yp["Toplam"]), short, long)
    if abs(values[0] - short - long) > 2:
        raise ValueError("BDDK 2020-12 TP vade toplamlarıyla genel toplam uyuşmuyor.")
    path = BRONZE_DIR / "bddk_monthly" / "2020" / "12" / "tablo_10_baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    import csv
    BASELINE_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    with BASELINE_SNAPSHOT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Period", "TP_Mevduat_Nominal", "YP_Mevduat_Nominal",
                         "TP_3AyaKadar_Nominal", "TP_3AydanFazla_Nominal", "Kaynak_URL"])
        writer.writerow(["2020-12", *values, report["source_url"]])
    con = connect_with_retry(LAKEHOUSE_DB_PATH, read_only=False)
    try:
        _ensure_baseline_table(con)
        con.execute(f"DELETE FROM {BASELINE_TABLE} WHERE Period='2020-12'")
        con.execute(f"INSERT INTO {BASELINE_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ["2020-12", *values, report["source_url"], datetime.now(timezone.utc).isoformat()])
    finally:
        con.close()
    return {"table": BASELINE_TABLE, "period": "2020-12", "values": values,
            "source_url": report["source_url"], "raw_path": str(path)}


def _ensure_baseline_table(con) -> None:
    con.execute(f"""CREATE TABLE IF NOT EXISTS {BASELINE_TABLE} (
        Period VARCHAR PRIMARY KEY,
        TP_Mevduat_Nominal DOUBLE,
        YP_Mevduat_Nominal DOUBLE,
        TP_3AyaKadar_Nominal DOUBLE,
        TP_3AydanFazla_Nominal DOUBLE,
        Kaynak_URL VARCHAR,
        Cekim_Zamani_UTC VARCHAR
    )""")
    if BASELINE_SNAPSHOT.exists() and con.execute(
            f"SELECT COUNT(*) FROM {BASELINE_TABLE} WHERE Period='2020-12'").fetchone()[0] == 0:
        import csv
        with BASELINE_SNAPSHOT.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0].get("Period") != "2020-12":
            raise ValueError("2020-12 BDDK baz anlık görüntüsü bozuk.")
        row = rows[0]
        values = [float(row[key]) for key in (
            "TP_Mevduat_Nominal", "YP_Mevduat_Nominal",
            "TP_3AyaKadar_Nominal", "TP_3AydanFazla_Nominal")]
        if min(values) <= 0 or abs(values[0] - values[2] - values[3]) > 2:
            raise ValueError("2020-12 BDDK baz tutarları uyuşmuyor.")
        con.execute(f"INSERT INTO {BASELINE_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ["2020-12", *values, row["Kaynak_URL"], "snapshot"])


def build_deposit_cpi_monthly() -> int:
    con = connect_with_retry(LAKEHOUSE_DB_PATH, read_only=False)
    try:
        _ensure_baseline_table(con)
        base = con.execute("""SELECT Deger FROM gold_fact_long
            WHERE Seri_Kod='TP.TUKFIY2025.GENEL' AND Period='2025-12'""").fetchone()
        if not base or base[0] <= 0:
            raise ValueError("2025-12 TÜFE baz ayı bulunamadı.")
        con.execute(f"""
            CREATE OR REPLACE TABLE {TABLE} AS
            WITH bddk AS (
                SELECT Period,
                    MAX(Deger) FILTER (WHERE Satir_No=1 AND Kolon='Toplam') AS TP_Mevduat_Nominal,
                    MAX(Deger) FILTER (WHERE Satir_No=5 AND Kolon='Toplam') AS YP_Mevduat_Nominal,
                    SUM(Deger) FILTER (WHERE Satir_No=1 AND Kolon IN
                        ('Vadesiz','1 Aya Kadar','1-3 Ay Arası')) AS TP_3AyaKadar_Nominal,
                    SUM(Deger) FILTER (WHERE Satir_No=1 AND Kolon IN
                        ('3-6 Ay Arası','6-12 Ay Arası','1  Yıl Üzeri')) AS TP_3AydanFazla_Nominal
                FROM gold_bddk_monthly_long WHERE Tablo='tablo_10'
                GROUP BY Period
            ), nominal AS (
                SELECT Period, TP_Mevduat_Nominal, YP_Mevduat_Nominal,
                       TP_3AyaKadar_Nominal, TP_3AydanFazla_Nominal FROM bddk
                UNION ALL
                SELECT Period, TP_Mevduat_Nominal, YP_Mevduat_Nominal,
                       TP_3AyaKadar_Nominal, TP_3AydanFazla_Nominal
                FROM gold_bddk_deposit_baseline WHERE Period='2020-12'
            ), changes AS (
                SELECT *,
                    TP_Mevduat_Nominal-LAG(TP_Mevduat_Nominal) OVER (ORDER BY Period) AS TP_Mevduat_Aylik_Degisim,
                    YP_Mevduat_Nominal-LAG(YP_Mevduat_Nominal) OVER (ORDER BY Period) AS YP_Mevduat_Aylik_Degisim,
                    TP_3AyaKadar_Nominal-LAG(TP_3AyaKadar_Nominal) OVER (ORDER BY Period) AS TP_3AyaKadar_Aylik_Degisim,
                    TP_3AydanFazla_Nominal-LAG(TP_3AydanFazla_Nominal) OVER (ORDER BY Period) AS TP_3AydanFazla_Aylik_Degisim
                FROM nominal
            ), aligned AS (
                SELECT x.*, c.Deger AS TUFE_2025_100
                FROM changes x JOIN gold_fact_long c ON x.Period=c.Period
                AND c.Seri_Kod='TP.TUKFIY2025.GENEL'
            )
            SELECT *,
                TP_Mevduat_Nominal * ? / TUFE_2025_100 AS TP_Mevduat_Reel,
                YP_Mevduat_Nominal * ? / TUFE_2025_100 AS YP_Mevduat_Reel,
                TP_3AyaKadar_Nominal * ? / TUFE_2025_100 AS TP_3AyaKadar_Reel,
                TP_3AydanFazla_Nominal * ? / TUFE_2025_100 AS TP_3AydanFazla_Reel,
                TP_Mevduat_Aylik_Degisim * ? / TUFE_2025_100 AS TP_Mevduat_Aylik_Degisim_Reel,
                YP_Mevduat_Aylik_Degisim * ? / TUFE_2025_100 AS YP_Mevduat_Aylik_Degisim_Reel,
                TP_3AyaKadar_Aylik_Degisim * ? / TUFE_2025_100 AS TP_3AyaKadar_Aylik_Degisim_Reel,
                TP_3AydanFazla_Aylik_Degisim * ? / TUFE_2025_100 AS TP_3AydanFazla_Aylik_Degisim_Reel,
                '2025-12' AS Reel_Fiyat_Baz_Ayi,
                'milyon TL' AS Mevduat_Birimi
            FROM aligned ORDER BY Period
        """, [base[0]] * 8)
        count, missing = con.execute(f"""SELECT COUNT(*), COUNT(*) FILTER
            (WHERE TUFE_2025_100 IS NULL OR TUFE_2025_100<=0 OR
             TP_Mevduat_Nominal IS NULL OR YP_Mevduat_Nominal IS NULL OR
             TP_3AyaKadar_Nominal IS NULL OR TP_3AydanFazla_Nominal IS NULL)
            FROM {TABLE}""").fetchone()
        if missing:
            raise ValueError(f"{missing} birleşik mevduat/TÜFE satırı eksik.")
        # These two presentation views match the requested two-dataset
        # layout. The CPI shown in the reference table has two decimals;
        # use that displayed series consistently in its derived columns.
        # The canonical table above retains the full-precision calculation.
        for view, suffix in ((REAL_CHANGE_VIEW, "_Aylik_Degisim"),
                             (REAL_STOCK_VIEW, "_Nominal")):
            con.execute(f"""
                CREATE OR REPLACE VIEW {view} AS
                SELECT REPLACE(Period, '-', '') AS Tarih,
                    ROUND(TUFE_2025_100, 2) AS TUFE,
                    ROUND(TP_Mevduat{suffix} *
                        (SELECT ROUND(TUFE_2025_100, 2) FROM {TABLE}
                         WHERE Period='2025-12') / ROUND(TUFE_2025_100, 2), 0)
                        AS TP_Mevduat,
                    ROUND(YP_Mevduat{suffix} *
                        (SELECT ROUND(TUFE_2025_100, 2) FROM {TABLE}
                         WHERE Period='2025-12') / ROUND(TUFE_2025_100, 2), 0)
                        AS YP_Mevduat,
                    ROUND(TP_3AyaKadar{suffix} *
                        (SELECT ROUND(TUFE_2025_100, 2) FROM {TABLE}
                         WHERE Period='2025-12') / ROUND(TUFE_2025_100, 2), 0)
                        AS TP_3AyaKadar_Mevduat,
                    ROUND(TP_3AydanFazla{suffix} *
                        (SELECT ROUND(TUFE_2025_100, 2) FROM {TABLE}
                         WHERE Period='2025-12') / ROUND(TUFE_2025_100, 2), 0)
                        AS TP_3AydanFazla_Mevduat
                FROM {TABLE}
            """)
        return count
    finally:
        con.close()
