"""Monthly BDDK/EVDS/TEFAS comparison with explicit stock/flow semantics."""

from __future__ import annotations

from config.settings import LAKEHOUSE_DB_PATH
from src.agent.db_utils import connect_with_retry

TABLE = "gold_savings_tefas_monthly"


def build_savings_tefas_monthly(start: str = "2021-01", end: str = "2025-12",
                                base: str = "2021-01", prices: str = "2025-12") -> dict:
    """Create a checked 60-month comparison, indexed on real TL values.

    BDDK deposits are month-end stocks in million TL; TEFAS volumes are
    monthly flows in TL. Indexes show their relative paths, not economic
    equivalence. The exact CPI is used before display rounding.
    """
    con = connect_with_retry(LAKEHOUSE_DB_PATH, read_only=False)
    try:
        months = con.execute("SELECT date_diff('month', ?::DATE, ?::DATE) + 1",
                             [start + "-01", end + "-01"]).fetchone()[0]
        if months <= 0 or not (start <= base <= end and start <= prices <= end):
            raise ValueError("Başlangıç, bitiş, endeks ve fiyat baz ayları uyumsuz.")
        sql = f"""
            CREATE OR REPLACE TEMP TABLE savings_work AS
            WITH b AS (
                SELECT Period,
                    MAX(Deger) FILTER (WHERE Satir_No=1 AND Kolon='Toplam') AS TP_Mevduat_Mn_TL,
                    MAX(Deger) FILTER (WHERE Satir_No=5 AND Kolon='Toplam') AS YP_Mevduat_Mn_TL,
                    SUM(Deger) FILTER (WHERE Satir_No=1 AND Kolon IN
                        ('Vadesiz','1 Aya Kadar','1-3 Ay Arası')) AS TP_3AyaKadar_Mn_TL,
                    SUM(Deger) FILTER (WHERE Satir_No=1 AND Kolon IN
                        ('3-6 Ay Arası','6-12 Ay Arası','1  Yıl Üzeri')) AS TP_3AydanFazla_Mn_TL
                FROM gold_bddk_monthly_long
                WHERE Tablo='tablo_10' AND Period BETWEEN ? AND ?
                GROUP BY Period
            ), macro AS (
                SELECT Period,
                    MAX(Deger) FILTER (WHERE Seri_Kod='TP.TUKFIY2025.GENEL') AS TUFE_2025_100,
                    MAX(Deger) FILTER (WHERE Seri_Kod='TP.MT210AGS.TRY.MT02') AS TP_Faiz_3AyaKadar_Yuzde,
                    MAX(Deger) FILTER (WHERE Seri_Kod='TP.DK.USD.A.YTL') AS USD_TL_AySonu
                FROM gold_fact_long
                WHERE Period BETWEEN ? AND ? AND Seri_Kod IN
                    ('TP.TUKFIY2025.GENEL','TP.MT210AGS.TRY.MT02','TP.DK.USD.A.YTL')
                GROUP BY Period
            )
            SELECT b.*, macro.TUFE_2025_100, macro.TP_Faiz_3AyaKadar_Yuzde,
                   macro.USD_TL_AySonu,
                   CAST(t.Toplam_Islem_Hacmi_TL AS DOUBLE) AS TEFAS_Islem_Hacmi_TL,
                   CAST(t.Net_Takas_Tutari_TL AS DOUBLE) AS TEFAS_Net_Takas_TL
            FROM b JOIN macro USING (Period)
            JOIN gold_tefas_monthly_volume t USING (Period)
            WHERE b.Period BETWEEN ? AND ?
        """
        con.execute(sql, [start, end, start, end, start, end])
        counts = con.execute("""SELECT COUNT(*), COUNT(DISTINCT Period),
            COUNT(*) FILTER (WHERE TP_Mevduat_Mn_TL IS NULL OR YP_Mevduat_Mn_TL IS NULL
                OR TP_3AyaKadar_Mn_TL IS NULL OR TP_3AydanFazla_Mn_TL IS NULL
                OR TUFE_2025_100 IS NULL OR TUFE_2025_100 <= 0
                OR TP_Faiz_3AyaKadar_Yuzde IS NULL OR USD_TL_AySonu IS NULL
                OR TEFAS_Islem_Hacmi_TL IS NULL OR TEFAS_Net_Takas_TL IS NULL),
            MIN(Period), MAX(Period) FROM savings_work""").fetchone()
        if counts != (months, months, 0, start, end):
            raise ValueError(f"Aylık birleşimde eksik/çift/bozuk dönem var: {counts}, beklenen {months} ay.")
        base_cpi, base_tp, base_yp, base_short, base_long, base_volume, base_net = con.execute(
            """SELECT TUFE_2025_100, TP_Mevduat_Mn_TL, YP_Mevduat_Mn_TL,
                      TP_3AyaKadar_Mn_TL, TP_3AydanFazla_Mn_TL,
                      TEFAS_Islem_Hacmi_TL, TEFAS_Net_Takas_TL
               FROM savings_work WHERE Period=?""", [base]).fetchone()
        price_cpi = con.execute(
            "SELECT TUFE_2025_100 FROM savings_work WHERE Period=?", [prices]).fetchone()[0]
        if min(base_cpi, base_tp, base_yp, base_short, base_long, base_volume, base_net, price_cpi) <= 0:
            raise ValueError("Endeks veya fiyat baz ayındaki tutarlar pozitif olmalı.")
        con.execute(f"""
            CREATE OR REPLACE TABLE {TABLE} AS
            WITH r AS (
                SELECT *,
                  TP_Mevduat_Mn_TL * ? / TUFE_2025_100 AS TP_Reel_Mn_TL,
                  YP_Mevduat_Mn_TL * ? / TUFE_2025_100 AS YP_Reel_Mn_TL,
                  TP_3AyaKadar_Mn_TL * ? / TUFE_2025_100 AS TP_3AyaKadar_Reel_Mn_TL,
                  TP_3AydanFazla_Mn_TL * ? / TUFE_2025_100 AS TP_3AydanFazla_Reel_Mn_TL,
                  TEFAS_Islem_Hacmi_TL * ? / TUFE_2025_100 AS TEFAS_Islem_Hacmi_Reel_TL,
                  TEFAS_Net_Takas_TL * ? / TUFE_2025_100 AS TEFAS_Net_Takas_Reel_TL
                FROM savings_work
            )
            SELECT *,
              100 * TP_Reel_Mn_TL / ? AS TP_Reel_Endeks_202101,
              100 * YP_Reel_Mn_TL / ? AS YP_Reel_Endeks_202101,
              100 * TP_3AyaKadar_Reel_Mn_TL / ? AS TP_3AyaKadar_Reel_Endeks_202101,
              100 * TP_3AydanFazla_Reel_Mn_TL / ? AS TP_3AydanFazla_Reel_Endeks_202101,
              100 * TEFAS_Islem_Hacmi_Reel_TL / ? AS TEFAS_Islem_Hacmi_Reel_Endeks_202101,
              100 * TEFAS_Net_Takas_Reel_TL / ? AS TEFAS_Net_Takas_Reel_Endeks_202101,
              100 * (TUFE_2025_100 / LAG(TUFE_2025_100)
                OVER (ORDER BY Period) - 1) AS Aylik_Enflasyon_Yuzde,
              100 * (TUFE_2025_100 / LAG(TUFE_2025_100, 12)
                OVER (ORDER BY Period) - 1) AS Yillik_Enflasyon_Yuzde,
              ? AS Reel_Fiyat_Baz_Ayi, ? AS Reel_Endeks_Baz_Ayi
            FROM r
            ORDER BY Period
        """, [price_cpi] * 6 + [base_tp * price_cpi / base_cpi,
             base_yp * price_cpi / base_cpi, base_short * price_cpi / base_cpi,
             base_long * price_cpi / base_cpi, base_volume * price_cpi / base_cpi,
             base_net * price_cpi / base_cpi, prices, base])
        return {"table": TABLE, "row_count": months, "start_period": start,
                "end_period": end, "price_month": prices, "index_base_month": base}
    finally:
        con.close()
