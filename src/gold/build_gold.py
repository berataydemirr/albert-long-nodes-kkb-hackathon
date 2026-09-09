"""
Gold layer builder (Faz A3 + A4): reads the aligned/corrected silver
sources, joins them on a common (Period, Kaynak, Seri) key, and loads
the result into a DuckDB file the Lakehouse Tool will query directly.

Produces three tables in data/gold/lakehouse.duckdb:

  gold_fact_long        Long/tidy fact table across ALL sources (EVDS,
                         TUIK, BDDK Monthly, BDDK Weekly) on a shared
                         monthly Period key - generic, joinable, is what
                         a text-to-SQL tool should be pointed at.

  gold_bddk_net_profit_monthly
                         BDDK Monthly tablo_2's YTD-cumulative "Donem
                         Net Kari (Zarari)" line, decumulated into a
                         true monthly flow (Faz A2 applied to a real,
                         scan-confirmed cumulative series).

  gold_konut_demo        Wide, demo-ready table built to answer the
                         hackathon brief's own worked example (see
                         "Verinin Analizi ve Dogrulama" slides): monthly
                         housing-loan stock, its interest rate, the
                         house price index, and an inflation-adjusted
                         (CPI-deflated, base = first period) housing
                         loan column - all four in one row per month.

Usage (from the repo root):
    python -m src.gold.build_gold
"""
import argparse

import duckdb
import pandas as pd

from config.settings import (
    BDDK_MONTHLY_SILVER_DIR, BDDK_WEEKLY_SILVER_DIR, EVDS_SILVER_DIR,
    TUIK_SILVER_DIR, GOLD_DIR,
)
from src.gold.align import donem_id_to_date, downsample_to_monthly, to_period
from src.gold.cumulative import decumulate
from src.gold.scan_cumulative import load_table_long
from src.common.logging_setup import get_logger

logger = get_logger("build_gold")

GOLD_DB_PATH = GOLD_DIR / "lakehouse.duckdb"

# EVDS series we pull into the long fact table, with human labels for
# the LLM-facing schema (see docs/gold_schema.md).
EVDS_SERIES_LABELS = {
    "TP.KFE.TR": "Konut Fiyat Endeksi (TR Geneli)",
    "TP.BKR.TRY.18": "Konut Kredisi Faiz Orani (TL, Stok, %)",
    "TP.BISPOLFAIZ.TUR": "TCMB Politika Faizi (%)",
    "TP.GENENDEKS.T1": "TUFE Genel Endeks (2003=100)",
    "TP.TSANAYMT2021.BCD": "Sanayi Uretim Endeksi (Mevsim/Takvim Etkisinden Arindirilmis)",
}


def _fact_row(period, kaynak, seri_kod, seri_adi, deger, birim, tip, kaynak_dosya):
    return {
        "Period": period, "Kaynak": kaynak, "Seri_Kod": seri_kod,
        "Seri_Adi": seri_adi, "Deger": deger, "Birim": birim,
        "Tip": tip, "Kaynak_Dosya": kaynak_dosya,
    }


def build_evds_facts() -> pd.DataFrame:
    rows = []
    for code, label in EVDS_SERIES_LABELS.items():
        path = EVDS_SILVER_DIR / f"{code.replace('.', '_')}.csv"
        if not path.exists():
            logger.warning(f"Missing EVDS silver file, skipped: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8")
        for _, r in df.iterrows():
            rows.append(_fact_row(r["Period"], "evds", code, label, r["Value"], None, "stock", path.name))

    # USD/TRY is daily -> downsample to monthly (last trading day of month = month-end rate)
    usd_path = EVDS_SILVER_DIR / "TP_DK_USD_A_YTL.csv"
    if usd_path.exists():
        usd = pd.read_csv(usd_path, encoding="utf-8")
        usd_monthly = downsample_to_monthly(usd, date_col="Period", value_cols=["Value"], how="last")
        for _, r in usd_monthly.iterrows():
            rows.append(_fact_row(r["Period"], "evds", "TP.DK.USD.A.YTL", "USD/TRY Alis Kuru (Ay Sonu)",
                                   r["Value"], "TRY", "stock", usd_path.name))

    return pd.DataFrame(rows)


def build_tuik_facts() -> pd.DataFrame:
    path = TUIK_SILVER_DIR / "TUIK_UNEMPLOYMENT_RATE.csv"
    if not path.exists():
        logger.warning(f"Missing TUIK silver file, skipped: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8")
    rows = [
        _fact_row(r["Period"], "tuik", "TUIK_UNEMPLOYMENT_RATE", "Issizlik Orani (%, 15+, Mevsimsellikten Arindirilmamis)",
                  r["Value"], "%", "stock", path.name)
        for _, r in df.iterrows()
    ]
    return pd.DataFrame(rows)


def build_bddk_monthly_konut_facts() -> pd.DataFrame:
    """tablo_4 / 'Tuketici Kredileri - Konut': confirmed a STOCK figure
    (no year-boundary reset, see cumulative_scan_report.csv) - used as-is."""
    tablo4_dir = BDDK_MONTHLY_SILVER_DIR / "tablo_4"
    long_df = load_table_long(tablo4_dir)
    item = long_df[long_df["Item"].str.contains("Konut", case=False, na=False)
                    & ~long_df["Item"].str.contains("Endeksli|Takip", case=False, na=False)]
    rows = [
        _fact_row(r["Period"], "bddk_monthly", "tablo_4.konut_kredisi", "Tuketici Kredileri - Konut (Toplam)",
                   r["Value"], "Bin TL", "stock", "tablo_4/*.csv")
        for _, r in item.iterrows()
    ]
    return pd.DataFrame(rows)


def build_bddk_weekly_konut_facts() -> pd.DataFrame:
    """tablo_289 (Loans) / 'a) Konut', weekly -> downsampled to monthly
    (last week of the month = month-end stock), Faz A1 applied for real."""
    tablo_dir = BDDK_WEEKLY_SILVER_DIR / "tablo_289"
    if not tablo_dir.exists():
        return pd.DataFrame()

    rows = []
    for csv_path in sorted(tablo_dir.glob("donem_*.csv")):
        donem_id = int(csv_path.stem.split("_")[1])
        date = donem_id_to_date(donem_id)
        df = pd.read_csv(csv_path, encoding="utf-8")
        match = df[df["Item"].astype(str).str.strip() == "a) Konut"]
        if match.empty:
            continue
        rows.append({"Date": date, "Value": float(match.iloc[0]["TOPLAM"])})

    if not rows:
        return pd.DataFrame()

    weekly_df = pd.DataFrame(rows)
    weekly_df["Date"] = pd.to_datetime(weekly_df["Date"])
    monthly = downsample_to_monthly(weekly_df, date_col="Date", value_cols=["Value"], how="last")

    return pd.DataFrame([
        _fact_row(r["Period"], "bddk_weekly", "tablo_289.konut", "Krediler - a) Konut (Haftalik Bulten, Ay Sonu)",
                   r["Value"], "Bin TL", "stock", "tablo_289/donem_*.csv")
        for _, r in monthly.iterrows()
    ])


def build_bddk_net_profit_monthly() -> pd.DataFrame:
    """A2 in action: tablo_2's net profit/loss line was scan-confirmed
    cumulative (see cumulative_scan_report.csv) - decumulate it here."""
    tablo2_dir = BDDK_MONTHLY_SILVER_DIR / "tablo_2"
    long_df = load_table_long(tablo2_dir)
    net_profit = long_df[long_df["Item"].astype(str).str.contains("NET KAR", case=False, na=False)]
    if net_profit.empty:
        logger.warning("Net profit line not found in tablo_2 - check the item name.")
        return pd.DataFrame()

    # There can be more than one matching line (gross vs net); keep the first exact match.
    item_name = net_profit["Item"].iloc[0]
    series = net_profit[net_profit["Item"] == item_name][["Period", "Value"]]
    result = decumulate(series, period_col="Period", value_col="Value")
    result["Item"] = item_name
    return result


def build_konut_demo_wide(fact_long: pd.DataFrame) -> pd.DataFrame:
    """Replicates the hackathon brief's own worked example end-to-end:
    monthly housing credit + its interest rate + house price index +
    a CPI-deflated ('enflasyondan arindirilmis') housing credit column,
    without disturbing the original nominal column - exactly what the
    brief's 3-step demo prompt sequence asks for."""
    def series(seri_kod, value_name):
        s = fact_long[fact_long["Seri_Kod"] == seri_kod][["Period", "Deger"]]
        return s.rename(columns={"Deger": value_name})

    konut = series("tablo_4.konut_kredisi", "Konut_Kredisi_Bin_TL")
    faiz = series("TP.BKR.TRY.18", "Konut_Kredisi_Faiz_Orani")
    endeks = series("TP.KFE.TR", "Konut_Fiyat_Endeksi")
    tufe = series("TP.GENENDEKS.T1", "TUFE_Endeksi")

    wide = konut.merge(faiz, on="Period", how="left") \
                .merge(endeks, on="Period", how="left") \
                .merge(tufe, on="Period", how="left") \
                .sort_values("Period").reset_index(drop=True)

    base_cpi = wide["TUFE_Endeksi"].iloc[0]
    wide["Konut_Kredisi_Reel_Bin_TL"] = wide["Konut_Kredisi_Bin_TL"] * (base_cpi / wide["TUFE_Endeksi"])
    wide["Reel_Baz_Donem"] = wide["Period"].iloc[0]

    return wide


def run(db_path=GOLD_DB_PATH) -> None:
    logger.info("Building gold_fact_long from evds, tuik, bddk_monthly, bddk_weekly ...")
    fact_long = pd.concat([
        build_evds_facts(),
        build_tuik_facts(),
        build_bddk_monthly_konut_facts(),
        build_bddk_weekly_konut_facts(),
    ], ignore_index=True)

    logger.info("Decumulating BDDK Monthly net profit (tablo_2) ...")
    net_profit = build_bddk_net_profit_monthly()

    logger.info("Building gold_konut_demo (hackathon worked example) ...")
    konut_demo = build_konut_demo_wide(fact_long)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.register("fact_long_df", fact_long)
    con.execute("CREATE OR REPLACE TABLE gold_fact_long AS SELECT * FROM fact_long_df")

    if not net_profit.empty:
        con.register("net_profit_df", net_profit)
        con.execute("CREATE OR REPLACE TABLE gold_bddk_net_profit_monthly AS SELECT * FROM net_profit_df")

    con.register("konut_demo_df", konut_demo)
    con.execute("CREATE OR REPLACE TABLE gold_konut_demo AS SELECT * FROM konut_demo_df")
    con.close()

    logger.info("=" * 50)
    logger.info("GOLD BUILD SUMMARY")
    logger.info(f"gold_fact_long                : {len(fact_long)} rows, {fact_long['Seri_Kod'].nunique()} series")
    logger.info(f"gold_bddk_net_profit_monthly   : {len(net_profit)} rows")
    logger.info(f"gold_konut_demo                : {len(konut_demo)} rows, {konut_demo['Period'].min()}..{konut_demo['Period'].max()}")
    logger.info(f"Written to                     : {db_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    argparse.ArgumentParser(description="Build the gold layer into DuckDB (no network)").parse_args()
    run()
