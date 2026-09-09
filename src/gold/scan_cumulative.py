"""
Runs cumulative.detect_cumulative() over every line item in every BDDK
Monthly table against the REAL silver data, and writes a report to
data/gold/cumulative_scan_report.csv. This is what Faz A2 ("BDDK'nin
hangi tablolarinin yil ici kumulatif oldugunu veriden tespit edip
duzeltme") actually means: not guessing per table, but testing all of
them and keeping the evidence.

Usage (from the repo root):
    python -m src.gold.scan_cumulative
"""
import argparse
import re

import pandas as pd

from config.settings import BDDK_MONTHLY_SILVER_DIR, GOLD_DIR
from src.gold.cumulative import detect_cumulative
from src.common.logging_setup import get_logger

logger = get_logger("scan_cumulative")

FILENAME_RE = re.compile(r"^(\d{4})_(\d{2})\.csv$")

# BUG FOUND WHILE WRITING THIS SCRIPT: silver CSVs on disk are correctly
# UTF-8 (verified against bronze JSON bytes), but pd.read_csv()/to_csv()
# without encoding='utf-8' fall back to the OS locale encoding. On a
# Windows machine with a Turkish locale that's cp1254, which cannot
# represent all UTF-8 sequences and silently turns 'İ' etc. into U+FFFD
# ('D�NEM NET KARI' instead of 'DÖNEM NET KARI'). Every read/write of
# these CSVs - here and in any future tool - MUST pass encoding='utf-8'
# explicitly, or Turkish item names corrupt silently on some machines
# and not others (a bug that looks fine on the author's Linux/Mac and
# breaks live in front of the jury on Windows).


def load_table_long(tablo_dir) -> pd.DataFrame:
    """Reads every month's CSV for one BDDK Monthly table and stacks them
    into (Period, Item, Value) using the 'Toplam' column (falls back to
    the last numeric column for tables that don't have one)."""
    frames = []
    for csv_path in sorted(tablo_dir.glob("*.csv")):
        m = FILENAME_RE.match(csv_path.name)
        if not m:
            continue
        period = f"{m.group(1)}-{m.group(2)}"

        df = pd.read_csv(csv_path, encoding="utf-8")
        value_col = "Toplam" if "Toplam" in df.columns else df.select_dtypes("number").columns[-1]
        item_col = "Column_2"
        if item_col not in df.columns or value_col not in df.columns:
            continue

        chunk = df[[item_col, value_col]].rename(columns={item_col: "Item", value_col: "Value"})
        chunk["Period"] = period
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["Period", "Item", "Value"])
    return pd.concat(frames, ignore_index=True)


def run() -> pd.DataFrame:
    results = []
    tablo_dirs = sorted(BDDK_MONTHLY_SILVER_DIR.glob("tablo_*"), key=lambda p: int(p.name.split("_")[1]))

    for tablo_dir in tablo_dirs:
        tablo_no = tablo_dir.name.split("_")[1]
        long_df = load_table_long(tablo_dir)
        if long_df.empty:
            continue

        for item, grp in long_df.groupby("Item"):
            diag = detect_cumulative(grp, period_col="Period", value_col="Value")
            results.append({"tablo": f"tablo_{tablo_no}", "item": item, **diag})

    report = pd.DataFrame(results)
    out_path = GOLD_DIR / "cumulative_scan_report.csv"
    report.to_csv(out_path, index=False, encoding="utf-8")

    n_cumulative = int(report["is_cumulative"].sum()) if not report.empty else 0
    logger.info("=" * 50)
    logger.info("CUMULATIVE SCAN SUMMARY (bddk_monthly)")
    logger.info(f"Line items scanned : {len(report)}")
    logger.info(f"Flagged cumulative : {n_cumulative}")
    logger.info(f"Report written to  : {out_path}")
    logger.info("=" * 50)
    if n_cumulative:
        flagged = report[report["is_cumulative"]][["tablo", "item", "monotonic_ratio", "reset_ratio"]]
        for _, row in flagged.iterrows():
            logger.info(f"  CUMULATIVE: {row['tablo']} / {row['item']}")

    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description="Scan BDDK Monthly for cumulative line items (no network)").parse_args()
    run()
