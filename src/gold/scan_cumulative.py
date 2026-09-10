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

from config.settings import BDDK_MONTHLY_SILVER_DIR, BDDK_WEEKLY_SILVER_DIR, BDDK_FINTURK_SILVER_DIR, GOLD_DIR
from src.gold.align import donem_id_to_date
from src.gold.cumulative import detect_cumulative
from src.common.logging_setup import get_logger

logger = get_logger("scan_cumulative")

FILENAME_RE = re.compile(r"^(\d{4})_(\d{2})\.csv$")
FINTURK_NON_VALUE_COLS = ("Eftodu", "Yıl", "Ay", "Şehir", "Grup")

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


def load_weekly_table_long(tablo_dir) -> pd.DataFrame:
    """Reads every donem's CSV for one BDDK Weekly table and stacks them
    into (Period, Item, Value) using the 'TOPLAM' column, same shape as
    load_table_long() above but keyed by donemId -> real date instead of
    a Y/M filename (donem_id_to_date, confirmed anchor - see align.py)."""
    frames = []
    for csv_path in sorted(tablo_dir.glob("donem_*.csv")):
        donem_id = int(csv_path.stem.split("_")[1])
        period = donem_id_to_date(donem_id).isoformat()

        df = pd.read_csv(csv_path, encoding="utf-8")
        if "Item" not in df.columns or "TOPLAM" not in df.columns:
            continue
        chunk = df[["Item", "TOPLAM"]].rename(columns={"TOPLAM": "Value"})
        chunk["Period"] = period
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["Period", "Item", "Value"])
    return pd.concat(frames, ignore_index=True)


def load_finturk_table_national_long(tablo_dir) -> pd.DataFrame:
    """Reads every quarter's CSV for one FinTurk table and sums each
    numeric column across all 82 rows (81 provinces + Yurt Disi) into a
    national total per quarter, long-shaped as (Period, Item, Value)
    where Item = the column name. Ratio columns ('Oran' in the name,
    e.g. tablo_5) are skipped - summing a percentage across provinces
    isn't a meaningful national figure, same reasoning the Monthly scan
    already applies to excluding ratio tables from decumulation."""
    frames = []
    for csv_path in sorted(tablo_dir.glob("*.csv")):
        year, quarter_month = csv_path.stem.split("-")
        period = f"{year}-{int(quarter_month):02d}"

        df = pd.read_csv(csv_path, encoding="utf-8")
        value_cols = [c for c in df.columns if c not in FINTURK_NON_VALUE_COLS and "Oran" not in c]
        national = df[value_cols].apply(pd.to_numeric, errors="coerce").sum(numeric_only=True)
        for item, value in national.items():
            frames.append({"Period": period, "Item": item, "Value": value})

    if not frames:
        return pd.DataFrame(columns=["Period", "Item", "Value"])
    return pd.DataFrame(frames)


def _scan_source(source_name: str, tablo_dirs, loader) -> pd.DataFrame:
    """Shared scan loop: runs detect_cumulative() over every (tablo,
    Item) pair a loader produces, logs a summary, writes a per-source
    report CSV. Used identically for Monthly/Weekly/FinTurk so the three
    scans can't silently drift out of sync with each other."""
    results = []
    for tablo_dir in tablo_dirs:
        tablo_no = tablo_dir.name.split("_")[-1]
        long_df = loader(tablo_dir)
        if long_df.empty:
            # INFO not WARNING: an empty result here is often EXPECTED (e.g.
            # FinTurk tablo_5 is 100% ratio columns, all deliberately
            # filtered out by load_finturk_table_national_long) rather than
            # a broken loader - agent_controller.py's pipeline runner treats
            # any WARNING as a step failure, so this must not cry wolf.
            # A genuinely broken loader still shows up as a low
            # "Line items scanned" count in the summary below.
            logger.info(f"_scan_source({source_name}): no data for {tablo_dir.name}, skipped (expected for ratio-only tables)")
            continue

        for item, grp in long_df.groupby("Item"):
            diag = detect_cumulative(grp, period_col="Period", value_col="Value")
            results.append({"tablo": f"tablo_{tablo_no}", "item": item, **diag})

    report = pd.DataFrame(results)
    out_path = GOLD_DIR / f"cumulative_scan_report_{source_name}.csv"
    report.to_csv(out_path, index=False, encoding="utf-8")

    n_cumulative = int(report["is_cumulative"].sum()) if not report.empty else 0
    logger.info("=" * 50)
    logger.info(f"CUMULATIVE SCAN SUMMARY ({source_name})")
    logger.info(f"Line items scanned : {len(report)}")
    logger.info(f"Flagged cumulative : {n_cumulative}")
    logger.info(f"Report written to  : {out_path}")
    logger.info("=" * 50)
    if n_cumulative:
        flagged = report[report["is_cumulative"]][["tablo", "item", "monotonic_ratio", "reset_ratio"]]
        for _, row in flagged.iterrows():
            logger.info(f"  CUMULATIVE: {row['tablo']} / {row['item']}")

    return report


def run_monthly() -> pd.DataFrame:
    tablo_dirs = sorted(BDDK_MONTHLY_SILVER_DIR.glob("tablo_*"), key=lambda p: int(p.name.split("_")[1]))
    report = _scan_source("bddk_monthly", tablo_dirs, load_table_long)
    # kept at the ORIGINAL path too (data/gold/cumulative_scan_report.csv)
    # since build_gold.py's docstrings/docs reference it by that name.
    report.to_csv(GOLD_DIR / "cumulative_scan_report.csv", index=False, encoding="utf-8")
    return report


def run_weekly() -> pd.DataFrame:
    tablo_dirs = sorted(BDDK_WEEKLY_SILVER_DIR.glob("tablo_*"), key=lambda p: int(p.name.split("_")[1]))
    return _scan_source("bddk_weekly", tablo_dirs, load_weekly_table_long)


def run_finturk() -> pd.DataFrame:
    tablo_dirs = sorted(BDDK_FINTURK_SILVER_DIR.glob("tablo_*"), key=lambda p: int(p.name.split("_")[1]))
    return _scan_source("bddk_finturk", tablo_dirs, load_finturk_table_national_long)


def run() -> pd.DataFrame:
    """Runs all 3 scans (Monthly/Weekly/FinTurk) - kept as the module's
    single entry point for `python -m src.gold.scan_cumulative` and for
    agent_controller.py's pipeline step. Returns the Monthly report for
    backward compatibility with any code that inspects run()'s return
    value; Weekly/FinTurk reports are logged and written to disk the
    same way."""
    monthly = run_monthly()
    run_weekly()
    run_finturk()
    return monthly


if __name__ == "__main__":
    argparse.ArgumentParser(description="Scan BDDK Monthly/Weekly/FinTurk for cumulative line items (no network)").parse_args()
    run()
