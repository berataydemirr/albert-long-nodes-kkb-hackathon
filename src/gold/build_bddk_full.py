"""Promote the FULL BDDK silver archive into queryable gold tables.

WHY THIS EXISTS: the collectors already pull the complete BDDK archive -
17 Monthly Bulletin tables x 66 months (~132k data points), 9 Weekly
Bulletin tables x 286 weeks, 7 FinTurk tables x 22 quarters - but
build_gold.py only promotes a hand-written whitelist of ~18 single-value
series into gold_fact_long (one row's "Toplam" column each). Everything
else sat on disk, invisible to the agent: a live question about deposit
maturity structure ("TP/YP mevduat, 3 aya kadar vs 3 aydan fazla") was
answered "lakehouse'da yok" even though tablo_10 carries exactly those
rows and columns for all 66 months.

The gap is not collection, it is exposure. gold_fact_long's shape is
(Period, Seri_Kod, Deger): one named series per row, discoverable by
reading a catalog that is inlined into every system prompt. That design
does not scale here - the full archive is ~2,000 distinct
(table, row, column) combinations, which would be ~60k tokens of catalog
against a prompt that already spends ~13k on the existing one.

So these tables keep the SOURCE's own two-dimensional shape instead of
inventing 2,000 series codes: one row per (period, row-label, column-
label). Discovery happens through search_bddk_items() - the same
"search, don't enumerate" pattern already proven for EVDS's huge catalog
(see finance_fetch.search_evds_series) - not through the prompt catalog.

Written ADDITIVELY: gold_fact_long and gold_regional_long are never read,
modified or dropped here. This module only creates its own three tables.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from config.settings import (
    BDDK_FINTURK_SILVER_DIR,
    BDDK_FINTURK_TABLES,
    BDDK_MONTHLY_SILVER_DIR,
    BDDK_MONTHLY_TABLES,
    BDDK_WEEKLY_SILVER_DIR,
    BDDK_WEEKLY_TABLES,
    GOLD_DIR,
)
from src.gold.align import donem_id_to_date

logger = logging.getLogger("build_bddk_full")

MONTHLY_TABLE = "gold_bddk_monthly_long"
WEEKLY_TABLE = "gold_bddk_weekly_long"
FINTURK_TABLE = "gold_bddk_finturk_long"

# Monthly silver carries four bookkeeping columns before the real data:
# Column_0 is the "taraf" (always Sektor here), Column_1 the source row
# number, Column_2 the row label, BasitFont the source's own bold/italic
# styling - which is the ONLY marker of hierarchy in the bulletin (bold =
# a top-level item, italic = a sub-breakdown of the bold row above it).
# It is kept as `Seviye` rather than dropped: without it nothing stops a
# SUM() over every row from double-counting parents and their children.
_MONTHLY_META_COLS = ("Column_0", "Column_1", "Column_2", "BasitFont")
_FINTURK_META_COLS = ("Eftodu", "Yıl", "Ay", "Şehir", "Grup")

_MONTHLY_FILENAME_RE = re.compile(r"^(\d{4})_(\d{2})\.csv$")
_WEEKLY_FILENAME_RE = re.compile(r"^donem_(\d+)\.csv$")
_FINTURK_FILENAME_RE = re.compile(r"^(\d{4})-(\d{1,2})\.csv$")

def _cumulative_flags() -> dict[tuple[str, str], bool | None]:
    """(tablo, item) -> is_cumulative, from the scan report build_gold
    already produces. Carried into the gold table because it changes what
    a correct query looks like: every row of tablo_2 (Kar Zarar) is a
    year-to-date cumulative, so a monthly figure there is a DIFFERENCE
    between consecutive months, not the stored value. Missing entries map
    to None ("unknown"), never silently to False."""
    path = GOLD_DIR / "cumulative_scan_report.csv"
    if not path.exists():
        logger.warning(f"_cumulative_flags: {path} missing, every row will be Kumulatif=None")
        return {}
    report = pd.read_csv(path, encoding="utf-8")
    flags: dict[tuple[str, str], bool | None] = {}
    for _, row in report.iterrows():
        value = row.get("is_cumulative")
        if pd.isna(value):
            flags[(str(row["tablo"]), str(row["item"]))] = None
        else:
            flags[(str(row["tablo"]), str(row["item"]))] = bool(value)
    return flags


def _melt(frame: pd.DataFrame, meta_cols, value_name: str = "Deger") -> pd.DataFrame:
    """Wide -> long over every column that is not bookkeeping. Values are
    coerced with errors='coerce' and non-numeric cells dropped: the
    bulletin uses '-' and blanks for "not applicable", and carrying those
    through as text would break every numeric comparison downstream."""
    value_cols = [c for c in frame.columns if c not in meta_cols]
    if not value_cols:
        return pd.DataFrame()
    melted = frame.melt(
        id_vars=[c for c in meta_cols if c in frame.columns],
        value_vars=value_cols,
        var_name="Kolon",
        value_name=value_name,
    )
    melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce")
    return melted.dropna(subset=[value_name])


# Period is attached to each source row before the files are stacked, so
# one melt can serve a whole table directory. Melting per FILE instead
# was measured at 77s for the full archive (3,850 tiny melt/concat pairs);
# stacking first and melting once per table is the same output in a
# fraction of the time, which matters because this runs inside the gold
# build and inside the test suite.
_PERIOD_COL = "__Period"


def _stack_dir(tablo_dir, filename_re, period_of, required_col: str) -> pd.DataFrame:
    """Read every CSV in one table directory into a single frame, tagging
    each row with the period its file covers. Returns empty if the
    directory holds nothing usable - a missing table must not abort the
    whole build."""
    frames = []
    for csv_path in sorted(tablo_dir.glob("*.csv")):
        match = filename_re.match(csv_path.name)
        if not match:
            continue
        frame = pd.read_csv(csv_path, encoding="utf-8")
        if required_col not in frame.columns:
            logger.warning(f"_stack_dir: {csv_path} has no {required_col} column, skipped")
            continue
        for key, value in period_of(match).items():
            frame[key] = value
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_monthly_long() -> pd.DataFrame:
    """Every (month, row, column) cell of all 17 Monthly Bulletin tables."""
    flags = _cumulative_flags()
    frames = []
    for tablo_no, meta in BDDK_MONTHLY_TABLES.items():
        tablo = f"tablo_{tablo_no}"
        tablo_dir = BDDK_MONTHLY_SILVER_DIR / tablo
        if not tablo_dir.exists():
            logger.warning(f"build_monthly_long: {tablo_dir} missing, skipped")
            continue
        raw = _stack_dir(
            tablo_dir, _MONTHLY_FILENAME_RE,
            lambda m: {_PERIOD_COL: f"{m.group(1)}-{m.group(2)}"},
            required_col="Column_2",
        )
        if raw.empty:
            continue
        melted = _melt(raw, _MONTHLY_META_COLS + (_PERIOD_COL,))
        if melted.empty:
            continue
        melted = melted.rename(columns={
            "Column_1": "Satir_No", "Column_2": "Satir",
            "BasitFont": "Seviye", _PERIOD_COL: "Period",
        })
        melted["Tablo"] = tablo
        melted["Tablo_Adi"] = meta["name"]
        melted["Birim"] = meta["unit"]
        melted["Kumulatif"] = melted["Satir"].astype(str).map(
            lambda s, _t=tablo: flags.get((_t, s))
        )
        frames.append(melted)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[["Period", "Tablo", "Tablo_Adi", "Satir_No", "Satir", "Seviye",
               "Kolon", "Deger", "Birim", "Kumulatif"]]
    logger.info(f"build_monthly_long: {len(out)} rows across {out.Tablo.nunique()} table(s)")
    return out


def build_weekly_long() -> pd.DataFrame:
    """Every (week, row, TP/YP/TOPLAM) cell of all 9 Weekly Bulletin tables.

    Carries BOTH `Tarih` (the real week date) and `Period` (YYYY-MM):
    Period is what every other gold_* table uses, so a weekly series can
    be joined/aggregated against monthly data - but several weeks share
    one Period, so a monthly figure needs an explicit aggregate (last
    week of the month for a stock, not SUM)."""
    def _weekly_period(match) -> dict:
        donem_id = int(match.group(1))
        tarih = donem_id_to_date(donem_id)
        return {"Donem_Id": donem_id, "Tarih": tarih.isoformat(),
                _PERIOD_COL: tarih.strftime("%Y-%m")}

    frames = []
    for tablo_no, name in BDDK_WEEKLY_TABLES.items():
        tablo = f"tablo_{tablo_no}"
        tablo_dir = BDDK_WEEKLY_SILVER_DIR / tablo
        if not tablo_dir.exists():
            logger.warning(f"build_weekly_long: {tablo_dir} missing, skipped")
            continue
        raw = _stack_dir(tablo_dir, _WEEKLY_FILENAME_RE, _weekly_period, required_col="Item")
        if raw.empty:
            continue
        melted = _melt(raw, ("Sequence", "Item", "Donem_Id", "Tarih", _PERIOD_COL))
        if melted.empty:
            continue
        melted = melted.rename(columns={
            "Sequence": "Satir_No", "Item": "Satir", _PERIOD_COL: "Period",
        })
        melted["Tablo"] = tablo
        melted["Tablo_Adi"] = name
        frames.append(melted)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[["Period", "Tarih", "Donem_Id", "Tablo", "Tablo_Adi",
               "Satir_No", "Satir", "Kolon", "Deger"]]
    logger.info(f"build_weekly_long: {len(out)} rows across {out.Tablo.nunique()} table(s)")
    return out


def build_finturk_long() -> pd.DataFrame:
    """Every (quarter, province, metric) cell of all 7 FinTurk tables.

    gold_regional_long already carries 35 curated metrics from this same
    source; this table is the uncurated remainder (76 metric columns in
    total - tablo_4's 31 sectoral loan columns and tablo_7's gold
    deposits are absent from the curated set entirely). Both are kept:
    gold_regional_long stays the tidy, unit-checked path for the metrics
    it covers, this one is the fallback for everything else."""
    frames = []
    for tablo_no, name in BDDK_FINTURK_TABLES.items():
        tablo = f"tablo_{tablo_no}"
        tablo_dir = BDDK_FINTURK_SILVER_DIR / tablo
        if not tablo_dir.exists():
            logger.warning(f"build_finturk_long: {tablo_dir} missing, skipped")
            continue
        raw = _stack_dir(
            tablo_dir, _FINTURK_FILENAME_RE,
            lambda m: {_PERIOD_COL: f"{m.group(1)}-{int(m.group(2)):02d}"},
            required_col="Şehir",
        )
        if raw.empty:
            continue
        melted = _melt(raw, _FINTURK_META_COLS + (_PERIOD_COL,))
        if melted.empty:
            continue
        melted = melted.rename(columns={"Şehir": "Bolge", _PERIOD_COL: "Period"})
        melted["Tablo"] = tablo
        melted["Tablo_Adi"] = name
        frames.append(melted)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[["Period", "Tablo", "Tablo_Adi", "Bolge", "Grup", "Kolon", "Deger"]]
    logger.info(f"build_finturk_long: {len(out)} rows across {out.Tablo.nunique()} table(s)")
    return out


def write_bddk_full_tables(con) -> dict[str, int]:
    """Create/replace ONLY this module's three tables on an open DuckDB
    connection. Never touches gold_fact_long or gold_regional_long."""
    written: dict[str, int] = {}
    for table, builder in (
        (MONTHLY_TABLE, build_monthly_long),
        (WEEKLY_TABLE, build_weekly_long),
        (FINTURK_TABLE, build_finturk_long),
    ):
        frame = builder()
        if frame.empty:
            logger.warning(f"write_bddk_full_tables: {table} would be empty, NOT written")
            written[table] = 0
            continue
        con.register("_bddk_full_tmp", frame)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _bddk_full_tmp")
        con.unregister("_bddk_full_tmp")
        written[table] = len(frame)
        logger.info(f"write_bddk_full_tables: wrote {table} ({len(frame)} rows)")
    return written


def main() -> None:
    import duckdb

    from config.settings import LAKEHOUSE_DB_PATH

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
    con = duckdb.connect(str(LAKEHOUSE_DB_PATH))
    try:
        written = write_bddk_full_tables(con)
    finally:
        con.close()
    for table, rows in written.items():
        print(f"{table}: {rows} rows")


if __name__ == "__main__":
    main()
