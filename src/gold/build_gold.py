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

  gold_finturk_regional_quarterly
                         BDDK FinTurk tablo_1 (Loans), kept at its NATIVE
                         quarterly, per-province grain rather than forced
                         onto the monthly axis - the first regional
                         (Bolge) dimension in the gold layer.

Usage (from the repo root):
    python -m src.gold.build_gold
"""
import argparse
import re

import duckdb
import pandas as pd

from config.settings import (
    BDDK_MONTHLY_SILVER_DIR, BDDK_WEEKLY_SILVER_DIR, BDDK_FINTURK_SILVER_DIR,
    EVDS_SILVER_DIR, TUIK_SILVER_DIR, GOLD_DIR, BDDK_MONTHLY_TABLES,
)
from src.gold.align import donem_id_to_date, downsample_to_monthly, to_period, upsample_quarterly_to_monthly
from src.gold.cumulative import decumulate
from src.gold.scan_cumulative import load_table_long
from src.common.logging_setup import get_logger

logger = get_logger("build_gold")

GOLD_DB_PATH = GOLD_DIR / "lakehouse.duckdb"

# EVDS series we pull into the long fact table, with human labels for
# the LLM-facing schema (see docs/gold_schema.md).
EVDS_SERIES_LABELS = {
    "TP.KFE.TR": ("Konut Fiyat Endeksi (TR Geneli)", "Endeks"),
    "TP.BKR.TRY.18": ("Konut Kredisi Faiz Orani (TL, Stok, %)", "%"),
    "TP.BISPOLFAIZ.TUR": ("TCMB Politika Faizi (%)", "%"),
    "TP.GENENDEKS.T1": ("TUFE Genel Endeks (2003=100)", "Endeks (2003=100)"),
    "TP.TSANAYMT2021.BCD": ("Sanayi Uretim Endeksi (Mevsim/Takvim Etkisinden Arindirilmis)", "Endeks"),
    "TP.MT210AGS.TRY.MT06": ("TL Mevduat Faiz Orani (Vadesiz Haric, Stok, %)", "%"),
    "TP.GY1.N2": ("Reel Kesim Guven Endeksi", "Endeks"),
    "TP.BKR.TRY.KTFTUK": ("Tuketici Kredisi Faiz Orani, Ihtiyac+Tasit+Konut Birlikte (TL, Stok, %)", "%"),
    # TP.ALTINPIYASA.KAP05 (gold price) deliberately NOT included here -
    # see the data-quality warning next to it in config/evds_series.py.
}

# Weekly-dated EVDS series that need downsampling before joining the
# monthly fact table - 'how' picked per series semantics (see align.py):
# a spending FLOW is meaningfully averaged over the weeks in a month,
# not just read off the last week like a stock/rate would be; M2 (a
# stock) uses the last observation of the month instead.
EVDS_WEEKLY_SERIES = {
    # birim=None for both: EVDS's own metadata for these two series states
    # no currency/scale (checked via get_series()), so it is left empty
    # rather than filled with a plausible-looking guess.
    "TP.KKHARTUT.KT1": ("Kredi/Banka Karti Harcamasi, Toplam (Haftalik, Aylik Ortalama)", None, "flow", "mean"),
    "TP.HPBITABLO1.11": ("M2 Para Arzi (Ay Sonu)", None, "stock", "last"),
}


def _fact_row(period, kaynak, seri_kod, seri_adi, deger, birim, tip, kaynak_dosya, bolge="TR"):
    return {
        "Period": period, "Bolge": bolge, "Kaynak": kaynak, "Seri_Kod": seri_kod,
        "Seri_Adi": seri_adi, "Deger": deger, "Birim": birim,
        "Tip": tip, "Kaynak_Dosya": kaynak_dosya,
    }


def build_evds_facts() -> pd.DataFrame:
    rows = []
    for code, (label, birim) in EVDS_SERIES_LABELS.items():
        path = EVDS_SILVER_DIR / f"{code.replace('.', '_')}.csv"
        if not path.exists():
            logger.warning(f"Missing EVDS silver file, skipped: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8")
        for _, r in df.iterrows():
            rows.append(_fact_row(r["Period"], "evds", code, label, r["Value"], birim, "stock", path.name))

    # USD/TRY is daily -> downsample to monthly (last trading day of month = month-end rate)
    usd_path = EVDS_SILVER_DIR / "TP_DK_USD_A_YTL.csv"
    if usd_path.exists():
        usd = pd.read_csv(usd_path, encoding="utf-8")
        usd_monthly = downsample_to_monthly(usd, date_col="Period", value_cols=["Value"], how="last")
        for _, r in usd_monthly.iterrows():
            rows.append(_fact_row(r["Period"], "evds", "TP.DK.USD.A.YTL", "USD/TRY Alis Kuru (Ay Sonu)",
                                   r["Value"], "TRY", "stock", usd_path.name))

    # Weekly-dated series (e.g. card spending) -> downsample to monthly too
    for code, (label, birim, tip, how) in EVDS_WEEKLY_SERIES.items():
        path = EVDS_SILVER_DIR / f"{code.replace('.', '_')}.csv"
        if not path.exists():
            logger.warning(f"Missing EVDS silver file, skipped: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8")
        monthly = downsample_to_monthly(df, date_col="Period", value_cols=["Value"], how=how)
        for _, r in monthly.iterrows():
            rows.append(_fact_row(r["Period"], "evds", code, label, r["Value"], birim, tip, path.name))

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
    (no year-boundary reset, see cumulative_scan_report.csv) - used as-is.

    Unit fixed from BDDK_MONTHLY_TABLES (the bronze JSON's own "caption"
    field, e.g. "Tuketici Kredileri (milyon TL), Donem:2021/1" - see
    config/settings.py) - it is MILYON TL, not "Bin TL" as this was
    mislabeled in an earlier version of this function. The number itself
    was never wrong, only its unit label - this changes what the figure
    should be called/reported as, not any value in gold_fact_long."""
    tablo4_dir = BDDK_MONTHLY_SILVER_DIR / "tablo_4"
    long_df = load_table_long(tablo4_dir)
    item = long_df[long_df["Item"].str.contains("Konut", case=False, na=False)
                    & ~long_df["Item"].str.contains("Endeksli|Takip", case=False, na=False)]
    unit = BDDK_MONTHLY_TABLES[4]["unit"]
    rows = [
        _fact_row(r["Period"], "bddk_monthly", "tablo_4.konut_kredisi", "Tuketici Kredileri - Konut (Toplam)",
                   r["Value"], unit, "stock", "tablo_4/*.csv")
        for _, r in item.iterrows()
    ]
    return pd.DataFrame(rows)


def build_bddk_monthly_item_facts(tablo_no: int, item_name: str, seri_kod: str, seri_adi: str, birim: str | None = None) -> pd.DataFrame:
    """Generic single-line-item puller for BDDK Monthly, used for items
    beyond the housing-credit theme (Faz A's "diger tablolardan kalem
    ekle" item). item_name must match Column_2 EXACTLY (checked against
    real silver data, not a substring guess) - confirmed non-cumulative
    against cumulative_scan_report_bddk_monthly.csv before being added
    here. birim defaults to BDDK_MONTHLY_TABLES's table-level unit, but
    can be overridden for a ratio/count row inside an otherwise TL-valued
    table (e.g. tablo_12's capital adequacy ratio row is '%', not the
    table's default 'milyon TL')."""
    tablo_dir = BDDK_MONTHLY_SILVER_DIR / f"tablo_{tablo_no}"
    long_df = load_table_long(tablo_dir)
    item = long_df[long_df["Item"] == item_name]
    if item.empty:
        logger.warning(f"build_bddk_monthly_item_facts: '{item_name}' not found in tablo_{tablo_no}")
        return pd.DataFrame()

    unit = birim if birim is not None else BDDK_MONTHLY_TABLES[tablo_no]["unit"]
    return pd.DataFrame([
        _fact_row(r["Period"], "bddk_monthly", seri_kod, seri_adi, r["Value"], unit, "stock", f"tablo_{tablo_no}/*.csv")
        for _, r in item.iterrows()
    ])


def build_bddk_monthly_kobi_facts() -> pd.DataFrame:
    """tablo_6 (SME Loans) bypasses the generic
    build_bddk_monthly_item_facts() on purpose - it has no column named
    exactly 'Toplam' (unlike every other table pulled from so far), it
    has several sub-metric columns instead: 'Nakdi Kredi TP/YP', 'Toplam
    Nakdi Krediler', 'Takip Kredi TP/YP', 'Gayri Nakdi Kredi TP/YP',
    'Musteri Sayisi (Net)'. load_table_long()'s generic 'last numeric
    column' fallback silently picked the customer-COUNT column (often
    0) instead of the loan amount - confirmed live: the generic path
    reported the SME total loan series as 0.0 TL for every recent
    month, which is obviously wrong for a major Turkish banking-sector
    total.
    Reads 'Toplam Nakdi Krediler' directly instead - SME cash loans,
    domestic + foreign currency combined, the real total."""
    tablo_dir = BDDK_MONTHLY_SILVER_DIR / "tablo_6"
    if not tablo_dir.exists():
        return pd.DataFrame()

    rows = []
    for csv_path in sorted(tablo_dir.glob("*_*.csv")):
        m = re.match(r"^(\d{4})_(\d{2})\.csv$", csv_path.name)
        if not m:
            continue
        period = f"{m.group(1)}-{m.group(2)}"
        df = pd.read_csv(csv_path, encoding="utf-8")
        item = df[df["Column_2"] == "Toplam KOBİ Kredileri (2+3+4)"]
        if item.empty or "Toplam Nakdi Krediler" not in df.columns:
            logger.warning(f"build_bddk_monthly_kobi_facts: item/column missing for {csv_path.name}")
            continue
        rows.append({"Period": period, "Value": float(item.iloc[0]["Toplam Nakdi Krediler"])})

    if not rows:
        return pd.DataFrame()
    unit = BDDK_MONTHLY_TABLES[6]["unit"]
    return pd.DataFrame([
        _fact_row(r["Period"], "bddk_monthly", "tablo_6.toplam_kobi_kredileri",
                   "Toplam KOBI Kredileri (Nakdi, TP+YP), Sektor Geneli", r["Value"], unit, "stock", "tablo_6/*.csv")
        for r in rows
    ])


def build_health_life_expectancy() -> pd.DataFrame:
    """TUIK_LIFE_EXPECTANCY (see config/tuik_series.py) - the hackathon's
    "not just finance" health-data requirement. Kept at its NATIVE
    annual grain in its own table rather than forced onto the monthly
    gold_fact_long axis (same reasoning as gold_finturk_regional_quarterly:
    forward-filling a yearly figure across 12 months would misrepresent
    it as a real monthly observation). Only 2021-2024 - TUIK hadn't
    published 2025/2026 life expectancy figures as of this build; that
    gap is real, not a bug, and is left visible rather than filled."""
    path = TUIK_SILVER_DIR / "TUIK_LIFE_EXPECTANCY.csv"
    if not path.exists():
        logger.warning(f"Missing TUIK silver file, skipped: {path}")
        return pd.DataFrame()
    # dtype=str: without it, pandas infers Period ('2021','2022',...) as
    # int64 since every value looks numeric, and it ends up stored as
    # 2021.0 instead of "2021" - inconsistent with every other Period
    # column in the gold layer, which is always a string.
    df = pd.read_csv(path, encoding="utf-8", dtype={"Period": str})
    return pd.DataFrame([
        {"Period": r["Period"], "Bolge": "TR", "Kaynak": "tuik", "Seri_Kod": "TUIK_LIFE_EXPECTANCY",
         "Seri_Adi": "Doguzda Beklenen Yasam Suresi, Toplam (Yil)", "Deger": r["Value"], "Birim": "Yil",
         "Tip": "stock", "Kaynak_Dosya": path.name}
        for _, r in df.iterrows()
    ])


def build_bddk_weekly_konut_facts() -> pd.DataFrame:
    """tablo_289 (Loans) / 'a) Konut', weekly -> downsampled to monthly.

    RECLASSIFIED FROM 'stock' TO 'flow': it does not disagree with BDDK
    Monthly's housing-loan stock (tablo_4) by a clean unit factor (see
    docs/gold_schema.md) - it FAILS to grow smoothly like a balance at
    all. Plotted week over week it oscillates in a ~35-40k band
    (38091, 37319, 37621, 38075, 39452, ...) instead of climbing like
    tablo_4's stock does. That is the signature of a per-week FLOW (new
    housing loans originated that week), not an account balance, so it
    is summed (not last-of-month) across the weeks in a month and kept
    as its own series rather than reconciled against tablo_4."""
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
    weekly_df["_period"] = weekly_df["Date"].apply(to_period)
    monthly = weekly_df.groupby("_period", as_index=False)["Value"].sum().rename(columns={"_period": "Period"})

    # Unit intentionally None: unlike BDDK Monthly, the Weekly Bulletin's
    # HTML page doesn't state a unit anywhere in its visible text
    # (checked) - claiming "Bin TL" here would be a guess, not a
    # confirmed fact like tablo_4's unit above is.
    return pd.DataFrame([
        _fact_row(r["Period"], "bddk_weekly", "tablo_289.konut", "Krediler - a) Konut, Haftalik Yeni Kullandirim Toplami (Bilgi: stok DEGIL)",
                   r["Value"], None, "flow", "tablo_289/donem_*.csv")
        for _, r in monthly.iterrows()
    ])


def _load_finturk_table(table_no: int) -> pd.DataFrame:
    """Reads every quarter's FinTurk table CSV and stacks them into one long
    DataFrame with a 'YYYY-Q' Donem column.

    All FinTurk tables share the same shape (Eftodu/Yil/Ay/Sehir/Grup plus
    metric columns) and the same province grain, so one loader serves all
    of them."""
    tablo_dir = BDDK_FINTURK_SILVER_DIR / f"tablo_{table_no}"
    if not tablo_dir.exists():
        logger.warning(f"_load_finturk_table: tablo_{table_no} not found in silver")
        return pd.DataFrame()

    frames = []
    for csv_path in sorted(tablo_dir.glob("*.csv")):
        year, quarter_month = csv_path.stem.split("-")
        df = pd.read_csv(csv_path, encoding="utf-8")
        df["Donem"] = f"{year}-{int(quarter_month)}"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True)
    logger.debug(f"_load_finturk_table: tablo_{table_no} -> {len(stacked)} rows, {len(frames)} quarter(s)")
    return stacked


def _load_finturk_table1() -> pd.DataFrame:
    """Kept as a named alias: tablo_1 feeds the national and wide-regional
    builders below, which predate the generic loader."""
    return _load_finturk_table(1)


# (table, source column) -> (metric slug, unit, additive)
#
# `additive` decides whether summing across provinces is meaningful. Money
# and counts add up; a ratio does not, and a "share of the national NPL
# ratio" would be a meaningless number, so compare_regions only reports
# shares and totals for additive metrics.
#
# Units are carried through verbatim rather than normalised. BDDK ships the
# SAME concept at two different scales - tablo_5's takip oranı is a
# percentage (1.56-9.29) while the derived NPL_Orani_Yaklasik column in the
# wide table is a 0-1 fraction (0.0248) - so silently rescaling either one
# would produce a figure that is off by 100x with nothing to catch it.
FINTURK_REGIONAL_METRICS: list[tuple[int, str, str, str, bool]] = [
    # tablo_1 - loans
    (1, "Toplam Nakdi Krediler", "kredi", "Bin TL", True),
    (1, "Nakdi Krediler", "nakdi_kredi", "Bin TL", True),
    (1, "Takipteki Alacaklar", "takipteki_alacak", "Bin TL", True),
    (1, "Gayrinakdi Krediler", "gayrinakdi_kredi", "Bin TL", True),
    # tablo_2 - deposits
    (2, "Toplam Mevduat", "mevduat", "Bin TL", True),
    (2, "Tasarruf Mevduatı", "mevduat_tasarruf", "Bin TL", True),
    (2, "Tasarruf Mevduatı (TL)", "mevduat_tasarruf_tl", "Bin TL", True),
    (2, "Tasarruf Mevduatı (DTH)", "mevduat_tasarruf_dth", "Bin TL", True),
    (2, "Diğer Mevduat", "mevduat_diger", "Bin TL", True),
    # tablo_3 - consumer credit by product, and its non-performing part
    (3, "Konut Kredisi", "konut_kredisi", "Bin TL", True),
    (3, "Taşıt Kredisi", "tasit_kredisi", "Bin TL", True),
    (3, "Kredili Mevduat Hesabı", "kmh", "Bin TL", True),
    (3, "Diğer Tüketici Kredileri", "diger_tuketici_kredisi", "Bin TL", True),
    (3, "Bireysel Kredi Kartları", "kredi_karti", "Bin TL", True),
    (3, "Takipteki Konut Kredisi", "takip_konut_kredisi", "Bin TL", True),
    (3, "Takipteki Taşıt Kredisi", "takip_tasit_kredisi", "Bin TL", True),
    (3, "Takipteki Diğer Tüketici Kredileri", "takip_diger_tuketici", "Bin TL", True),
    # tablo_5 - BDDK's own ratios, already percentages
    (5, "Takipteki Alacaklar Toplam Nakdi Kredi Oranı", "npl", "%", False),
    (5, "Toplam Nakdi Kredi Toplam Mevduat Oranı", "kredi_mevduat_orani", "%", False),
    (5, "Toplam Nakdi Kredi Tasarruf Mevduatı Oranı", "kredi_tasarruf_orani", "%", False),
    (5, "Konut Kredisi Performans Oranı", "konut_takip_orani", "%", False),
    (5, "Taşıt Kredisi Performans Oranı", "tasit_takip_orani", "%", False),
    (5, "Diğer Tüketici Kredileri Performans Oranı", "diger_tuketici_takip_orani", "%", False),
    (5, "İnşaat Kredi Performans Oranı", "insaat_takip_orani", "%", False),
    (5, "Tekstil ve Tekstil Ürünleri Kredi Performans Oranı", "tekstil_takip_orani", "%", False),
    (5, "Turizm Kredi Performans Oranı", "turizm_takip_orani", "%", False),
    (5, "Ziraat ve Balıkçılık Kredi Performans Oran", "ziraat_takip_orani", "%", False),
    (5, "Enerji Kredi Performans Oranı", "enerji_takip_orani", "%", False),
    (5, "Gıda Meşrubat ve Tütün Kredi Performans Oranı", "gida_takip_orani", "%", False),
    # tablo_6 - structural / per-capita
    (6, "Yurtiçi Şube Sayısı", "sube_sayisi", "adet", True),
    (6, "Şubeye Düşen Nüfus", "subeye_dusen_nufus", "kisi", False),
    (6, "Kişi Başı Nakdi Kredi", "kisi_basi_kredi", "TL", False),
    (6, "Kişi Başı Takipteki Alacak", "kisi_basi_takipteki_alacak", "TL", False),
    (6, "Kişi Başı Tasarruf Mevduatı", "kisi_basi_tasarruf_mevduati", "TL", False),
    (6, "Kişi Başı Toplam Mevduat", "kisi_basi_mevduat", "TL", False),
]


def build_finturk_regional_long() -> pd.DataFrame:
    """Every province-level FinTurk metric in one long table.

    WHY LONG: the metric set is data, not schema. A wide table would need a
    column (and a code change) per metric, and the agent tool would need a
    matching enum entry - which is paid for in the model's context on every
    single request. As rows, 36 metrics cost the prompt nothing; the tool
    takes the metric name as a string and validates it against this table.

    Only 'SEKTÖR' rows are kept. The source also ships per-bank-group rows
    in some tables, and summing across groups would double-count.
    """
    logger.info("build_finturk_regional_long: start")
    frames = []
    by_table: dict[int, pd.DataFrame] = {}

    for table_no, column, slug, unit, additive in FINTURK_REGIONAL_METRICS:
        if table_no not in by_table:
            by_table[table_no] = _load_finturk_table(table_no)
        source = by_table[table_no]
        if source.empty:
            logger.warning(f"build_finturk_regional_long: tablo_{table_no} empty, skipping {slug}")
            continue
        if column not in source.columns:
            # A renamed source column must be loud: silently dropping a
            # metric would leave the tool reporting "unknown metric" with
            # no clue why.
            logger.error(
                f"build_finturk_regional_long: column {column!r} missing from tablo_{table_no}; "
                f"metric {slug!r} will not be built"
            )
            continue

        rows = source[source["Grup"] == "SEKTÖR"] if "Grup" in source.columns else source
        part = pd.DataFrame({
            "Period": rows["Donem"].astype(str),
            "Bolge": rows["Şehir"].astype(str).str.strip(),
            "Metrik": slug,
            "Deger": pd.to_numeric(rows[column], errors="coerce"),
            "Birim": unit,
            "Toplanabilir": additive,
            "Kaynak_Tablo": f"finturk/tablo_{table_no}",
        })
        dropped = int(part["Deger"].isna().sum())
        if dropped:
            logger.warning(f"build_finturk_regional_long: {slug} dropped {dropped} non-numeric row(s)")
        part = part[part["Deger"].notna()]
        logger.debug(f"build_finturk_regional_long: {slug} -> {len(part)} rows")
        frames.append(part)

    if not frames:
        logger.error("build_finturk_regional_long: nothing built")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    logger.info(
        f"build_finturk_regional_long: done, {len(out)} rows, "
        f"{out['Metrik'].nunique()} metric(s), {out['Bolge'].nunique()} bolge, "
        f"{out['Period'].nunique()} period(s)"
    )
    return out


def build_finturk_national_facts() -> pd.DataFrame:
    """National total credit from FinTurk (sum across all 82 rows: 81
    provinces + 'YURT DISI'/abroad), quarterly -> upsampled to monthly
    via forward-fill (Faz A1's explicit approximation, see align.py) so
    it can sit in the same monthly gold_fact_long as everything else."""
    long_df = _load_finturk_table1()
    if long_df.empty:
        return pd.DataFrame()

    national_quarterly = long_df.groupby("Donem", as_index=False)["Toplam Nakdi Krediler"].sum()
    national_quarterly = national_quarterly.rename(columns={"Toplam Nakdi Krediler": "Value"})

    monthly = upsample_quarterly_to_monthly(national_quarterly, quarter_col="Donem", value_cols=["Value"])

    # Unit "Bin TL" here comes from BDDK_FINTURK_TABLES in config/settings.py
    # ("Loans (Thousand TRY)") - UNLIKE BDDK_MONTHLY_TABLES's units above,
    # this one is NOT confirmed by a caption field (checked: FinTurk's
    # bronze JSON has none) - carried over as an existing assumption, not
    # independently verified in this pass. See docs/gold_schema.md.
    rows = []
    for _, r in monthly.iterrows():
        tip = "flow" if r["_period_type"] == "observed" else "flow_quarter_ffill"
        rows.append(_fact_row(r["Period"], "bddk_finturk", "tablo_1.toplam_nakdi_kredi_ulusal",
                               "Toplam Nakdi Krediler, Turkiye Geneli (81 il + Yurt Disi Toplami)",
                               r["Value"], "Bin TL (dogrulanmadi)", tip, "tablo_1/*.csv", bolge="TR"))
    return pd.DataFrame(rows)


def build_finturk_regional_quarterly() -> pd.DataFrame:
    """Province-level credit, kept at its NATIVE quarterly grain (not
    forced onto the monthly axis - 81 provinces x monthly forward-fill
    would triple the row count for no real information gain over the
    quarterly figure). First regional (Bolge) dimension in the gold
    layer - answers "which il" questions the other tables can't."""
    long_df = _load_finturk_table1()
    if long_df.empty:
        return pd.DataFrame()

    out = long_df[["Donem", "Şehir", "Toplam Nakdi Krediler", "Takipteki Alacaklar"]].rename(columns={
        "Donem": "Period", "Şehir": "Bolge",
        "Toplam Nakdi Krediler": "Toplam_Nakdi_Kredi_Bin_TL",
        "Takipteki Alacaklar": "Takipteki_Alacak_Bin_TL",
    })
    out["NPL_Orani_Yaklasik"] = out["Takipteki_Alacak_Bin_TL"] / out["Toplam_Nakdi_Kredi_Bin_TL"]
    return out.sort_values(["Period", "Bolge"]).reset_index(drop=True)


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

    # Column named _Milyon_TL, not _Bin_TL: BDDK's own bronze caption for
    # tablo_4 says "(milyon TL)" (see BDDK_MONTHLY_TABLES in
    # config/settings.py) - an earlier version of this table mislabeled
    # the unit as "Bin TL". Fixed here; the underlying numbers never changed.
    konut = series("tablo_4.konut_kredisi", "Konut_Kredisi_Milyon_TL")
    faiz = series("TP.BKR.TRY.18", "Konut_Kredisi_Faiz_Orani")
    endeks = series("TP.KFE.TR", "Konut_Fiyat_Endeksi")
    tufe = series("TP.GENENDEKS.T1", "TUFE_Endeksi")

    wide = konut.merge(faiz, on="Period", how="left") \
                .merge(endeks, on="Period", how="left") \
                .merge(tufe, on="Period", how="left") \
                .sort_values("Period").reset_index(drop=True)

    base_cpi = wide["TUFE_Endeksi"].iloc[0]
    wide["Konut_Kredisi_Reel_Milyon_TL"] = wide["Konut_Kredisi_Milyon_TL"] * (base_cpi / wide["TUFE_Endeksi"])
    wide["Reel_Baz_Donem"] = wide["Period"].iloc[0]

    return wide


def run(db_path=GOLD_DB_PATH) -> None:
    logger.info("Building gold_fact_long from evds, tuik, bddk_monthly, bddk_weekly, bddk_finturk ...")
    fact_long = pd.concat([
        build_evds_facts(),
        build_tuik_facts(),
        build_bddk_monthly_konut_facts(),
        build_bddk_monthly_item_facts(3, "Toplam Krediler", "tablo_3.toplam_krediler", "Toplam Krediler, Sektor Geneli"),
        build_bddk_monthly_item_facts(9, "Toplam Mevduat", "tablo_9.toplam_mevduat", "Toplam Mevduat, Sektor Geneli"),
        build_bddk_monthly_item_facts(12, "Sermaye Yeterliliği Standart Rasyosu ((5/7)*100) (YÜZDE)",
                                       "tablo_12.sermaye_yeterlilik_rasyosu", "Sermaye Yeterliliği Standart Rasyosu", birim="%"),
        # 3 new items from tablo_1 (Balance Sheet), all bronze/silver already
        # collected - no new network calls. Each item_name is matched
        # EXACTLY against real Column_2 text (not a substring guess) and
        # confirmed non-cumulative against
        # cumulative_scan_report_bddk_monthly.csv before being added
        # here (same discipline as the pre-existing items above).
        # Chosen as a set on purpose: TOPLAM AKTIFLER (assets) and
        # TOPLAM PASIFLER (liabilities) must be equal by the balance
        # sheet's own accounting identity - a genuine cross-check on the
        # pipeline's correctness, not just "did it run without
        # crashing" (see tests/test_gold_data_integrity.py).
        build_bddk_monthly_item_facts(1, "TOPLAM AKTİFLER", "tablo_1.toplam_aktifler", "Toplam Aktifler, Sektor Geneli"),
        build_bddk_monthly_item_facts(1, "TOPLAM PASİFLER", "tablo_1.toplam_pasifler", "Toplam Pasifler, Sektor Geneli"),
        build_bddk_monthly_item_facts(1, "TOPLAM ÖZKAYNAKLAR", "tablo_1.toplam_ozkaynaklar", "Toplam Ozkaynaklar (Sermaye), Sektor Geneli"),
        # 3 more, same zero-new-network-call discipline as above, this
        # time from tablo_6 (SME Loans), tablo_11 (Liquidity Position),
        # tablo_15 (Ratios) - all confirmed non-cumulative.
        # NOTE on what was deliberately NOT added here: tablo_15's ROA/
        # ROE-style rows ("Donem Net Kari / Ortalama Toplam Aktifler
        # (%)" etc.) came back is_cumulative=True in the scan report -
        # their numerator (period net profit) is itself YTD-cumulative
        # (same root cause as tablo_2's net profit, see
        # build_bddk_net_profit_monthly below), so the raw ratio is NOT
        # a comparable month-to-month figure without properly
        # decumulating the numerator first, not just diffing the ratio
        # itself. Left out rather than shipped wrong - a real trap this
        # scan caught, not an oversight.
        build_bddk_monthly_kobi_facts(),
        build_bddk_monthly_item_facts(11, "Likidite Yeterlilik Oranı", "tablo_11.likidite_yeterlilik_orani", "Likidite Yeterlilik Orani", birim="%"),
        build_bddk_monthly_item_facts(15, "Takipteki Alacaklar (Brüt) / Toplam Nakdi Krediler (%)", "tablo_15.takipteki_alacaklar_orani", "Takipteki Alacaklar Orani (NPL), Sektor Geneli", birim="%"),
        build_bddk_weekly_konut_facts(),
        build_finturk_national_facts(),
    ], ignore_index=True)

    logger.info("Decumulating BDDK Monthly net profit (tablo_2) ...")
    net_profit = build_bddk_net_profit_monthly()

    logger.info("Building gold_konut_demo (hackathon worked example) ...")
    konut_demo = build_konut_demo_wide(fact_long)

    logger.info("Building gold_finturk_regional_quarterly (province-level, native quarterly grain) ...")
    finturk_regional = build_finturk_regional_quarterly()

    logger.info("Building gold_health_life_expectancy (native annual grain) ...")
    life_expectancy = build_health_life_expectancy()

    logger.info("Building gold_regional_long (all province-level FinTurk metrics) ...")
    regional_long = build_finturk_regional_long()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.register("fact_long_df", fact_long)
    con.execute("CREATE OR REPLACE TABLE gold_fact_long AS SELECT * FROM fact_long_df")

    if not net_profit.empty:
        con.register("net_profit_df", net_profit)
        con.execute("CREATE OR REPLACE TABLE gold_bddk_net_profit_monthly AS SELECT * FROM net_profit_df")

    con.register("konut_demo_df", konut_demo)
    con.execute("CREATE OR REPLACE TABLE gold_konut_demo AS SELECT * FROM konut_demo_df")

    if not finturk_regional.empty:
        con.register("finturk_regional_df", finturk_regional)
        con.execute("CREATE OR REPLACE TABLE gold_finturk_regional_quarterly AS SELECT * FROM finturk_regional_df")

    if not life_expectancy.empty:
        con.register("life_expectancy_df", life_expectancy)
        con.execute("CREATE OR REPLACE TABLE gold_health_life_expectancy AS SELECT * FROM life_expectancy_df")

    if not regional_long.empty:
        con.register("regional_long_df", regional_long)
        con.execute("CREATE OR REPLACE TABLE gold_regional_long AS SELECT * FROM regional_long_df")

    con.close()

    logger.info("=" * 50)
    logger.info("GOLD BUILD SUMMARY")
    logger.info(f"gold_fact_long                 : {len(fact_long)} rows, {fact_long['Seri_Kod'].nunique()} series")
    logger.info(f"gold_bddk_net_profit_monthly   : {len(net_profit)} rows")
    logger.info(
        f"gold_regional_long             : {len(regional_long)} rows, "
        f"{regional_long['Metrik'].nunique() if not regional_long.empty else 0} metrics, "
        f"{regional_long['Bolge'].nunique() if not regional_long.empty else 0} bolge"
    )
    logger.info(f"gold_konut_demo                : {len(konut_demo)} rows, {konut_demo['Period'].min()}..{konut_demo['Period'].max()}")
    logger.info(f"gold_finturk_regional_quarterly: {len(finturk_regional)} rows, {finturk_regional['Bolge'].nunique() if not finturk_regional.empty else 0} bolge")
    logger.info(f"Written to                     : {db_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    argparse.ArgumentParser(description="Build the gold layer into DuckDB (no network)").parse_args()
    run()
