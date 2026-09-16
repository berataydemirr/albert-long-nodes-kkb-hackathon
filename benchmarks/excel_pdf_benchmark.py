"""Excel/PDF import benchmark - real files, not synthetic fixtures.

Every other benchmark and test fixture in this project builds its own
Excel/PDF with known, controlled content (see hard_benchmark.py's
_setup_synthetic_fixtures, or the reportlab-built PDFs in
tests/test_pdf_import.py). Those are the right tool for pinning down a
SPECIFIC scenario, but they cannot catch what a real government export
actually looks like - a real file was never designed to be easy to parse.

This benchmark downloads REAL, currently-published files from two Turkish
public institutions (chosen from the same research pass that widened the
project's data sources - see config/evds_series.py's TP.GY1.N2 entry and
this session's README for the others that turned out to have no real
public API: KKB/Findeks and TBB Risk Merkezi) and runs them through the
actual import pipeline (src/agent/data_import.py, src/agent/pdf_import.py)
- not a mock, not a trimmed-down copy of the file:

  1. TBB (Turkiye Bankalar Birligi) - a real legacy .xls export
     ("Banka ve Gruplar Bazinda Sube ve Calisan Sayilari"). Found two real
     bugs live while first testing this file (see git history around this
     benchmark's own commit): openpyxl cannot open .xls at all despite it
     being listed as a supported extension, and a real full-width merged
     title (not this project's target "year over months" merge shape)
     crashed schema inference with duplicate column names. Both fixed in
     src/agent/data_import.py; this benchmark is the regression check that
     exercises them against the real file, not just a minimal repro.

  2. Hazine ve Maliye Bakanligi - a real "Kamu Borc Yonetimi Raporu"
     (public debt management report) PDF, 47 pages, dozens of real tables.
     Found: the 30-page cap rejected this ordinary report outright (raised
     to 60, see pdf_import.py's MAX_PDF_PAGES comment), and PDF tables had
     no merged-header handling at all despite Excel already having one for
     the identical "year over quarters/months" shape (added
     _flatten_pdf_merged_header). One real, honest, NOT fixed finding is
     also recorded here: a currency-unit header cell ('Milyon TL') comes
     back as 'Milyon \xa8' on some pages - the source PDF's embedded font
     has a broken ToUnicode CMap for that specific glyph, which is
     information already lost at PDF-text-extraction time, not a byte-
     encoding mismatch repair_mojibake can undo.

Network dependency: this benchmark downloads both files fresh unless a
cached copy already exists in data/debug/real_world_samples/ (gitignored -
see data/debug/ in .gitignore) - same convention as hard_benchmark.py's own
live-API calls, kept out of pytest for the same reason (network + non-
determinism). Run with:

    python -m benchmarks.excel_pdf_benchmark
"""
import json
from pathlib import Path

import requests

from config.settings import DEBUG_DIR
from src.agent.data_import import LAKEHOUSE_DB_PATH, detect_schema, import_dataset
from src.agent.pdf_import import import_pdf_table, list_pdf_tables
from src.common.logging_setup import get_logger

logger = get_logger("excel_pdf_benchmark")

SAMPLES_DIR = DEBUG_DIR / "real_world_samples"

REAL_FILES = {
    "tbb_sube_calisan_xls": {
        "url": "https://www.tbb.org.tr/sites/default/files/raporlar/51246/tablolar/tablo_2-banka_ve_gruplar_bazinda_Sube_ve_Calisan_sayilari_0.xls",
        "filename": "tbb_sube_calisan.xls",
        "source": "Turkiye Bankalar Birligi - Banka ve Gruplar Bazinda Sube ve Calisan Sayilari (legacy .xls)",
    },
    "hmb_kamu_borc_pdf": {
        "url": "https://ms.hmb.gov.tr/uploads/2025/12/Web_Kamu_Borc_Yonetimi_Raporu_Aralik_2025-468fb79b5ceac818.pdf",
        "filename": "hmb_kamu_borc_aralik_2025.pdf",
        "source": "Hazine ve Maliye Bakanligi - Kamu Borc Yonetimi Raporu, Aralik 2025",
    },
}

# Real pages in the HMB report worth exercising - picked by hand after a
# live scan of every page's extract_tables() output (see this benchmark's
# own commit message / session notes), not guessed:
#   5  - the merged-header + multi-page-continuation case (the headline
#        find: a year merged over month columns, stitched across 3 pages)
#   39 - an ordinary single-row-header table (the "nothing went wrong"
#        control case - a benchmark that only ever reports failures proves
#        nothing about the successes)
#   41 - a genuinely nested two-level header ("Vadesi Gecmis" /
#        "Vadesi Gelmemis" > "Anapara"/"Diger"/"Toplam") - looked like it
#        might be a harder shape than the flat "top spans sub" case
#        _flatten_pdf_merged_header targets, but the same top-forward-fill
#        logic resolves it correctly (verified live: 51 rows, columns
#        like 'Vadesi Gelmemis Alacaklar_Anapara' come out right) since
#        the nesting still reduces to one blank-or-not cell per column.
HMB_TEST_PAGES = [5, 39, 41]


def _ensure_downloaded(key: str) -> Path:
    meta = REAL_FILES[key]
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / meta["filename"]
    if path.exists():
        logger.info(f"Cache hit: {key} -> {path}")
        return path
    logger.info(f"Downloading real file: {key} <- {meta['url']}")
    resp = requests.get(meta["url"], timeout=40)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    logger.info(f"Saved: {path} ({len(resp.content)} bytes)")
    return path


def _test_excel(path: Path) -> dict:
    result = {"file": path.name, "format": path.suffix.lower(), "tests": []}

    try:
        schema = detect_schema(path)
        result["tests"].append({
            "step": "detect_schema", "ok": True,
            "row_count": schema["row_count"], "column_count": schema["column_count"],
            "data_quality": schema["data_quality"],
        })
    except Exception as e:
        result["tests"].append({"step": "detect_schema", "ok": False, "error": f"{type(e).__name__}: {e}"})
        return result

    try:
        imported = import_dataset(path, "xlpdfbench_tbb_sube_calisan")
        result["tests"].append({
            "step": "import_dataset", "ok": True,
            "row_count": imported["row_count"],
            "columns": [c["column"] for c in imported["columns"]],
        })
    except Exception as e:
        result["tests"].append({"step": "import_dataset", "ok": False, "error": f"{type(e).__name__}: {e}"})

    return result


def _test_pdf(path: Path) -> dict:
    result = {"file": path.name, "format": ".pdf", "pages": {}}

    preview = list_pdf_tables(path)
    result["page_count"] = preview["page_count"]
    result["tables_found_total"] = len(preview["tables"])

    for page in HMB_TEST_PAGES:
        try:
            imported = import_pdf_table(path, f"xlpdfbench_hmb_pg{page}", page=page)
            result["pages"][str(page)] = {
                "ok": True,
                "row_count": imported["row_count"],
                "column_count": imported["column_count"],
                "columns": [c["column"] for c in imported["columns"]],
                "data_quality": imported["data_quality"],
                "stitched": imported.get("stitched"),
            }
        except Exception as e:
            result["pages"][str(page)] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return result


def _cleanup_bench_tables():
    """Benchmark datasets get written into the REAL lakehouse.duckdb (this
    script deliberately does not monkeypatch LAKEHOUSE_DB_PATH - the whole
    point is exercising the real write path against real data), so they
    are dropped again afterward the same way manual verification tables
    have been cleaned up elsewhere this session.

    Real bug found live in an earlier version of this function: a LIKE
    'bench_%' pattern also matched hard_benchmark.py's OWN persistent
    fixtures ('benchmark_synthetic_csv' / 'benchmark_synthetic_pdf', see
    that module's _setup_synthetic_fixtures) and silently deleted them -
    the 'xlpdfbench_' prefix below (matching the one import_dataset/
    import_pdf_table calls above actually use) exists specifically so
    this LIKE pattern cannot ever collide with an unrelated table again."""
    import duckdb

    if not LAKEHOUSE_DB_PATH.exists():
        return
    con = duckdb.connect(str(LAKEHOUSE_DB_PATH))
    try:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'xlpdfbench_%'"
        ).fetchall()]
        for t in tables:
            con.execute(f'DROP TABLE IF EXISTS "{t}"')
            con.execute("DELETE FROM _import_registry WHERE dataset_name = ?", [t])
        if tables:
            logger.info(f"Cleaned up {len(tables)} benchmark table(s): {tables}")
    finally:
        con.close()


def main():
    logger.info("=" * 70)
    logger.info("EXCEL/PDF IMPORT BENCHMARK - real files, not synthetic fixtures")
    logger.info("=" * 70)

    report = {"excel": None, "pdf": None}

    xls_path = _ensure_downloaded("tbb_sube_calisan_xls")
    report["excel"] = _test_excel(xls_path)
    report["excel"]["source"] = REAL_FILES["tbb_sube_calisan_xls"]["source"]

    pdf_path = _ensure_downloaded("hmb_kamu_borc_pdf")
    report["pdf"] = _test_pdf(pdf_path)
    report["pdf"]["source"] = REAL_FILES["hmb_kamu_borc_pdf"]["source"]

    _cleanup_bench_tables()

    out_path = Path(__file__).parent / "excel_pdf_benchmark_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("-" * 70)
    logger.info("EXCEL (TBB, legacy .xls):")
    for t in report["excel"]["tests"]:
        status = "OK" if t["ok"] else "FAIL"
        logger.info(f"  [{status}] {t['step']}: {t.get('row_count', t.get('error'))}")

    logger.info("PDF (HMB, Kamu Borc Yonetimi Raporu, 47 pages):")
    for page, r in report["pdf"]["pages"].items():
        status = "OK" if r["ok"] else "FAIL"
        detail = f"{r['row_count']} rows, {r['column_count']} cols, stitched={r.get('stitched')}" if r["ok"] else r["error"]
        logger.info(f"  [{status}] page {page}: {detail}")

    logger.info("-" * 70)
    logger.info(f"Full report: {out_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
