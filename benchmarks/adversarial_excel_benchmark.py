"""Adversarial benchmark against PRIVATE, real user-uploaded Excel/CSV/PDF
files - not synthetic fixtures, not the public BDDK/TUIK/EVDS gold layer.

The files themselves live in test_excels/ (gitignored - see .gitignore's
"Private real-world test Excel/PDF files" entry), a folder the user drops
new files into over time. This script and its question sets ARE committed;
the files are not, and neither their names nor their content belong in any
committed file, commit message, or README.

Unlike benchmarks/hard_benchmark.py (which checks tool selection and
numeric grounding across ~150 varied questions against the known gold
schema), this benchmark asks a SMALL number of deliberately hard,
trap-laden questions against ONE real file's actual, quirky content -
questions designed to catch a specific class of failure: the model
treating superficially similar columns as interchangeable, or reporting a
plausible-sounding but WRONG fact instead of reading the file's own
ground truth. Each question below states its trap and its ground truth
explicitly, both verified by hand against the real file before being
written down here (see the commit that added this file for how).

Question sets are keyed by filename stem, since each real file has its
own quirks that only make sense against that file's actual schema - a
file with no matching question set is still imported and schema-checked
(so nothing in test_excels/ is silently skipped), just without the
hand-crafted traps.

Run with:
    python -m benchmarks.adversarial_excel_benchmark
    python -m benchmarks.adversarial_excel_benchmark --file EVDS_16-09-2026
"""
import argparse
import json
import re
import time
from pathlib import Path

from config.settings import PROJECT_ROOT
from src.agent.data_import import detect_schema, import_dataset
from src.agent.orchestrator import ask
from src.agent.pdf_import import import_pdf_table
from src.common.logging_setup import get_logger

logger = get_logger("adversarial_excel_benchmark")

TEST_EXCELS_DIR = PROJECT_ROOT / "test_excels"
REPORT_PATH = Path(__file__).parent / "adversarial_excel_benchmark_report.json"


# --- question sets, one per known file stem -------------------------------
#
# Each question carries:
#   trap        - what specific failure this is fishing for (documentation,
#                 not shown to the model)
#   ground_truth - what was verified by hand against the real file
#   must_include / must_not_include - lowercased substrings the answer is
#                 auto-checked against (best-effort signal, not a hard
#                 grade - see report's own "auto_check" field: a genuine
#                 trap question often has no single-substring answer, so
#                 "review" is a legitimate, expected outcome, not a bug in
#                 the check).

EVDS_16_09_2026_QUESTIONS = [
    {
        "id": "ibbs_region_name_trap",
        "trap": (
            "The obvious-looking answer is wrong: TR10 is Istanbul, not "
            "Balikesir/Canakkale (that's TR22) - a model that pattern-"
            "matches 'TRxx = some Marmara pair of provinces' without "
            "actually reading the file's own 'Seri Aciklamalari' block "
            "(rows 10+, already surfaced in an earlier real answer in this "
            "same session) will get this wrong with total confidence."
        ),
        "ground_truth": "TR22 = Balikesir, Canakkale (verified against the file's own row 17: 'TP.KFE.TR22, TR22 (Balikesir, Canakkale)-Duzey').",
        "question": (
            "Veri setindeki bolgesel kodlarin (TR10, TR22 vb.) Istatistiki "
            "Bolge Birimleri Siniflandirmasi (IBBS) kodlari oldugunu fark "
            "et. TR22 kodunun hangi illeri kapsadigini bul (dosyanin kendi "
            "aciklama bolumunu kullan, tahmin etme). Ardindan TR22 "
            "endeksinin 2021-2026 getirisini 'Balikesir-Canakkale bolgesi' "
            "olarak raporla."
        ),
        "must_include": ["balıkesir", "çanakkale"],
        "must_not_include": ["istanbul"],
    },
    {
        "id": "ibbs_classification_level_mismatch",
        "trap": (
            "TP_KFE_TR7/TR8/TR9/TRA/TRB/TRC are IBBS LEVEL-1 codes (each "
            "covers 6-9 provinces), while TP_KFE_TR10/TR51/TR31/etc are "
            "LEVEL-2 codes (each covers 1-4 provinces) - genuinely different "
            "granularity mixed in the same flat column list. A model that "
            "averages all 21 regional columns as if they were peer, "
            "equal-weight regions is silently comparing a single city "
            "(TR10 = Istanbul alone) against a 9-province aggregate (TRC) "
            "on equal footing."
        ),
        "ground_truth": (
            "Confirmed by hand from the file's own descriptions: TR7 covers "
            "8 provinces (Nevsehir..Yozgat), TRC covers 9 (Gaziantep..Siirt), "
            "while TR10 covers exactly 1 (Istanbul) and TR51 exactly 1 "
            "(Ankara) - NOT the same classification level, despite the "
            "shared 'TRxx' naming pattern."
        ),
        "question": (
            "Sutun isimlerindeki tum bolgesel kodlari (TR10, TR51, TR31, "
            "TR21, TR22, TR32, TR33, TR41, TR42, TR52, TR61, TR62, TR63, "
            "TR7, TR8, TR9, TRA, TRB, TRC) listele. Bunlarin hepsi ayni "
            "Istatistiki Bolge Birimleri Siniflandirmasi (IBBS) seviyesinde "
            "mi, yoksa farkli seviyeler mi karisik? Bu 21 sutunun basit "
            "ortalamasini alarak 'Turkiye genelinde bolgesel KFE ortalamasi' "
            "hesaplamak metodolojik olarak neden yanlis olabilir, acikla."
        ),
        "must_include": [],  # no single substring proves this - review by hand
        "must_not_include": [],
    },
    {
        "id": "observation_method_mismatch",
        "trap": (
            "The file's own metadata (rows 11 and 12) states two DIFFERENT "
            "observation conventions in the same workbook: "
            "TP.ALTINPIYASA.KAP02 is 'Gozlem Degeri: Bitis' (period-END "
            "value) while TP.KFE.TR is 'Gozlem Degeri: Aritmetik Ortalama' "
            "(period-AVERAGE value). An earlier real answer in this exact "
            "session computed a gold-vs-housing CAGR/real-return comparison "
            "WITHOUT ever surfacing this - directly comparing a period-end "
            "price to a period-average index as if they were the same kind "
            "of number."
        ),
        "ground_truth": (
            "Confirmed: row 11's third cell literally reads 'Gozlem Degeri: "
            "Bitis' for gold, row 12's reads 'Gozlem Degeri: Aritmetik "
            "Ortalama' for KFE - both present verbatim in the file."
        ),
        "question": (
            "TP_ALTINPIYASA_KAP02 (altin) ve TP_KFE_TR (konut endeksi) "
            "serilerinin 'Seri Aciklamalari' bolumunde belirtilen gozlem "
            "yontemleri (Gozlem Degeri sutunu) ayni mi, farkli mi? Ikisini "
            "doğrudan karsilastirarak bir getiri/CAGR analizi yaparken bu "
            "fark onemli mi, aciklaya biliyor musun?"
        ),
        "must_include": ["bitiş", "ortalama"],
        "must_not_include": [],
    },
    {
        "id": "nonexistent_region_code",
        "trap": (
            "TR11 does not exist as a column in this file (real level-2 "
            "codes present are TR10/TR21/TR22/TR31/TR32/TR33/TR41/TR42/"
            "TR51/TR52/TR61/TR62/TR63, plus level-1 TR7/TR8/TR9/TRA/TRB/"
            "TRC) - a model must say the column/series does not exist, "
            "not silently substitute a nearby code's data or invent a "
            "number."
        ),
        "ground_truth": "TP_KFE_TR11 is not a column in this dataset - confirmed against the real 22-column header.",
        "question": "TP_KFE_TR11 bolgesinin 2021'den 2026'ya buyume orani nedir?",
        "must_include": [],
        "must_not_include": [],  # checked via admits_missing below instead
        "expect_admits_missing": True,
    },
]

QUESTION_SETS = {
    "EVDS_16-09-2026": EVDS_16_09_2026_QUESTIONS,
}

_MISSING_DATA_RE = re.compile(
    r"bulunam|mevcut de[gğ]il|yok\b|eri[sş]ilem|veri(?:si)? yok|"
    r"s[uü]tun(?:u)? yok|kolon(?:u)? yok|tanımlı de[gğ]il",
    re.IGNORECASE,
)


def _import_file(path: Path) -> str:
    """Returns the dataset_name it was imported under."""
    stem = re.sub(r"[^a-zA-Z0-9_]", "_", path.stem).lower()
    dataset_name = f"advtest_{stem}"
    if path.suffix.lower() == ".pdf":
        result = import_pdf_table(path, dataset_name)
    else:
        result = import_dataset(path, dataset_name)
    logger.info(f"_import_file: {path.name} -> '{dataset_name}' ({result['row_count']} rows, {result['column_count']} cols)")
    return dataset_name


def _check_answer(item: dict, answer: str) -> dict:
    lowered = answer.lower()
    must_include = item.get("must_include", [])
    must_not_include = item.get("must_not_include", [])
    include_ok = all(s in lowered for s in must_include)
    exclude_ok = all(s not in lowered for s in must_not_include)

    if item.get("expect_admits_missing"):
        admitted = bool(_MISSING_DATA_RE.search(lowered))
        return {"verdict": "PASS" if admitted else "FAIL", "detail": f"admits_missing={admitted}"}

    if not must_include and not must_not_include:
        return {"verdict": "REVIEW", "detail": "No auto-checkable substring for this trap - read the answer by hand."}

    verdict = "PASS" if (include_ok and exclude_ok) else "FAIL"
    return {
        "verdict": verdict,
        "detail": f"must_include_ok={include_ok}, must_not_include_ok={exclude_ok}",
    }


def run_file(path: Path) -> dict:
    logger.info(f"run_file: {path.name}")
    schema = detect_schema(path)
    dataset_name = _import_file(path)

    questions = QUESTION_SETS.get(path.stem, [])
    if not questions:
        logger.warning(f"run_file: no hand-crafted question set for '{path.stem}' - imported and schema-checked only.")

    results = []
    for item in questions:
        logger.info(f"run_file: asking '{item['id']}'")
        t0 = time.monotonic()
        answer, trace = ask(item["question"], return_trace=True)
        elapsed = round(time.monotonic() - t0, 1)
        check = _check_answer(item, answer)
        results.append({
            "id": item["id"], "trap": item["trap"], "ground_truth": item["ground_truth"],
            "question": item["question"], "answer": answer, "elapsed_sec": elapsed,
            "tools_called": [t["name"] for t in trace], "auto_check": check,
        })
        logger.info(f"  [{check['verdict']}] {item['id']} ({elapsed}s)")

    return {
        "file": path.name, "dataset_name": dataset_name,
        "schema": {"row_count": schema["row_count"], "column_count": schema["column_count"],
                   "data_quality": schema["data_quality"]},
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Adversarial benchmark against private real Excel/CSV/PDF files in test_excels/")
    parser.add_argument("--file", help="Only run this file's stem (e.g. 'EVDS_16-09-2026')")
    args = parser.parse_args()

    if not TEST_EXCELS_DIR.exists():
        logger.error(f"main: {TEST_EXCELS_DIR} does not exist - nothing to test.")
        raise SystemExit(1)

    files = sorted(
        p for p in TEST_EXCELS_DIR.iterdir()
        if p.suffix.lower() in (".xlsx", ".xls", ".csv", ".pdf") and (not args.file or p.stem == args.file)
    )
    if not files:
        logger.error(f"main: no supported file found in {TEST_EXCELS_DIR} (looked for .xlsx/.xls/.csv/.pdf).")
        raise SystemExit(1)

    reports = [run_file(p) for p in files]

    REPORT_PATH.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("ADVERSARIAL EXCEL/PDF BENCHMARK (private test_excels/ files)")
    print("=" * 70)
    for r in reports:
        print(f"\n{r['file']}  ({r['schema']['row_count']} rows, {r['schema']['column_count']} cols)")
        if not r["results"]:
            print("  (no hand-crafted question set for this file)")
        for item in r["results"]:
            print(f"  [{item['auto_check']['verdict']:6s}] {item['id']:35s} {item['elapsed_sec']:5.1f}s")
    print("-" * 70)
    print(f"Full report: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
