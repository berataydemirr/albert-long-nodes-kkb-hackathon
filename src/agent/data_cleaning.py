"""Shared data-quality helpers for CSV/Excel/PDF import.

Every function here handles one specific real-world messiness a file a
user hands the system can contain - each one was built against a
concrete scenario (see the module-level docstring on each function) and
tested with a realistic synthetic fixture, not a hypothetical. None of
this replaces the vetted BDDK/EVDS/TUIK pipeline; it only makes runtime
import (data_import.py, pdf_import.py) more resilient to the kind of
mess a real spreadsheet or PDF export actually contains.

Deliberately conservative throughout: every function here either
normalizes something unambiguous, or - where the input is genuinely
ambiguous (see normalize_numeric_text) - returns a best-effort answer
without silently pretending certainty it does not have. Nothing here
invents a value that was not derivable from the input.
"""
import difflib
import re
from datetime import datetime

from src.common.logging_setup import get_logger

logger = get_logger("data_cleaning")

# --- 5. Farklı boş veri gösterimleri ----------------------------------------
# A cell meaning "no data" shows up in wildly different ways across real
# files - a bare dash, "N/A" in either case, a literal empty string, or a
# few less common markers. pandas' own read_csv/read_excel already treats
# SOME of these as NaN by default (its na_values list), but not all real
# variants seen in Turkish public-sector exports (".." for a suppressed
# statistic is a TUIK/Eurostat convention, for one).
NULL_TOKENS = {
    "", "-", "--", "---", "n/a", "na", "n.a.", "n.a", "null", "none",
    "nan", "yok", "veri yok", "bilinmiyor", "belirsiz", "..", "...",
    "#n/a", "#yok",
    # Excel formula-error results (scenario 7: a hidden row left over
    # from a broken formula) - these are not data, they are Excel
    # reporting that a CELL'S OWN FORMULA failed, and must never be
    # summed/averaged in as if they were a real zero or missing-but-valid
    # observation.
    "#ref!", "#div/0!", "#value!", "#name?", "#null!", "#num!",
}


def is_null_token(value) -> bool:
    """True if a raw cell value (already stringified) means 'no data' by
    one of the conventions above, case/whitespace-insensitive."""
    if value is None:
        return True
    s = str(value).strip().lower()
    return s in NULL_TOKENS


# --- 3. Avrupa/Amerikan sayı formatı + 8. Kısaltılmış büyüklükler ----------
# A PDF or a pasted-in Excel cell has no cell dtype/locale metadata - every
# number arrives as plain text in whatever convention the source used, and
# on top of that may be abbreviated ("1.5M", "2.3B", "1,5 milyon").
_SUFFIX_MULTIPLIERS = {
    "k": 1_000, "bin": 1_000,
    "m": 1_000_000, "mn": 1_000_000, "milyon": 1_000_000, "mio": 1_000_000,
    "b": 1_000_000_000, "mlr": 1_000_000_000, "milyar": 1_000_000_000, "bn": 1_000_000_000,
    "t": 1_000_000_000_000, "trilyon": 1_000_000_000_000, "tn": 1_000_000_000_000,
}
# Longest keys first so 'milyar' is tried before a shorter accidental
# prefix match; word boundary anchored so 'M' does not match inside 'MB'.
_SUFFIX_RE = re.compile(
    r"^(-?[\d.,]+)\s*(" + "|".join(sorted(_SUFFIX_MULTIPLIERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def normalize_numeric_text(raw: str) -> float | None:
    """Best-effort text -> float, handling:
      - Turkish thousands/decimal separators ('1.234,56') vs plain
        international decimal notation ('2.009') - only a value that ALSO
        contains a comma is treated as Turkish-formatted; a period-only
        value is read as an ordinary decimal point. This is the same
        disambiguation rule pdf_import.py uses (see its own
        _normalize_numeric_cell) - kept here as the one shared
        implementation instead of two that can drift.
      - Abbreviated magnitude suffixes ('1.5M' -> 1_500_000, '2,3 milyar'
        -> 2_300_000_000), Turkish and English, case-insensitive.
    Returns None (not 0 or a guess) when the text does not parse as a
    number at all - the caller decides whether that means "leave as
    text" or "treat as missing"."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or is_null_token(s):
        return None

    m = _SUFFIX_RE.match(s)
    multiplier = 1.0
    if m:
        number_part, suffix = m.group(1), m.group(2).lower()
        multiplier = _SUFFIX_MULTIPLIERS[suffix]
        s = number_part

    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    # else: period-only or no separator at all - read as an ordinary
    # decimal point / plain integer, the international convention.

    try:
        return float(s) * multiplier
    except ValueError:
        return None


# --- 2. Karmaşık tarih formatları -------------------------------------------
_TR_MONTHS = {
    "ocak": 1, "şubat": 2, "subat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "mayis": 5,
    "haziran": 6, "temmuz": 7, "ağustos": 8, "agustos": 8, "eylül": 9, "eylul": 9,
    "ekim": 10, "kasım": 11, "kasim": 11, "aralık": 12, "aralik": 12,
}
_QUARTER_END_MONTH = {1: "03", 2: "06", 3: "09", 4: "12"}

_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?$")
_ISO_DATE_SLASH_RE = re.compile(r"^(\d{4})/(\d{1,2})(?:/(\d{1,2}))?$")
_DAY_MONTHNAME_YEAR_RE = re.compile(r"^(\d{1,2})[-\s]([A-Za-zçğıöşüÇĞİÖŞÜ]+)[-\s](\d{4})$")
_MONTHNAME_YEAR_RE = re.compile(r"^([A-Za-zçğıöşüÇĞİÖŞÜ]+)\s+(\d{4})$")
# Two orderings both occur in real exports: 'YYYY-Q2' (marker then
# number, EVDS/international style) and 'YYYY 2. Çeyrek' (number then
# marker, TUIK/Turkish style).
_QUARTER_RE = re.compile(r"^(\d{4})\s*[-/ ]?\s*([1-4])\s*\.?\s*(?:Ç|Çeyrek|Ceyrek|çeyrek|ceyrek|quarter)", re.IGNORECASE)
_QUARTER_RE_MARKER_FIRST = re.compile(r"^(\d{4})\s*[-/ ]?\s*Q\s*([1-4])$", re.IGNORECASE)
_DMY_SLASH_RE = re.compile(r"^(\d{1,2})[./](\d{1,2})[./](\d{4})$")


def parse_flexible_date(raw: str) -> str | None:
    """Normalize a date/period written in one of several real-world
    formats into this project's 'YYYY-MM' Period convention (quarters
    are mapped to their closing month, matching how the rest of the
    lakehouse already represents a quarterly series upsampled to
    monthly - see build_gold.py's own 'flow_quarter_ffill' convention).
    Formats handled, all found in real BDDK/EVDS/TUIK-adjacent exports or
    plausible user uploads:
      - '2026-06', '2026-06-15'        (ISO, day silently dropped - this
                                         project's Period grain is monthly)
      - '2026/06'
      - '01-Mart-2023', '01 Mart 2023' (day + Turkish month name + year)
      - 'Mart 2023'                    (Turkish month name + year)
      - '15.03.2023'                   (D.M.Y, dot-separated)
      - '2023 1. Çeyrek', '2023-Q1'    (quarter -> its closing month)
    Returns None (never a guess) when nothing matches - the caller must
    decide whether an unparseable date means the whole row is unusable."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or is_null_token(s):
        return None

    m = _ISO_DATE_RE.match(s) or _ISO_DATE_SLASH_RE.match(s)
    if m:
        year, month = m.group(1), m.group(2)
        return f"{year}-{int(month):02d}"

    m = _QUARTER_RE.match(s) or _QUARTER_RE_MARKER_FIRST.match(s)
    if m:
        year, q = m.group(1), int(m.group(2))
        return f"{year}-{_QUARTER_END_MONTH[q]}"

    m = _DMY_SLASH_RE.match(s)
    if m:
        _, month, year = m.groups()
        return f"{year}-{int(month):02d}"

    m = _DAY_MONTHNAME_YEAR_RE.match(s)
    if m:
        _, month_name, year = m.groups()
        month_no = _TR_MONTHS.get(month_name.lower())
        if month_no:
            return f"{year}-{month_no:02d}"

    m = _MONTHNAME_YEAR_RE.match(s)
    if m:
        month_name, year = m.groups()
        month_no = _TR_MONTHS.get(month_name.lower())
        if month_no:
            return f"{year}-{month_no:02d}"

    return None


# --- 1. Birebir olmayan kategori/sektör isimleri + 10. bulanık kolon eşleşme -
def fuzzy_best_match(query: str, candidates: list[str], threshold: float = 0.6) -> tuple[str, float] | None:
    """Best fuzzy match for `query` among `candidates` (difflib
    SequenceMatcher ratio, stdlib - no extra dependency). Used for two
    real scenarios:
      - reconciling a long official category name ('Tarım, Avcılık ve
        Ormancılık', a BDDK-style sector label) against a shorter one
        from a different source ('Tarım', TÜİK-style) when a JOIN needs
        to line the two up and no exact match exists.
      - recovering a column name after character-encoding damage (see
        repair_mojibake) by matching the recovered guess against a known
        expected list.
    A SUBSTRING match (either string contained in the other, case/
    whitespace-insensitive) is checked first and scored as a near-perfect
    match - this is what actually resolves 'Tarım, Avcılık ve Ormancılık'
    against 'Tarım', which plain SequenceMatcher ratio alone would score
    only moderately given how much of the longer string doesn't overlap.
    Returns None if nothing clears `threshold`."""
    if not query or not candidates:
        return None
    q_norm = " ".join(query.strip().lower().split())

    best_name, best_score = None, 0.0
    for cand in candidates:
        c_norm = " ".join(str(cand).strip().lower().split())
        if not c_norm:
            continue
        if q_norm == c_norm:
            return cand, 1.0
        if q_norm in c_norm or c_norm in q_norm:
            # Length ratio rewards a substring match that covers most of
            # the shorter string ('Tarım' covering all of itself inside
            # the long BDDK name) over one that covers only a sliver.
            score = 0.9 + 0.1 * (min(len(q_norm), len(c_norm)) / max(len(q_norm), len(c_norm)))
        else:
            score = difflib.SequenceMatcher(None, q_norm, c_norm).ratio()
        if score > best_score:
            best_name, best_score = cand, score

    if best_name is not None and best_score >= threshold:
        return best_name, round(best_score, 3)
    return None


# --- 10. Bozuk Türkçe karakter kodlaması ------------------------------------
# The classic case: a file's real bytes are UTF-8, but something along the
# way (a tool that defaulted to Windows-1254 or Latin-1) decoded them
# wrong, turning 'İşgücü' into 'Ä°ÅŸgÃ¼cÃ¼'-style garbage. Re-encoding the
# WRONGLY-decoded text back to bytes using the wrong codec, then decoding
# those bytes correctly as UTF-8, recovers the original text - this only
# works if the text was read via the wrong codec in the first place (not
# if the underlying bytes are genuinely lost), so a repair is verified by
# checking real Turkish letters appear in the result before trusting it.
_MOJIBAKE_HINT_RE = re.compile(r"[ÃÂ].|Å[^A-Za-z0-9\s]|â€")
_TURKISH_LETTERS_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]")


def repair_mojibake(text: str) -> str:
    """Attempt to recover Turkish text that was UTF-8 bytes decoded with
    the wrong codec. Returns the input UNCHANGED if it does not look like
    mojibake, or if the repair attempt does not actually produce more
    real Turkish letters than the input had - never returns something
    worse than what came in."""
    if not text or not _MOJIBAKE_HINT_RE.search(text):
        return text
    original_tr_count = len(_TURKISH_LETTERS_RE.findall(text))
    for wrong_codec in ("windows-1254", "latin-1", "iso-8859-9"):
        try:
            repaired = text.encode(wrong_codec).decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
        if len(_TURKISH_LETTERS_RE.findall(repaired)) > original_tr_count:
            logger.info(f"repair_mojibake: recovered text via {wrong_codec} round-trip")
            return repaired
    return text


# --- 9. Ara toplamların ham veriye karışması --------------------------------
_SUBTOTAL_RE = re.compile(
    r"\b(toplam|ara toplam|genel toplam|alt toplam|subtotal|total|grand total|sum)\b",
    re.IGNORECASE,
)


def is_subtotal_label(label: str) -> bool:
    """True if a row/category label looks like a subtotal or grand-total
    line rather than a real observation - TUIK/BDDK-style tables commonly
    insert one at the end of each quarter or section, in the SAME column
    as the real category values, with no structural marker distinguishing
    it. Counting it as a normal row would double-count whatever it
    summarizes."""
    if not label:
        return False
    return bool(_SUBTOTAL_RE.search(str(label)))
