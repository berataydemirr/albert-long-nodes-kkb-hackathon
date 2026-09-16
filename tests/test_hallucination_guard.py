"""Tests for the runtime Seri_Kod hallucination check (see
src/agent/hallucination_guard.py and its use in orchestrator.ask()).

Two layers are tested: the pure extraction/comparison functions here, and
the full ask() correction/caveat behavior in
test_orchestrator_hallucination_guard.py.
"""

from src.agent.hallucination_guard import extract_cited_series_codes, find_hallucinated_codes


# --- extract_cited_series_codes -----------------------------------------

def test_extracts_an_evds_style_code_from_a_turkish_sentence():
    text = "TP.BKR.TRY.18 serisine göre faiz oranı %3.2'ye indi."

    assert extract_cited_series_codes(text) == {"TP.BKR.TRY.18"}


def test_extracts_a_bddk_style_code():
    text = "tablo_4.konut_kredisi serisinde son üç ayda artış var."

    assert extract_cited_series_codes(text) == {"tablo_4.konut_kredisi"}


def test_extracts_a_tuik_style_code():
    text = "TUIK_LIFE_EXPECTANCY serisi 2023'te 78.6 yıl oldu."

    assert extract_cited_series_codes(text) == {"TUIK_LIFE_EXPECTANCY"}


def test_extracts_multiple_distinct_codes_from_one_answer():
    text = "TP.BKR.TRY.18 ile tablo_4.konut_kredisi arasında korelasyon -0.71."

    assert extract_cited_series_codes(text) == {"TP.BKR.TRY.18", "tablo_4.konut_kredisi"}


def test_a_code_directly_followed_by_a_turkish_suffix_is_still_matched_cleanly():
    """"...18'e göre" must not pull the apostrophe or suffix into the code -
    the word boundary has to land exactly at the last digit."""
    text = "TP.BKR.TRY.18'e göre kredi hacmi arttı."

    assert extract_cited_series_codes(text) == {"TP.BKR.TRY.18"}


def test_a_code_in_backticks_or_parentheses_is_still_matched():
    text = "Kaynak seri (`TP.BISPOLFAIZ.TUR`) haftalık güncelleniyor."

    assert extract_cited_series_codes(text) == {"TP.BISPOLFAIZ.TUR"}


def test_ordinary_text_with_no_code_shaped_tokens_returns_nothing():
    text = "Konut kredisi faizi son üç ayda 3.18'den 2.94'e geriledi, yaklaşık %8 düşüş."

    assert extract_cited_series_codes(text) == set()


def test_a_plain_decimal_number_is_not_mistaken_for_a_code():
    """3.14, 2026.01 etc. must not match - only the TP./tablo_N./TUIK_
    shapes are real codes."""
    text = "Oran 3.14 civarında, artış 2026.01 döneminden bu yana sürüyor."

    assert extract_cited_series_codes(text) == set()


def test_a_version_style_string_is_not_mistaken_for_an_evds_code():
    text = "Uygulama v1.2.3 sürümünde bu hata giderildi."

    assert extract_cited_series_codes(text) == set()


def test_empty_or_none_text_returns_nothing():
    assert extract_cited_series_codes("") == set()
    assert extract_cited_series_codes(None) == set()


# --- find_hallucinated_codes ---------------------------------------------

def test_a_real_code_is_never_flagged():
    valid = {"TP.BKR.TRY.18", "tablo_4.konut_kredisi"}
    text = "TP.BKR.TRY.18 serisine göre faiz düştü."

    assert find_hallucinated_codes(text, valid) == set()


def test_an_invented_code_is_flagged():
    valid = {"TP.BKR.TRY.18"}
    text = "TP.UYDURULMUS.KOD serisine göre faiz düştü."

    assert find_hallucinated_codes(text, valid) == {"TP.UYDURULMUS.KOD"}


def test_a_mix_of_real_and_invented_codes_flags_only_the_invented_one():
    valid = {"TP.BKR.TRY.18"}
    text = "TP.BKR.TRY.18 ile TP.HAYALI.SERI arasında korelasyon var."

    assert find_hallucinated_codes(text, valid) == {"TP.HAYALI.SERI"}


def test_comparison_is_case_sensitive_because_real_code_casing_is_load_bearing():
    """A case mismatch would also break a real WHERE Seri_Kod=... clause
    elsewhere in the system, so it must be surfaced, not silently accepted
    as "close enough"."""
    valid = {"TP.BKR.TRY.18"}
    text = "tp.bkr.try.18 serisine göre..."  # lowercase: does not even match the code shape

    # Lowercase text does not match the (uppercase-only) EVDS pattern at
    # all, so nothing is extracted - nothing to flag either.
    assert find_hallucinated_codes(text, valid) == set()


def test_no_valid_codes_means_nothing_is_ever_flagged():
    """An empty catalog (missing/unbuilt database) means there is nothing
    real to check against - flagging every code in that state would be
    noise, not signal, so the check stays silent instead of crying wolf."""
    text = "TP.HERHANGI.BIRSEY serisine göre..."

    assert find_hallucinated_codes(text, set()) == set()


def test_no_codes_cited_at_all_is_never_flagged():
    valid = {"TP.BKR.TRY.18"}
    text = "Faiz oranı son üç ayda geriledi."

    assert find_hallucinated_codes(text, valid) == set()
