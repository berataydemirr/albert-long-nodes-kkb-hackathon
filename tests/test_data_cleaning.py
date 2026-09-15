"""Tests for src/agent/data_cleaning.py - one section per real-world data
quality scenario it was built to handle. Every test uses a concrete
example close to what a real BDDK/TUIK/user-uploaded file would contain,
not an abstract case.
"""
import pytest

from src.agent.data_cleaning import (
    fuzzy_best_match,
    is_null_token,
    is_subtotal_label,
    normalize_numeric_text,
    parse_flexible_date,
    repair_mojibake,
)


# --- 1. Birebir olmayan sektör isimleri eşleşmesi ---------------------------

def test_fuzzy_match_long_bddk_name_against_short_tuik_name():
    candidates = ["Tarım", "Sanayi", "İnşaat", "Hizmetler", "Ticaret"]
    result = fuzzy_best_match("Tarım, Avcılık ve Ormancılık", candidates)
    assert result is not None
    name, score = result
    assert name == "Tarım"
    assert score >= 0.9


def test_fuzzy_match_the_other_direction_short_query_long_candidate():
    candidates = ["Tarım, Avcılık ve Ormancılık", "İmalat Sanayi", "İnşaat Sektörü"]
    result = fuzzy_best_match("Tarım", candidates)
    assert result[0] == "Tarım, Avcılık ve Ormancılık"


def test_fuzzy_match_exact_match_scores_perfectly():
    assert fuzzy_best_match("Sanayi", ["Sanayi", "Tarım"]) == ("Sanayi", 1.0)


def test_fuzzy_match_returns_none_below_threshold():
    assert fuzzy_best_match("Sanayi", ["Tarım", "İnşaat"], threshold=0.6) is None


def test_fuzzy_match_returns_none_for_empty_inputs():
    assert fuzzy_best_match("", ["Tarım"]) is None
    assert fuzzy_best_match("Tarım", []) is None


# --- 2. Karmaşık tarih formatları -------------------------------------------

def test_parse_flexible_date_iso():
    assert parse_flexible_date("2026-06") == "2026-06"
    assert parse_flexible_date("2026-06-15") == "2026-06"  # day dropped, monthly grain


def test_parse_flexible_date_iso_slash():
    assert parse_flexible_date("2026/6") == "2026-06"


def test_parse_flexible_date_quarter_turkish():
    assert parse_flexible_date("2023 1. Çeyrek") == "2023-03"
    assert parse_flexible_date("2023-4.Ceyrek") == "2023-12"


def test_parse_flexible_date_quarter_q_notation():
    assert parse_flexible_date("2023-Q2") == "2023-06"


def test_parse_flexible_date_day_turkish_month_year():
    assert parse_flexible_date("01-Mart-2023") == "2023-03"
    assert parse_flexible_date("15 Aralık 2024") == "2024-12"


def test_parse_flexible_date_turkish_month_year_only():
    assert parse_flexible_date("Mart 2023") == "2023-03"
    assert parse_flexible_date("Ağustos 2024") == "2024-08"


def test_parse_flexible_date_dmy_dotted():
    assert parse_flexible_date("15.03.2023") == "2023-03"


def test_parse_flexible_date_unparseable_returns_none_not_a_guess():
    assert parse_flexible_date("geçen hafta") is None
    assert parse_flexible_date("") is None
    assert parse_flexible_date(None) is None


def test_parse_flexible_date_null_token_returns_none():
    assert parse_flexible_date("N/A") is None


# --- 3. Avrupa/Amerikan sayı formatı çakışması ------------------------------

def test_normalize_numeric_text_turkish_thousands_decimal():
    assert normalize_numeric_text("1.234,56") == 1234.56


def test_normalize_numeric_text_international_decimal_not_corrupted():
    """The exact bug found live in pdf_import.py: '2.009' must read as
    2.009, not 2009 (period-only, no comma present)."""
    assert normalize_numeric_text("2.009") == 2.009


def test_normalize_numeric_text_plain_comma_decimal():
    assert normalize_numeric_text("37,5") == 37.5


def test_normalize_numeric_text_plain_integer():
    assert normalize_numeric_text("42") == 42.0


def test_normalize_numeric_text_none_for_unparseable():
    assert normalize_numeric_text("yaklaşık bir milyon civarı") is None


def test_normalize_numeric_text_none_for_null_tokens():
    assert normalize_numeric_text("-") is None
    assert normalize_numeric_text("N/A") is None
    assert normalize_numeric_text("") is None
    assert normalize_numeric_text(None) is None


# --- 8. Kısaltılmış finansal büyüklükler ------------------------------------

def test_normalize_numeric_text_english_abbreviations():
    assert normalize_numeric_text("1.5M") == 1_500_000.0
    assert normalize_numeric_text("2.3B") == 2_300_000_000.0
    assert normalize_numeric_text("500K") == 500_000.0


def test_normalize_numeric_text_turkish_abbreviations():
    assert normalize_numeric_text("1,5 milyon") == 1_500_000.0
    assert normalize_numeric_text("2,3 milyar") == 2_300_000_000.0
    assert normalize_numeric_text("750 bin") == 750_000.0


def test_normalize_numeric_text_abbreviation_case_insensitive():
    assert normalize_numeric_text("1.5m") == 1_500_000.0
    assert normalize_numeric_text("2.3b") == 2_300_000_000.0


# --- 5. Farklı boş veri gösterimleri -----------------------------------------

@pytest.mark.parametrize("token", ["-", "--", "N/A", "n/a", "NA", "null", "", "  ", "..", "yok", "Yok", "#N/A"])
def test_is_null_token_recognizes_every_real_variant(token):
    assert is_null_token(token)


def test_is_null_token_rejects_a_real_value():
    assert not is_null_token("42")
    assert not is_null_token("Ankara")
    assert not is_null_token("0")  # zero is a real value, not missing


# --- 9. Ara toplamların ham veriye karışması --------------------------------

@pytest.mark.parametrize("label", ["Toplam", "TOPLAM", "Ara Toplam", "Genel Toplam", "Alt Toplam", "Subtotal", "Grand Total"])
def test_is_subtotal_label_recognizes_common_markers(label):
    assert is_subtotal_label(label)


def test_is_subtotal_label_rejects_a_real_category():
    assert not is_subtotal_label("Tarım")
    assert not is_subtotal_label("İstanbul")


def test_is_subtotal_label_matches_within_a_longer_string():
    """A real BDDK/TUIK export often puts the marker inside a longer
    label ('2024 Ç1 Toplamı') rather than as the whole cell."""
    assert is_subtotal_label("2024 1. Çeyrek Toplam")


# --- 10. Bozuk Türkçe karakter kodlaması ------------------------------------

def test_repair_mojibake_recovers_windows1254_misdecoded_text():
    original = "İşgücü Göstergeleri"
    mangled = original.encode("utf-8").decode("windows-1254")
    assert mangled != original  # sanity: the fixture is actually broken
    assert repair_mojibake(mangled) == original


def test_repair_mojibake_leaves_clean_text_unchanged():
    clean = "İşgücü Göstergeleri"
    assert repair_mojibake(clean) == clean


def test_repair_mojibake_leaves_unrelated_text_unchanged():
    assert repair_mojibake("Hello world 123") == "Hello world 123"


def test_repair_mojibake_handles_empty_string():
    assert repair_mojibake("") == ""
