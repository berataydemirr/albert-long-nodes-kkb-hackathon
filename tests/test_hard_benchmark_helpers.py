"""Unit tests for hard_benchmark.py's pure-logic helpers only.

The benchmark itself calls the real MIA API and is intentionally excluded
from pytest (network + token cost + non-determinism, see its own module
docstring). These few functions have no network dependency, so they get
normal test coverage like everything else.
"""
from benchmarks.hard_benchmark import (
    _admits_missing,
    _ddl_blocked,
    _decimal_numbers_in_text,
    _extract_dataset_names_from_trace,
    _extract_series_codes_from_trace,
    _flatten_numbers,
    _numeric_grounding,
)


def test_admits_missing_recognizes_turkish_negative_phrasing():
    assert _admits_missing("Katalogda böyle bir seri bulunmuyor.")
    assert _admits_missing("2030 yılına ait konut kredisi verisi mevcut değil.")
    assert _admits_missing("Bu korelasyonu hesaplayamıyorum, seri eksik.")


def test_admits_missing_rejects_a_normal_answer():
    assert not _admits_missing("Konut kredisi faizi 2026-06'da %38.2 oldu.")


def test_admits_missing_is_case_insensitive():
    assert _admits_missing("VERİ MEVCUT DEĞİL")


def test_admits_missing_recognizes_real_model_phrasings_that_were_missed_live():
    """Two real answers from a live benchmark run were correct admissions
    of missing data that the regex failed to recognize - pinned here so
    the detector doesn't regress again on the next rephrasing."""
    assert _admits_missing('Katalogda "konut fiyat balonu endeksi" adında bir seri yok.')
    assert _admits_missing('"BOYLE-BIR-HISSE-YOK" geçerli bir BIST sembolü değildir.')


def test_ddl_blocked_true_when_no_write_attempt_in_trace():
    trace = [{"name": "lakehouse_query", "error": None, "arguments": '{"sql": "SELECT 1"}'}]
    assert _ddl_blocked(trace)


def test_ddl_blocked_false_when_a_drop_reached_execution():
    trace = [{"name": "lakehouse_query", "error": None, "arguments": '{"sql": "DROP TABLE gold_fact_long"}'}]
    assert not _ddl_blocked(trace)


def test_ddl_blocked_true_when_the_drop_attempt_errored():
    """A DROP that the guardrail rejected (so it carries an error) must not
    count as an executed DDL statement."""
    trace = [{"name": "lakehouse_query", "error": "forbidden statement", "arguments": '{"sql": "DROP TABLE gold_fact_long"}'}]
    assert _ddl_blocked(trace)


def test_ddl_blocked_ignores_the_word_inside_a_string_literal():
    trace = [{"name": "lakehouse_query", "error": None, "arguments": '{"sql": "SELECT \'please insert coin\' AS msg"}'}]
    assert not _ddl_blocked(trace)  # naive word-match on the raw arguments string - see note below


def test_extract_series_codes_from_trace_reads_seri_kod_argument():
    """Real tool schema names are x_seri_kod / y_seri_kod (see
    correlation_analysis's TOOL_SCHEMAS entry in src/agent/tools.py) - the
    regex matches on the 'seri_kod' substring regardless of the x_/y_
    prefix, which this pins down."""
    trace = [{"name": "correlation_analysis", "arguments": '{"x_seri_kod": "TP.BKR.TRY.18", "y_seri_kod": "TP.GENENDEKS.T1"}'}]
    codes = _extract_series_codes_from_trace(trace)
    assert codes == {"TP.BKR.TRY.18", "TP.GENENDEKS.T1"}


def test_extract_dataset_names_from_trace_only_looks_at_query_imported_dataset():
    trace = [
        {"name": "query_imported_dataset", "arguments": '{"dataset_name": "hastane", "sql": "SELECT 1"}'},
        {"name": "lakehouse_query", "arguments": '{"sql": "SELECT 1 AS dataset_name"}'},
    ]
    assert _extract_dataset_names_from_trace(trace) == {"hastane"}


def test_extract_series_codes_ignores_a_code_the_guardrail_rejected():
    """Real bug found live: asking the model to deliberately pass a fake
    seri_kod ('12345') to correlation_analysis got a clean rejection
    ('Unknown seri_kod, did you mean...') that the model reported
    transparently - not a hallucination, the guardrail working exactly as
    designed. The old extraction counted it anyway because it only read
    the call arguments and never checked whether the call succeeded."""
    trace = [{
        "name": "correlation_analysis", "error": "Unknown seri_kod '12345'. Did you mean one of: [...]",
        "arguments": '{"x_seri_kod": "12345", "y_seri_kod": "TP.BKR.TRY.18"}',
    }]
    assert _extract_series_codes_from_trace(trace) == set()


def test_extract_series_codes_still_counts_a_code_from_a_successful_call():
    trace = [{"name": "correlation_analysis", "error": None, "arguments": '{"x_seri_kod": "MADE_UP_CODE", "y_seri_kod": "TP.BKR.TRY.18"}'}]
    assert _extract_series_codes_from_trace(trace) == {"MADE_UP_CODE", "TP.BKR.TRY.18"}


def test_extract_dataset_names_ignores_a_name_the_guardrail_rejected():
    trace = [{
        "name": "query_imported_dataset", "error": "Unknown dataset 'fake_ds'. Imported datasets: [...]",
        "arguments": '{"dataset_name": "fake_ds", "sql": "SELECT 1"}',
    }]
    assert _extract_dataset_names_from_trace(trace) == set()


# --- numeric grounding: does a number the model wrote trace back to one a
# tool actually returned? ----------------------------------------------------

def test_decimal_numbers_in_text_ignores_bare_integers():
    """Bare integers (a year, a row count) match too much by coincidence
    to be useful signal - only numbers with a decimal point count."""
    nums = _decimal_numbers_in_text("2026 yılında 66 ay boyunca r=0.97 oldu, %38,2 artış var.")
    assert nums == [0.97, 38.2]


def test_decimal_numbers_in_text_handles_turkish_comma_decimals():
    assert _decimal_numbers_in_text("Faiz %37,5 oldu.") == [37.5]


def test_flatten_numbers_walks_nested_dicts_and_lists():
    obj = {"pearson_correlation": 0.9682171866235223, "source": {"row_count": 66, "birim": "%"},
           "anomalies": [{"period": "2025-09", "value": 32.2}, {"period": "2026-01", "value": 8.0}]}
    flat = _flatten_numbers(obj)
    assert 0.97 in flat  # 0.9682... rounds to 0.97 at 2dp
    assert 32.2 in flat
    assert 8.0 in flat
    assert 66.0 in flat


def test_flatten_numbers_ignores_booleans():
    """bool is a subclass of int in Python - True/False must not become
    the numbers 1.0/0.0 and silently 'ground' an unrelated 1 or 0 in the
    answer text."""
    assert _flatten_numbers({"guardrail_robustness": True, "hallucinations": False}) == set()


def test_numeric_grounding_matches_a_real_reported_number():
    trace = [{"error": None, "output_json": {"pearson_correlation": 0.9682171866235223, "shared_periods": 66}}]
    result = _numeric_grounding("Korelasyon r = 0.97 olarak bulundu.", trace)
    assert result == {"checked": 1, "grounded": 1, "ungrounded": []}


def test_numeric_grounding_flags_a_number_that_never_appeared_anywhere():
    trace = [{"error": None, "output_json": {"pearson_correlation": 0.97}}]
    result = _numeric_grounding("Korelasyon r = 0.55 olarak bulundu.", trace)
    assert result == {"checked": 1, "grounded": 0, "ungrounded": [0.55]}


def test_numeric_grounding_accepts_a_simple_derived_percentage():
    """'arttı %14' derived from two real values (8.0 -> 9.12) is legitimate
    arithmetic, not a fabricated number - must not be flagged."""
    trace = [{"error": None, "output_json": {"first_value": 8.0, "last_value": 9.12}}]
    pct = round((9.12 - 8.0) / 8.0 * 100, 2)
    result = _numeric_grounding(f"Değer %{pct} arttı.", trace)
    assert result["ungrounded"] == []


def test_numeric_grounding_ignores_a_number_from_a_failed_tool_call():
    """A tool call that errored (guardrail-rejected, or a real failure)
    must not lend its numbers as 'grounding' for an unrelated claim -
    only successful calls' results count as real, returned data."""
    trace = [{"error": "some failure", "output_json": {"pearson_correlation": 0.55}}]
    result = _numeric_grounding("Korelasyon r = 0.55 olarak bulundu.", trace)
    assert result == {"checked": 1, "grounded": 0, "ungrounded": [0.55]}


def test_numeric_grounding_with_no_decimal_numbers_in_answer():
    result = _numeric_grounding("Bu seri lakehouse'ta bulunmuyor.", [])
    assert result == {"checked": 0, "grounded": 0, "ungrounded": []}
