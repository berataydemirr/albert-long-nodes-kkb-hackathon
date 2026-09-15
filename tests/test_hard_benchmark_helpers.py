"""Unit tests for hard_benchmark.py's pure-logic helpers only.

The benchmark itself calls the real MIA API and is intentionally excluded
from pytest (network + token cost + non-determinism, see its own module
docstring). These few functions have no network dependency, so they get
normal test coverage like everything else.
"""
from benchmarks.hard_benchmark import _admits_missing, _ddl_blocked, _extract_dataset_names_from_trace, _extract_series_codes_from_trace


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
