"""Tests for src/collectors/tuik.py's dataflow search (metadata only, no
actual series data - see search_dataflows' own docstring for why this is
safe to expose more broadly than fetch_series).

list_dataflows()/search_dataflows() hit the real TUIK API and are not
covered here (network-free, per this project's convention - see
test_data_import.py's refresh_tuik_series tests for the same reasoning);
_tr_casefold has no network dependency and gets normal coverage.
"""
import pytest

from src.collectors.tuik import _tr_casefold, search_dataflows


def test_tr_casefold_fixes_the_dotted_capital_i_bug():
    """Real bug found live: plain str.lower() turns 'İ' into 'i' plus a
    COMBINING DOT ABOVE (U+0307), not the ASCII 'i' a query types - so
    searching 'işgücü' against a catalog entry titled 'Temel İşgücü
    Göstergeleri' (the real registered TUIK_UNEMPLOYMENT_RATE dataflow's
    own name) returned zero matches even though it's an exact match by
    eye."""
    assert _tr_casefold("İşgücü") == _tr_casefold("işgücü") == "işgücü"


def test_tr_casefold_handles_dotless_capital_i_too():
    assert _tr_casefold("IRAK") == "ırak"


def test_tr_casefold_is_a_substring_after_normalization():
    title = "Temel İşgücü Göstergeleri"
    assert "işgücü" in _tr_casefold(title)


def test_search_dataflows_rejects_empty_query():
    with pytest.raises(ValueError):
        search_dataflows("")


def test_search_dataflows_rejects_non_string_query():
    with pytest.raises(ValueError):
        search_dataflows(12345)
