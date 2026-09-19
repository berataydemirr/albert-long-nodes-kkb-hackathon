"""Tests for the fuzzy-match fallback in the EVDS and TUIK catalog search
tools - added after a live benchmark run: "ihtiyaç kredisi faiz oranı"
returned 0 rows from search_evds_series even though a close series exists,
purely because EVDS's own naming doesn't contain that exact phrase as a
substring. Network-free: both catalogs are monkeypatched, matching this
project's convention for these tools (see test_progressive_finance.py's
test_evds_fetch_verifies_code_and_preserves_monthly_data)."""
import pandas as pd
import pytest

from src.agent import finance_fetch


def _catalog_frame():
    return pd.DataFrame({
        "SERIE_CODE": ["TP.BKR.TRY.KTFTUK", "TP.BKR.TRY.18", "TP.BKR.TRY.99"],
        # Deliberately word-order-shuffled vs the query, so the exact
        # substring search finds nothing but a real close match exists.
        "SERIE_NAME": ["Faiz Oranı (Tüketici Kredisi Birleşik)", "Konut Kredisi Faiz Oranı",
                       "Otomobil Sigortası Prim Endeksi"],
    })


def test_exact_substring_match_is_not_marked_fuzzy(monkeypatch):
    class Client:
        def get_series(self, group):
            return _catalog_frame()
    monkeypatch.setattr(finance_fetch, "_get_client", Client)
    result = finance_fetch.search_evds_series("konut kredisi faiz oranı", datagroup="BKR")
    assert result["fuzzy_match"] is False
    assert result["row_count"] == 1
    assert result["rows"][0]["SERIE_CODE"] == "TP.BKR.TRY.18"


def test_no_exact_match_falls_back_to_fuzzy_candidates(monkeypatch):
    class Client:
        def get_series(self, group):
            return _catalog_frame()
    monkeypatch.setattr(finance_fetch, "_get_client", Client)
    result = finance_fetch.search_evds_series("ihtiyaç kredisi faiz oranı", datagroup="BKR")
    assert result["fuzzy_match"] is True
    assert result["row_count"] >= 1
    codes = {row["SERIE_CODE"] for row in result["rows"]}
    # The word-order-shuffled consumer-loan-rate row must surface; the
    # unrelated insurance-index row must not.
    assert "TP.BKR.TRY.KTFTUK" in codes
    assert "TP.BKR.TRY.99" not in codes


def test_unrelated_query_returns_nothing_rather_than_forced_noise(monkeypatch):
    class Client:
        def get_series(self, group):
            return _catalog_frame()
    monkeypatch.setattr(finance_fetch, "_get_client", Client)
    result = finance_fetch.search_evds_series("bambaşka bir konu zzzqqq", datagroup="BKR")
    assert result["fuzzy_match"] is True
    assert result["row_count"] == 0


def test_tuik_search_dataflows_falls_back_to_fuzzy_and_flags_results(monkeypatch):
    from src.collectors import tuik

    dataflows = [
        {"dataflow_id": "DF_ISGUCU", "version": "1.0", "name_tr": "Faiz Oranı (İşgücü Piyasası)", "name_en": "x"},
        {"dataflow_id": "DF_UNRELATED", "version": "1.0", "name_tr": "Otomobil Sigortası", "name_en": "y"},
    ]
    monkeypatch.setattr(tuik, "list_dataflows", lambda refresh=False: dataflows)
    result = tuik.search_dataflows("işgücü faiz oranı")
    assert result and all(m["fuzzy"] for m in result)
    assert result[0]["dataflow_id"] == "DF_ISGUCU"

    exact = tuik.search_dataflows("işgücü")
    assert exact[0]["fuzzy"] is False
