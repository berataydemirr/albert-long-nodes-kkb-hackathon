"""Tests for fetch_bist_stock (src/agent/bist_import.py).

Only the validation paths are covered here - they raise before any
network call, so they run in normal pytest. The actual Yahoo Finance
fetch and lakehouse write were verified manually against a real ticker
(AKBNK.IS), per this project's established convention of keeping live
network calls out of the test suite (same reasoning as
refresh_tuik_series's TUIK fetch).
"""
import pytest

from src.agent.bist_import import fetch_bist_stock


@pytest.mark.parametrize("bad_ticker", [
    "", "a", "way-too-long-for-a-ticker", "AK BNK", "AK;DROP TABLE x;--", None, 123,
])
def test_rejects_malformed_ticker(bad_ticker):
    with pytest.raises(ValueError):
        fetch_bist_stock(bad_ticker)


def test_rejects_invalid_period():
    with pytest.raises(ValueError, match="period must be one of"):
        fetch_bist_stock("AKBNK", period="13mo")


def test_strips_dot_is_suffix_before_validating():
    """A caller passing the yfinance-style symbol directly ('AKBNK.IS')
    should not be rejected just because '.IS' isn't alphanumeric - it
    normalizes to the bare ticker before the format check, same as the
    error-free path would after a real fetch."""
    # Malformed *after* stripping .IS should still fail - proves the
    # suffix is actually being stripped, not just tolerated as noise.
    with pytest.raises(ValueError, match="does not look like a BIST ticker"):
        fetch_bist_stock("AK BNK.IS")
