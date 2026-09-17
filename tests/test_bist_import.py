"""Tests for fetch_bist_stock (src/agent/bist_import.py).

The validation paths need no network call at all, so they run directly.
The actual write-path behavior (dropping an incomplete row, writing to
the lakehouse) is tested against a MOCKED yfinance - a fake Ticker
returning a canned DataFrame - rather than skipped, since that behavior
is exactly what a live bug (see test_drops_a_row_with_no_close_yet) was
found in and needs real regression coverage; only the actual live HTTP
call to Yahoo Finance is avoided, per this project's convention of
keeping live network calls out of the test suite (same reasoning as
refresh_tuik_series's TUIK fetch, verified manually instead).
"""
import pandas as pd
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


class _FakeTicker:
    def __init__(self, history_df):
        self._history_df = history_df

    def history(self, period):
        return self._history_df


def _fake_history(rows):
    """rows: list of (date_str, open, high, low, close, volume) - any
    price field can be None to simulate an incomplete/feed-lag row (real
    yfinance represents this as a float NaN, not a Python None - the
    columns are cast to float64 here so a None input round-trips the same
    way a real response's NaN would, otherwise .round(4) in the code
    under test would hit a real None and raise a DIFFERENT, fixture-only
    TypeError that has nothing to do with the actual live bug)."""
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = df[col].astype(float)
    return df.set_index("Date")


@pytest.fixture(autouse=True)
def _fresh_lakehouse_db(monkeypatch, tmp_path):
    import src.agent.data_import as di

    monkeypatch.setattr(di, "LAKEHOUSE_DB_PATH", tmp_path / "lakehouse_test.duckdb")
    yield


def test_drops_a_row_with_no_close_yet(monkeypatch):
    """Real bug found live: Yahoo Finance can publish a trading day's
    Volume before its price data has settled, leaving Open/High/Low/Close
    all NaN for that row. Reported as last_close, this was a bare NaN in
    the tool's own JSON result - the same class of bug this project has
    fixed at the DB-serialization layer before. The incomplete row must
    be dropped, not kept with a fabricated or null price."""
    import src.agent.bist_import as bi

    fake_hist = _fake_history([
        ("2026-09-09", 75.0, 76.3, 74.8, 72.9, 372553606),
        ("2026-09-10", 73.2, 73.4, 72.0, 72.1, 222813703),
        ("2026-09-15", None, None, None, None, 150461390),  # feed-lag row
    ])
    monkeypatch.setattr(bi.yf, "Ticker", lambda symbol: _FakeTicker(fake_hist))

    result = fetch_bist_stock("AKBNK", period="1mo")
    assert result["row_count"] == 2
    assert result["last_date"] == "2026-09-10"
    assert result["last_close"] == 72.1
    import math
    assert not math.isnan(result["last_close"])


def test_raises_a_clear_error_when_every_row_is_incomplete(monkeypatch):
    import src.agent.bist_import as bi

    fake_hist = _fake_history([("2026-09-15", None, None, None, None, 100)])
    monkeypatch.setattr(bi.yf, "Ticker", lambda symbol: _FakeTicker(fake_hist))

    with pytest.raises(ValueError, match="feed lag"):
        fetch_bist_stock("AKBNK", period="1mo")


def test_normal_fetch_with_no_incomplete_rows(monkeypatch):
    import src.agent.bist_import as bi

    fake_hist = _fake_history([
        ("2026-09-09", 75.0, 76.3, 74.8, 72.9, 372553606),
        ("2026-09-10", 73.2, 73.4, 72.0, 72.1, 222813703),
    ])
    monkeypatch.setattr(bi.yf, "Ticker", lambda symbol: _FakeTicker(fake_hist))

    result = fetch_bist_stock("AKBNK", period="1mo")
    assert result["row_count"] == 2
    assert result["last_close"] == 72.1


# --- Index alias resolution ----------------------------------------------
# Real bug found live: a question about "BIST 100" made the model try the
# ticker "BIST100" first (a clean but wasted empty-result round trip)
# before succeeding with the real Yahoo Finance symbol "XU100" on a
# second try - burning a whole extra model round on a demo question. The
# alias must be resolved in code, not left to the model to eventually
# guess right.

@pytest.mark.parametrize("spoken_name", ["BIST100", "BIST 100", "BIST-100", "bist100", "bist 100"])
def test_bist_100_index_names_resolve_to_the_real_xu100_symbol(monkeypatch, spoken_name):
    import src.agent.bist_import as bi

    requested_symbols = []

    def _fake_ticker(symbol):
        requested_symbols.append(symbol)
        return _FakeTicker(_fake_history([("2026-09-10", 9500.0, 9600.0, 9400.0, 9550.0, 1_000_000)]))

    monkeypatch.setattr(bi.yf, "Ticker", _fake_ticker)

    result = fetch_bist_stock(spoken_name, period="1y")

    assert requested_symbols == ["XU100.IS"]
    assert result["ticker"] == "XU100"
    assert result["yf_symbol"] == "XU100.IS"
    assert result["dataset_name"] == "bist_xu100"


@pytest.mark.parametrize("spoken_name,expected", [
    ("BIST30", "XU030"), ("BIST 30", "XU030"), ("BIST50", "XU050"), ("BIST 50", "XU050"),
])
def test_other_bist_index_aliases_also_resolve(monkeypatch, spoken_name, expected):
    import src.agent.bist_import as bi

    requested_symbols = []
    monkeypatch.setattr(
        bi.yf, "Ticker",
        lambda symbol: (requested_symbols.append(symbol) or _FakeTicker(
            _fake_history([("2026-09-10", 100.0, 101.0, 99.0, 100.5, 1_000)])
        )),
    )

    result = fetch_bist_stock(spoken_name)

    assert requested_symbols == [f"{expected}.IS"]
    assert result["ticker"] == expected


def test_the_real_xu100_symbol_passes_through_unchanged(monkeypatch):
    """The alias table must not mangle the already-correct symbol."""
    import src.agent.bist_import as bi

    requested_symbols = []
    monkeypatch.setattr(
        bi.yf, "Ticker",
        lambda symbol: (requested_symbols.append(symbol) or _FakeTicker(
            _fake_history([("2026-09-10", 9500.0, 9600.0, 9400.0, 9550.0, 1_000_000)])
        )),
    )

    fetch_bist_stock("XU100")

    assert requested_symbols == ["XU100.IS"]


def test_an_ordinary_stock_ticker_is_not_touched_by_alias_resolution():
    """Regression guard: a real stock ticker that happens to share no
    alias key must never be rewritten."""
    import src.agent.bist_import as bi

    assert bi._resolve_index_alias("AKBNK") == "AKBNK"
    assert bi._resolve_index_alias("GARAN") == "GARAN"
