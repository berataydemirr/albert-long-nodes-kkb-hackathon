"""Fetch live BIST (Borsa Istanbul) stock price history via yfinance and
load it into the lakehouse as its own table - the same "live-fetch, write
into the shared file, queryable immediately" pattern as
refresh_tuik_series in data_import.py, for a finance data source outside
anything the team pre-collected (team feedback: "IMKB'den hisse verileri
alinacak").

Unlike a TUIK series, a BIST ticker is not a small closed registry that
needs hand-resolved SDMX keys - yfinance takes a plain ticker symbol
directly, so this is NOT restricted to a pre-registered list the way
refresh_tuik_series is. The risk profile is different: worst case for a
wrong/unknown ticker is an empty result (checked and rejected below), not
a real-but-wrong series silently mislabeled.

yfinance addresses a BIST stock with a '.IS' suffix (e.g. AKBNK.IS for
Akbank) - appended automatically since a question would give the bare
ticker, not yfinance's own suffix convention.
"""
import re
from typing import Final

import pandas as pd
import yfinance as yf

from src.agent.data_import import _write_dataset
from src.common.logging_setup import get_logger

logger = get_logger("bist_import")

_TICKER_RE = re.compile(r"^[A-Za-z0-9]{2,10}$")
VALID_PERIODS: Final = ("5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max")


def fetch_bist_stock(ticker: str, period: str = "6mo") -> dict:
    """Pull OHLCV daily history for one BIST-listed stock and write it
    into the lakehouse as 'bist_<ticker>' (e.g. 'bist_akbnk'), through
    the same write path (collision check, registry, retry-safe
    connection) as a CSV/Excel/PDF import - so it is immediately
    queryable and joinable via lakehouse_query, exactly like an imported
    dataset."""
    if not isinstance(ticker, str):
        raise ValueError("ticker must be a string.")
    bare = ticker.strip().upper()
    if bare.endswith(".IS"):
        bare = bare[:-3]
    if not _TICKER_RE.match(bare):
        raise ValueError(f"'{ticker}' does not look like a BIST ticker (letters/digits, 2-10 chars, e.g. 'AKBNK').")
    if period not in VALID_PERIODS:
        raise ValueError(f"period must be one of {VALID_PERIODS}, got {period!r}.")

    yf_symbol = f"{bare}.IS"
    logger.info(f"fetch_bist_stock: start ticker={bare!r}, yf_symbol={yf_symbol!r}, period={period!r}")

    hist = yf.Ticker(yf_symbol).history(period=period)
    if hist.empty:
        logger.warning(f"fetch_bist_stock: no data returned for {yf_symbol!r}")
        raise ValueError(
            f"No price data returned for '{bare}' (tried Yahoo Finance symbol '{yf_symbol}'). "
            f"Check it is a real BIST ticker (e.g. AKBNK, THYAO, GARAN)."
        )

    df = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col].round(4)

    # Found live: the most recent row can have a real Volume but NaN
    # Open/High/Low/Close - Yahoo Finance publishing a trading day's
    # volume before its price data has settled (a feed-lag artifact, not
    # a real "zero price" day). Reported as last_close, this produced a
    # bare NaN in the tool's JSON result - invalid JSON a browser's
    # JSON.parse rejects, the same class of bug this project has fixed
    # at the DB-serialization layer before, here one step upstream of
    # it. A day with no real close is not a data point; dropped rather
    # than kept with a fabricated or null price.
    incomplete = df["Close"].isna()
    if incomplete.any():
        logger.warning(f"fetch_bist_stock: dropping {int(incomplete.sum())} row(s) with no Close yet (feed lag) for {yf_symbol!r}")
        df = df[~incomplete].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No usable price data for '{bare}' yet - every row in this period is incomplete (feed lag). Try again shortly.")

    dataset_name = f"bist_{bare.lower()}"
    display_name = f"{yf_symbol} (Yahoo Finance, BIST)"
    imported_at = _write_dataset(dataset_name, display_name, df)

    result = {
        "dataset_name": dataset_name,
        "ticker": bare,
        "yf_symbol": yf_symbol,
        "period": period,
        "imported_at": imported_at,
        "row_count": len(df),
        "first_date": df["Date"].iloc[0],
        "last_date": df["Date"].iloc[-1],
        "last_close": float(df["Close"].iloc[-1]),
        "note": (
            f"Live from Yahoo Finance for BIST symbol '{yf_symbol}', written into the lakehouse as "
            f"'{dataset_name}' (columns: Date, Open, High, Low, Close, Volume) - query it directly with "
            f"lakehouse_query, including JOINs against gold_* tables."
        ),
    }
    logger.info(f"fetch_bist_stock: done, {dataset_name!r} <- {len(df)} rows, last_close={result['last_close']}")
    return result
