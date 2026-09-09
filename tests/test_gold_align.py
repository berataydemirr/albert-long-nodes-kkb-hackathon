"""
Tests for src/gold/align.py - no network, no fixtures needed (pure
pandas/datetime functions). The donemId anchor and downsampling
behavior are locked in against the real values used in
src/gold/build_gold.py, so a change here can't silently drift from
what the gold layer actually relies on.

Run: pytest tests/test_gold_align.py -v
"""
import datetime as dt

import pandas as pd

from src.gold.align import (
    donem_id_to_date, to_period, downsample_to_monthly,
    upsample_quarterly_to_monthly, forward_fill_gaps,
)


def test_donem_id_to_date_matches_the_confirmed_anchor():
    # Anchor from config/settings.py's own comment: donem 367 = week 1
    # of 2021, 2021-01-08.
    assert donem_id_to_date(367) == dt.date(2021, 1, 8)


def test_donem_id_to_date_advances_one_week_per_donem():
    assert donem_id_to_date(368) == dt.date(2021, 1, 15)
    assert donem_id_to_date(367 + 52) == dt.date(2022, 1, 7)


def test_to_period_normalizes_various_date_inputs():
    assert to_period("2021-03-15") == "2021-03"
    assert to_period(dt.date(2024, 12, 1)) == "2024-12"
    assert to_period(pd.Timestamp("2026-06-30")) == "2026-06"


def test_downsample_to_monthly_last_keeps_month_end_value():
    # A daily/weekly STOCK figure (e.g. an FX rate): the right monthly
    # representative is the last observation of the month, not an average.
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2021-01-05", "2021-01-20", "2021-01-31", "2021-02-10"]),
        "Value": [7.40, 7.50, 7.60, 7.80],
    })
    out = downsample_to_monthly(df, date_col="Date", value_cols=["Value"], how="last")
    jan = out[out["Period"] == "2021-01"].iloc[0]
    assert jan["Value"] == 7.60


def test_downsample_to_monthly_mean_averages_the_month():
    # A daily/weekly FLOW figure (e.g. weekly card spending): averaged
    # over the weeks in the month, not just the last week's value.
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2021-01-01", "2021-01-08", "2021-01-15"]),
        "Value": [10.0, 20.0, 30.0],
    })
    out = downsample_to_monthly(df, date_col="Date", value_cols=["Value"], how="mean")
    jan = out[out["Period"] == "2021-01"].iloc[0]
    assert jan["Value"] == 20.0


def test_upsample_quarterly_to_monthly_expands_and_flags_the_fill():
    # FinTurk's own convention: 'YYYY-Q' with Q in {3,6,9,12}.
    df = pd.DataFrame({"Donem": ["2021-3"], "Value": [900]})
    out = upsample_quarterly_to_monthly(df, quarter_col="Donem", value_cols=["Value"])

    assert sorted(out["Period"]) == ["2021-01", "2021-02", "2021-03"]
    # Only the quarter's own end month is a real observation; the other
    # two months are forward-filled and must be flagged as such, since
    # gold_fact_long relies on this to tag Tip='flow_quarter_ffill'.
    types = dict(zip(out["Period"], out["_period_type"]))
    assert types["2021-03"] == "observed"
    assert types["2021-01"] == "quarter_ffill"
    assert types["2021-02"] == "quarter_ffill"
    # The value itself is just repeated (an approximation, not invented data).
    assert (out["Value"] == 900).all()


def test_forward_fill_gaps_flags_the_months_it_invents():
    df = pd.DataFrame({"Period": ["2021-01", "2021-03"], "Value": [10, 30]})
    out = forward_fill_gaps(df, period_col="Period", value_cols=["Value"],
                             start_period="2021-01", end_period="2021-03")

    feb = out[out["Period"] == "2021-02"].iloc[0]
    assert feb["_filled"] == True  # noqa: E712 - explicit bool comparison reads clearer here
    assert feb["Value"] == 10  # carried forward from January

    jan = out[out["Period"] == "2021-01"].iloc[0]
    assert jan["_filled"] == False  # noqa: E712
