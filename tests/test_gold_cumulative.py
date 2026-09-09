"""
Tests for src/gold/cumulative.py against small, hand-built series - no
network, no fixtures needed (the module is a pure pandas function, see
its own docstring). Chosen to lock in the behavior confirmed against
real BDDK data in src/gold/scan_cumulative.py's report: tablo_2 (income
statement) is cumulative, tablo_4 (a loan balance) is not.

Run: pytest tests/test_gold_cumulative.py -v
"""
import pandas as pd

from src.gold.cumulative import detect_cumulative, decumulate


def _cumulative_series() -> pd.DataFrame:
    """Two years of a year-to-date cumulative series: monthly flows of
    10, 20, 30, ... reset every January - the exact shape confirmed for
    BDDK Monthly tablo_2's income-statement lines."""
    periods, values = [], []
    for year in (2021, 2022):
        running_total = 0
        for month in range(1, 13):
            running_total += 10 * month
            periods.append(f"{year}-{month:02d}")
            values.append(running_total)
    return pd.DataFrame({"Period": periods, "Value": values})


def _stock_series() -> pd.DataFrame:
    """Two years of a smoothly growing balance - no reset at the year
    boundary, the shape confirmed for BDDK Monthly tablo_4's housing-loan
    line (Dec 2021: 298896 -> Jan 2022: 300667)."""
    periods, values = [], []
    value = 100_000
    for year in (2021, 2022):
        for month in range(1, 13):
            value += 1_000  # keeps climbing across the Dec->Jan boundary too
            periods.append(f"{year}-{month:02d}")
            values.append(value)
    return pd.DataFrame({"Period": periods, "Value": values})


def test_detect_cumulative_flags_a_ytd_resetting_series():
    result = detect_cumulative(_cumulative_series())
    assert result["is_cumulative"] is True
    assert result["monotonic_ratio"] >= 0.9
    assert result["reset_ratio"] >= 0.5


def test_detect_cumulative_does_not_flag_a_growing_stock():
    result = detect_cumulative(_stock_series())
    assert result["is_cumulative"] is False


def test_detect_cumulative_reports_insufficient_data_instead_of_guessing():
    tiny = pd.DataFrame({"Period": ["2021-01", "2021-02"], "Value": [10, 20]})
    result = detect_cumulative(tiny)
    assert result["is_cumulative"] is False
    assert result["reason"] == "insufficient_data"


def test_decumulate_recovers_the_original_monthly_flow():
    result = decumulate(_cumulative_series())

    # January of each year keeps its own value (it IS the first month's flow).
    jan_2021 = result[result["Period"] == "2021-01"].iloc[0]
    assert jan_2021["Value_Monthly"] == 10

    # A later month's flow is this month's cumulative total minus last month's.
    mar_2021 = result[result["Period"] == "2021-03"].iloc[0]
    assert mar_2021["Value_Monthly"] == 30  # 10*3, the March-only flow

    # The reset at the year boundary must not leak into January's flow
    # (a naive global .diff() would compute Jan 2022 - Dec 2021, a huge
    # negative number, instead of Jan 2022's own value).
    jan_2022 = result[result["Period"] == "2022-01"].iloc[0]
    assert jan_2022["Value_Monthly"] == 10
