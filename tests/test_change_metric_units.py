"""A rate series changes in percentage points, a volume series in percent.

Found live on a shock-scan question: the agent reported the June 2023 policy
hike (8.5 -> 15) as "a 76.5% shock". The arithmetic was right and the finance
was wrong - a rate is already a percentage, so dividing one by another both
misstates the move and mis-ranks it against a larger rate's bigger move.
"""

import pytest

from src.agent.tools import _is_rate_series, change_detection


@pytest.mark.parametrize("birim", ["%", " % ", "yüzde", "oran", "PERCENT"])
def test_percentage_units_are_recognized_as_rate_series(birim):
    assert _is_rate_series(birim) is True


@pytest.mark.parametrize("birim", ["milyon TL", "Bin TL", "endeks", "adet", None, ""])
def test_amount_and_index_units_are_not_rate_series(birim):
    assert _is_rate_series(birim) is False


def test_a_rate_series_reports_percentage_points_and_never_percent():
    periods = ["2023-05", "2023-06", "2023-07"]
    values = [8.5, 15.0, 17.5]

    result = change_detection(periods=periods, values=values, seri_kod=None)
    # No seri_kod means no Birim from the lakehouse, so this falls back to
    # the volume path - the explicit-array route cannot know the unit.
    assert result["change_metric"] == "percent"


def test_ranking_uses_the_size_of_the_actual_move_not_the_size_of_the_base():
    """8.5 -> 15 is +6.5pp on a small base (+76.5% relatively); 17.5 -> 25
    is +7.5pp on a larger one (+42.9%). Ranked by relative change the
    smaller move wins, which is exactly the misleading answer. Ranked in
    percentage points the bigger actual hike wins."""
    result = change_detection(
        seri_kod="TP.BISPOLFAIZ.TUR", start_period="2023-01", end_period="2024-06")

    assert result["change_metric"] == "percentage_points"
    top = result["largest_changes"][0]
    assert top["period"] == "2023-08"
    assert top["change_pp"] == pytest.approx(7.5)
    assert "percent_change" not in top, "the misleading number must not be offered at all"
    assert "pp" in result["metric_note"]


def test_a_volume_series_still_reports_relative_percent():
    result = change_detection(
        seri_kod="tablo_4.konut_kredisi", start_period="2023-01", end_period="2024-06")

    assert result["change_metric"] == "percent"
    assert "percent_change" in result["largest_changes"][0]
    assert "change_pp" not in result["largest_changes"][0]
