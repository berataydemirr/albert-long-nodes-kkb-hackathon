"""C4 - Anomaly Tool (src/agent/anomaly_tool.py).

Backward-compatible behaviour (MAD on short series, None/NaN handling,
MAD=0 fallback) is covered in test_agent_tools.py and
test_agent_tools_stress.py; this file covers the new methods, filters,
episodes and the JSON contract the agent sees.
"""
import json
import math
import random

import pytest

from src.agent.anomaly_tool import anomaly_analysis, infer_season_length
from src.agent.tools import execute_tool, lakehouse_query


def _monthly(n: int, start_year: int = 2021) -> list[str]:
    return [f"{start_year + i // 12}-{i % 12 + 1:02d}" for i in range(n)]


def _noisy(n: int, level: float = 100.0, noise: float = 1.0, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [level + rng.uniform(-noise, noise) for _ in range(n)]


def _growing(n: int, growth: float = 0.02, noise: float = 0.01, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [100 * (1 + growth) ** i * (1 + rng.uniform(-noise, noise)) for i in range(n)]


def test_auto_on_trending_series_flags_the_spike_not_the_trend_end():
    """Global MAD on a trending series flags the END of the trend - the
    reason rolling_zscore is the auto default for long series."""
    periods, values = _monthly(48), _growing(48)
    values[30] *= 1.30
    result = anomaly_analysis(periods, values)
    flagged = {a["period"] for a in result["anomalies"]}
    assert result["method"] == "rolling_zscore"
    assert periods[30] in flagged
    assert not flagged & set(periods[-6:])


def test_spike_does_not_make_following_points_look_like_drops():
    """Moving MEDIAN baseline: a spike inside the trailing window must not
    inflate the baseline for the next `window` points."""
    periods, values = _monthly(40), _noisy(40)
    values[20] = 160.0
    result = anomaly_analysis(periods, values)
    assert [a["period"] for a in result["anomalies"]] == [periods[20]]


def test_direction_filter_keeps_only_requested_side():
    periods, values = _monthly(36), _noisy(36)
    values[12] = 150.0
    values[24] = 50.0
    up = anomaly_analysis(periods, values, direction="up")
    down = anomaly_analysis(periods, values, direction="down")
    assert [a["period"] for a in up["anomalies"]] == [periods[12]]
    assert [a["period"] for a in down["anomalies"]] == [periods[24]]
    assert down["anomalies"][0]["direction"] == "down"


def test_min_deviation_pct_drops_statistically_odd_but_tiny_moves():
    periods, values = _monthly(36), _noisy(36, noise=0.1)
    values[20] = 102.0  # ~20 noise-units away, but only +2%
    assert periods[20] in {a["period"] for a in anomaly_analysis(periods, values)["anomalies"]}
    assert anomaly_analysis(periods, values, min_deviation_pct=5.0)["anomalies"] == []


def test_severity_critical_for_large_scores():
    periods, values = _monthly(36), _noisy(36)
    values[20] = 200.0
    result = anomaly_analysis(periods, values)
    assert result["anomalies"][0]["severity"] == "critical"


def test_adjacent_anomalies_form_one_resolved_episode_with_recovery():
    periods, values = _monthly(36), _noisy(36)
    values[12] = values[13] = 150.0
    result = anomaly_analysis(periods, values)
    assert len(result["episodes"]) == 1
    episode = result["episodes"][0]
    assert (episode["start_period"], episode["end_period"], episode["n_points"]) == (periods[12], periods[13], 2)
    assert episode["status"] == "resolved"
    assert episode["recovered_on"] == periods[14]
    assert result["latest"]["status"] == "normal"
    assert "back to normal from" in result["event_signatures"][0]


def test_anomaly_on_last_point_is_active():
    periods, values = _monthly(30), _noisy(30)
    values[-1] = 170.0
    result = anomaly_analysis(periods, values)
    assert result["latest"]["status"] == "anomalous"
    assert result["episodes"][-1]["status"] == "active"
    assert result["episodes"][-1]["recovered_on"] is None
    assert "still ongoing" in result["event_signatures"][-1]


def test_stl_removes_seasonality_and_finds_the_real_spike():
    rng = random.Random(3)
    periods = _monthly(48)
    values = [100 + 10 * math.sin(2 * math.pi * i / 12) + rng.uniform(-0.5, 0.5) for i in range(48)]
    values[30] += 8  # smaller than the seasonal swing
    result = anomaly_analysis(periods, values, method="stl")
    assert result["method"] == "stl"
    assert result["season_length"] == 12
    assert periods[30] in {a["period"] for a in result["anomalies"]}
    assert result["anomaly_count"] <= 3


def test_stl_falls_back_without_a_seasonal_period():
    periods = [str(y) for y in range(2001, 2031)]
    result = anomaly_analysis(periods, _noisy(30), method="stl")
    assert result["method"] != "stl"
    assert any("stl not possible" in n for n in result["notes"])


def test_rolling_iqr_flags_spike():
    periods, values = _monthly(36), _noisy(36)
    values[18] = 150.0
    result = anomaly_analysis(periods, values, method="rolling_iqr")
    assert result["method"] == "rolling_iqr"
    assert periods[18] in {a["period"] for a in result["anomalies"]}


def test_short_series_uses_mad_for_backward_compatibility():
    result = anomaly_analysis(["2021-01", "2021-02", "2021-03", "2021-04"], [10, 11, 10, 100])
    assert result["method"] == "mad"


def test_infer_season_length():
    assert infer_season_length(_monthly(24)) == 12
    assert infer_season_length(["2021-3", "2021-6", "2021-9", "2021-12"]) == 4
    assert infer_season_length(["2021", "2022"]) is None


@pytest.mark.parametrize("kwargs", [{"method": "prophet"}, {"direction": "sideways"}, {"threshold": 0}, {"window": 1}, {"min_deviation_pct": -1}])
def test_invalid_parameters_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        anomaly_analysis(_monthly(10), _noisy(10), **kwargs)


def test_non_numeric_value_raises_value_error():
    with pytest.raises(ValueError, match="not numeric"):
        anomaly_analysis(["2021-01", "2021-02", "2021-03"], [1.0, "abc", 3.0])


def test_execute_tool_round_trip_is_valid_json():
    periods, values = _monthly(30), _noisy(30)
    values[10] = 150.0
    payload = json.loads(execute_tool("anomaly_analysis", json.dumps({"periods": periods, "values": values, "direction": "up"})))
    assert payload["anomalies"][0]["period"] == periods[10]
    assert {"episodes", "latest", "event_signatures", "warning"} <= payload.keys()


def test_real_data_usdtry_flags_the_known_shocks():
    """Real gold data: the Nov-2021 lira crash and the Jun-2023
    post-election devaluation must both surface as upward episodes."""
    rows = lakehouse_query("SELECT Period, Deger FROM gold_fact_long WHERE Seri_Kod = 'TP.DK.USD.A.YTL' ORDER BY Period")["rows"]
    result = anomaly_analysis([r["Period"] for r in rows], [r["Deger"] for r in rows])
    flagged = {a["period"] for a in result["anomalies"] if a["direction"] == "up"}
    assert "2021-11" in flagged or "2021-12" in flagged
    assert "2023-06" in flagged or "2023-07" in flagged
