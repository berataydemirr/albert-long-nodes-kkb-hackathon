"""C6 - Change Detection Tool (src/agent/change_detection_tool.py).

The original `largest_changes` contract is covered in
test_agent_tools.py and test_agent_tools_stress.py; this file covers
break points, segments, the latest regime and the JSON contract.
"""
import json
import random

import pytest

import src.agent.change_detection_tool as cd
from src.agent.change_detection_tool import change_detection
from src.agent.tools import execute_tool, lakehouse_query


def _monthly(n: int, start_year: int = 2021) -> list[str]:
    return [f"{start_year + i // 12}-{i % 12 + 1:02d}" for i in range(n)]


def _steps(levels: list[float], seg_len: int = 15, noise: float = 0.3, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [level + rng.uniform(-noise, noise) for level in levels for _ in range(seg_len)]


def test_single_level_shift_is_found_at_the_first_new_period():
    periods, values = _monthly(30), _steps([10, 20])
    result = change_detection(periods, values)
    assert result["method"] == "pelt"
    assert [c["period"] for c in result["change_points"]] == [periods[15]]
    cp = result["change_points"][0]
    assert cp["direction"] == "up"
    assert cp["shift"] == pytest.approx(10, abs=0.5)
    assert result["latest_regime"]["since"] == periods[15]
    assert "[CHANGE POINT]" in result["event_signatures"][0]


def test_two_shifts_give_three_segments():
    periods, values = _monthly(45), _steps([10, 20, 10])
    result = change_detection(periods, values)
    assert [c["period"] for c in result["change_points"]] == [periods[15], periods[30]]
    assert [c["direction"] for c in result["change_points"]] == ["up", "down"]
    assert len(result["segments"]) == 3
    assert all(s["n_periods"] >= result["min_segment"] for s in result["segments"])


def test_moving_average_method_finds_the_same_shift():
    periods, values = _monthly(30), _steps([10, 20])
    result = change_detection(periods, values, method="moving_average")
    assert result["method"] == "moving_average"
    assert [c["period"] for c in result["change_points"]] == [periods[15]]


def test_pct_change_signal_finds_growth_acceleration():
    rng = random.Random(1)
    periods = _monthly(40)
    values = [100.0]
    for i in range(1, 40):
        growth = 0.01 if i <= 20 else 0.05
        values.append(values[-1] * (1 + growth) * (1 + rng.uniform(-0.002, 0.002)))
    result = change_detection(periods, values, signal="pct_change")
    assert result["signal"] == "pct_change"
    assert len(result["change_points"]) == 1
    cp = result["change_points"][0]
    assert cp["period"] in periods[20:23]
    assert cp["after_mean"] == pytest.approx(5, abs=0.5)
    assert "pp" in result["event_signatures"][0]


def test_pct_change_with_zero_falls_back_to_diff():
    periods = _monthly(20)
    values = [0.0] + [float(i) for i in range(1, 20)]
    result = change_detection(periods, values, signal="pct_change")
    assert result["signal"] == "diff"
    assert any("used diff" in n for n in result["notes"])


def test_constant_series_has_no_change_points():
    result = change_detection(_monthly(20), [5.0] * 20)
    assert result["change_points"] == []
    assert len(result["segments"]) == 1
    assert any("constant" in n for n in result["notes"])


def test_short_series_skips_break_search_but_keeps_largest_changes():
    result = change_detection(["2021-01", "2021-02", "2021-03"], [100, 110, 55])
    assert result["change_points"] == []
    assert result["largest_changes"][0]["period"] == "2021-03"
    assert any("usable points" in n for n in result["notes"])


def test_lower_sensitivity_never_finds_more_breaks():
    rng = random.Random(5)
    values, level = [], 50.0
    for _ in range(60):
        level += rng.uniform(-1, 1)
        values.append(level)
    periods = _monthly(60)
    counts = [len(change_detection(periods, values, sensitivity=s)["change_points"]) for s in ("low", "medium", "high")]
    assert counts == sorted(counts)


def test_summary_reports_half_vs_half_trend():
    result = change_detection(_monthly(30), _steps([10, 20]))
    assert result["summary"]["trend_direction"] == "increasing"
    assert result["summary"]["trend_pct"] > 5


def test_missing_values_are_skipped_and_json_safe():
    periods, values = _monthly(30), _steps([10, 20])
    values[3], values[20] = None, float("nan")
    result = change_detection(periods, values)
    json.loads(json.dumps(result))  # strict: no NaN tokens
    assert [c["period"] for c in result["change_points"]] == [periods[15]]


def test_falls_back_to_moving_average_without_ruptures(monkeypatch):
    monkeypatch.setattr(cd, "rpt", None)
    result = change_detection(_monthly(30), _steps([10, 20]))
    assert result["method"] == "moving_average"
    assert any("ruptures is not installed" in n for n in result["notes"])


@pytest.mark.parametrize("kwargs", [{"method": "bocpd"}, {"signal": "log"}, {"sensitivity": "extreme"}, {"min_segment": 1}])
def test_invalid_parameters_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        change_detection(_monthly(10), list(range(10)), **kwargs)


def test_execute_tool_round_trip_is_valid_json():
    periods, values = _monthly(30), _steps([10, 20])
    payload = json.loads(execute_tool("change_detection", json.dumps({"periods": periods, "values": values})))
    assert payload["change_points"][0]["period"] == periods[15]
    assert {"segments", "latest_regime", "largest_changes", "event_signatures", "warning"} <= payload.keys()


def test_real_data_policy_rate_breaks_at_the_2023_tightening_turn():
    """Real gold data: the policy rate went 8.5% -> 42.5% in monthly
    steps between 2023-06 and 2023-12; the upward break must fall inside
    that staircase (the exact step PELT picks within it is arbitrary)."""
    rows = lakehouse_query("SELECT Period, Deger FROM gold_fact_long WHERE Seri_Kod = 'TP.BISPOLFAIZ.TUR' ORDER BY Period")["rows"]
    result = change_detection([r["Period"] for r in rows], [r["Deger"] for r in rows])
    ups = [c for c in result["change_points"] if c["direction"] == "up" and "2023-06" <= c["period"] <= "2023-12"]
    assert len(ups) == 1
