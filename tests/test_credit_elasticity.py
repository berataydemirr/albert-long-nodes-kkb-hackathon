import json

import pytest

from src.agent import orchestrator
from src.agent.tools import TOOL_SPECS, credit_elasticity_analysis, execute_tool


def test_credit_elasticity_uses_the_literal_requested_months_not_the_nearest_quarter():
    """Real repro: the policy rate is 9.0 in 2023-01 and first hits 50.0 in
    2024-03. The old implementation snapped to whatever quarter-end months
    happened to be shared with the quarterly vehicle series inside a
    generic 2023-01..2024-12 request, landing on 2023-03 (8.5%) and
    2024-12 (47.5%, already past the peak) - missing the actual shock
    window the caller asked about."""
    result = credit_elasticity_analysis(start_period="2023-01", end_period="2024-03")

    assert result["compared_period"] == {"start": "2023-01", "end": "2024-03", "months": 14}
    assert result["policy_rate"]["start"] == 9.0
    assert result["policy_rate"]["end"] == 50.0
    assert result["policy_rate"]["peak_in_range"] == 50.0


def test_credit_elasticity_converts_every_amount_to_a_common_billion_try_unit():
    """Real repro: housing is stored in 'milyon TL' and vehicle credit in
    'Bin TL' - returned unconverted, the answer showed '391.782 M TL' next
    to '65.261.409 Bin TL', unreadable and three orders of magnitude apart
    at a glance. Both must come back in the same 'milyar TL' unit."""
    result = credit_elasticity_analysis(start_period="2023-01", end_period="2024-03")
    credits = {item["label"]: item for item in result["credits"]}

    assert result["unit"].startswith("milyar TL")
    assert credits["konut_kredisi"]["nominal_start_milyar_tl"] == pytest.approx(362.695, abs=0.01)
    assert credits["tasit_kredisi"]["nominal_start_milyar_tl"] == pytest.approx(50.261, abs=0.01)
    # Same unit on both sides, so a direct magnitude comparison is now valid.
    assert credits["konut_kredisi"]["nominal_start_milyar_tl"] > credits["tasit_kredisi"]["nominal_start_milyar_tl"]


def test_credit_elasticity_reports_real_values_and_the_more_sensitive_side():
    result = credit_elasticity_analysis(start_period="2023-01", end_period="2024-03")
    credits = {item["label"]: item for item in result["credits"]}

    assert credits["konut_kredisi"]["real_change_pct"] == pytest.approx(-31.0888)
    assert credits["tasit_kredisi"]["real_change_pct"] == pytest.approx(5.412)
    # Housing contracted in real terms over the shock window; vehicle credit's
    # nominal growth (aided by forward-filled quarterly data) briefly outran
    # CPI. abs(elasticity) is therefore larger on the housing side here -
    # a different, period-dependent answer from a looser date range, which
    # is exactly why the exact months matter (see the test above).
    assert result["more_rate_sensitive_by_absolute_elasticity"] == "konut_kredisi"


def test_credit_elasticity_surfaces_bank_facing_consumer_rates_not_just_the_policy_rate():
    """Real repro: the policy rate moved +41pp, but that is not what a
    borrower actually pays. Housing's own consumer rate moved only
    +3.97pp over the same window (rate-suppressed housing campaigns), a
    gap the policy-rate-only elasticity number hides entirely. No
    standalone vehicle-only consumer rate exists in the lakehouse, so the
    combined ihtiyac+tasit+konut rate is used and labeled as a proxy, not
    invented as if it were vehicle-specific."""
    result = credit_elasticity_analysis(start_period="2023-01", end_period="2024-03")
    context = result["consumer_rate_context"]

    assert context["housing_consumer_rate"]["change_pp"] == pytest.approx(3.97, abs=0.01)
    assert context["combined_consumer_rate_incl_vehicle"]["change_pp"] == pytest.approx(17.23, abs=0.01)
    assert context["policy_rate_change_pp"] == 41.0
    assert "combined ihtiyac+tasit+konut" in context["note"]


def test_credit_elasticity_fails_fast_when_a_boundary_month_has_no_real_observation():
    with pytest.raises(ValueError, match="no observation at start_period"):
        credit_elasticity_analysis(start_period="1999-01", end_period="2024-03")


def test_credit_elasticity_is_registered_as_one_analysis_tool():
    spec = TOOL_SPECS["credit_elasticity_analysis"]

    assert spec.group == "analysis"
    payload = json.loads(execute_tool("credit_elasticity_analysis", json.dumps({
        "start_period": "2023-01", "end_period": "2024-03"
    })))
    assert payload["policy_rate"]["start"] == 9.0
    assert payload["policy_rate"]["end"] == 50.0


def test_deterministic_fallback_can_render_a_complete_elasticity_answer():
    payload = credit_elasticity_analysis()
    answer = orchestrator._deterministic_fallback([{
        "name": "credit_elasticity_analysis", "error": None, "output_json": payload
    }], "model_error")

    assert "Konut kredisi" in answer
    assert "nedensellik tahmini değildir" in answer
    assert "bankanın müşteriye yansıttığı faiz" in answer
