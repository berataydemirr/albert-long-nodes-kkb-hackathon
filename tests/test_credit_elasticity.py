import json

import pytest

from src.agent import orchestrator
from src.agent.tools import TOOL_SPECS, credit_elasticity_analysis, execute_tool


def test_credit_elasticity_uses_shared_quarters_and_real_values():
    result = credit_elasticity_analysis(start_period="2023-01", end_period="2024-12")

    assert result["effective_shared_period"] == {"start": "2023-03", "end": "2024-12", "months": 21}
    assert result["shared_observations"] == 8
    assert result["policy_rate"]["peak_in_shared_periods"] == 50.0
    credits = {item["label"]: item for item in result["credits"]}
    assert credits["konut_kredisi"]["real_change_pct"] == pytest.approx(-38.1568)
    assert credits["tasit_kredisi"]["real_change_pct"] == pytest.approx(-43.9476)
    assert result["more_rate_sensitive_by_absolute_elasticity"] == "tasit_kredisi"


def test_credit_elasticity_is_registered_as_one_analysis_tool():
    spec = TOOL_SPECS["credit_elasticity_analysis"]

    assert spec.group == "analysis"
    payload = json.loads(execute_tool("credit_elasticity_analysis", json.dumps({
        "start_period": "2023-01", "end_period": "2024-12"
    })))
    assert payload["shared_observations"] == 8


def test_deterministic_fallback_can_render_a_complete_elasticity_answer():
    payload = credit_elasticity_analysis()
    answer = orchestrator._deterministic_fallback([{
        "name": "credit_elasticity_analysis", "error": None, "output_json": payload
    }], "model_error")

    assert "Taşıt kredisi" in answer
    assert "-43.9476" in answer
    assert "nedensellik tahmini değildir" in answer
