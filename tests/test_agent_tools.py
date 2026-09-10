import pytest

from src.agent.tools import anomaly_analysis, change_detection, correlation_analysis
from src.agent.web_search import web_search
import src.agent.web_search as web_search_module


def test_anomaly_analysis_finds_outlier():
    result = anomaly_analysis(["2021-01", "2021-02", "2021-03", "2021-04"], [10, 11, 10, 100])
    assert result["anomalies"][0]["period"] == "2021-04"


def test_change_detection_orders_by_relative_change():
    result = change_detection(["2021-01", "2021-02", "2021-03"], [100, 110, 55])
    assert result["largest_changes"][0]["period"] == "2021-03"


def test_correlation_warns_against_causal_claim():
    result = correlation_analysis(["2021-01", "2021-02", "2021-03"], [1, 2, 3], ["2021-01", "2021-02", "2021-03"], [2, 4, 6])
    assert result["pearson_correlation"] == pytest.approx(1.0)
    assert "not proof of causality" in result["warning"]


def test_web_search_parses_compact_citation_ready_results(monkeypatch):
    class FakeResponse:
        text = """
        <div class='result'>
          <a class='result__a' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.tcmb.gov.tr%2F'>TCMB</a>
          <a class='result__snippet'>Policy rate announcement</a>
        </div>
        <div class='result'>
          <a class='result__a' href='https://www.tcmb.gov.tr/'>Duplicate TCMB</a>
        </div>
        """

        def raise_for_status(self):
            return None

    captured = {}

    def fake_get(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(web_search_module.requests, "get", fake_get)
    monkeypatch.setattr(web_search_module, "WEB_SEARCH_PROVIDER", "duckduckgo")

    result = web_search("TCMB politika faizi", max_results=3)

    assert captured["params"] == {"q": "TCMB politika faizi"}
    assert captured["timeout"] == 12
    assert result["result_count"] == 1
    assert result["results"] == [{
        "title": "TCMB",
        "url": "https://www.tcmb.gov.tr/",
        "snippet": "Policy rate announcement",
        "source": "www.tcmb.gov.tr",
    }]
    assert "not verified" in result["citation_guidance"]


@pytest.mark.parametrize("query,max_results", [(None, 5), ("", 5), ("x" * 501, 5), ("valid", 0), ("valid", 11), ("valid", True)])
def test_web_search_validates_its_input(query, max_results):
    with pytest.raises(ValueError):
        web_search(query, max_results)


def test_web_search_is_exposed_to_the_model():
    from src.agent.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

    schema = next(item for item in TOOL_SCHEMAS if item["function"]["name"] == "web_search")
    assert TOOL_FUNCTIONS["web_search"] is web_search
    assert schema["function"]["parameters"]["properties"]["max_results"]["maximum"] == 10
