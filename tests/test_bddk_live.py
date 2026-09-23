from src.agent import bddk_live
from src.agent.router import Route, route_question
from src.agent.tools import TOOL_SCHEMAS
from src.agent.untrusted import is_untrusted_call


def test_live_bddk_report_parses_caption_and_rows(monkeypatch):
    response = {
        "success": True,
        "Json": {
            "caption": "Mevduat Vade İtibarıyla (milyon TL), Dönem:2025/12",
            "colNames": ["", "", "", "Vadesiz", "Toplam"],
            "data": {"total": "1", "rows": [
                {"cell": ["Sektör", "1", "TP Mevduat / Katılım Fonları - Yurt İçi Yerleşik", "20", "50"]},
                {"cell": ["Sektör", "2", "Başka kalem", "10", "30"]},
            ]},
        },
    }
    monkeypatch.setattr(bddk_live, "make_session", lambda: object())
    monkeypatch.setattr(bddk_live, "fetch_page", lambda *args: response)

    result = bddk_live.fetch_bddk_monthly_table(2025, 12, 10, "Yurt İçi Yerleşik")

    assert result["unit"] == "milyon TL"
    assert result["row_count"] == 1
    assert result["rows"][0]["Vadesiz"] == "20"
    assert result["source_url"] == bddk_live.URL


def test_bddk_url_route_exposes_live_report_tool():
    decision = route_question("https://www.bddk.org.tr/BultenAylik/tr/Home/Index mevduat", TOOL_SCHEMAS)
    assert decision.route is Route.WEB_URL
    assert "fetch_bddk_monthly_table" in decision.allowed_tool_names
    assert is_untrusted_call("fetch_bddk_monthly_table", {"year": 2025, "month": 12, "table_id": 10})
