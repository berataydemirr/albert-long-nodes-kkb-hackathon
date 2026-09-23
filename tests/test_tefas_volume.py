"""The TEFAS report footer and partial responses must never become data."""

from decimal import Decimal

import pytest

from src.agent.tefas_volume import _validated_rows


def test_volume_report_keeps_both_amounts_and_excludes_footer():
    expected = ["2021-01", "2021-02"]
    payload = {"errorCode": None, "data": [
        {"donem": "2021-01", "toplamIslemHacmi": Decimal("17867617148.62"),
         "netTakasTutar": Decimal("3357468920.89")},
        {"donem": "2021-02", "toplamIslemHacmi": Decimal("15069010334.81"),
         "netTakasTutar": Decimal("3918184474.76")},
        {"donem": "Toplam", "toplamIslemHacmi": Decimal("32936627483.43"),
         "netTakasTutar": Decimal("7275653395.65")},
    ]}
    rows = _validated_rows(payload, expected)
    assert len(rows) == 2
    assert rows[0] == {"Period": "2021-01",
                       "Toplam_Islem_Hacmi_TL": "17867617148.62",
                       "Net_Takas_Tutari_TL": "3357468920.89"}


@pytest.mark.parametrize("data", [
    [{"donem": "2021-01", "toplamIslemHacmi": 1, "netTakasTutar": 2}],
    [{"donem": "2021-01", "toplamIslemHacmi": 1, "netTakasTutar": 2},
     {"donem": "2021-01", "toplamIslemHacmi": 1, "netTakasTutar": 2}],
    [{"donem": "2021-01", "toplamIslemHacmi": 1, "netTakasTutar": None},
     {"donem": "2021-02", "toplamIslemHacmi": 1, "netTakasTutar": 2}],
])
def test_volume_report_rejects_incomplete_duplicate_or_invalid_rows(data):
    with pytest.raises(ValueError):
        _validated_rows({"errorCode": None, "data": data}, ["2021-01", "2021-02"])
