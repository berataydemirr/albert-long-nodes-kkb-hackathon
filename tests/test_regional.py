"""Tests for province-level access.

These run against the real regional table, because the two failure modes
this module exists to prevent are both data-shaped: a period ordering
that is wrong only inside a year, and a non-province row that is large
enough to sit inside a top-10 list. Neither reproduces against a fixture
that does not carry the real quirks.
"""

import pytest

from src.agent.regional import (
    DB_PATH,
    NON_PROVINCE_LABELS,
    _quarter_sort_key,
    available_periods,
    compare_regions,
)

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(), reason="gold database not built; run `python -m src.gold.build_gold`"
)


# --- the period ordering trap -------------------------------------------

def test_quarter_sort_key_orders_within_a_year():
    """Plain string ordering puts '2021-12' before '2021-3'. This is the
    single assertion the whole module's correctness rests on."""
    raw = ["2021-9", "2021-12", "2021-3", "2021-6"]

    assert sorted(raw) != ["2021-3", "2021-6", "2021-9", "2021-12"]
    assert sorted(raw, key=_quarter_sort_key) == ["2021-3", "2021-6", "2021-9", "2021-12"]


def test_quarter_sort_key_orders_across_years():
    raw = ["2026-3", "2021-12", "2025-9"]
    assert sorted(raw, key=_quarter_sort_key) == ["2021-12", "2025-9", "2026-3"]


def test_unparseable_period_sorts_last_instead_of_raising():
    """A malformed period must not take a whole ranking down."""
    assert _quarter_sort_key("bozuk") == (9999, 99)
    assert sorted(["bozuk", "2026-6"], key=_quarter_sort_key) == ["2026-6", "bozuk"]


def test_available_periods_are_correctly_ordered():
    periods = available_periods()

    assert periods == sorted(periods, key=_quarter_sort_key)
    assert periods != sorted(periods), "the table really does contain the string-sort trap"


def test_default_period_is_the_true_latest_not_the_string_max():
    result = compare_regions()

    assert result["period"] == available_periods()[-1]


# --- the non-province trap ----------------------------------------------

def test_yurt_disi_is_excluded_from_province_rankings_by_default():
    result = compare_regions(metric="kredi", top_n=25)

    ranked_names = {row["bolge"] for row in result["ranked"]}
    assert not (ranked_names & set(NON_PROVINCE_LABELS))


def test_yurt_disi_is_reported_rather_than_silently_dropped():
    """Excluding it is correct; hiding it would be a different bug."""
    result = compare_regions(metric="kredi")

    assert result["non_province_rows"]
    assert result["non_province_rows"][0]["bolge"] in NON_PROVINCE_LABELS


def test_non_provinces_can_be_included_deliberately():
    included = compare_regions(metric="kredi", top_n=25, include_non_provinces=True)

    names = {row["bolge"] for row in included["ranked"]}
    assert names & set(NON_PROVINCE_LABELS), "YURT DISI is big enough to make a top-25"


def test_province_count_matches_turkey():
    result = compare_regions()
    assert result["province_count"] == 81


# --- ranking behaviour --------------------------------------------------

def test_ranking_is_ordered_and_ranks_are_sequential():
    result = compare_regions(metric="kredi", top_n=10)

    values = [row["value"] for row in result["ranked"]]
    assert values == sorted(values, reverse=True)
    assert [row["rank"] for row in result["ranked"]] == list(range(1, len(values) + 1))


def test_ascending_order_returns_the_smallest_first():
    result = compare_regions(metric="kredi", top_n=5, order="asc")

    values = [row["value"] for row in result["ranked"]]
    assert values == sorted(values)


def test_istanbul_leads_national_lending():
    """A sanity anchor: if this ever fails, the join or the metric is wrong,
    not Turkish economic geography."""
    result = compare_regions(metric="kredi", top_n=1)

    assert result["ranked"][0]["bolge"] == "İSTANBUL"


def test_shares_are_reported_for_credit_and_sum_sensibly():
    result = compare_regions(metric="kredi", top_n=25)

    shares = [row["share_pct"] for row in result["ranked"]]
    assert all(0 <= s <= 100 for s in shares)
    assert sum(shares) <= 100.01


def test_share_is_not_reported_for_a_ratio_metric():
    """A percentage share of an NPL ratio would be meaningless."""
    result = compare_regions(metric="npl", top_n=5)

    assert all("share_pct" not in row for row in result["ranked"])
    assert "total_value" not in result


def test_npl_values_are_plausible_ratios():
    result = compare_regions(metric="npl", top_n=81)

    values = [row["value"] for row in result["ranked"]]
    assert all(0 <= v < 1 for v in values), "NPL is stored as a 0-1 ratio, not a percentage"


# --- growth -------------------------------------------------------------

def test_growth_is_added_against_an_earlier_quarter():
    periods = available_periods()
    result = compare_regions(period=periods[-1], compare_to_period=periods[-5], top_n=5)

    assert result["compare_to_period"] == periods[-5]
    assert all("growth_pct" in row for row in result["ranked"])


def test_growth_against_a_later_quarter_is_rejected():
    """Silently returning a negative 'growth' would misread as a decline."""
    periods = available_periods()

    with pytest.raises(ValueError, match="must be earlier"):
        compare_regions(period=periods[0], compare_to_period=periods[-1])


def test_growth_against_the_same_quarter_is_rejected():
    periods = available_periods()

    with pytest.raises(ValueError, match="must be earlier"):
        compare_regions(period=periods[-1], compare_to_period=periods[-1])


# --- input validation ---------------------------------------------------

def test_unknown_period_lists_what_is_available():
    with pytest.raises(ValueError) as excinfo:
        compare_regions(period="2019-3")

    assert "Available" in str(excinfo.value)


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="Unknown metric"):
        compare_regions(metric="kar")


def test_bad_order_is_rejected():
    with pytest.raises(ValueError, match="desc"):
        compare_regions(order="yukari")


@pytest.mark.parametrize("bad", [0, -3, "5", 2.5, True])
def test_bad_top_n_is_rejected(bad):
    with pytest.raises(ValueError, match="positive integer"):
        compare_regions(top_n=bad)


def test_top_n_is_capped_so_the_context_cannot_blow_up():
    """The tool exists so the model can ask about regions WITHOUT pulling
    all 82 rows into context; an uncapped top_n would defeat that."""
    result = compare_regions(top_n=500)

    assert result["returned"] <= 25


def test_period_is_passed_as_a_parameter_not_concatenated():
    with pytest.raises(ValueError) as excinfo:
        compare_regions(period="' OR 1=1 --")

    assert "Unknown period" in str(excinfo.value)
