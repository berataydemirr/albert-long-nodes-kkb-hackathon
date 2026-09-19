"""
Cumulative-series detection & correction (Faz A2).

BDDK's own note (hackathon brief, page 8): some published tables are
year-to-date cumulative, some are not, and it doesn't say which -
"verilerin ayristirilarak duzenlenmesi onemlidir". This module answers
that from the data itself instead of guessing per table:

Heuristic for "this line item is YTD-cumulative within a calendar year":
  1. Within each year, consecutive months are (near-)monotonically
     non-decreasing - a cumulative total can't shrink.
  2. At every year boundary, the value drops back down substantially
     (next January < last December) - the running total resets.
A stock figure (an account balance, a rate, an index) fails #2: it
carries its December level into January and keeps drifting from there,
as confirmed for BDDK Monthly tablo_4's 'Tuketici Kredileri - Konut'
line in the exploration notes (Dec 2021: 298896 -> Jan 2022: 300667,
no reset).
"""
import pandas as pd

NONDECREASING_TOLERANCE = -0.005  # allow -0.5% relative noise/rounding, still counts as "non-decreasing"
RESET_DROP_THRESHOLD = 0.5        # next Jan must fall below 50% of last Dec to count as a reset

# BDDK Monthly's tablo_2 ("Kar Zarar") is the income statement: every
# line in it - revenue AND expense/loss lines alike - accumulates from
# the start of the calendar year by Turkish accounting convention. That
# is true by definition, not something the monotonic-increase test below
# can safely decide: a net/loss line (e.g. "Kambiyo Karlari Zararlari
# Net") legitimately falls within a year whenever the bank posts a loss
# that month, so the monotonic test alone misreads it as "not
# cumulative" - confirmed live, monotonic_ratio ~0.60 for that exact
# line despite it being YTD-cumulative by construction. Table membership
# is ground truth here and overrides the math instead of being outvoted
# by it.
INCOME_STATEMENT_TABLES = {"tablo_2"}


def detect_cumulative(df: pd.DataFrame, period_col: str = "Period", value_col: str = "Value",
                       table: str | None = None) -> dict:
    """Runs the two-part heuristic on a single (already sorted-by-period)
    series and returns a diagnostic dict - never a silent True/False,
    so a human can sanity-check borderline cases before trusting it.
    is_cumulative is True/False/None - None ("uncertain") means the data
    itself cannot answer the question (see the boundaries==0 case below),
    which is a different, weaker claim than a confident False and must
    not be treated as one.

    table, when given (e.g. "tablo_2"), lets the income-statement rule
    above override the math for a line item the accounting rule already
    settles - see INCOME_STATEMENT_TABLES."""
    work = df[[period_col, value_col]].dropna().copy()
    work["_year"] = work[period_col].str.slice(0, 4).astype(int)
    work = work.sort_values(period_col)

    if len(work) < 3:
        return {"is_cumulative": False, "status": "not_cumulative", "reason": "insufficient_data", "n_obs": len(work)}

    # Part 1: within-year monotonic non-decreasing?
    within_year_drops = 0
    within_year_pairs = 0
    for _year, grp in work.groupby("_year"):
        vals = grp[value_col].to_numpy()
        for prev, curr in zip(vals[:-1], vals[1:]):
            within_year_pairs += 1
            if prev == 0:
                continue
            rel_change = (curr - prev) / abs(prev)
            if rel_change < NONDECREASING_TOLERANCE:
                within_year_drops += 1

    monotonic_ratio = 1 - (within_year_drops / within_year_pairs) if within_year_pairs else 0

    # Part 2: does it reset at year boundaries?
    yearly_last = work.groupby("_year")[value_col].last()
    yearly_first = work.groupby("_year")[value_col].first()
    years = sorted(work["_year"].unique())

    resets, boundaries = 0, 0
    for y_prev, y_next in zip(years[:-1], years[1:]):
        boundaries += 1
        dec_val = yearly_last.get(y_prev)
        jan_val = yearly_first.get(y_next)
        if dec_val and jan_val is not None and jan_val < dec_val * RESET_DROP_THRESHOLD:
            resets += 1

    reset_ratio = resets / boundaries if boundaries else 0

    if table in INCOME_STATEMENT_TABLES:
        is_cumulative, status, reason = True, "cumulative", "income_statement_rule"
    elif boundaries == 0:
        # No year boundary was ever observed (a short series, e.g. only
        # 5 months on record) - the reset test literally cannot run, so
        # this is NOT the same claim as "tested and failed". Reporting a
        # confident False here previously threw away real cumulative
        # candidates (e.g. "Kredi Riskine Esas Tutar") purely because
        # they hadn't lived through a January yet.
        is_cumulative, status, reason = None, "uncertain_short_series", "no_year_boundary_observed"
    else:
        is_cumulative = monotonic_ratio >= 0.9 and reset_ratio >= 0.5
        status = "cumulative" if is_cumulative else "not_cumulative"
        reason = "monotonic+resets" if is_cumulative else "fails_monotonic_or_reset_test"

    return {
        "is_cumulative": is_cumulative,
        "status": status,
        "monotonic_ratio": round(monotonic_ratio, 3),
        "reset_ratio": round(reset_ratio, 3),
        "n_obs": len(work),
        "year_boundaries_checked": boundaries,
        "reason": reason,
    }


def decumulate(df: pd.DataFrame, period_col: str = "Period", value_col: str = "Value") -> pd.DataFrame:
    """Converts a YTD-cumulative series into its true per-period (monthly)
    flow: January keeps its original value (it IS the first month's
    flow), every later month in the same year becomes
    current_cumulative - previous_cumulative. Crosses year boundaries
    cleanly because the diff is computed per-year, not globally."""
    work = df[[period_col, value_col]].dropna().copy()
    work["_year"] = work[period_col].str.slice(0, 4)
    work = work.sort_values(period_col)

    work["Value_Monthly"] = work.groupby("_year")[value_col].diff()
    periods = pd.PeriodIndex(work[period_col], freq="M")
    if periods.duplicated().any():
        raise ValueError("Duplicate periods cannot be decumulated.")
    consecutive = pd.Series(periods.asi8, index=work.index).diff().eq(1)
    work.loc[~consecutive, "Value_Monthly"] = float("nan")
    is_january = periods.month == 1
    work.loc[is_january, "Value_Monthly"] = work.loc[is_january, value_col]

    return work[[period_col, value_col, "Value_Monthly"]].rename(
        columns={value_col: "Value_Cumulative"}
    ).reset_index(drop=True)
