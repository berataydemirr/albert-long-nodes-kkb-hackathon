"""
Frequency alignment (Faz A1): puts every silver source - daily, weekly
(BDDK donem id), monthly, quarterly - onto one common monthly axis.

DECISION: the common axis is MONTHLY, not weekly. Reasoning: 4 of 5
sources are already monthly-native (EVDS mostly, TUIK, BDDK Monthly),
the hackathon's own demo scenario asks for "60 aylik veriden ... aylik"
output (see docs), and downsampling weekly/daily -> monthly is a
straightforward last-observation-of-month reduction, while upsampling
monthly -> weekly would mean inventing data through interpolation. A
weekly gold axis is still possible later using `bddk_weekly_donem_to_date`
below without touching the other sources.

Every function takes/returns a plain 'Period' column in 'YYYY-MM' form
so callers can join on it directly - no datetime objects leak across
module boundaries.
"""
import datetime as dt

import pandas as pd

# Anchor confirmed in config/settings.py: donem 367 = week 1 of 2021,
# starting 2021-01-08. BDDK's weekly bulletin advances one donem per week.
_DONEM_ANCHOR_ID = 367
_DONEM_ANCHOR_DATE = dt.date(2021, 1, 8)


def donem_id_to_date(donem_id: int) -> dt.date:
    """Converts a BDDK weekly bulletin 'donemId' to its real calendar date."""
    weeks_after_anchor = donem_id - _DONEM_ANCHOR_ID
    return _DONEM_ANCHOR_DATE + dt.timedelta(weeks=weeks_after_anchor)


def to_period(date_like) -> str:
    """Any date/Timestamp/'YYYY-MM-DD' string -> 'YYYY-MM'."""
    ts = pd.Timestamp(date_like)
    return f"{ts.year:04d}-{ts.month:02d}"


def downsample_to_monthly(df: pd.DataFrame, date_col: str, value_cols: list[str],
                           group_cols: list[str] | None = None, how: str = "last") -> pd.DataFrame:
    """Collapses daily/weekly rows to one row per calendar month.

    how='last'  -> last observation of the month (right choice for STOCK
                   figures: balances, rates, indices - the number that
                   was true at month end).
    how='mean'  -> average over the month (right choice for FLOW figures
                   sampled sub-monthly, e.g. a daily FX rate you want as
                   a monthly reference rate rather than a single day's
                   value).
    `group_cols` (e.g. ['Item'] for BDDK weekly's per-line-item rows) are
    preserved as an extra grouping key so unrelated rows don't get mixed.
    """
    work = df.copy()
    work["_period"] = pd.to_datetime(work[date_col]).apply(to_period)

    keys = ["_period"] + (group_cols or [])
    work = work.sort_values(date_col)

    if how == "last":
        out = work.groupby(keys, as_index=False)[value_cols].last()
    elif how == "mean":
        out = work.groupby(keys, as_index=False)[value_cols].mean(numeric_only=True)
    else:
        raise ValueError(f"Unknown how='{how}', expected 'last' or 'mean'")

    return out.rename(columns={"_period": "Period"})


def upsample_quarterly_to_monthly(df: pd.DataFrame, quarter_col: str, value_cols: list[str],
                                   group_cols: list[str] | None = None) -> pd.DataFrame:
    """Expands FinTurk-style 'YYYY-Q' quarterly rows (Q in 3,6,9,12) into
    3 monthly rows via forward-fill. EXPLICIT LIMITATION, stated rather
    than hidden: this repeats the quarter-end value across its 3 months
    because no finer-grained figure exists - it is an approximation, not
    a real monthly observation. A '_period_type' column marks which rows
    are forward-filled so downstream tools/LLM prompts can say so.
    """
    rows = []
    for _, row in df.iterrows():
        year, q = row[quarter_col].split("-")
        year, q = int(year), int(q)
        for m in range(q - 2, q + 1):  # e.g. q=6 -> months 4,5,6
            period = f"{year:04d}-{m:02d}"
            new_row = row.to_dict()
            new_row["Period"] = period
            new_row["_period_type"] = "observed" if m == q else "quarter_ffill"
            rows.append(new_row)

    out = pd.DataFrame(rows)
    cols = ["Period", "_period_type"] + (group_cols or []) + value_cols
    return out[cols]


def forward_fill_gaps(df: pd.DataFrame, period_col: str, value_cols: list[str],
                       start_period: str, end_period: str,
                       group_cols: list[str] | None = None) -> pd.DataFrame:
    """Fills any missing months in [start_period, end_period] by carrying
    the last known value forward (e.g. a monthly series that has a
    silent gap). Rows created this way are flagged in '_filled'."""
    full_index = pd.period_range(start_period, end_period, freq="M").strftime("%Y-%m")

    if group_cols:
        parts = []
        for keys, grp in df.groupby(group_cols):
            grp = grp.set_index(period_col).reindex(full_index)
            grp["_filled"] = grp[value_cols[0]].isna()
            grp[value_cols] = grp[value_cols].ffill()
            if not isinstance(keys, tuple):
                keys = (keys,)
            for col, val in zip(group_cols, keys):
                grp[col] = val
            grp[period_col] = grp.index
            parts.append(grp.reset_index(drop=True))
        return pd.concat(parts, ignore_index=True)

    out = df.set_index(period_col).reindex(full_index)
    out["_filled"] = out[value_cols[0]].isna()
    out[value_cols] = out[value_cols].ffill()
    out[period_col] = out.index
    return out.reset_index(drop=True)
