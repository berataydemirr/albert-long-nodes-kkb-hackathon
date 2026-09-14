"""Fetch a gold-layer series by Seri_Kod so the model never has to retype it.

WHY THIS EXISTS (measured, not assumed):
The analysis tools originally accepted only raw `periods`/`values` arrays.
That forced this chain on every analytical question:

    lakehouse_query  ->  66 rows come back
    model            ->  retypes all 66 periods + 66 values into the next
                         tool call's JSON arguments, by hand, token by token

A single causality_analysis call over two 66-month series measured 2442
characters (~610 tokens) of arguments the model had to GENERATE. Output
generation is the slowest phase of inference, those 610 tokens then live
in the message history for every later round, and a small model that
mistypes or truncates one value silently corrupts the statistics.

Resolving the series inside Python instead turns that into ~30 tokens
("x_seri_kod": "TP.BKR.TRY.18"). Same numbers, ~20x fewer tokens, and the
values are read from the database rather than transcribed by an LLM.

Every step logs its state so a wrong series can be traced to the exact
query, filter and row count that produced it.
"""

import re

import duckdb

from config.settings import GOLD_DIR
from src.common.logging_setup import get_logger

logger = get_logger("series_resolver")

DB_PATH = GOLD_DIR / "lakehouse.duckdb"
FACT_TABLE = "gold_fact_long"

# A Seri_Kod is written by the model into a parameterized query, never
# interpolated - but it is also used in log lines and error messages, so
# an absurdly long or newline-stuffed value is rejected early rather than
# polluting the log file.
MAX_SERI_KOD_LEN = 120
_PERIOD_RE = re.compile(r"^\d{4}(-\d{1,2})?$")


def _db() -> duckdb.DuckDBPyConnection:
    logger.debug(f"_db: opening {DB_PATH} (read_only=True)")
    if not DB_PATH.exists():
        logger.error(f"_db: gold database missing at {DB_PATH}")
        raise FileNotFoundError(
            f"Gold database not found: {DB_PATH}. Run `python -m src.gold.build_gold` first."
        )
    return duckdb.connect(str(DB_PATH), read_only=True)


def _validate_seri_kod(seri_kod) -> str:
    logger.debug(f"_validate_seri_kod: raw={seri_kod!r}")
    if not isinstance(seri_kod, str) or not seri_kod.strip():
        logger.error(f"_validate_seri_kod: rejected (not a non-empty string): {seri_kod!r}")
        raise ValueError("seri_kod must be a non-empty string.")
    cleaned = seri_kod.strip()
    if len(cleaned) > MAX_SERI_KOD_LEN:
        logger.error(f"_validate_seri_kod: rejected (length {len(cleaned)} > {MAX_SERI_KOD_LEN})")
        raise ValueError(f"seri_kod is too long (max {MAX_SERI_KOD_LEN} characters).")
    logger.debug(f"_validate_seri_kod: accepted={cleaned!r}")
    return cleaned


def _validate_period(period, field_name: str) -> str | None:
    """Periods are 'YYYY-MM' in the monthly fact table, but quarterly and
    annual rows exist too ('2021-9', '2021'), so the check stays loose on
    zero-padding and only rejects shapes that cannot be a period at all."""
    if period is None:
        return None
    logger.debug(f"_validate_period: {field_name}={period!r}")
    if not isinstance(period, str) or not _PERIOD_RE.match(period.strip()):
        logger.error(f"_validate_period: rejected {field_name}={period!r}")
        raise ValueError(f"{field_name} must look like 'YYYY-MM' or 'YYYY', got {period!r}.")
    return period.strip()


def _suggest_similar(con: duckdb.DuckDBPyConnection, seri_kod: str, limit: int = 8) -> list[str]:
    """On an unknown Seri_Kod, hand back real codes that share a token with
    it. Without this the model burns a whole extra round calling
    lakehouse_schema again just to re-read the catalog it already saw."""
    logger.debug(f"_suggest_similar: looking for codes similar to {seri_kod!r}")
    rows = con.execute(f'SELECT DISTINCT Seri_Kod FROM "{FACT_TABLE}"').fetchall()
    all_codes = [row[0] for row in rows]
    needle_parts = {part for part in re.split(r"[^A-Za-z0-9]+", seri_kod.lower()) if len(part) > 2}
    logger.debug(f"_suggest_similar: needle_parts={needle_parts}, catalog_size={len(all_codes)}")

    scored = []
    for code in all_codes:
        code_parts = {part for part in re.split(r"[^A-Za-z0-9]+", code.lower()) if len(part) > 2}
        overlap = len(needle_parts & code_parts)
        if overlap:
            scored.append((overlap, code))
    scored.sort(key=lambda item: (-item[0], item[1]))
    suggestions = [code for _, code in scored[:limit]] or sorted(all_codes)[:limit]
    logger.info(f"_suggest_similar: {len(suggestions)} suggestion(s) for {seri_kod!r}: {suggestions}")
    return suggestions


def resolve_series(seri_kod: str, bolge: str | None = None,
                   start_period: str | None = None, end_period: str | None = None) -> dict:
    """Read one series out of the gold fact table, ordered by Period.

    Returns {"seri_kod", "seri_adi", "birim", "tip", "bolge", "periods",
    "values", "row_count"}. `periods`/`values` are plain lists so callers
    can feed them straight into the existing array-based analysis code
    without a second conversion step.

    Raises ValueError with concrete suggestions when the code is unknown,
    so the model can correct itself in the same round instead of falling
    back to another schema lookup.
    """
    seri_kod = _validate_seri_kod(seri_kod)
    start_period = _validate_period(start_period, "start_period")
    end_period = _validate_period(end_period, "end_period")
    logger.info(
        f"resolve_series: start seri_kod={seri_kod!r}, bolge={bolge!r}, "
        f"start_period={start_period!r}, end_period={end_period!r}"
    )

    con = _db()
    try:
        # Parameterized throughout: the Seri_Kod comes from the model, so it
        # is data, never SQL text - the string is never concatenated in.
        sql = f'SELECT Period, Deger, Seri_Adi, Birim, Tip, Bolge FROM "{FACT_TABLE}" WHERE Seri_Kod = ?'
        params: list = [seri_kod]
        if bolge is not None:
            sql += " AND Bolge = ?"
            params.append(bolge)
        if start_period is not None:
            sql += " AND Period >= ?"
            params.append(start_period)
        if end_period is not None:
            sql += " AND Period <= ?"
            params.append(end_period)
        sql += " ORDER BY Period"
        logger.debug(f"resolve_series: sql={sql!r}, params={params}")

        frame = con.execute(sql, params).fetchdf()
        logger.info(f"resolve_series: query returned {len(frame)} row(s) for {seri_kod!r}")

        if frame.empty:
            exists = con.execute(
                f'SELECT COUNT(*) FROM "{FACT_TABLE}" WHERE Seri_Kod = ?', [seri_kod]
            ).fetchone()[0]
            logger.warning(f"resolve_series: empty result for {seri_kod!r} (code exists in table: {exists > 0})")
            if exists == 0:
                suggestions = _suggest_similar(con, seri_kod)
                raise ValueError(
                    f"Unknown seri_kod {seri_kod!r}. Did you mean one of: {suggestions}? "
                    f"Call lakehouse_schema for the full catalog."
                )
            raise ValueError(
                f"seri_kod {seri_kod!r} exists but no rows match the given filters "
                f"(bolge={bolge!r}, start_period={start_period!r}, end_period={end_period!r})."
            )

        # Metadata is constant per series; the first row is representative.
        first = frame.iloc[0]
        # NaN values would become the invalid JSON token 'NaN' downstream and
        # would also corrupt every statistic, so they are dropped here with
        # the count logged rather than silently passed on.
        before_drop = len(frame)
        frame = frame[frame["Deger"].notna()]
        dropped = before_drop - len(frame)
        if dropped:
            logger.warning(f"resolve_series: dropped {dropped} row(s) with a null Deger for {seri_kod!r}")

        result = {
            "seri_kod": seri_kod,
            "seri_adi": None if first["Seri_Adi"] is None else str(first["Seri_Adi"]),
            "birim": None if first["Birim"] is None else str(first["Birim"]),
            "tip": None if first["Tip"] is None else str(first["Tip"]),
            "bolge": None if first["Bolge"] is None else str(first["Bolge"]),
            "periods": [str(value) for value in frame["Period"].tolist()],
            "values": [float(value) for value in frame["Deger"].tolist()],
            "row_count": len(frame),
        }
        logger.info(
            f"resolve_series: done seri_kod={seri_kod!r}, row_count={result['row_count']}, "
            f"range={result['periods'][0] if result['periods'] else None}..{result['periods'][-1] if result['periods'] else None}, "
            f"birim={result['birim']!r}, tip={result['tip']!r}"
        )
        return result
    except Exception:
        logger.exception(f"resolve_series: failed for seri_kod={seri_kod!r}")
        raise
    finally:
        con.close()
