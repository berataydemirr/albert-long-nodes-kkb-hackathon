"""Shared DuckDB connection helper with a lock-contention retry.

lakehouse.duckdb now holds both the curated gold tables and anything a
user imports or the agent fetches live at runtime (see data_import.py),
so a short write (an import, a TUIK refresh) can land in the same window
as one of the many read-only connections the live-serving agent opens
and closes per tool call. DuckDB allows either N concurrent read-only
connections OR one exclusive read-write connection to a file, never a
mix - confirmed live: opening a second connection while another is still
open raises duckdb.ConnectionException ("Can't open a connection to same
database file with a different configuration than existing
connections"). This is an ordinary race under concurrent use, not a rare
edge case, so it is retried rather than left to surface as a 500: the
losing side is almost always waiting on a write that finishes in well
under a second, not a real deadlock.
"""
import time

import duckdb

from src.common.logging_setup import get_logger

logger = get_logger("db_utils")

MAX_ATTEMPTS = 20
BASE_DELAY_SEC = 0.1


def connect_with_retry(path, *, read_only: bool, max_attempts: int = MAX_ATTEMPTS,
                       base_delay: float = BASE_DELAY_SEC) -> duckdb.DuckDBPyConnection:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return duckdb.connect(str(path), read_only=read_only)
        except duckdb.ConnectionException as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            logger.warning(
                f"connect_with_retry: {path} busy (read_only={read_only}), "
                f"attempt {attempt}/{max_attempts}: {exc}"
            )
            time.sleep(base_delay)
    logger.error(f"connect_with_retry: gave up on {path} (read_only={read_only}) after {max_attempts} attempts")
    raise last_exc
