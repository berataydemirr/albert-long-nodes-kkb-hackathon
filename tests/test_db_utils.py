"""Tests for connect_with_retry (src/agent/db_utils.py).

The lakehouse merge (imports/TUIK-refresh/BIST-fetch now write into the
same file lakehouse_query reads from) makes a brief lock conflict a
normal event under concurrent use, not a rare edge case - reproduced
live before this was written: a second duckdb.connect() to a file
another connection still holds open raises duckdb.ConnectionException
immediately, it does not block and wait.
"""
import threading
import time

import duckdb
import pytest

from src.agent.db_utils import connect_with_retry


def test_connect_with_retry_succeeds_immediately_when_uncontended(tmp_path):
    db_path = tmp_path / "uncontended.duckdb"
    duckdb.connect(str(db_path)).close()

    con = connect_with_retry(db_path, read_only=True)
    try:
        assert con.execute("SELECT 1").fetchone() == (1,)
    finally:
        con.close()


def test_connect_with_retry_waits_out_a_brief_writer(tmp_path):
    db_path = tmp_path / "contended.duckdb"
    duckdb.connect(str(db_path)).execute("CREATE TABLE t AS SELECT 1 AS x").close()

    writer = duckdb.connect(str(db_path), read_only=False)

    def release_after_delay():
        time.sleep(0.3)
        writer.close()

    threading.Thread(target=release_after_delay).start()

    started = time.monotonic()
    reader = connect_with_retry(db_path, read_only=True, max_attempts=30, base_delay=0.05)
    elapsed = time.monotonic() - started
    try:
        assert reader.execute("SELECT x FROM t").fetchone() == (1,)
    finally:
        reader.close()
    # Must have actually waited (not raced past the writer via luck) and
    # must not have taken anywhere near the full 30-attempt budget.
    assert 0.2 < elapsed < 3.0


def test_connect_with_retry_gives_up_and_raises_after_max_attempts(tmp_path):
    db_path = tmp_path / "permanently_locked.duckdb"
    duckdb.connect(str(db_path)).close()
    writer = duckdb.connect(str(db_path), read_only=False)

    try:
        with pytest.raises(duckdb.ConnectionException):
            connect_with_retry(db_path, read_only=True, max_attempts=3, base_delay=0.01)
    finally:
        writer.close()
