"""Durable prompt history and immutable tool evidence, addressed by opaque IDs.

SQLite keeps the audit log separate from mutable Gold tables. Only result
indexes enter model context; DuckDB calculates against the saved rows.
IDs are bearer capabilities, not user-provided filenames or SQL identifiers.
"""
import json
import re
import secrets
import sqlite3
import numpy as np
from datetime import datetime, timezone
from contextlib import contextmanager

import duckdb
import pandas as pd

from config.settings import GOLD_DIR
from src.agent.sql_guard import run_guarded_select

STORE_PATH = GOLD_DIR / "evidence.sqlite3"
_ID = re.compile(r"^[a-f0-9]{32}$")


@contextmanager
def _connect():
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(STORE_PATH, timeout=15)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT, payload TEXT)")
        with con:
            yield con
    finally:
        con.close()


def _save(kind, payload):
    identifier = secrets.token_hex(16)
    with _connect() as con:
        con.execute("INSERT INTO records VALUES (?, ?, ?)",
                    (identifier, kind, json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False)))
    return identifier


def _read(identifier, kind):
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
        raise ValueError("Geçersiz kayıt kimliği.")
    with _connect() as con:
        row = con.execute("SELECT payload FROM records WHERE id=? AND kind=?", (identifier, kind)).fetchone()
    if not row:
        raise ValueError("Kayıt bulunamadı; önceki veriyi yeniden yükleyin.")
    return json.loads(row[0])


def snapshot_result(name, arguments, payload):
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    # Every successful structured result is retained before prompt truncation.
    identifier = _save("evidence", {"tool": name, "arguments": arguments, "result": payload,
                                  "created_at": datetime.now(timezone.utc).isoformat()})
    return {**payload, "evidence_id": identifier}


def save_turn(question, answer, note, trace):
    artifacts = []
    for entry in trace:
        result = entry.get("output_json") or {}
        if not isinstance(result, dict) or not result.get("evidence_id"):
            continue
        artifacts.append({"evidence_id": result["evidence_id"], "tool": entry["name"],
                          "arguments": entry.get("arguments"),
                          "columns": result.get("columns"), "row_count": result.get("row_count"),
                          "dataset_name": result.get("dataset_name"), "source_url": result.get("source_url")})
    payload = {"question": question, "answer": answer, "note": note, "artifacts": artifacts}
    return _save("turn", payload), artifacts


def load_turns(turn_ids):
    return [_read(identifier, "turn") for identifier in dict.fromkeys(turn_ids)]


def recall_turn(turn_id: str) -> dict:
    return _read(turn_id, "turn")


def evidence_query(evidence_ids: list[str], sql: str) -> dict:
    """Query original tabular results as data_1, data_2, ... without retyping."""
    if not 1 <= len(evidence_ids) <= 6:
        raise ValueError("1–6 veri kaydı seçin.")
    con = duckdb.connect(":memory:", config={"enable_external_access": "false"})
    try:
        for index, identifier in enumerate(evidence_ids, 1):
            payload = _read(identifier, "evidence")["result"]
            if payload.get("truncated") or payload.get("truncated_to"):
                raise ValueError("Kayıt kısaltılmış; tam dönem verisini kaynak tablodan sorgulayın.")
            rows = payload.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError("Bu kayıt tablo içermiyor; dataset_name tablosunu lakehouse_query ile sorgulayın.")
            frame = pd.DataFrame(rows, columns=payload.get("columns") or None)
            con.register(f"data_{index}", frame)
        result = run_guarded_select(con, sql, label="evidence_query")
        return {**result, "evidence_ids": evidence_ids, "source": "immutable_tool_results"}
    finally:
        con.close()


def evidence_correlation(x_id: str, y_id: str, x_period: str = "Period",
                         y_period: str = "Period", x_value: str = "Value",
                         y_value: str = "Value", y_transform: str = "none") -> dict:
    """Checked monthly Pearson correlation over exactly the saved observations."""
    if y_transform not in {"none", "monthly_pct_change"}:
        raise ValueError("Geçersiz dönüşüm.")

    def frame(identifier, period, value):
        payload = _read(identifier, "evidence")["result"]
        if payload.get("truncated") or payload.get("truncated_to"):
            raise ValueError("Kısaltılmış veriyle korelasyon hesaplanamaz.")
        data = pd.DataFrame(payload.get("rows", []), columns=payload.get("columns") or None)
        if period not in data or value not in data:
            raise ValueError("Dönem/değer sütunu bulunamadı; önce kayıt şemasını sorgulayın.")
        data = data[[period, value]].copy()
        data.columns = ["period", "value"]
        if not data.period.astype(str).str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
            raise ValueError("Aylık korelasyon için YYYY-MM dönemleri gerekli.")
        data["value"] = pd.to_numeric(data.value, errors="raise")
        data = data.drop_duplicates()
        if data.period.duplicated().any():
            raise ValueError("Aynı ay için çelişkili değerler var; kaynağı veya seriyi daraltın.")
        if np.isinf(data.value.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("Sonsuz sayısal değer kabul edilmez.")
        return data.sort_values("period")

    left, right = frame(x_id, x_period, x_value), frame(y_id, y_period, y_value)
    if y_transform == "monthly_pct_change":
        ordinals = pd.PeriodIndex(right.period, freq="M").asi8
        consecutive = pd.Series(ordinals, index=right.index).diff().eq(1)
        previous = right.value.shift(1)
        right["value"] = (100 * (right.value / previous - 1)).where(consecutive & previous.gt(0))
    x_valid, y_valid = left.dropna(), right.dropna()
    con = duckdb.connect(":memory:", config={"enable_external_access": "false"})
    try:
        con.register("x", x_valid)
        con.register("y", y_valid)
        joined = con.execute("SELECT period, x.value AS x, y.value AS y FROM x JOIN y USING (period) ORDER BY period").fetchdf()
        if len(joined) < 3:
            raise ValueError("En az 3 ortak ve geçerli ay gerekli.")
        con.register("aligned", joined)
        coefficient = con.execute("SELECT corr(x, y) FROM aligned").fetchone()[0]
        if coefficient is not None and not np.isfinite(coefficient):
            coefficient = None
        shared = set(joined.period)
        return {"pearson_correlation": coefficient, "shared_periods": len(joined),
                "first_period": joined.period.iloc[0], "last_period": joined.period.iloc[-1],
                "excluded_x_periods": sorted(set(left.period) - shared),
                "excluded_y_periods": sorted(set(right.period) - shared),
                "x_evidence_id": x_id, "y_evidence_id": y_id, "y_transform": y_transform,
                "columns": list(joined.columns), "rows": joined.to_dict("records"), "row_count": len(joined),
                "warning": "Pearson r −1 ile +1 arasındadır. Korelasyon nedensellik değildir; birimler ve yıllıklandırma ayrıca değerlendirilmelidir."}
    finally:
        con.close()
