"""
Shared parsing helpers for BDDK's jqGrid-style JSON responses.

Root-cause fix (isolated here, applied once): the real API response
looks like {"success": true, "Json": {"colNames": [...], "colModels":
[...], "data": {"page": "1", "total": "1", "records": "1", "rows":
[...]}, "caption": ..., "uyari": ..., "duyurular": []}}. Rows and the
total page count live under "Json.data", not directly under "Json".
The original code read these as "Json.rows"/"Json.total"; since
dict.get(key, default) silently falls back instead of raising, this
never crashed - it just ALWAYS returned 0 rows and 1 page. Verified
live by comparing against the raw JSON: table 1 / 2021-01 actually
has 62 rows.
"""
from typing import Any


def get_columns_and_rows(json_data: dict) -> tuple[list, list]:
    """Extracts column names and raw (jqGrid-format) rows from an API response."""
    grid = json_data.get("Json", {})
    col_names = grid.get("colNames", [])
    rows = grid.get("data", {}).get("rows", [])
    return col_names, rows


def get_total_pages(json_data: dict) -> int:
    """Extracts the total page count (data.total arrives as a numeric STRING, e.g. "1")."""
    total = json_data.get("Json", {}).get("data", {}).get("total")
    try:
        return int(total) if total not in (None, "") else 1
    except (TypeError, ValueError):
        return 1


def rows_to_records(col_names: list, rows: list) -> list:
    """Converts jqGrid rows ({"id": .., "cell": [...]}) into a list of flat dicts."""
    records = []
    for row in rows:
        cell_data = row.get("cell", [])
        record: dict[str, Any] = {}
        for i, col in enumerate(col_names):
            col_name = col if col.strip() != "" else f"Column_{i}"
            record[col_name] = cell_data[i] if i < len(cell_data) else None
        records.append(record)
    return records
