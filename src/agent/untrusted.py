"""Explicit markers around tool output that carries third-party text.

The router only ever sees the user's question. The realistic injection path
is text a tool brings back - a web page, a PDF table, a cell in an uploaded
Excel file - so that text is fenced before it enters the model context, and
the system prompt tells the model that nothing inside the fence is an
instruction. The fence is data-level only: sql_guard and the read-only
connection remain the actual security boundary.
"""

from __future__ import annotations

import re

BEGIN_MARKER = "<<<UNTRUSTED_CONTENT"
END_MARKER = "<<<END_UNTRUSTED_CONTENT>>>"

# Tools whose output is (or can be) text written by someone other than the
# team: remote pages/documents, and user-uploaded or live-fetched datasets.
UNTRUSTED_TOOLS = frozenset({
    "recall_turn",
    "evidence_correlation",
    "evidence_query",
    "import_url_dataset",
    "search_evds_series",
    "fetch_evds_series",
    "web_search",
    "web_url_extract",
    "list_imported_datasets",
    "query_imported_dataset",
})

_TABLE_REF_RE = re.compile(r'\b(?:from|join)\s+"?([A-Za-z_][\w]*)"?', re.IGNORECASE)
_WRAPPED_RE = re.compile(
    r"\A" + re.escape(BEGIN_MARKER) + r" source=(?P<source>[\w.-]+)>>>\n(?P<body>.*)\n"
    + re.escape(END_MARKER) + r"\Z",
    re.DOTALL,
)


def is_untrusted_call(name: str, arguments: dict | None) -> bool:
    """lakehouse_query is untrusted only when it reads a non-gold table,
    i.e. an uploaded or live-fetched dataset living in the same database."""
    if name in UNTRUSTED_TOOLS:
        return True
    if name == "lakehouse_query" and isinstance(arguments, dict):
        sql = arguments.get("sql")
        if isinstance(sql, str):
            return any(not t.lower().startswith("gold_") for t in _TABLE_REF_RE.findall(sql))
    return False


def wrap_untrusted(content: str, source: str) -> str:
    # A payload that spells out the end marker must not be able to close the
    # fence early and continue as if it were trusted text.
    body = content.replace(END_MARKER, "[marker removed]").replace(BEGIN_MARKER, "[marker removed]")
    return f"{BEGIN_MARKER} source={source}>>>\n{body}\n{END_MARKER}"


def unwrap_untrusted(content: str) -> tuple[str, str | None]:
    """Return (body, source) for wrapped content, or (content, None)."""
    match = _WRAPPED_RE.match(content)
    if not match:
        return content, None
    return match.group("body"), match.group("source")
