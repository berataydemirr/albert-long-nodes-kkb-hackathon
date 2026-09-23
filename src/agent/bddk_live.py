"""Read a BDDK monthly bulletin table from the report endpoint used by its UI."""

from config.settings import BDDK_MONTHLY_TABLES
from src.collectors.bddk_monthly import URL, fetch_page
from src.common.http import make_session
from src.common.jqgrid import get_columns_and_rows, get_total_pages, rows_to_records


def fetch_bddk_monthly_table(year: int, month: int, table_id: int, row_filter: str = "") -> dict:
    """Fetch one month's complete table, optionally returning matching rows only."""
    if type(year) is not int or not 2000 <= year <= 2100:
        raise ValueError("year must be an integer from 2000 to 2100")
    if type(month) is not int or not 1 <= month <= 12:
        raise ValueError("month must be an integer from 1 to 12")
    if type(table_id) is not int or table_id not in BDDK_MONTHLY_TABLES:
        raise ValueError("table_id must be a published BDDK monthly table number")
    if not isinstance(row_filter, str) or len(row_filter) > 200:
        raise ValueError("row_filter must be a string of at most 200 characters")

    session = make_session()
    rows = []
    columns = []
    caption = ""
    page = 1
    while True:
        response = fetch_page(session, year, month, table_id, page)
        if not response or response.get("success") is not True:
            raise RuntimeError(f"BDDK report could not be fetched for {year}-{month:02d}, table {table_id}, page {page}.")
        page_columns, page_rows = get_columns_and_rows(response)
        if page == 1:
            columns = page_columns
            caption = response.get("Json", {}).get("caption", "")
        elif page_columns != columns:
            raise ValueError("BDDK report columns changed between pages.")
        rows.extend(page_rows)
        total_pages = get_total_pages(response)
        if total_pages > 20:
            raise ValueError("BDDK report exceeds the 20-page limit.")
        if page >= total_pages:
            break
        page += 1

    records = rows_to_records(columns, rows)
    if row_filter:
        needle = row_filter.casefold()
        records = [r for r in records if needle in str(r.get("Column_2", "")).casefold()]
    return {
        "source_url": URL,
        "report_url": "https://www.bddk.org.tr/BultenAylik/tr/Home/Index",
        "period": f"{year}-{month:02d}",
        "table_id": table_id,
        "caption": caption,
        "unit": BDDK_MONTHLY_TABLES[table_id]["unit"],
        "columns": [c if c.strip() else f"Column_{i}" for i, c in enumerate(columns)],
        "rows": records[:100],
        "row_count": len(records),
        "truncated": len(records) > 100,
        "content_trust": "untrusted_remote_content",
    }
