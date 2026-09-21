"""Regression tests for the URL-shape and outbound-header fixes.

Both come from live failures: a GitHub page link reported as a dead 404
instead of being rewritten to the raw host, and a self-identifying bot
User-Agent being served a WAF error page by public data hosts.
"""

import pytest

from src.agent.web_url import BROWSER_HEADERS, sanitize_url


@pytest.mark.parametrize("broken", [
    "https://github.com/vega/vega-datasets/blob/main/data/sp500.csv",
    "https://github.com/vega/vega-datasets/raw/main/data/sp500.csv",
    "https://www.github.com/vega/vega-datasets/blob/main/data/sp500.csv",
    "https://media.githubusercontent.com/media/vega/vega-datasets/main/data/sp500.csv",
])
def test_github_page_links_are_rewritten_to_the_raw_file_host(broken):
    fixed, note = sanitize_url(broken)

    assert fixed == "https://raw.githubusercontent.com/vega/vega-datasets/main/data/sp500.csv"
    assert note, "a silent rewrite would hide which address was actually fetched"


@pytest.mark.parametrize("already_fine", [
    "https://raw.githubusercontent.com/vega/vega-datasets/main/data/sp500.csv",
    "https://evds2.tcmb.gov.tr/service/series.csv",
    "https://api.worldbank.org/v2/country/TUR/indicator/FP.CPI.TOTL.ZG?format=json",
])
def test_urls_that_already_serve_bytes_are_left_untouched(already_fine):
    fixed, note = sanitize_url(already_fine)

    assert fixed == already_fine
    assert note is None


def test_a_non_string_url_does_not_raise_before_validation_can_report_it():
    fixed, note = sanitize_url(None)

    assert fixed is None and note is None


def test_outbound_requests_are_browser_shaped_not_bot_shaped():
    """A UA containing the project's own name and a repo URL is what the
    WAF was matching on. The Accept header matters too: some hosts
    content-negotiate a CSV endpoint into HTML without it."""
    assert "Mozilla/5.0" in BROWSER_HEADERS["User-Agent"]
    assert "KKB-Agentic" not in BROWSER_HEADERS["User-Agent"]
    assert "csv" in BROWSER_HEADERS["Accept"]
