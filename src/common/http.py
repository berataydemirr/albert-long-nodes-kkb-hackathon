"""
Shared requests.Session factory + WAF-friendly delay helper, used when
talking to sources like BDDK/TCMB.
"""
import time
import random
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import WAF_SLEEP_MIN_SEC, WAF_SLEEP_MAX_SEC

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
}


def make_session(extra_headers: dict | None = None) -> requests.Session:
    session = requests.Session()

    # Not a security shortcut: bddk.org.tr genuinely serves a broken/incomplete
    # TLS certificate chain (independently confirmed via WebFetch and plain
    # requests - both hit the same certificate error). This is a known
    # server-side misconfiguration on BDDK's end, not a MITM risk we are
    # choosing to ignore. Before reverting verify=False "for security",
    # re-confirm the certificate chain is still broken.
    session.verify = False

    headers = dict(DEFAULT_HEADERS)
    if extra_headers:
        headers.update(extra_headers)
    session.headers.update(headers)

    retry = Retry(
        total=3,
        backoff_factor=2.0,
        status_forcelist=[500, 502, 503, 504],  # 403 deliberately excluded - don't auto-retry a WAF block
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def waf_sleep() -> None:
    """Random delay between requests so we don't trip the site's bot-protection WAF."""
    time.sleep(random.uniform(WAF_SLEEP_MIN_SEC, WAF_SLEEP_MAX_SEC))
