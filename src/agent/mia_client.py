"""OpenAI-compatible client for the MIA/Kloudeks endpoint. Logs every
request (model, message count, tool count) so a bad/slow call is
traceable without re-running the whole agent loop."""

from openai import OpenAI

from config.settings import MIA_API_KEY, MIA_BASE_URL
from src.common.logging_setup import get_logger

logger = get_logger("mia_client")

_client = None

# No response in this time = something is stuck (network, an overloaded
# shared model host) - fail fast and let orchestrator.py's retry loop
# handle it instead of hanging the whole agent silently.
#
# Raised from 60 to 240: found live that a real question needing heavy
# reasoning PLUS a real web_search round-trip (now that SearXNG is
# actually running, instead of failing instantly) can legitimately take
# longer than 60s for a single model completion - a real user decision:
# correctness over speed, explicitly fine with a query taking up to ~10
# minutes end to end rather than cutting a real tool call short and
# risking a rushed or unsupported answer.
REQUEST_TIMEOUT_SEC = 240
# 0, not 1: this used to be 1 on the assumption that the SDK's own retry
# "only fires on connection-level failures" - that assumption was wrong,
# and finding out cost a live 1163s run. Read the SDK's own request()
# source (openai._base_client.SyncAPIClient.request): it catches
# httpx's timeout exceptions directly (`except timeout_exceptions()`) and
# retries internally BEFORE ever raising APITimeoutError to our code -
# max_retries=1 means every one of orchestrator.py's own outer attempts
# could silently become 2 real HTTP attempts, EACH allowed to run the
# full configured timeout. That is exactly what was observed live: a
# round whose deadline-aware timeout should have capped it at ~563s
# instead ran ~1127s (2.00x) before _call_model's own except block ever
# saw the exception - the SDK had quietly retried once underneath it,
# doubling the wait with no log line to show it happened.
# orchestrator._call_model already owns retry/backoff/deadline for this
# endpoint end to end; a second, invisible retry layer underneath it
# only defeats that budget, it never improves it. 0 makes the SDK raise
# on the first real failure, so _call_model's own accounting is accurate.
SDK_MAX_RETRIES = 0


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    if not MIA_API_KEY:
        logger.error("get_client: MIA_API_KEY missing from environment")
        raise RuntimeError("MIA_API_KEY is missing. Add it to .env; never hard-code it.")
    logger.info(f"get_client: creating OpenAI client, base_url={MIA_BASE_URL}, timeout={REQUEST_TIMEOUT_SEC}s")
    _client = OpenAI(api_key=MIA_API_KEY, base_url=MIA_BASE_URL, timeout=REQUEST_TIMEOUT_SEC, max_retries=SDK_MAX_RETRIES)
    return _client
