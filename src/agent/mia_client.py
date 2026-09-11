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
REQUEST_TIMEOUT_SEC = 60
# The OpenAI SDK's own retry only fires on connection-level failures; MIA
# is a shared hackathon endpoint that can also return 429/5xx under load,
# so orchestrator.py adds its own retry on top - keep the SDK's low so a
# transient failure surfaces to that layer quickly instead of silently
# eating tens of seconds inside the SDK first.
SDK_MAX_RETRIES = 1


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
