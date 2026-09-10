"""OpenAI-compatible client for the MIA/Kloudeks endpoint. Logs every
request (model, message count, tool count) so a bad/slow call is
traceable without re-running the whole agent loop."""

from openai import OpenAI

from config.settings import MIA_API_KEY, MIA_BASE_URL
from src.common.logging_setup import get_logger

logger = get_logger("mia_client")

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    if not MIA_API_KEY:
        logger.error("get_client: MIA_API_KEY missing from environment")
        raise RuntimeError("MIA_API_KEY is missing. Add it to .env; never hard-code it.")
    logger.info(f"get_client: creating OpenAI client, base_url={MIA_BASE_URL}")
    _client = OpenAI(api_key=MIA_API_KEY, base_url=MIA_BASE_URL)
    return _client
