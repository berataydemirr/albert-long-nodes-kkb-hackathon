"""Small, configuration-driven client for the MIA OpenAI-compatible API."""

from openai import OpenAI

from config.settings import MIA_API_KEY, MIA_BASE_URL


def get_client() -> OpenAI:
    if not MIA_API_KEY:
        raise RuntimeError("MIA_API_KEY is missing. Add it to .env; never hard-code it.")
    return OpenAI(api_key=MIA_API_KEY, base_url=MIA_BASE_URL)
