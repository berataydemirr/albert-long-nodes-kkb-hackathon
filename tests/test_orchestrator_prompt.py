"""Tests for the injected series catalog and the reasoning_effort control.

Both are latency work, so the risk they carry is silent: a catalog that
stops being injected, or a rejected parameter that takes the agent down,
would show up as "it got slow again" or "it stopped answering" rather
than as a failing assertion somewhere obvious.
"""

import openai
import pytest

from src.agent import orchestrator
from src.agent.series_resolver import DB_PATH, catalog_block

needs_db = pytest.mark.skipif(
    not DB_PATH.exists(), reason="gold database not built; run `python -m src.gold.build_gold`"
)


@needs_db
def test_catalog_block_lists_every_series_once():
    block = catalog_block(refresh=True)

    lines = [line for line in block.splitlines() if line.strip()]
    codes = [line.split("|")[0].strip() for line in lines]
    assert codes, "catalog should not be empty when the database exists"
    assert len(codes) == len(set(codes)), "each series should appear exactly once"


@needs_db
def test_catalog_block_is_small_enough_to_inline():
    """It is paid for on every request, so a catalog that grows without
    bound would quietly undo the saving it exists to create."""
    block = catalog_block(refresh=True)

    assert len(block) < 8_000, f"catalog is {len(block)} chars, too large to inline in the prompt"


@needs_db
def test_system_prompt_carries_the_catalog_and_the_real_column_names():
    prompt = orchestrator._system_prompt()

    assert "gold_fact_long" in prompt
    # Naming only the concept ("dönem") led the model to write
    # WHERE "Dönem" >= ... against a table whose column is Period.
    assert "Period" in prompt
    assert "Seri_Kod" in prompt
    assert "Deger" in prompt


@needs_db
def test_system_prompt_tells_the_model_not_to_look_the_schema_up_again():
    """The whole point of injecting the catalog is removing that round."""
    prompt = orchestrator._system_prompt()

    assert "lakehouse_schema" in prompt


def test_system_prompt_falls_back_cleanly_without_a_database(monkeypatch):
    """A fresh clone has no gold database. The agent must still start and
    answer with the web tools instead of failing at prompt-build time."""
    monkeypatch.setattr(orchestrator, "catalog_block", lambda: "")

    prompt = orchestrator._system_prompt()

    assert prompt == orchestrator.SYSTEM_PROMPT_BASE


class _FakeCompletions:
    """Records the kwargs it was called with and fails the first call the
    way an endpoint that rejects reasoning_effort does."""

    def __init__(self, reject_reasoning: bool):
        self.reject_reasoning = reject_reasoning
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_reasoning and "reasoning_effort" in kwargs:
            raise openai.BadRequestError(
                message="Unexpected reasoning effort",
                response=type("R", (), {"status_code": 400, "headers": {}, "request": None})(),
                body=None,
            )
        return "completion"


class _FakeClient:
    def __init__(self, reject_reasoning: bool = False):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(reject_reasoning)})()


def test_reasoning_effort_is_sent_to_the_model():
    client = _FakeClient()

    orchestrator._call_model(client, [{"role": "user", "content": "x"}], [], reasoning_effort="low")

    assert client.chat.completions.calls[0]["reasoning_effort"] == "low"


def test_a_rejected_reasoning_effort_is_retried_without_it():
    """Swapping in a model that does not accept the parameter must degrade
    to a normal call, not take the agent down."""
    client = _FakeClient(reject_reasoning=True)

    result = orchestrator._call_model(
        client, [{"role": "user", "content": "x"}], [], reasoning_effort="low"
    )

    assert result == "completion"
    assert "reasoning_effort" in client.chat.completions.calls[0]
    assert "reasoning_effort" not in client.chat.completions.calls[1]


def test_no_reasoning_effort_is_sent_when_it_is_switched_off():
    client = _FakeClient()

    orchestrator._call_model(client, [{"role": "user", "content": "x"}], [], reasoning_effort="")

    assert "reasoning_effort" not in client.chat.completions.calls[0]
