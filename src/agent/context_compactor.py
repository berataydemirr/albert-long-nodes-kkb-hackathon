"""Keep the tool-loop message history from growing without bound.

WHY THIS EXISTS:
Chat completions are stateless, so every round resends the ENTIRE message
list. A tool result produced in round 1 is paid for again in rounds 2..8.
With a 12,000-character cap per tool output and up to 8 rounds, a long
chain can carry tens of thousands of tokens of stale tool output into
every later request - the cost grows roughly quadratically in the number
of rounds, which shows up as "the agent gets slower the longer it thinks".

WHAT IT DOES:
Once the estimated history size crosses a budget, the *content* of older
tool messages is replaced with a short digest that keeps the shape of the
result (row counts, column names, first rows) and states plainly that the
full output was trimmed. Recent rounds are always left untouched, because
those are the ones the model is actively reasoning over.

WHAT IT MUST NEVER DO:
  - delete a message. A 'tool' message must stay paired with the
    tool_call_id in the preceding assistant message, or the next API call
    fails with a 400. Compaction rewrites content in place, never removes.
  - touch the system prompt or the user's question. Those are the task
    definition; losing them changes what is being answered.

Every decision is logged (sizes before/after, which messages were
rewritten, why compaction did or did not run) so an unexpectedly short
answer can be traced to whether compaction removed something it should
have kept.
"""

import json
import os

from src.agent.untrusted import unwrap_untrusted, wrap_untrusted
from src.common.logging_setup import get_logger

logger = get_logger("context_compactor")

# Chars-per-token is model-specific; 4 is the usual rough estimate for
# English/Turkish mixed text and is only used to decide WHEN to compact,
# never reported as a real token count.
CHARS_PER_TOKEN = 4

# Compaction starts once the estimated history exceeds this.
# Calibrated, not guessed: benchmarks/context_cost_benchmark.py measures a
# realistic 8-round chain at ~5,100 estimated tokens, so a 6,000 budget
# never fired at all. 3,000 engages partway through a long chain while
# still leaving the recent rounds (which keep_recent protects) untouched.
DEFAULT_BUDGET_TOKENS = 3_000

# Tool outputs from the most recent N rounds are never touched - the model
# is still actively using them.
DEFAULT_KEEP_RECENT_TOOL_MESSAGES = 3

# How much of a digested payload's head is preserved as a sample.
DIGEST_SAMPLE_CHARS = 400

COMPACTED_MARKER = "[COMPACTED]"


def compact_context(messages: list[dict], tool_schemas: list[dict]) -> tuple[list[dict], dict]:
    """Harness-owned compact action at 80% of configured input capacity.

    Capacity is a deployment setting, not an assertion about the provider's
    model limit. Include schemas and reserve output space in the estimate.

    The defaults used to be 16,384/2,048, which left almost nothing for the
    conversation: the system prompt alone is ~4,700 estimated tokens and the
    22 tool schemas another ~4,400, so 80% of the 14,336 usable tokens was
    already spent before the first tool result arrived and compaction fired
    on nearly every multi-round question. The MIA endpoint reports
    max_model_len=262,144 for Qwen3.8-27B (its own 400 names the number), so
    the old value was ~16x under what the server allows. 65,536 is still a
    deliberate fraction of that - prefill time grows with the prompt, and a
    hackathon endpoint is shared - but it leaves ~40k tokens of real room
    for tool output, which is more than the longest measured 8-round chain.
    """
    capacity = int(os.getenv("AGENT_CONTEXT_TOKENS", "65536"))
    reserve = int(os.getenv("AGENT_OUTPUT_RESERVE_TOKENS", "8192"))
    if capacity <= reserve or reserve < 0:
        raise ValueError("Context capacity must exceed the nonnegative output reserve.")
    overhead = len(json.dumps(tool_schemas, ensure_ascii=False)) // CHARS_PER_TOKEN
    usable = capacity - reserve
    budget = max(0, int(usable * 0.8) - overhead)
    compacted, stats = compact_messages(messages, budget_tokens=budget)
    stats.update({"action": "compact", "threshold": 0.8, "capacity_tokens": capacity,
                  "reserved_output_tokens": reserve, "schema_est_tokens": overhead,
                  "utilization_before": (stats["before_tokens"] + overhead) / usable,
                  "utilization_after": (stats["after_tokens"] + overhead) / usable,
                  "threshold_exceeded": stats["before_tokens"] + overhead > usable * 0.8})
    return compacted, stats


def estimate_tokens(messages: list[dict]) -> int:
    """Rough size of the history, used only as a compaction trigger."""
    total_chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif content is not None:
            total_chars += len(json.dumps(content, ensure_ascii=False, default=str))
        # Tool call arguments live outside 'content' but are still sent.
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if function:
                total_chars += len(str(function.get("arguments", "")))
    estimate = total_chars // CHARS_PER_TOKEN
    logger.debug(f"estimate_tokens: total_chars={total_chars}, estimated_tokens={estimate}")
    return estimate


def _digest_tool_output(content: str) -> str:
    """Shrink one tool result to a short, still-informative stand-in.

    Structured results keep their shape (row count, columns, a sample row)
    because that is what the model actually reasons over later - "I already
    have 66 rows of this series" is enough to stop it re-querying, while
    the full 66 rows are not needed once it has moved on.
    """
    original_len = len(content)
    # A fenced third-party result keeps its fence in the digest; the sample
    # row it carries is still untrusted text.
    content, untrusted_source = unwrap_untrusted(content)
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        payload = None

    if isinstance(payload, dict):
        summary = {key: value for key, value in payload.items()
                   if key in ("row_count", "columns", "truncated", "error", "shared_periods", "source", "seri_kod", "evidence_id", "dataset_name", "source_url", "start_period", "end_period", "missing_periods", "aggregation", "pearson_correlation", "x_source", "y_source")}
        rows = payload.get("rows")
        if isinstance(rows, list) and rows:
            summary["first_row"] = rows[0]
            summary["rows_omitted"] = len(rows)
        digest = json.dumps(summary, ensure_ascii=False, default=str)
        logger.debug(f"_digest_tool_output: structured digest, keys={list(summary.keys())}")
    else:
        digest = content[:DIGEST_SAMPLE_CHARS]
        logger.debug("_digest_tool_output: non-JSON payload, kept head sample")
    if untrusted_source:
        digest = wrap_untrusted(digest, untrusted_source)

    result = (
        f"{COMPACTED_MARKER} Full output ({original_len} chars) was trimmed to save context. "
        f"Summary: {digest} "
        f"If you need the full result again, call the tool again with a narrower filter."
    )
    logger.debug(f"_digest_tool_output: {original_len} chars -> {len(result)} chars")
    return result


def compact_messages(messages: list[dict], budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                     keep_recent: int = DEFAULT_KEEP_RECENT_TOOL_MESSAGES) -> tuple[list[dict], dict]:
    """Return (possibly compacted copy of messages, stats).

    Pure with respect to the caller's list: the input is never mutated, so
    a caller that keeps its own reference sees no surprise rewrites.
    """
    before_tokens = estimate_tokens(messages)
    stats = {
        "ran": False,
        "before_tokens": before_tokens,
        "after_tokens": before_tokens,
        "budget_tokens": budget_tokens,
        "compacted_messages": 0,
        "saved_tokens": 0,
    }

    if before_tokens <= budget_tokens:
        logger.debug(f"compact_messages: skipped, {before_tokens} tokens <= budget {budget_tokens}")
        return messages, stats

    # Index every tool message; only the older ones are eligible.
    tool_indexes = [i for i, message in enumerate(messages) if message.get("role") == "tool"]
    eligible = tool_indexes[:-keep_recent] if keep_recent > 0 else tool_indexes
    logger.info(
        f"compact_messages: START over budget - estimated={before_tokens} > budget={budget_tokens}, "
        f"tool_messages={len(tool_indexes)}, eligible={len(eligible)} (keeping {keep_recent} most recent)"
    )

    if not eligible:
        logger.warning(
            "compact_messages: over budget but nothing eligible to compact "
            f"(tool_messages={len(tool_indexes)}, keep_recent={keep_recent}) - history left untouched"
        )
        return messages, stats

    compacted = list(messages)
    rewritten = 0
    for index in eligible:
        message = compacted[index]
        content = message.get("content")
        if not isinstance(content, str) or content.startswith(COMPACTED_MARKER):
            logger.debug(f"compact_messages: index={index} skipped (already compacted or non-text)")
            continue
        digest = _digest_tool_output(content)
        if len(digest) >= len(content):
            # Digesting a short result can make it longer; leave it alone.
            logger.debug(f"compact_messages: index={index} skipped (digest {len(digest)} >= original {len(content)})")
            continue
        compacted[index] = {**message, "content": digest}
        rewritten += 1
        logger.info(
            f"compact_messages: rewrote index={index} tool_call_id={message.get('tool_call_id')} "
            f"{len(content)} chars -> {len(digest)} chars"
        )

    after_tokens = estimate_tokens(compacted)
    stats.update({
        "ran": rewritten > 0,
        "after_tokens": after_tokens,
        "compacted_messages": rewritten,
        "saved_tokens": max(before_tokens - after_tokens, 0),
    })
    logger.info(
        f"compact_messages: DONE rewrote={rewritten} message(s), "
        f"{before_tokens} -> {after_tokens} estimated tokens (saved {stats['saved_tokens']})"
    )
    return compacted, stats
