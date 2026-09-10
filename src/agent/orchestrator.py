"""Tool-calling conversation loop for the MIA hackathon agent.

Every round is logged (model call, tool calls + results, round number)
so a wrong/stuck answer can be traced back to which tool call caused it,
without re-running the whole thing with print statements added by hand.
"""

import json
import time

from config.settings import MIA_CHAT_MODEL
from src.agent.mia_client import get_client
from src.agent.tools import TOOL_SCHEMAS, execute_tool
from src.common.logging_setup import get_logger

logger = get_logger("orchestrator")

SYSTEM_PROMPT = """Sen KKB Agentic Data Analytics asistanısın.
Önce local Lakehouse tool ile veriyi keşfet ve sorgula; hesapları Python
tool'larına yaptır. Correlation/causality sonucunu kesin nedensellik diye
sunma - tool'un kendi warning'ini oldugu gibi aktar. Her yanıtta kullanılan
kaynak tablosunu, seriyi, dönemi ve gerekiyorsa forward-fill veya birim
belirsizliğini açıkça belirt. Veri yetersizse uydurma; eksikliği söyle.
Yanıtı Türkçe, kısa ama kanıtlı ver."""


def ask(question: str, max_rounds: int = 8, return_trace: bool = False):
    """return_trace=True also returns the round-by-round tool-call trace
    (name + arguments + whether it raised) - used by benchmarks/
    agent_benchmark.py to score tool-selection accuracy and hallucinated
    Seri_Kod values without re-parsing the log file."""
    logger.info(f"ask: question={question!r}, max_rounds={max_rounds}")
    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    trace = []

    for round_no in range(1, max_rounds + 1):
        logger.info(f"ask: round {round_no}/{max_rounds}, calling model={MIA_CHAT_MODEL}")
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=MIA_CHAT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
        )
        elapsed = time.monotonic() - t0
        logger.info(f"ask: round {round_no} model call took {elapsed:.2f}s")

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else message)

        if not message.tool_calls:
            content = message.content or "Model boş yanıt döndürdü."
            logger.info(f"ask: round {round_no} - no tool calls, final answer len={len(content)}")
            return (content, trace) if return_trace else content

        logger.info(f"ask: round {round_no} - {len(message.tool_calls)} tool call(s): {[c.function.name for c in message.tool_calls]}")
        for call in message.tool_calls:
            logger.debug(f"ask: tool_call id={call.id}, name={call.function.name}, arguments={call.function.arguments}")
            entry = {"round": round_no, "name": call.function.name, "arguments": call.function.arguments, "error": None}
            try:
                output = execute_tool(call.function.name, call.function.arguments)
            except Exception as exc:
                logger.warning(f"ask: tool '{call.function.name}' failed: {exc}")
                output = json.dumps({"error": str(exc)}, ensure_ascii=False)
                entry["error"] = str(exc)
            trace.append(entry)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    logger.error(f"ask: exceeded {max_rounds} rounds without a final answer")
    raise RuntimeError(f"Agent tool loop exceeded {max_rounds} rounds")
