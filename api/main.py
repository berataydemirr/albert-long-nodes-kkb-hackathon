"""FastAPI backend for the Next.js frontend (Faz E, professional deploy
path). Python stays the only backend language per the hackathon's own
rule ("Python ana gelistirme dili olacak") - this just exposes the
existing src/agent and src/report logic as JSON over HTTP instead of
rendering it directly in Streamlit.

Run (dev):    uvicorn api.main:app --reload --port 8000
Run (prod):   uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs:         http://localhost:8000/docs (FastAPI auto-generated)
"""
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import MIA_API_KEY
from api.gold_queries import get_dashboard_snapshot, DB_PATH, query_series_from_gold
from src.report.citations import extract_citations
from src.common.logging_setup import get_logger

logger = get_logger("api")

app = FastAPI(title="KKB Agentic Data Analytics API", version="1.0.0")

# Next.js dev server (localhost:3000) and the production build served
# from the same origin both need to reach this API. Kept as an explicit
# list, not "*" - this API can execute SQL (guardrailed, but still) and
# call a paid MIA endpoint, so wildcard CORS is the wrong default even
# for a hackathon demo.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    t0 = time.monotonic()
    logger.info(f"request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"request: {request.method} {request.url.path} raised unhandled")
        raise
    elapsed = time.monotonic() - t0
    logger.info(f"request: {request.method} {request.url.path} -> {response.status_code} ({elapsed:.2f}s)")
    return response


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[dict]
    tool_failures: list[dict]
    citation_warning: str | None
    charts: list[dict]
    elapsed_sec: float


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gold_db_exists": DB_PATH.exists(),
        "mia_configured": bool(MIA_API_KEY),
    }


@app.get("/api/dashboard")
def dashboard():
    if not DB_PATH.exists():
        logger.error("dashboard: gold database missing")
        raise HTTPException(status_code=503, detail="Gold veritabani yok. `python -m src.gold.build_gold` calistirin.")
    try:
        return get_dashboard_snapshot()
    except Exception as exc:
        logger.exception("dashboard: snapshot failed")
        raise HTTPException(status_code=500, detail=f"Dashboard verisi hazirlanamadi: {exc}")


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if not MIA_API_KEY:
        raise HTTPException(status_code=503, detail="MIA_API_KEY .env dosyasinda tanimli degil.")

    from src.agent.orchestrator import ask as agent_ask  # imported lazily: touches MIA_API_KEY at import time via mia_client

    t0 = time.monotonic()
    try:
        answer, trace = agent_ask(payload.question, return_trace=True)
    except Exception as exc:
        logger.exception("ask: agent_ask failed")
        raise HTTPException(status_code=502, detail=f"Agent hata verdi: {exc}")
    elapsed = time.monotonic() - t0

    citation_result = extract_citations(trace)
    charts = []
    for c in citation_result["citations"]:
        if c.get("tool") != "lakehouse_query":
            continue
        for seri_kod in c.get("seri_kodlar", []):
            df = query_series_from_gold(seri_kod)
            if df.empty:
                continue
            charts.append({
                "seri_kod": seri_kod,
                "periods": df["Period"].tolist(),
                "values": [None if pd_isna(v) else v for v in df["Deger"].tolist()],
            })

    return AskResponse(
        answer=answer,
        grounded=citation_result["grounded"],
        citations=citation_result["citations"],
        tool_failures=citation_result["tool_failures"],
        citation_warning=citation_result["warning"],
        charts=charts,
        elapsed_sec=round(elapsed, 2),
    )


def pd_isna(v):
    import math
    return v is None or (isinstance(v, float) and math.isnan(v))
