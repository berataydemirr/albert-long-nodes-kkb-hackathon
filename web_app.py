"""Local browser UI for exercising routing and function calling."""

import re
import secrets
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import DEMO_AUTH_PASS, DEMO_AUTH_USER, USER_UPLOADS_DIR
from src.agent.data_import import (
    MAX_FILE_BYTES,
    SUPPORTED_EXTENSIONS,
    import_dataset,
    list_sheets,
)
from src.agent.pdf_import import import_pdf_table, list_pdf_tables
from src.agent.orchestrator import ask, summarize_turn
from src.agent.router import route_question
from src.agent.charts import derive_charts
from src.agent.tools import TOOL_SCHEMAS
from src.common.logging_setup import get_logger

logger = get_logger("web_app")

app = FastAPI(title="KKB Agentic Analytics", docs_url=None, redoc_url=None)


_LOCAL_HOSTNAMES = ("127.0.0.1", "localhost", "[::1]")


@app.middleware("http")
async def demo_basic_auth(request: Request, call_next):
    """HTTP Basic Auth gate - only active when DEMO_AUTH_USER/DEMO_AUTH_PASS
    are both set (e.g. in .env for a public-tunnel demo). Off by default so
    normal local dev needs no credentials. secrets.compare_digest avoids a
    timing side-channel on the comparison.

    Skipped for a request whose Host header names 127.0.0.1/localhost - the
    developer working directly against the local server should never see a
    login prompt. This can't be done by checking request.client.host
    instead: cloudflared itself connects to the origin from 127.0.0.1 too
    (it's a local reverse proxy), so the TCP peer address is identical for
    "me, in my own browser" and "a visitor coming through the public
    tunnel" - only the Host header (which cloudflared forwards unchanged
    from the original public request) tells them apart."""
    if not (DEMO_AUTH_USER and DEMO_AUTH_PASS):
        return await call_next(request)

    host_header = request.headers.get("host", "").split(":")[0]
    if host_header in _LOCAL_HOSTNAMES:
        return await call_next(request)

    auth = request.headers.get("authorization", "")
    if auth.startswith("Basic "):
        import base64
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            user, _, pw = decoded.partition(":")
        except Exception:
            user, pw = "", ""
        if secrets.compare_digest(user, DEMO_AUTH_USER) and secrets.compare_digest(pw, DEMO_AUTH_PASS):
            return await call_next(request)

    logger.warning(f"demo_basic_auth: rejected request from {request.client.host if request.client else '?'} to {request.url.path}")
    return Response(status_code=401, headers={"WWW-Authenticate": "Basic realm=\"KKB Agent Demo\""})


# Design-canvas UI (KKB Agent Lab), wired to the same /api endpoints below.
# Served from this app (not a separate origin) so its fetch() calls need no CORS.
app.mount("/lab", StaticFiles(directory="static/design", html=True), name="lab")


class HistoryTurn(BaseModel):
    """One prior (question, answer) pair from the SAME browser tab's chat.
    The server holds no session state between requests - the client
    resends its own transcript, and orchestrator.ask()/_history_messages
    further truncates and caps it before it ever reaches the model. The
    field-level max_length here is a request-size guard, not the memory
    budget itself."""
    question: str = Field(min_length=1, max_length=2_000)
    answer: str = Field(min_length=1, max_length=8_000)
    # The turn_note this server returned with that answer: tools, series
    # and tables used, so a follow-up question keeps that context.
    note: str | None = Field(default=None, max_length=400)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    # Oldest first. Capped at list level too so a request can't smuggle in
    # an unbounded transcript; orchestrator.ask() only ever looks at the
    # last few entries anyway (see HISTORY_MAX_TURNS).
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)
    # Client-generated id; /api/progress/{request_id} streams this run's events.
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")


# request_id -> {"events": [...], "updated": monotonic}. In-memory and
# per-process: progress is a UI nicety, not state worth persisting.
_PROGRESS: dict[str, dict] = {}
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_TTL_SEC = 600
_PROGRESS_MAX_EVENTS = 200


def _progress_recorder(request_id: str):
    with _PROGRESS_LOCK:
        now = time.monotonic()
        for stale in [k for k, v in _PROGRESS.items() if now - v["updated"] > _PROGRESS_TTL_SEC]:
            del _PROGRESS[stale]
        _PROGRESS[request_id] = {"events": [], "updated": now, "done": False}

    def record(event: dict) -> None:
        with _PROGRESS_LOCK:
            entry = _PROGRESS.get(request_id)
            if entry is not None and len(entry["events"]) < _PROGRESS_MAX_EVENTS:
                entry["events"].append(event)
                entry["updated"] = time.monotonic()

    return record


@app.get("/api/progress/{request_id}")
def progress(request_id: str, after: int = 0) -> dict:
    """Events of a running /api/ask since index `after`. Polled by the UI
    while it waits, so it can say which tool is running right now."""
    with _PROGRESS_LOCK:
        entry = _PROGRESS.get(request_id)
        if entry is None:
            return {"events": [], "next": after, "done": False}
        events = entry["events"][max(after, 0):]
        return {"events": events, "next": max(after, 0) + len(events), "done": entry["done"]}


@app.get("/")
def index_redirect() -> RedirectResponse:
    return RedirectResponse(url="/lab/")


_CLASSIC_UI_PATH = Path(__file__).parent / "static" / "classic.html"


@app.get("/classic", response_class=HTMLResponse)
def classic_ui() -> str:
    return _CLASSIC_UI_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/tools")
def list_tools() -> dict:
    """Expose the model-visible registry for local MVP inspection."""
    return {
        "count": len(TOOL_SCHEMAS),
        "tools": [
            {
                "name": schema["function"]["name"],
                "description": schema["function"].get("description", ""),
                "parameters": schema["function"].get("parameters", {}),
            }
            for schema in TOOL_SCHEMAS
        ],
    }


_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_]")
_PDF_EXTENSION = ".pdf"
_ALL_UPLOAD_EXTENSIONS = SUPPORTED_EXTENSIONS + (_PDF_EXTENSION,)


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), dataset_name: str | None = Form(None),
                         sheet: str | None = Form(None), page: int = Form(0),
                         table_index: int = Form(0)) -> dict:
    """Save an uploaded CSV/Excel/PDF file and import it into the
    lakehouse (see data_import.py / pdf_import.py for why it's the same
    lakehouse.duckdb the curated gold tables live in, not a separate
    file).

    Runs schema detection + import synchronously and returns the result,
    rather than handing back a job id to poll - these are small files
    (25 MB cap) meant to be usable in the same conversation, not a batch
    pipeline. A malformed or oversized file is a 422, not a 500: the
    server did its job correctly by refusing it. page/table_index only
    apply to a PDF (which page, which of possibly several tables on it -
    both default to the first, like sheet defaults to the first Excel
    sheet); a caller that gets a poor extraction can inspect
    list_pdf_tables' output and retry with a different pair.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALL_UPLOAD_EXTENSIONS:
        logger.warning(f"upload_dataset: rejected {file.filename!r}, unsupported extension {suffix!r}")
        raise HTTPException(status_code=422, detail=f"Unsupported file type {suffix!r}. Supported: {_ALL_UPLOAD_EXTENSIONS}.")

    body = await file.read()
    if len(body) > MAX_FILE_BYTES:
        logger.warning(f"upload_dataset: rejected {file.filename!r}, {len(body)} bytes over the {MAX_FILE_BYTES} cap")
        raise HTTPException(status_code=422, detail=f"File is {len(body) / 1_048_576:.1f} MB; the limit is {MAX_FILE_BYTES / 1_048_576:.0f} MB.")

    # Saved under a random id, not the original filename: two uploads
    # named the same thing must not collide on disk, and the original
    # name is preserved separately (source_file in the registry) for
    # display.
    USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = USER_UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    saved_path.write_bytes(body)
    logger.info(f"upload_dataset: saved {file.filename!r} ({len(body)} bytes) -> {saved_path}")

    default_name = _SAFE_STEM_RE.sub("_", Path(file.filename or "dataset").stem.lower())[:60] or "dataset"
    if default_name[0].isdigit():
        default_name = f"d_{default_name}"
    name = dataset_name.strip() if dataset_name and dataset_name.strip() else default_name

    try:
        if suffix == _PDF_EXTENSION:
            preview = list_pdf_tables(saved_path)
            result = import_pdf_table(saved_path, name, page=page, table_index=table_index, source_label=file.filename)
            result["pdf_tables_found"] = preview["tables"]
        else:
            sheets = list_sheets(saved_path)
            result = import_dataset(saved_path, name, sheet=sheet, source_label=file.filename)
            result["sheets"] = sheets
        logger.info(f"upload_dataset: imported as {name!r}, {result['row_count']} rows")
        return result
    except ValueError as exc:
        logger.warning(f"upload_dataset: import failed for {file.filename!r}: {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error(f"upload_dataset: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/ask")
def answer_question(request: AskRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Soru boş olamaz.")
    history = [turn.model_dump(exclude_none=True) for turn in request.history]
    try:
        decision = route_question(question, TOOL_SCHEMAS)
        progress_kwargs = {"on_event": _progress_recorder(request.request_id)} if request.request_id else {}
        try:
            answer, trace, timing = ask(question, history=history, return_trace=True, return_timing=True,
                                        **progress_kwargs)
        finally:
            if request.request_id:
                with _PROGRESS_LOCK:
                    if request.request_id in _PROGRESS:
                        _PROGRESS[request.request_id]["done"] = True

        charts = derive_charts(trace)
        # output_json carries the full tool result so charts can be built
        # from every row; the debug panel only ever shows output_preview,
        # so it is dropped here rather than sent (a 200-row query result
        # would otherwise be serialized twice into the same response).
        lean_trace = [{k: v for k, v in entry.items() if k != "output_json"} for entry in trace]

        return {
            "answer": answer,
            # Charts derived from the tool results, not generated by the
            # model - see src/agent/charts.py for why.
            "charts": charts,
            # Debug timing: a flat event timeline (model calls + tool calls,
            # each with a start offset and a duration) plus rolled-up totals.
            # Rendered as the timeline strip in the Function calls sidebar.
            "timing": timing,
            "route": {
                "name": decision.route.value,
                "reason": decision.reason,
                "available": decision.available,
                "allowed_tools": list(decision.allowed_tool_names),
                "tool_groups": list(decision.tool_groups),
                "injection_suspected": decision.injection_suspected,
                "reasoning_effort": decision.settings.reasoning_effort if decision.settings else None,
                "max_rounds": decision.settings.max_rounds if decision.settings else None,
            },
            "trace": lean_trace,
            # Sent back with this turn's history entry (see HistoryTurn.note).
            "turn_note": summarize_turn(trace),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent yanıt üretemedi. Sunucu günlüklerini kontrol edin.") from exc
