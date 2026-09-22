"""Local browser UI for exercising routing and function calling."""

import json
import queue
import re
import hashlib
import secrets
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
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
from src.agent.evidence import load_turns, save_turn, snapshot_result
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
    Legacy clients resend a transcript; clients with turn_ids use the
    server's saved history instead. The orchestrator bounds model context. The
    field-level max_length here is a request-size guard, not the memory
    budget itself."""
    question: str = Field(min_length=1, max_length=2_000)
    answer: str = Field(min_length=1, max_length=8_000)
    # The turn_note this server returned with that answer: tools, series
    # and tables used, so a follow-up question keeps that context.
    note: str | None = Field(default=None, max_length=400)


class AskRequest(BaseModel):
    turn_ids: list[str] = Field(default_factory=list, max_length=100)
    question: str = Field(min_length=1, max_length=2_000)
    # Oldest first. Capped at list level too so a request can't smuggle in
    # an unbounded transcript; orchestrator.ask() only ever looks at the
    # last few entries anyway (see HISTORY_MAX_TURNS).
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)
    # Rolled-up notes from turns older than `history` covers (see
    # historyForRequest() in static/design/index.html and
    # HISTORY_DIGEST_MAX_CHARS in orchestrator.py) - key points, not prose,
    # so a session that has run long still carries forward what it was
    # working on.
    history_digest: str | None = Field(default=None, max_length=800)
    # Client-generated id; /api/progress/{request_id} streams this run's events.
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")
    # True: /api/ask answers as NDJSON - {"type": "delta", ...} lines with
    # the model's text as it is generated, then one {"type": "result", ...}
    # (or {"type": "error", ...}) line. False keeps the single-JSON body.
    stream: bool = False


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
    content_id = hashlib.sha256(body + f"{sheet}:{page}:{table_index}".encode()).hexdigest()[:16]
    name = dataset_name.strip() if dataset_name and dataset_name.strip() else default_name[:40] + "_" + content_id

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
        result = snapshot_result("upload_dataset", "{}", result)
        note = f"Yüklenen tablo: {name}; satır: {result['row_count']}"
        turn_id, artifacts = save_turn(f"Dosya yükle: {file.filename}", note, note, [
            {"name": "upload_dataset", "arguments": "{}", "output_json": result}])
        result.update({"turn_id": turn_id, "artifacts": artifacts})
        return result
    except ValueError as exc:
        logger.warning(f"upload_dataset: import failed for {file.filename!r}: {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error(f"upload_dataset: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# A single silent multi-minute POST is exactly what a proxy/tunnel's
# idle-connection timeout kills - confirmed live: the cloudflared quick
# tunnel was truncating /api/ask around the 4-minute mark, which the
# client then surfaced as "SyntaxError: not valid JSON" (see the comment
# above requestJsonWithRetry's catch in static/design/index.html). Bytes
# on the wire, not total duration, are what idle timeouts key off, so
# answer_question streams a newline heartbeat every _ASK_HEARTBEAT_SEC
# while the real work runs on a background thread, keeping the connection
# classified as active for as long as the agent needs (comfortably past
# the ~10 minute worst case in mia_client.REQUEST_TIMEOUT_SEC's own
# comment). The HTTP status is always 200 because it is already sent by
# the time a failure could be known; failures ride in the JSON body's
# "error" field instead, and the client checks that field itself.
_ASK_HEARTBEAT_SEC = 15


def _run_ask(request: AskRequest, question: str, on_delta=None) -> dict:
    history = [turn.model_dump(exclude_none=True) for turn in request.history]
    evidence_kwargs = {}
    if request.turn_ids:
        try:
            saved = load_turns(request.turn_ids)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        history = [{k: turn[k] for k in ("question", "answer", "note")} for turn in saved]
        evidence_kwargs["evidence_context"] = [
            {"turn_id": identifier, "question": turn["question"][:300],
             "artifacts": turn["artifacts"] if i >= len(saved) - 3 else "recall_turn ile getir"}
            for i, (identifier, turn) in enumerate(zip(dict.fromkeys(request.turn_ids), saved))
        ]
    try:
        decision = route_question(question, TOOL_SCHEMAS)
        record = _progress_recorder(request.request_id) if request.request_id else None
        progress_kwargs = {}
        if on_delta is not None:
            # Deltas go straight down the response stream; they would blow
            # through _PROGRESS_MAX_EVENTS and the poller only needs stages.
            def on_event(event: dict) -> None:
                if event.get("kind") == "delta":
                    on_delta(event)
                elif record is not None:
                    record(event)
            progress_kwargs = {"on_event": on_event, "stream": True}
        elif record is not None:
            progress_kwargs = {"on_event": record}
        try:
            answer, trace, timing = ask(question, history=history, history_digest=request.history_digest,
                                        return_trace=True, return_timing=True, **progress_kwargs, **evidence_kwargs)
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
        note = summarize_turn(trace)
        turn_id, artifacts = save_turn(question, answer, note, trace)

        return {
            "turn_id": turn_id,
            "artifacts": artifacts,
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
            "turn_note": note,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Agent yanıt üretemedi. Sunucu günlüklerini kontrol edin.") from exc


@app.post("/api/ask")
def answer_question(request: AskRequest) -> StreamingResponse:
    # Cheap, purely local validation stays a real synchronous HTTPException -
    # no background thread or streaming needed for a request that fails
    # before any agent work would even start.
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Soru boş olamaz.")

    box: dict = {}
    deltas: queue.Queue = queue.Queue()

    def worker():
        try:
            box["payload"] = _run_ask(request, question, on_delta=deltas.put if request.stream else None)
        except HTTPException as exc:
            box["error"] = exc.detail
        except Exception as exc:
            logger.error(f"answer_question: unhandled error in worker thread: {exc}", exc_info=True)
            box["error"] = "Agent yanıt üretemedi. Sunucu günlüklerini kontrol edin."
        finally:
            deltas.put(None)  # wakes generate() the moment the run ends

    def generate():
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            try:
                item = deltas.get(timeout=_ASK_HEARTBEAT_SEC)
            except queue.Empty:
                yield b"\n"  # heartbeat - keeps the connection from looking idle
                continue
            if item is None:
                break
            # Coalesce whatever else is already queued into one line per
            # round, so a fast token stream isn't one tiny write per token.
            batch, done = [item], False
            while True:
                try:
                    nxt = deltas.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    done = True
                    break
                batch.append(nxt)
            merged: list[dict] = []
            for ev in batch:
                if merged and merged[-1]["round"] == ev.get("round"):
                    merged[-1]["text"] += ev.get("text", "")
                else:
                    merged.append({"type": "delta", "round": ev.get("round"), "text": ev.get("text", "")})
            for line in merged:
                yield json.dumps(line, ensure_ascii=False).encode("utf-8") + b"\n"
            if done:
                break
        thread.join()
        if request.stream:
            final = {"type": "error", "error": box["error"]} if "error" in box else {"type": "result", **box["payload"]}
            yield json.dumps(final, default=str).encode("utf-8") + b"\n"
            return
        # Valid JSON permits insignificant whitespace before the value, so
        # the heartbeats already written do not corrupt the body - the
        # client's response.json() parses this exactly as if no heartbeats
        # had been sent at all.
        yield json.dumps({"error": box["error"]} if "error" in box else box["payload"]).encode("utf-8")

    media_type = "application/x-ndjson" if request.stream else "application/json"
    return StreamingResponse(generate(), media_type=media_type)
