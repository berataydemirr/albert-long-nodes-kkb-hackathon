"""Harness reliability (#17): retries after failure, structural truncation,
argument validation, timeouts, parallel tool calls, the run deadline,
runtime numeric grounding, follow-up notes, the prompt cache, ToolSpec and
the pipeline controller."""

import io
import json
import time

import agent_controller
from src.agent import orchestrator, tools
from src.agent.router import TOOL_GROUPS
from tests.test_loop_guard import _FakeMessage, _FakeToolCall
from tests.test_routing_orchestrator import _ScriptedClient, _patch

LOOKUP = "Konut kredisi faizinin son değeri nedir?"


def _history():
    return {"signatures": set(), "pending": set(), "per_tool": {}}


def _call(name, args, call_id="c1"):
    return _FakeToolCall(name, json.dumps(args), call_id)


def _tool_messages(client):
    return [m for m in client.requests[-1]["messages"] if m.get("role") == "tool"]


# --- loop guard ------------------------------------------------------------

def test_a_failed_call_can_be_retried_with_the_same_arguments():
    history = _history()
    assert orchestrator._loop_guard("web_search", '{"query": "faiz"}', history) is None
    orchestrator._record_call_outcome("web_search", '{"query": "faiz"}', history, ok=False)

    assert orchestrator._loop_guard("web_search", '{"query": "faiz"}', history) is None


def test_a_successful_call_is_still_blocked_on_repeat():
    history = _history()
    orchestrator._loop_guard("web_search", '{"query": "faiz"}', history)
    orchestrator._record_call_outcome("web_search", '{"query": "faiz"}', history, ok=True)

    blocked = orchestrator._loop_guard("web_search", '{"query": "faiz"}', history)

    assert json.loads(blocked)["reason"] == "identical_repeat"


def test_an_error_result_does_not_count_as_success(monkeypatch):
    client = _ScriptedClient([
        _FakeMessage(tool_calls=[_call("lakehouse_query", {"sql": "SELECT 1"})]),
        _FakeMessage(tool_calls=[_call("lakehouse_query", {"sql": "SELECT 1"}, "c2")]),
        _FakeMessage(content="cevap"),
    ])
    _patch(monkeypatch, client, tool_output='{"error": "database is locked"}')

    _, trace = orchestrator.ask(LOOKUP, return_trace=True)

    assert not trace[1].get("blocked")


# --- truncation ------------------------------------------------------------

def test_truncation_keeps_valid_json_and_leading_rows():
    rows = [{"Period": f"2020-{i:04d}", "Deger": i * 1.5, "Seri_Adi": "x" * 80} for i in range(400)]
    output = json.dumps({"columns": ["Period", "Deger", "Seri_Adi"], "rows": rows, "row_count": 400})

    truncated = orchestrator._truncate_tool_output(output, "lakehouse_query")
    data = json.loads(truncated)

    assert len(truncated) <= orchestrator.TOOL_OUTPUT_MAX_CHARS
    assert data["truncated"] is True
    assert data["row_count"] == 400
    assert data["rows_total"] == 400
    assert 0 < data["rows_shown"] == len(data["rows"]) < 400
    assert data["rows"][0] == rows[0]


def test_truncation_of_output_without_a_list_is_still_json():
    output = json.dumps({"text": "a\"b" * 10_000})

    data = json.loads(orchestrator._truncate_tool_output(output, "web_url_extract"))

    assert data["truncated"] is True
    assert data["preview"]


def test_the_trace_keeps_every_row_for_charts(monkeypatch):
    rows = [{"Period": f"p{i}", "Deger": i, "pad": "y" * 100} for i in range(300)]
    raw = json.dumps({"columns": ["Period", "Deger", "pad"], "rows": rows, "row_count": 300})
    client = _ScriptedClient([
        _FakeMessage(tool_calls=[_call("lakehouse_query", {"sql": "SELECT * FROM gold_fact_long"})]),
        _FakeMessage(content="cevap"),
    ])
    _patch(monkeypatch, client, tool_output=raw)

    _, trace = orchestrator.ask(LOOKUP, return_trace=True)

    assert len(trace[0]["output_json"]["rows"]) == 300
    assert json.loads(_tool_messages(client)[0]["content"])["truncated"] is True


# --- argument validation ---------------------------------------------------

def test_bad_arguments_get_an_actionable_error_and_the_tool_does_not_run(monkeypatch):
    client = _ScriptedClient([
        _FakeMessage(tool_calls=[_call("lakehouse_query", {"query": "SELECT 1"})]),
        _FakeMessage(content="cevap"),
    ])
    _patch(monkeypatch, client)
    ran = []
    monkeypatch.setattr(orchestrator, "execute_tool", lambda name, args: ran.append(name) or "{}")

    _, trace = orchestrator.ask(LOOKUP, return_trace=True)

    assert ran == []
    error = json.loads(trace[0]["output_preview"])
    assert error["error_type"] == "invalid_arguments"
    assert error["retryable"] is True
    assert any("'sql' is required" in p for p in error["problems"])


def test_validate_tool_arguments_checks_types_and_enums():
    _, problems = tools.validate_tool_arguments("compare_regions", '{"top_n": "5", "order": "up"}')

    assert "field 'top_n' must be integer, got str" in problems
    assert any("'order' must be one of" in p for p in problems)
    assert tools.validate_tool_arguments("compare_regions", '{"top_n": 5}')[1] == []
    assert tools.validate_tool_arguments("lakehouse_query", "not json")[1]


# --- timeouts, parallelism, deadline ----------------------------------------

def test_a_hanging_tool_times_out_and_the_run_continues(monkeypatch):
    client = _ScriptedClient([
        _FakeMessage(tool_calls=[_call("web_search", {"query": "faiz"})]),
        _FakeMessage(content="cevap"),
    ])
    _patch(monkeypatch, client)
    monkeypatch.setattr(orchestrator, "execute_tool", lambda name, args: time.sleep(1.0) or "{}")
    spec = tools.TOOL_SPECS["web_search"]
    monkeypatch.setitem(tools.TOOL_SPECS, "web_search", tools.ToolSpec(**{**spec.__dict__, "timeout_sec": 0.05}))

    answer, trace = orchestrator.ask("Bugünkü politika faizi için internette ara.", return_trace=True)

    assert answer == "cevap"
    assert json.loads(trace[0]["output_preview"])["error_type"] == "timeout"


def test_read_only_tools_in_one_round_run_in_parallel(monkeypatch):
    calls = [_call("lakehouse_query", {"sql": f"SELECT {i}"}, f"c{i}") for i in range(3)]
    client = _ScriptedClient([_FakeMessage(tool_calls=calls), _FakeMessage(content="cevap")])
    _patch(monkeypatch, client)
    monkeypatch.setattr(orchestrator, "execute_tool", lambda name, args: time.sleep(0.3) or '{"ok": true}')

    started = time.monotonic()
    _, trace = orchestrator.ask(LOOKUP, return_trace=True)

    assert time.monotonic() - started < 0.8
    assert [entry["arguments"] for entry in trace] == [c.function.arguments for c in calls]


def test_duckdb_writers_are_serialized(monkeypatch):
    active, peak = [0], [0]

    def fake(name, args):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.1)
        active[0] -= 1
        return '{"ok": true}'

    calls = [_call("fetch_bist_stock", {"ticker": t}, t) for t in ("AKBNK", "GARAN")]
    client = _ScriptedClient([_FakeMessage(tool_calls=calls), _FakeMessage(content="cevap")])
    _patch(monkeypatch, client)
    monkeypatch.setattr(orchestrator, "execute_tool", fake)

    orchestrator.ask("AKBNK ve GARAN hisselerini getir", return_trace=True)

    assert peak[0] == 1


def test_the_run_deadline_forces_an_answer(monkeypatch):
    client = _ScriptedClient([_FakeMessage(content="süre doldu, kısa cevap")])
    _patch(monkeypatch, client)

    answer = orchestrator.ask(LOOKUP, deadline_sec=0)

    assert answer == "süre doldu, kısa cevap"
    assert len(client.requests) == 1
    assert "tools" not in client.requests[0]
    assert "Süre limitine" in client.requests[0]["messages"][-1]["content"]


# --- runtime numeric grounding ---------------------------------------------

def test_an_ungrounded_number_triggers_one_repair_round(monkeypatch):
    client = _ScriptedClient([
        _FakeMessage(tool_calls=[_call("correlation_analysis", {"x_seri_kod": "A", "y_seri_kod": "B"})]),
        _FakeMessage(content="Korelasyon r = 0.55 bulundu."),
        _FakeMessage(content="Korelasyon r = 0.97 bulundu."),
    ])
    _patch(monkeypatch, client, tool_output='{"pearson_correlation": 0.97}')
    monkeypatch.setattr(orchestrator, "AGENT_GROUNDING_REPAIR", True)

    answer, _, timing = orchestrator.ask("Faiz ile kredi arasında korelasyon var mı?",
                                         return_trace=True, return_timing=True)

    assert answer == "Korelasyon r = 0.97 bulundu."
    assert "0.55" in client.requests[2]["messages"][-1]["content"]
    assert "tools" not in client.requests[2]
    assert timing["summary"]["grounding"]["repaired"] is True


def test_a_grounded_answer_is_left_alone(monkeypatch):
    client = _ScriptedClient([
        _FakeMessage(tool_calls=[_call("correlation_analysis", {"x_seri_kod": "A", "y_seri_kod": "B"})]),
        _FakeMessage(content="Korelasyon r = 0.97 bulundu."),
    ])
    _patch(monkeypatch, client, tool_output='{"pearson_correlation": 0.97}')

    answer = orchestrator.ask("Faiz ile kredi arasında korelasyon var mı?")

    assert answer == "Korelasyon r = 0.97 bulundu."
    assert len(client.requests) == 2


# --- follow-up context -----------------------------------------------------

def test_summarize_turn_names_tools_series_and_tables():
    trace = [
        {"name": "lakehouse_query", "error": None,
         "arguments": json.dumps({"sql": "SELECT * FROM gold_fact_long WHERE Seri_Kod IN ('KONUT_FAIZ', 'TUFE')"})},
        {"name": "compare_regions", "error": None, "arguments": json.dumps({"metric": "npl", "period": "2025-12"})},
        {"name": "web_search", "error": "boom", "arguments": json.dumps({"query": "x"})},
    ]

    note = orchestrator.summarize_turn(trace)

    assert "lakehouse_query, compare_regions" in note
    assert "KONUT_FAIZ, TUFE" in note
    assert "gold_fact_long" in note
    assert "metric=npl" in note and "period=2025-12" in note
    assert "web_search" not in note


def test_history_note_reaches_the_model():
    messages = orchestrator._history_messages([
        {"question": "Konut faizi?", "answer": "%3,2", "note": "araçlar: lakehouse_query; seriler: KONUT_FAIZ"},
    ])

    assert "seriler: KONUT_FAIZ" in messages[1]["content"]


def test_history_digest_from_older_turns_is_carried_as_a_system_note():
    """Turns beyond HISTORY_MAX_TURNS are gone from `history` itself, but
    their key points survive as `digest` - this is what keeps a long
    session from forgetting what it was working on entirely."""
    messages = orchestrator._history_messages(
        [{"question": "Konut faizi?", "answer": "%3,2"}],
        digest="araçlar: lakehouse_query; seriler: KONUT_FAIZ, TUFE",
    )

    assert messages[0]["role"] == "system"
    assert "KONUT_FAIZ, TUFE" in messages[0]["content"]
    # The digest is background, not a live turn - it never displaces the
    # recent (question, answer) pair that follows it.
    assert messages[1]["content"] == "Konut faizi?"
    assert messages[2]["content"] == "%3,2"


def test_history_digest_alone_with_no_recent_turns_still_produces_a_message():
    messages = orchestrator._history_messages(None, digest="seriler: KONUT_FAIZ")

    assert len(messages) == 1
    assert messages[0]["role"] == "system"


# --- system prompt cache ---------------------------------------------------

def test_other_tables_block_is_rebuilt_only_when_the_database_changes(monkeypatch, tmp_path):
    db = tmp_path / "lakehouse.duckdb"
    db.write_text("v1")
    builds = []
    monkeypatch.setattr(orchestrator, "LAKEHOUSE_DB_PATH", db)
    monkeypatch.setattr(orchestrator, "_other_tables_cache", {"key": None, "block": ""})
    monkeypatch.setattr(orchestrator, "_build_other_tables_block", lambda: builds.append(1) or f"block{len(builds)}")

    assert orchestrator._other_tables_block() == "block1"
    assert orchestrator._other_tables_block() == "block1"
    db.write_text("v2 longer")

    assert orchestrator._other_tables_block() == "block2"
    assert len(builds) == 2


# --- ToolSpec --------------------------------------------------------------

def test_tool_specs_are_the_single_source():
    assert set(tools.TOOL_FUNCTIONS) == set(tools.TOOL_SPECS)
    assert {name for group in TOOL_GROUPS.values() for name in group} == set(tools.TOOL_SPECS)
    assert orchestrator._tool_call_limit("anomaly_analysis") == tools.TOOL_SPECS["anomaly_analysis"].call_limit
    assert not tools.TOOL_SPECS["fetch_bist_stock"].parallel_safe
    assert not tools.TOOL_SPECS["refresh_tuik_series"].parallel_safe


def test_lakehouse_schema_description_no_longer_contradicts_the_prompt():
    description = tools.TOOL_SPECS["lakehouse_schema"].schema["function"]["description"]

    assert "before writing SQL" not in description
    assert "already in the system prompt" in description


# --- pipeline controller ---------------------------------------------------

class _FakeProcess:
    def __init__(self, output, returncode):
        self.stdout = io.StringIO(output)
        self.returncode = returncode

    def wait(self):
        return self.returncode


def test_a_warning_does_not_retry_a_successful_step(monkeypatch):
    runs = []

    def popen(*args, **kwargs):
        runs.append(1)
        return _FakeProcess("2026 - [x] - WARNING - slow page\n", 0)

    monkeypatch.setattr(agent_controller.subprocess, "Popen", popen)

    assert agent_controller.run_module("src.collectors.evds", []) is True
    assert len(runs) == 1


def test_a_non_zero_exit_is_retried(monkeypatch):
    runs = []

    def popen(*args, **kwargs):
        runs.append(1)
        return _FakeProcess("", 1)

    monkeypatch.setattr(agent_controller.subprocess, "Popen", popen)
    monkeypatch.setattr(agent_controller.time, "sleep", lambda s: None)

    assert agent_controller.run_module("src.collectors.evds", []) is False
    assert len(runs) == agent_controller.MAX_RETRIES


def test_steps_depending_on_a_failed_step_are_skipped():
    results = dict(agent_controller.run_pipeline(
        agent_controller.build_pipeline("test"),
        runner=lambda module, args: module != "src.processing.evds_silver",
    ))

    assert results["src.processing.evds_silver"] == "FAIL"
    assert results["src.processing.tuik_silver"] == "OK"
    assert results["src.gold.scan_cumulative"] == "SKIP"
    assert results["src.gold.build_gold"] == "SKIP"


def test_a_scaled_and_rounded_value_counts_as_grounded():
    from src.agent.grounding import _numeric_grounding

    trace = [{"error": None, "output_json": {"avg_change_in_window": 429410414.34782606,
                                             "avg_change_outside_window": 327542600.04761904}}]

    result = _numeric_grounding("Ortalama artış 429,4 milyar TL, dışında 327,5 milyar TL.", trace)

    assert result["ungrounded"] == []


def test_progress_events_name_the_running_tool(monkeypatch):
    client = _ScriptedClient([
        _FakeMessage(tool_calls=[_call("lakehouse_query", {"sql": "SELECT 1"})]),
        _FakeMessage(content="cevap"),
    ])
    _patch(monkeypatch, client)
    events = []

    orchestrator.ask(LOOKUP, on_event=events.append)

    kinds = [e["kind"] for e in events]
    assert kinds[:3] == ["route", "llm", "tools_start"]
    assert events[2]["names"] == ["lakehouse_query"]
    assert events[3]["kind"] == "tool_done" and events[3]["name"] == "lakehouse_query"


def test_repair_is_off_by_default_and_only_reported(monkeypatch):
    client = _ScriptedClient([
        _FakeMessage(tool_calls=[_call("correlation_analysis", {"x_seri_kod": "A", "y_seri_kod": "B"})]),
        _FakeMessage(content="Korelasyon r = 0.55 bulundu."),
    ])
    _patch(monkeypatch, client, tool_output='{"pearson_correlation": 0.97}')
    monkeypatch.setattr(orchestrator, "AGENT_GROUNDING_REPAIR", False)

    answer, _, timing = orchestrator.ask("Faiz ile kredi arasında korelasyon var mı?",
                                         return_trace=True, return_timing=True)

    assert answer == "Korelasyon r = 0.55 bulundu."
    assert len(client.requests) == 2
    assert timing["summary"]["grounding"]["ungrounded"] == [0.55]
