"""Measure routing against real agent runs, offline.

router_benchmark.py only proves the regexes match hand-picked prompts. This
replays every question from hard_benchmark.py through the router and
compares the prediction with the tools the model ACTUALLY called in the
last recorded run (benchmarks/hard_benchmark_report.json):

  - confusion matrix: predicted route vs. the route implied by the tools
    that were called (same precedence the router uses).
  - coverage: runs where the model called a tool outside the routed groups.
    Each of these would have cost one request_tools round under tool
    groups - or an answer, if the model did not ask.
  - round budget: runs that used more rounds than the route now allows.
  - schema savings: average share of tool schemas no longer sent.

No model or network calls. Re-run hard_benchmark first for fresh traces.

Run: python -m benchmarks.routing_coverage_benchmark
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from benchmarks.hard_benchmark import BENCHMARK_QUESTIONS
from benchmarks.hard_benchmark import REPORT_PATH as HARD_REPORT_PATH
from src.agent.router import TOOL_GROUPS, Route, route_question
from src.agent.tools import TOOL_SCHEMAS

REPORT_PATH = Path(__file__).with_name("routing_coverage_report.json")

_GROUP_OF = {name: group for group, names in TOOL_GROUPS.items() for name in names}
# Mirrors router._SIGNAL_PRECEDENCE; a URL question cannot be told apart
# from a web search by the tools alone, so both land on web_search here.
_OBSERVED_PRECEDENCE = (
    ("live_fetch", Route.LIVE_FETCH),
    ("imports", Route.IMPORTS),
    ("regional", Route.REGIONAL),
    ("web", Route.WEB_SEARCH),
    ("analysis", Route.ANALYSIS),
)


def observed_route(called_tools: list[str]) -> Route | None:
    """None when nothing was called - there is no evidence either way."""
    if not called_tools:
        return None
    groups = {_GROUP_OF.get(name) for name in called_tools}
    for group, route in _OBSERVED_PRECEDENCE:
        if group in groups:
            return route
    return Route.LOCAL_ANALYTICS


def evaluate(report: dict) -> dict:
    runs = {entry["id"]: entry for entry in report.get("questions", [])}
    total_schemas = len(TOOL_SCHEMAS)
    confusion: Counter = Counter()
    cases = []
    for item in BENCHMARK_QUESTIONS:
        run = runs.get(item["id"])
        if run is None:
            continue
        called = [name for name in run.get("called_tools", []) if name in _GROUP_OF]
        decision = route_question(item["question"], TOOL_SCHEMAS)
        observed = observed_route(called)
        predicted = decision.route.value
        if observed is not None:
            confusion[(predicted, observed.value)] += 1
        missing = sorted(set(called) - set(decision.allowed_tool_names))
        budget = decision.settings.max_rounds if decision.settings else 0
        cases.append({
            "id": item["id"],
            "predicted_route": predicted,
            "observed_route": observed.value if observed else None,
            "tool_groups": list(decision.tool_groups),
            "called_tools": called,
            "missing_tools": missing if decision.route is not Route.REFUSE else [],
            "refused": decision.route is Route.REFUSE,
            "injection_suspected": decision.injection_suspected,
            "rounds_used": run.get("rounds_used", 0),
            "max_rounds": budget,
            "schemas_sent": len(decision.allowed_tool_names),
        })

    routed = [c for c in cases if not c["refused"]]
    with_tools = [c for c in routed if c["called_tools"]]
    covered = [c for c in with_tools if not c["missing_tools"]]
    matched = sum(n for (p, o), n in confusion.items() if p == o)
    return {
        "source_report": str(HARD_REPORT_PATH.name),
        "questions": len(cases),
        "refused": sum(c["refused"] for c in cases),
        "route_agreement": round(matched / max(sum(confusion.values()), 1), 3),
        "tool_coverage": round(len(covered) / max(len(with_tools), 1), 3),
        "runs_missing_a_tool": [
            {"id": c["id"], "missing": c["missing_tools"], "groups": c["tool_groups"]}
            for c in with_tools if c["missing_tools"]
        ],
        "runs_over_round_budget": [
            {"id": c["id"], "rounds_used": c["rounds_used"], "max_rounds": c["max_rounds"]}
            for c in routed if c["rounds_used"] > c["max_rounds"]
        ],
        "avg_schemas_sent": round(sum(c["schemas_sent"] for c in routed) / max(len(routed), 1), 1),
        "total_schemas": total_schemas,
        "confusion_matrix": [
            {"predicted": p, "observed": o, "count": n} for (p, o), n in sorted(confusion.items())
        ],
        "cases": cases,
    }


def main() -> None:
    report = json.loads(HARD_REPORT_PATH.read_text(encoding="utf-8"))
    result = evaluate(report)
    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"questions: {result['questions']} (refused: {result['refused']})")
    print(f"route agreement (predicted == observed): {result['route_agreement']:.1%}")
    print(f"tool coverage (no called tool cut off):  {result['tool_coverage']:.1%}")
    print(f"schemas sent per run: {result['avg_schemas_sent']} of {result['total_schemas']}")
    routes = sorted({row["predicted"] for row in result["confusion_matrix"]}
                    | {row["observed"] for row in result["confusion_matrix"]})
    cell = {(row["predicted"], row["observed"]): row["count"] for row in result["confusion_matrix"]}
    print("\nconfusion matrix (rows = predicted, columns = observed from tool calls)")
    print(f"{'':>16}" + "".join(f"{r[:10]:>12}" for r in routes))
    for p in routes:
        print(f"{p:>16}" + "".join(f"{cell.get((p, o), 0):>12}" for o in routes))
    for miss in result["runs_missing_a_tool"]:
        print(f"  cut off: {miss['id']}: {miss['missing']} (groups: {miss['groups']})")
    for over in result["runs_over_round_budget"]:
        print(f"  over budget: {over['id']}: used {over['rounds_used']} > {over['max_rounds']}")
    print(f"\nreport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
