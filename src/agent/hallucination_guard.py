"""Runtime check that a final answer only cites Seri_Kod values that
actually exist in the lakehouse, instead of relying purely on prompt
wording to stop the model from inventing one.

WHY THIS EXISTS: benchmarks/agent_benchmark.py already MEASURES this exact
failure mode offline (its `hallucination_rate` metric), but a benchmark
only reports the problem after the fact - it runs on demand, in a
terminal, and protects nobody actually using the live app. This module is
the same check (a cited code that is not in the real catalog), moved into
the live answer path in orchestrator.ask(), paired with an actual
remediation instead of only a log line: see orchestrator.ask() for how a
hallucinated code gets one corrective round before the answer ships with
a visible caveat.

Deliberately narrow: this catches an INVENTED IDENTIFIER (a Seri_Kod that
does not exist), which is unambiguous - a code either is or is not in the
catalog. It does not attempt to verify arbitrary numeric claims in prose
(e.g. "faiz %3.2'ye indi"), because there is no reliable, low-false-
positive way to match a rounded/rephrased number in free text back to a
specific tool result. An invented identifier is the one hallucination
shape that can be checked with confidence.
"""

from __future__ import annotations

import re

# Three real code shapes seen in the lakehouse (see
# series_resolver.catalog_block and the SYSTEM_PROMPT_BASE catalog):
#   EVDS-style:  TP.BKR.TRY.18, TP.DK.USD.A.YTL, TP.BISPOLFAIZ.TUR
#   BDDK-style:  tablo_4.konut_kredisi, tablo_15.takipteki_alacaklar_orani
#   TUIK-style:  TUIK_LIFE_EXPECTANCY, TUIK_UNEMPLOYMENT_RATE
# These are matched in prose text, not just SQL/tool arguments - a
# hallucinated code is just as wrong stated in a sentence as it is in a
# WHERE clause, and the answer's own prose is what a user actually reads.
_EVDS_CODE_RE = re.compile(r"\bTP\.[A-Z0-9]+(?:\.[A-Z0-9]+)+\b")
_BDDK_CODE_RE = re.compile(r"\btablo_\d+\.[a-z_]+\b", re.IGNORECASE)
_TUIK_CODE_RE = re.compile(r"\bTUIK_[A-Z][A-Z_]+\b")

_ALL_PATTERNS = (_EVDS_CODE_RE, _BDDK_CODE_RE, _TUIK_CODE_RE)


def extract_cited_series_codes(text: str) -> set[str]:
    """Every code-shaped token found in free text."""
    if not text:
        return set()
    found: set[str] = set()
    for pattern in _ALL_PATTERNS:
        found.update(match.group(0) for match in pattern.finditer(text))
    return found


def find_hallucinated_codes(answer_text: str, valid_codes: set[str]) -> set[str]:
    """Codes the answer cites that are not in the real catalog.

    An empty valid_codes (no database built yet, or the catalog failed to
    load) never flags anything - there is nothing real to check against,
    and flagging every code in that state would be noise, not signal.
    Comparison is case-sensitive on purpose: these codes are copied
    verbatim from the catalog into SQL WHERE clauses elsewhere in the
    system, so a case mismatch is itself a real bug worth surfacing, not
    something to paper over with a case-insensitive match.
    """
    if not valid_codes:
        return set()
    cited = extract_cited_series_codes(answer_text)
    return {code for code in cited if code not in valid_codes}
