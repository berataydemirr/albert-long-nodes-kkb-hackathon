"""Numeric grounding: does a number the model wrote trace back to a number a
tool returned?

Shared by benchmarks/hard_benchmark.py (reported as a rate) and
orchestrator.ask() (one repair round at runtime).
"""

import math
import re


# --- numeric grounding: does a number the model WROTE actually trace back
# to a number a tool RETURNED? -----------------------------------------------
#
# Every other check here validates which tool was called and whether its
# name/argument looked hallucinated - none of them look at whether the
# prose answer's own numbers are real. The right tool can be called,
# return a real number, and the model can still misreport it in the final
# sentence - a different, more damaging failure mode that tool_selection_ok
# and the series/dataset-name checks are structurally blind to. This is a
# heuristic, not a proof (legitimate derived arithmetic - a percentage
# change, a difference of two values - will not appear as a literal
# number in the tool's own JSON), so it is reported as a rate and every
# "ungrounded" number is kept in the per-question result for manual
# review, not used as a hard pass/fail gate.
# Handles Turkish model prose, in priority order:
#  1. Thousands-grouped ('1.234.567' or '1.234.567,89') - TWO OR MORE
#     exactly-3-digit '.' groups, optional ',' decimal tail. Unambiguous:
#     a real decimal number never has more than one '.'. A SINGLE group
#     ('9.268') is deliberately excluded from this fast path and falls
#     through to the ambiguous case below instead - found live, this
#     regex originally used '+' (one or more) and silently swallowed
#     '9.268' as "unambiguously 9268" before the ambiguous-candidate
#     logic ever got a chance to also offer the 9.268 decimal reading.
#  2. A lone comma-decimal ('9,3') - unambiguous, comma is the Turkish
#     decimal separator.
#  3. A lone period number with exactly 3 trailing digits ('66.583' or
#     '9.268') - GENUINELY ambiguous, not resolvable from the digits
#     alone: it can be the thousands-grouped integer 66583 (a raw count,
#     Turkish style), or a literal decimal like 9.268 (e.g. "9.268
#     milyar TL", a value already scaled to billions with 3 decimals of
#     precision) - both patterns were found live in real answers. Rather
#     than guess, both readings are kept and grounding accepts either.
#  4. A lone period number with 1, 2, or 4+ trailing digits ('0.97') -
#     Turkish thousands grouping is always exactly 3 digits, so this
#     cannot be that; read as an ordinary decimal point.
_TR_GROUPED_RE = re.compile(r"-?\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?")
_LONE_NUMBER_RE = re.compile(r"-?\d+[.,]\d+")


def _decimal_numbers_in_text(text: str) -> list[list[float]]:
    """One entry per number FOUND in the text; each entry is the list of
    its plausible numeric readings (almost always one, two for the
    genuinely ambiguous case above)."""
    out: list[list[float]] = []
    masked = list(text)
    for m in _TR_GROUPED_RE.finditer(text):
        raw = m.group(0).replace(".", "").replace(",", ".")
        try:
            out.append([round(float(raw), 2)])
        except ValueError:
            continue
        for i in range(m.start(), m.end()):
            masked[i] = " "

    for m in _LONE_NUMBER_RE.finditer("".join(masked)):
        raw = m.group(0)
        if "," in raw:
            try:
                out.append([round(float(raw.replace(",", ".")), 2)])
            except ValueError:
                pass
            continue
        candidates: list[float] = []
        try:
            candidates.append(round(float(raw), 2))
        except ValueError:
            pass
        int_part, _, frac_part = raw.rpartition(".")
        if int_part.lstrip("-") and len(frac_part) == 3:
            try:
                candidates.append(round(float(int_part + frac_part), 2))
            except ValueError:
                pass
        if candidates:
            out.append(candidates)
    return out


def _flatten_numbers(obj) -> set[float]:
    found: set[float] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            found |= _flatten_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _flatten_numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        # A crash found live: a tool result can legitimately contain a
        # NaN (fetch_bist_stock, before its own fix - Yahoo Finance
        # publishing a day's Volume before its price data settles) -
        # round(nan) with no ndigits raises ValueError (float.__round__
        # can't return an int for NaN), taking down the whole benchmark
        # run partway through. A NaN is not a real returned number
        # either way, so it is simply excluded here rather than crashing
        # or being treated as grounding evidence for anything.
        value = float(obj)
        if not math.isnan(value) and not math.isinf(value):
            found.add(round(value, 2))
    return found


def _numeric_grounding(answer: str, trace: list[dict]) -> dict:
    answer_numbers = _decimal_numbers_in_text(answer)
    if not answer_numbers:
        return {"checked": 0, "grounded": 0, "ungrounded": []}

    tool_numbers: set[float] = set()
    for entry in trace:
        if entry.get("error") is not None:
            continue
        tool_numbers |= _flatten_numbers(entry.get("output_json"))

    # Legitimate derived arithmetic a model does ALL the time, none of it
    # a fabrication - every case here was found live, flagged as
    # "ungrounded", and manually confirmed to be correct math the first
    # version of this check just didn't recognize:
    #   - percent <-> decimal (a value / 100 or * 100): "%32,21 = 0,3221"
    #   - squaring (r -> r^2): "r=0.72, r^2=0.51"
    #   - a ratio between two tool numbers (a currency conversion):
    #     "69.55 TL / 46.56 kur = 1.49 USD"
    #   - a value repeated across two rows/periods (dedup collapses it to
    #     one set member, so a-b never gets tried against itself) makes a
    #     correct "0 degisim" claim - 0.0 is always allowed regardless,
    #     since a claimed zero difference carries essentially no
    #     fabrication risk on its own.
    # Capped list size keeps the pairwise loop cheap even for a large
    # result set.
    derived: set[float] = {0.0}
    tn_list = list(tool_numbers)[:60]
    for x in tn_list:
        # Same 2dp rounding as the answer-side extraction (and as
        # _flatten_numbers itself) - a mismatched precision here would
        # silently defeat the whole comparison (0.3221 vs the answer's
        # rounded 0.32 would never be equal).
        derived.add(round(x / 100, 2))
        derived.add(round(x * 100, 2))
        derived.add(round(x ** 2, 2))
        # A Turkish "Bin TL" (thousand-TRY) raw value is routinely
        # reported in prose as milyon/milyar/trilyon TL - found live,
        # real BDDK-derived answers doing exactly this (a compare_regions
        # figure shown as "9.268 milyar TL"). All three directions are
        # added since either the raw or the reported figure could be the
        # larger one depending on which unit the source table used.
        for scale in (1_000, 1_000_000, 1_000_000_000):
            derived.add(round(x / scale, 2))
            derived.add(round(x * scale, 2))
            # A scaled value is usually also rounded in prose: 429410414
            # Bin TL reported as "429,4 milyar TL" was flagged live and
            # cost a pointless repair round.
            derived.add(round(x / scale, 1))
            derived.add(float(round(x / scale)))
        # A model reporting fewer decimal places than the raw value
        # (e.g. "%9,3" in prose for a raw 9.2934...) is normal rounding,
        # not fabrication - found live (chain_01: tool returned an NPL of
        # 9.29-ish, answer rounded to one decimal for readability).
        derived.add(round(x, 1))
        derived.add(round(x))
    for i, a in enumerate(tn_list):
        for b in tn_list[i + 1:]:
            derived.add(round(a - b, 2))
            derived.add(round(b - a, 2))
            if a:
                derived.add(round((b - a) / a * 100, 2))
                derived.add(round(b / a, 2))
            if b:
                derived.add(round((a - b) / b * 100, 2))
                derived.add(round(a / b, 2))

    grounded = 0
    ungrounded = []
    for candidates in answer_numbers:
        if any(c in tool_numbers or c in derived for c in candidates):
            grounded += 1
        else:
            ungrounded.append(candidates[0])

    return {"checked": len(answer_numbers), "grounded": grounded, "ungrounded": ungrounded}
