"""Tier-2 semantic fallacy checks (PRD v1.2 1.3): claims whose numbers are all right but whose meaning is wrong.

Four fallacy classes, each checked deterministically against the claim's own evidence (so they run offline and cost
no tokens; the online model judge receives the same taxonomy as well):

  TREND_REVERSAL     the claim says a figure fell while the cited growth metric rose, or the other way round
  SCOPE_SWITCH       the claim talks about a full year while every cited figure is a partial-year cumulative
  CONDITION_OMISSION an announcement said a thing is planned / expected / proposed, the claim says it is done
  CAUSAL_INVERSION   the claim asserts that A caused B, while the source text only lists them side by side

Each check is conservative: it needs positive evidence of a mismatch, never just the absence of a match.
"""
from __future__ import annotations

import re
from typing import Optional

from ..context import RunContext
from ..state import Claim

FALLACIES = ("TREND_REVERSAL", "SCOPE_SWITCH", "CONDITION_OMISSION", "CAUSAL_INVERSION")
UP = ("增长", "上升", "增加", "提升", "提高", "增")
DOWN = ("下降", "下滑", "减少", "降低", "下跌", "降")
GROWTH = re.compile(r"(同比|环比)\s*(增长|上升|增加|提升|提高|下降|下滑|减少|降低|下跌|降|增)?\s*([-+]?\d+(?:\.\d+)?)\s*%")
FULL_YEAR = re.compile(r"全年|全年度|\d{4}\s*年度(?!报告)")
PLAN_WORDS = ("预计", "拟", "计划", "将", "拟定", "有望", "目标")
DONE_RE = re.compile(r"已(完成|实现|达成|回购|落地|兑现|支付|发放|实施|完毕|成功|签订|取得)")
STRONG_CAUSE = ("因此导致", "从而导致", "进而导致", "导致", "致使", "造成", "引起", "使得", "归因于")
ANY_CAUSE = ("因此", "从而", "进而", "导致", "致使", "造成", "引起", "使得", "归因于", "所致", "由于", "因为", "主要系", "受.{0,8}影响", "系因")


def _records(ctx: RunContext, c: Claim):
    out = []
    for e in c.evidence_ids:
        ev = ctx.evidence.get(e)
        for m in (ev.metric_ids if ev else []):
            r = ctx.metric(m)
            if r is not None and r.value is not None:
                out.append(r)
    return out


def _doc_text(ctx: RunContext, c: Claim) -> str:
    return " ".join(ctx.evidence[e].excerpt for e in c.evidence_ids if e in ctx.evidence and ctx.evidence[e].kind == "document")


def growth_phrases(text: str):
    """(start, end, direction_word or None, signed_number_in_percent, matched_text)."""
    for m in GROWTH.finditer(text):
        yield m.start(), m.end(), m.group(2), float(m.group(3)), m.group(0)


def trend_reversal(ctx: RunContext, c: Claim) -> Optional[str]:
    recs = [r for r in _records(ctx, c) if r.unit == "ratio"]
    for _, _, word, num, raw in growth_phrases(c.text):
        if word is None:
            continue                       # "同比6.54%": the sign alone speaks for itself, Tier-1 checked the magnitude
        implied = abs(num) if word in UP else -abs(num)
        for r in recs:
            if abs(abs(r.value) * 100 - abs(num)) <= 0.006 and r.value != 0 and (r.value > 0) != (implied > 0):
                return f"“{raw}” says {'rose' if implied > 0 else 'fell'}, but the cited metric {r.metric} is {r.value:+.4f}"
    return None


QUOTE_PREFIX = re.compile(r"^(说明|追查|交叉核验|年报附注|语义探针|管理层讨论与分析披露)[\[：]")


def scope_switch(ctx: RunContext, c: Claim) -> Optional[str]:
    # a sentence copied from a filing says whatever the filing says; only our own wording can switch scope
    if QUOTE_PREFIX.match(c.text) or c.text[:12].count("公告《"):
        return None
    if not FULL_YEAR.search(c.text):
        return None
    periods = [r.period for r in _records(ctx, c) if re.fullmatch(r"\d{4}Q[1-4]", r.period or "")]
    if periods and all(not p.endswith("Q4") for p in periods):
        return f"text speaks of a full year, but every cited figure is a partial-year cumulative ({sorted(set(periods))})"
    return None


def condition_omission(ctx: RunContext, c: Claim) -> Optional[str]:
    doc = _doc_text(ctx, c)
    if not doc:
        return None
    body = c.text
    for m in DONE_RE.finditer(body):
        verb = m.group(1)
        planned = any(re.search(re.escape(w) + r"[^，。；\s]{0,4}" + re.escape(verb), doc) for w in PLAN_WORDS)
        done_in_source = re.search("已" + re.escape(verb), doc) is not None
        if planned and not done_in_source:
            return f"claim says “已{verb}”, the source only says it is planned / expected"
    return None


def causal_inversion(ctx: RunContext, c: Claim) -> Optional[str]:
    doc = _doc_text(ctx, c)
    if not doc:
        return None                        # model arguments citing metrics are judged elsewhere
    strong = next((w for w in STRONG_CAUSE if w in c.text), None)
    if strong and not any(re.search(p, doc) for p in ANY_CAUSE):
        return f"claim asserts causation (“{strong}”), the source text states none"
    return None


CHECKS = {"TREND_REVERSAL": trend_reversal, "SCOPE_SWITCH": scope_switch,
          "CONDITION_OMISSION": condition_omission, "CAUSAL_INVERSION": causal_inversion}


def fallacies(ctx: RunContext, c: Claim) -> list[tuple[str, str]]:
    out = []
    for name, fn in CHECKS.items():
        why = fn(ctx, c)
        if why:
            out.append((name, why))
    return out
