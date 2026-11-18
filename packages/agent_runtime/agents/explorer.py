"""Bounded ReAct-Lite (v2.0): inside a sub-graph, after the fixed operators ran, look for evidence the planner's
sub-hypotheses need but the standard pipeline did not collect.

Per hypothesis, at most `react_lite_steps` rounds:
  step 1  every search phrase whose terms are not already covered by evidence in this run is searched in the latest
          periodic report first, then the whole PIT-filtered corpus; the best sentence becomes a cited claim.
  step 2  (online only, and only if step 1 found nothing) one light-model call rewrites the failed phrases into one new
          phrase; it is searched once.
No hypothesis found nothing -> an open question names what evidence was needed. Nothing here writes a number: claims
quote the filing, and Tier-1 / Tier-2 verify them like any other disclosed fact.
"""
from __future__ import annotations

import re
from typing import Optional

from ..context import RunContext
from ..llm import call_json
from ..state import SubHypothesis
from .. import tools

SECTION_OF = {"fundamental": "financials", "market": "valuation", "news": "events"}
SENT = re.compile(r"[^。；\n|]{12,200}[。；]")


def _terms(query: str) -> list[str]:
    return [t for t in query.split() if len(t) >= 2]


def covered(ctx: RunContext, query: str) -> bool:
    """True when some evidence already collected in this run contains most of the phrase's terms."""
    terms = _terms(query)
    if not terms:
        return True
    need = max(1, round(len(terms) * 0.6))
    return any(sum(1 for t in terms if t in (ev.excerpt + ev.summary)) >= need for ev in ctx.evidence.values()
               if ev.kind in ("document", "event"))


def best_sentence(text: str, terms: list[str]) -> Optional[str]:
    best: tuple[int, str] | None = None
    for m in SENT.finditer(text):
        sent = m.group(0).strip()
        score = sum(1 for t in terms if t in sent)
        if score and (best is None or score > best[0]):
            best = (score, sent)
    return best[1] if best and best[0] >= max(1, min(2, len(terms))) else None


def _search(ctx: RunContext, h: SubHypothesis, query: str, agent: str, used: set[str]) -> Optional[str]:
    section = ["ANNOUNCEMENT"] if agent == "news" else None
    hits = tools.retrieve_latest_first(ctx, query, section, top_k=3)
    terms = _terms(query)
    for hit in hits:
        chunk = hit.parent or hit.chunk
        sent = best_sentence(chunk.text, terms)
        if not sent or sent in used:
            continue
        used.add(sent)
        ev = ctx.evidence_from_hit(hit, agent, focus=sent)
        cl = ctx.add_claim(SECTION_OF[agent], "disclosed_fact", f"假设[{h.sub_id}]{h.hypothesis_statement}：{sent}", [ev], agent)
        return cl.claim_id
    return None


def _rewrite(ctx: RunContext, h: SubHypothesis, failed: list[str]) -> Optional[str]:
    if not ctx.llm.online:
        return None
    sys_p = ("把检索失败的查询改写成一个更可能命中上市公司年报或公告原文的中文检索短语（2到4个词，空格分隔，不含数字）。"
             '只输出JSON {"query":"..."}')
    user = f"待验证：{h.hypothesis_statement}\n失败的查询：{'；'.join(failed)}"
    d = call_json(ctx.llm, ctx.tracer, "explorer", sys_p, user, "light", validate=lambda x: isinstance(x.get("query"), str))
    q = str((d or {}).get("query", "")).strip()
    return q if 2 <= len(q) <= 30 and not re.search(r"\d", q) and q not in failed else None


def explore(ctx: RunContext, h: SubHypothesis, agent: str) -> list[str]:
    steps = ctx.config.react_lite_steps
    if steps <= 0:
        return []
    ids: list[str] = []
    used: set[str] = set()
    todo = [q for q in h.custom_rag_queries if not covered(ctx, q)]
    if not todo:
        ctx.tracer.tool("explore_hypothesis", {"sub_id": h.sub_id, "step": 0}, result="already covered by collected evidence")
        return []
    failed: list[str] = []
    for q in todo[:2]:
        cid = _search(ctx, h, q, agent, used)
        ctx.tracer.tool("explore_hypothesis", {"sub_id": h.sub_id, "step": 1, "query": q}, result=cid or "no matching sentence")
        if cid:
            ids.append(cid)
        else:
            failed.append(q)
    if not ids and steps >= 2 and failed:
        q2 = _rewrite(ctx, h, failed)
        if q2:
            cid = _search(ctx, h, q2, agent, used)
            ctx.tracer.tool("explore_hypothesis", {"sub_id": h.sub_id, "step": 2, "query": q2}, result=cid or "no matching sentence")
            if cid:
                ids.append(cid)
    if not ids:
        ctx.open_questions.append({"topic": f"假设[{h.sub_id}]{h.hypothesis_statement}：未找到披露证据",
                                   "reason": f"需要：{h.verification_criterion or '相关披露原文'}"})
    return ids
