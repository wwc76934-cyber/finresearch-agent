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


# words that appear in almost any paragraph of a report: they can add to a match but never make one
GENERIC = {"收入", "占比", "变动", "原因", "投资", "金额", "影响", "情况", "公司", "变化", "增长", "下降", "比例", "政策", "风险",
           "价格", "成本", "波动", "子公司", "协议", "方案", "安排", "周转", "现金", "支出", "计提", "关联方"}


def best_sentence(text: str, terms: list[str]) -> Optional[str]:
    """The sentence that carries the most of the phrase's terms, and at least one term that is specific to the topic
    (an overseas / customer / capex word, not just 收入 or 变动), plus a second term when the phrase has one."""
    anchors = [t for t in terms if t not in GENERIC]
    best: tuple[int, str] | None = None
    for m in SENT.finditer(text):
        sent = m.group(0).strip()
        if anchors and not any(a in sent for a in anchors):
            continue
        score = sum(1 for t in terms if t in sent)
        if score and (best is None or score > best[0]):
            best = (score, sent)
    return best[1] if best and best[0] >= max(1, min(2, len(terms))) else None


def best_row(text: str, terms: list[str]) -> Optional[tuple[str, str]]:
    """Much of what a filing says about customers, overseas revenue or capex is a table row, not a sentence. A row counts
    when its label (first cell) holds a topic-specific term. The quote pairs each value with its column heading, so the
    claim is readable and checkable against the table in the evidence. Returns (claim text, the table line itself): the
    evidence excerpt is centred on that literal line so every number in the claim is visible in it."""
    anchors = [t for t in terms if t not in GENERIC]
    best: tuple[int, str, str] | None = None
    header: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        if set(line) <= set("|- :"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if nxt.startswith("|") and set(nxt) <= set("|- :"):
            header = cells                    # the row above a separator line is the column heading
            continue
        label = cells[0] if cells else ""
        if len(label) < 2 or not any(a in label for a in anchors) or not any(ch.isdigit() for ch in line):
            continue
        pairs = [f"{h} {v}" if h else v for h, v in zip(header[1:] + [""] * len(cells), cells[1:]) if v]
        row = f"{label}：" + "，".join(pairs)
        score = sum(1 for t in terms if t in label)
        if best is None or score > best[0]:
            best = (score, row[:200], line)
    return (f"年报表格披露：{best[1]}（单位见原表）", best[2]) if best else None


def _covering_claim(ctx: RunContext, h: SubHypothesis) -> Optional[str]:
    """The claim already in this run whose cited evidence covers one of the hypothesis' phrases."""
    for q in h.custom_rag_queries:
        terms = _terms(q)
        need = max(1, round(len(terms) * 0.6))
        ev_ids = {e.evidence_id for e in ctx.evidence.values() if e.kind in ("document", "event")
                  and sum(1 for t in terms if t in (e.excerpt + e.summary)) >= need}
        for c in ctx.claims.values():
            if c.claim_type == "disclosed_fact" and ev_ids & set(c.evidence_ids):
                return c.claim_id
    return None


def _search(ctx: RunContext, h: SubHypothesis, query: str, agent: str, used: set[str]) -> Optional[str]:
    section = ["ANNOUNCEMENT"] if agent == "news" else None
    terms = _terms(query)
    latest = tools.retrieve_latest_first(ctx, query, section, top_k=4)
    hits = latest + [x for x in tools.retrieve_document_evidence(ctx, query, section, top_k=8)
                     if x.chunk.chunk_id not in {y.chunk.chunk_id for y in latest}]     # then every other visible document
    for hit in hits:
        chunk = hit.parent or hit.chunk
        sentence = best_sentence(chunk.text, terms)
        found = (sentence, sentence) if sentence else best_row(chunk.text, terms)
        if not found or found[0] in used:
            continue
        sent, focus = found
        used.add(sent)
        ev = ctx.evidence_from_hit(hit, agent, focus=focus)
        cl = ctx.add_claim(SECTION_OF[agent], "disclosed_fact", f"年报/公告披露（对应假设[{h.sub_id}]）：{sent}", [ev], agent)
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


def _metric_covered(ctx: RunContext, h: SubHypothesis) -> bool:
    """A hypothesis such as 'leverage is under control' or 'valuation is low' is answered by metrics the pipeline already
    computed and cited, not by a sentence in a filing: no document hit is not a gap then."""
    have = {m.metric for m in ctx.calc.store.values() if m.ticker == ctx.ticker}
    return bool(h.required_metrics) and all(any(r in m for m in have) for r in h.required_metrics)


def explore(ctx: RunContext, h: SubHypothesis, agent: str) -> list[str]:
    steps = ctx.config.react_lite_steps
    if steps <= 0:
        return []
    if agent == "fundamental":
        tools.ensure_report_docs(ctx, annual=True)      # notes questions need the annual report, once per run
    ids: list[str] = []
    used: set[str] = set()
    todo = [q for q in h.custom_rag_queries if not covered(ctx, q)]
    if not todo:
        link = _covering_claim(ctx, h)
        if link:
            ctx.hypothesis_links[h.sub_id] = link
        ctx.tracer.tool("explore_hypothesis", {"sub_id": h.sub_id, "step": 0}, result=f"already covered by {link or 'collected evidence'}")
        return []
    failed: list[str] = []
    for q in todo[:2]:
        cid = _search(ctx, h, q, agent, used)
        ctx.tracer.tool("explore_hypothesis", {"sub_id": h.sub_id, "step": 1, "query": q}, result=cid or "no matching sentence")
        if not cid:
            core = " ".join(t for t in _terms(q) if t not in GENERIC)       # the topic words alone, without the generic ones
            if core and core != q:
                cid = _search(ctx, h, core, agent, used)
                ctx.tracer.tool("explore_hypothesis", {"sub_id": h.sub_id, "step": 1, "query": core}, result=cid or "no matching sentence")
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
    if ids:
        ctx.hypothesis_links[h.sub_id] = ids[0]
    if not ids and _metric_covered(ctx, h):
        ctx.tracer.tool("explore_hypothesis", {"sub_id": h.sub_id, "step": 3}, result="answered by computed metrics")
    elif not ids:
        ctx.open_questions.append({"topic": f"假设[{h.sub_id}]{h.hypothesis_statement}：未找到披露证据",
                                   "reason": f"需要：{h.verification_criterion or '相关披露原文'}"})
    return ids
