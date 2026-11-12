"""Chaos / fault injection used by the evaluation platform.

fault_injection keys on ResearchRequest:
  fake_number:     {"sticky": n}  corrupt one numeric claim; re-corrupt on the first n rewrites
  future_evidence: {"document_id": "..."}  attach a document visible only after as_of (a leaky tool)
  tool_error:      {"tool": "get_financial_statements", "remaining": k}  first k calls fail
  semantic:        {"type": "TREND_REVERSAL|SCOPE_SWITCH|CONDITION_OMISSION|CAUSAL_INVERSION"}  rewrite one claim so every
                   number is still right but the meaning is wrong (Tier-1 cannot see it; only Tier-2 can)
"""
from __future__ import annotations

import re
from datetime import timedelta

from sqlalchemy import select

from data_platform.db import SourceDocumentRow

from ..context import RunContext
from ..state import Claim, EvidenceCard

PCT = re.compile(r"(-?\d+\.\d+)%")


def _corrupt(text: str) -> str:
    m = PCT.search(text)
    if not m:
        return text
    bad = float(m.group(1)) * 1.37 + 3.1
    return text[: m.start()] + f"{bad:.2f}%" + text[m.end():]


def _semantic_target(ctx: RunContext, kind: str):
    """A real claim of the right shape, and the rewritten text. None when this run has no suitable claim."""
    from . import semantic_check as S

    for c in sorted(ctx.claims.values(), key=lambda c: c.claim_id):
        if c.status != "DRAFT":
            continue
        if kind == "TREND_REVERSAL":
            for start, end, word, num, raw in S.growth_phrases(c.text):
                if word in S.UP:
                    return c, c.text[:start] + raw.replace(word, "下降") + c.text[end:]
                if word in S.DOWN:
                    return c, c.text[:start] + raw.replace(word, "增长") + c.text[end:]
            m = re.search(r"同比(-?\d+\.\d+)%", c.text)
            if m and c.section in ("financials", "answer"):
                # "同比6.54%" carries the sign by itself; name the direction wrongly
                up = not m.group(1).startswith("-")
                return c, c.text[:m.start()] + f"同比{'下降' if up else '增长'}{abs(float(m.group(1))):.2f}%" + c.text[m.end():]
        if kind == "SCOPE_SWITCH" and c.section in ("financials", "answer") and re.search(r"20\d\dQ[123]", c.text) is None:
            recs = [ctx.metric(m) for e in c.evidence_ids for m in (ctx.evidence[e].metric_ids if e in ctx.evidence else [])]
            if recs and all(r is not None and re.fullmatch(r"\d{4}Q[123]", r.period or "") for r in recs) and PCT.search(c.text):
                return c, "全年" + c.text
        if kind == "CONDITION_OMISSION" and c.claim_type == "disclosed_fact":
            doc = " ".join(ctx.evidence[e].excerpt for e in c.evidence_ids if e in ctx.evidence and ctx.evidence[e].kind == "document")
            m = re.search(r"(拟|计划|预计)([^，。；\s]{0,4})(回购|实施|发放|支付|落地|签订)", doc)
            if m and "回购" in c.text + doc[:0] or (m and m.group(3) in c.text):
                return c, c.text.rstrip("。") + f"；公司已{m.group(3)}。"
        if kind == "CAUSAL_INVERSION" and c.claim_type == "disclosed_fact" and c.section in ("events", "bear", "bull", "financials"):
            doc = " ".join(ctx.evidence[e].excerpt for e in c.evidence_ids if e in ctx.evidence and ctx.evidence[e].kind == "document")
            if doc and not any(re.search(p, doc) for p in S.ANY_CAUSE):
                return c, c.text.rstrip("。") + "，这一事项因此导致了公司业绩的变化。"
    return None, None


def inject(ctx: RunContext) -> list[str]:
    done = []
    sem = ctx.fault_state.get("semantic")
    if sem is not None and not sem.get("applied"):
        target, new_text = _semantic_target(ctx, sem["type"])
        if target is not None:
            sem["original"] = target.text
            target.text = new_text
            sem["applied"], sem["injected_text"] = target.claim_id, new_text
            done.append(f"semantic:{sem['type']}->{target.claim_id}")
        else:
            sem["applied"] = "none"
            done.append(f"semantic:{sem['type']}->no suitable claim")
    fn = ctx.fault_state.get("fake_number")
    if fn is not None and not fn.get("applied"):
        ordered = sorted(ctx.claims.values(), key=lambda c: {"answer": 0, "financials": 1}.get(c.section, 2))
        for c in ordered:
            if c.section in ("financials", "answer") and PCT.search(c.text):
                c.text = _corrupt(c.text)
                fn["applied"] = c.claim_id
                done.append(f"fake_number->{c.claim_id}")
                break
    fe = ctx.fault_state.get("future_evidence")
    if fe is not None and not fe.get("applied"):
        with ctx.svc.Session() as s:
            q = select(SourceDocumentRow).where(SourceDocumentRow.ticker == ctx.ticker,
                                                SourceDocumentRow.available_at > ctx.as_of)
            if fe.get("document_id"):
                q = q.where(SourceDocumentRow.document_id == fe["document_id"])
            doc = s.scalars(q.order_by(SourceDocumentRow.available_at)).first()
        if doc:
            ev = EvidenceCard(evidence_id=f"E{next(ctx._ev):03d}", kind="document", source_ref=doc.document_id,
                              summary=doc.title, excerpt=doc.title, available_at=doc.available_at,
                              document_id=doc.document_id, produced_by="leaky_tool")
            ctx.evidence[ev.evidence_id] = ev
            target = next((c for c in ctx.claims.values() if c.section in ("bear", "events", "risk")), None) or \
                next(iter(ctx.claims.values()))
            target.evidence_ids.append(ev.evidence_id)
            target.text = target.text.rstrip("。") + f"；另据《{doc.title}》。"
            fe["applied"] = target.claim_id
            fe["leaked_document_id"] = doc.document_id
            done.append(f"future_evidence->{target.claim_id}")
    if done:
        ctx.tracer.tool("fault_injection", {"faults": done})
    return done


def maybe_recorrupt(ctx: RunContext, c: Claim) -> None:
    fn = ctx.fault_state.get("fake_number")
    if fn and fn.get("applied") == c.claim_id and fn.get("sticky", 0) > 0:
        fn["sticky"] -= 1
        c.text = _corrupt(c.text)
