"""Chaos / fault injection used by the evaluation platform.

fault_injection keys on ResearchRequest:
  fake_number:     {"sticky": n}  corrupt one numeric claim; re-corrupt on the first n rewrites
  future_evidence: {"document_id": "..."}  attach a document visible only after as_of (a leaky tool)
  tool_error:      {"tool": "get_financial_statements", "remaining": k}  first k calls fail
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


def inject(ctx: RunContext) -> list[str]:
    done = []
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
