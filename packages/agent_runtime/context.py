"""Run-scoped context: PIT data service, calculators, retriever, stores, tracer, LLM.

Kept outside LangGraph state on purpose: graph state carries pointers only.
"""
from __future__ import annotations

import itertools
from datetime import datetime
from typing import Optional

from data_platform import FinancialCalculator, LineageRecorder, PITDataService
from data_platform.schema import MetricRecord
from evaluation_platform.tracing import Tracer
from rag_engine.index import get_retriever
from rag_engine.retrieval import Hit

from .llm import BaseLLM, get_llm
from .state import ArchConfig, Claim, EvidenceCard, ResearchRequest, UncertaintyStatement

_CONTEXTS: dict[str, "RunContext"] = {}


def fmt(rec: MetricRecord) -> str:
    if rec.status != "OK" or rec.value is None:
        return {"NEGATIVE_EARNINGS_NA": "不适用(亏损/基数为负)", "ZERO_DENOMINATOR_NA": "不适用(分母为零)",
                "NEGATIVE_EQUITY_NA": "不适用(净资产为负)", "INSUFFICIENT_DATA": "数据不足",
                "RESTATEMENT_BREAKS_SPLIT": "不可拆分(期间重述)"}.get(rec.status, rec.status)
    v = rec.value
    if rec.unit == "ratio":
        return f"{v * 100:.2f}%"
    if rec.unit == "CNY":
        return f"{v / 1e8:.2f}亿元"
    if rec.unit == "days":
        return f"{v:.1f}天"
    if rec.unit == "x":
        return f"{v:.2f}倍"
    return f"{v:.4g}"


class RunContext:
    def __init__(self, run_id: str, request: ResearchRequest, config: ArchConfig, snapshot_id: str,
                 db_url: Optional[str] = None, llm: Optional[BaseLLM] = None, listener=None):
        self.run_id = run_id
        self.request = request
        self.config = config
        self.db_url = db_url
        self.snapshot_id = snapshot_id
        self.svc = PITDataService(request.as_of, snapshot_id, db_url)
        self.lineage = LineageRecorder(db_url, run_id)
        self.calc = FinancialCalculator(self.svc, self.lineage)
        self.retriever = get_retriever(db_url, anchors=config.rag_anchors_prefilter)
        self.tracer = Tracer(run_id, listener)
        self.llm = llm or get_llm()
        self.evidence: dict[str, EvidenceCard] = {}
        self.claims: dict[str, Claim] = {}
        self.uncertainties: list[UncertaintyStatement] = []
        self.open_questions: list[dict] = []
        self.answers: dict[str, dict] = {}
        self.anomalies: dict[str, dict] = {}
        self.verification_log: list[dict] = []
        self.raw_outputs: list[str] = []   # private scratchpads, joined only when isolation is off
        self._ev = itertools.count(1)
        self._cl = itertools.count(1)
        self.fault_state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in request.fault_injection.items()}
        self.renderers: dict = {}          # claim_id -> callable that re-renders text from MetricRecords
        self.charts: dict = {}
        self.judge_notes: dict = {}        # claim_id -> Tier-2 LLM judge label + reason
        self.timeline: list[dict] = []
        self.plan = None
        _CONTEXTS[run_id] = self

    @property
    def ticker(self) -> str:
        return self.request.ticker

    @property
    def as_of(self) -> datetime:
        return self.request.as_of

    # ---------- evidence ----------
    def evidence_from_metric(self, rec: MetricRecord, produced_by: str) -> EvidenceCard:
        for e in self.evidence.values():
            if e.kind == "metric" and e.source_ref == rec.metric_id:
                return e
        from data_platform.catalog import cn_name

        name = rec.extra.get("cn") or cn_name(rec.extra.get("base_metric", rec.metric))
        if rec.metric == "single_quarter":
            name = f"单季{cn_name(rec.extra.get('base_metric', ''))}"
        scope = "(母公司)" if rec.statement_scope.value == "parent" else ""
        ev = EvidenceCard(evidence_id=f"E{next(self._ev):03d}", kind="fact" if rec.extra.get("kind") == "fact" else "metric",
                          source_ref=rec.metric_id, summary=f"{rec.period} {name}{scope} = {fmt(rec)}",
                          excerpt=f"formula: {rec.formula}; inputs: {', '.join(rec.input_ids[:4])}",
                          available_at=rec.available_at, metric_ids=[rec.metric_id], produced_by=produced_by)
        self.evidence[ev.evidence_id] = ev
        self.lineage.add(rec.metric_id, ev.evidence_id, "metric", "evidence", f"agent.{produced_by}")
        return ev

    def evidence_from_hit(self, hit: Hit, produced_by: str) -> EvidenceCard:
        c = hit.parent or hit.chunk
        for e in self.evidence.values():
            if e.source_ref == c.chunk_id:
                return e
        ev = EvidenceCard(evidence_id=f"E{next(self._ev):03d}", kind="document", source_ref=c.chunk_id,
                          summary=c.section_path.split(" > ")[-1], excerpt=c.text[:1200], available_at=c.available_at,
                          document_id=c.document_id, page=f"p{c.page_start}-{c.page_end}", produced_by=produced_by)
        self.evidence[ev.evidence_id] = ev
        self.lineage.add(c.document_id, c.chunk_id, "document", "chunk", "rag.chunking")
        self.lineage.add(c.chunk_id, ev.evidence_id, "chunk", "evidence", f"agent.{produced_by}")
        return ev

    def add_claim(self, section: str, claim_type: str, text: str, evidence: list[EvidenceCard],
                  produced_by: str) -> Claim:
        cl = Claim(claim_id=f"C{next(self._cl):03d}", section=section, claim_type=claim_type, text=text,
                   evidence_ids=[e.evidence_id for e in evidence],
                   metric_ids=[m for e in evidence for m in e.metric_ids], produced_by=produced_by)
        self.claims[cl.claim_id] = cl
        for e in evidence:
            self.lineage.add(e.evidence_id, cl.claim_id, "evidence", "claim", f"agent.{produced_by}")
        return cl

    def metric(self, metric_id: str) -> Optional[MetricRecord]:
        return self.calc.store.get(metric_id)

    def note(self, text: str) -> None:
        self.raw_outputs.append(text)


def get_context(run_id: str) -> RunContext:
    return _CONTEXTS[run_id]


def drop_context(run_id: str) -> None:
    _CONTEXTS.pop(run_id, None)
