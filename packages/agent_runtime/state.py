"""Agent-facing contracts. Parent graph state only carries ids and summaries (pointers);
full evidence lives in the run-scoped EvidenceStore / MetricStore."""
from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class ArchConfig(BaseModel):
    name: str = "B4"
    parallel_subgraphs: bool = True
    replanning: bool = True
    debate: bool = True
    risk_agent: bool = True
    tier1_verifier: bool = True
    tier2_verifier: bool = True
    context_isolation: bool = True
    rag_anchors_prefilter: bool = True
    hybrid_rerank: bool = True
    self_rag: bool = True
    max_replan_depth: int = 2
    max_debate_rounds: int = 2
    max_claim_retries: int = 2


ARCHITECTURES: dict[str, ArchConfig] = {
    "B3": ArchConfig(name="B3", replanning=False, debate=False, risk_agent=False, tier1_verifier=False,
                     tier2_verifier=False),
    "B4": ArchConfig(name="B4"),
    "B4-no_anchor_prefilter": ArchConfig(name="B4-no_anchor_prefilter", rag_anchors_prefilter=False),
    "B4-no_hybrid_rerank": ArchConfig(name="B4-no_hybrid_rerank", hybrid_rerank=False),
    "B4-no_context_isolation": ArchConfig(name="B4-no_context_isolation", context_isolation=False),
    "B4-no_replanning": ArchConfig(name="B4-no_replanning", replanning=False),
    "B4-no_debate": ArchConfig(name="B4-no_debate", debate=False),
    "B4-no_tier1": ArchConfig(name="B4-no_tier1", tier1_verifier=False),
    "B4-no_tier2": ArchConfig(name="B4-no_tier2", tier2_verifier=False),
}


class ResearchRequest(BaseModel):
    ticker: str
    as_of: datetime
    question: str = "全面研究公司基本面、估值与风险"
    mode: Literal["fast", "deep"] = "deep"
    focus: list[str] = Field(default_factory=list)
    token_budget: int = 200_000
    fault_injection: dict[str, Any] = Field(default_factory=dict)


class Ask(BaseModel):
    """A structured requirement parsed from the question."""
    ask_id: str
    kind: Literal["metric", "anomalies", "valuation", "events", "peer_compare", "unanswerable"]
    metric: Optional[str] = None
    period: Optional[str] = None
    scope: str = "consolidated"
    unit: Optional[str] = None
    topic: str = ""


class SubTask(BaseModel):
    task_id: str
    agent: Literal["fundamental", "market", "news", "investigate"]
    objective: str
    tool_budget: int = 8
    depends_on: list[str] = Field(default_factory=list)
    acceptance: str = ""
    query: Optional[str] = None
    section_types: Optional[list[str]] = None
    anomaly_code: Optional[str] = None


class ResearchPlan(BaseModel):
    plan_id: str
    subtasks: list[SubTask]
    asks: list[Ask]
    budget_tokens: int
    rationale: str = ""


class EvidenceCard(BaseModel):
    evidence_id: str
    kind: Literal["fact", "metric", "document", "event"]
    source_ref: str              # fact_id / metric_id / chunk_id / document_id
    summary: str
    excerpt: str = ""
    available_at: datetime
    metric_ids: list[str] = Field(default_factory=list)
    document_id: Optional[str] = None
    page: Optional[str] = None
    produced_by: str = ""


class SubTaskSummary(BaseModel):
    task_id: str
    agent: str
    summary: str
    evidence_ids: list[str]
    metric_ids: list[str]
    anomalies: list[dict] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    claim_id: str
    section: Literal["overview", "financials", "valuation", "bull", "bear", "events", "risk", "answer", "falsification"]
    claim_type: Literal["disclosed_fact", "derived_metric", "model_inference"]
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    metric_ids: list[str] = Field(default_factory=list)
    produced_by: str
    status: Literal["DRAFT", "VERIFIED", "PARTIALLY_SUPPORTED", "DOWNGRADED", "REJECTED"] = "DRAFT"
    attempts: int = 0


class VerificationFailure(BaseModel):
    claim_id: str
    tier: int
    error_type: Literal["NUMBER_MISMATCH", "WILD_NUMBER", "PIT_VIOLATION", "MISSING_CITATION", "UNKNOWN_EVIDENCE",
                        "SCOPE_MIX", "UNSUPPORTED", "CONTRADICTED"]
    target_agent: str
    fix_instruction: str


class UncertaintyStatement(BaseModel):
    claim_id: str
    text: str
    reason: str


def merge_dict(a: dict, b: dict) -> dict:
    return {**a, **b}


class GraphState(TypedDict, total=False):
    run_id: str
    request: dict
    snapshot_id: str
    plan: dict
    summaries: Annotated[list[dict], operator.add]          # SubTaskSummary pointers
    pending_tasks: list[dict]
    replan_depth: int
    handled_anomalies: Annotated[list[str], operator.add]
    claim_ids: Annotated[list[str], operator.add]
    debate: dict
    risk: dict
    verification_round: int
    report: dict
    shared_history: Annotated[list[str], operator.add]      # only used when context isolation is OFF
