"""Agent-facing contracts. Parent graph state only carries ids and summaries (pointers);
full evidence lives in the run-scoped EvidenceStore / MetricStore."""
from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from data_platform.data_mode import DataMode, default_data_mode


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
    react_lite_steps: int = 2            # sub-graph self-directed evidence search per hypothesis (v2.0); 0 = off
    complexity_router: bool = True       # ComplexityRouter: lookups skip debate/risk/probe/LLM judge; topic analyses skip debate/risk
    self_rag: bool = True
    # LLM-written bull/bear arguments (online only). On: an independent pairwise judge preferred these reports
    # 8/10 vs 0/10 (EVALUATION.md section 11). Costs about 2x per report. Offline always uses templates.
    llm_debate: bool = True
    subgraph_crosscheck: bool = True     # Fundamental subgraph checks revenue/receivables/profit/cash-flow and reads NOTES itself
    semantic_probe: bool = True          # LLM scan of notes / announcements for qualitative red flags (online only)
    anomaly_mode: str = "statistical"    # "statistical" (z-scores vs own history and peers) | "rules" (fixed cut-offs)
    max_replan_depth: int = 2
    max_debate_rounds: int = 2
    max_claim_retries: int = 2


ARCHITECTURES: dict[str, ArchConfig] = {
    "B3": ArchConfig(name="B3", replanning=False, debate=False, risk_agent=False, tier1_verifier=False,
                     tier2_verifier=False),
    "B4": ArchConfig(name="B4"),
    "B4-template_debate": ArchConfig(name="B4-template_debate", llm_debate=False),
    "B4-no_anchor_prefilter": ArchConfig(name="B4-no_anchor_prefilter", rag_anchors_prefilter=False),
    "B4-no_react_lite": ArchConfig(name="B4-no_react_lite", react_lite_steps=0),
    "B4-no_router": ArchConfig(name="B4-no_router", complexity_router=False),
    "B4-no_hybrid_rerank": ArchConfig(name="B4-no_hybrid_rerank", hybrid_rerank=False),
    "B4-no_context_isolation": ArchConfig(name="B4-no_context_isolation", context_isolation=False),
    "B4-no_crosscheck": ArchConfig(name="B4-no_crosscheck", subgraph_crosscheck=False),
    "B4-no_semantic_probe": ArchConfig(name="B4-no_semantic_probe", semantic_probe=False),
    "B4-rule_anomalies": ArchConfig(name="B4-rule_anomalies", anomaly_mode="rules"),
    "B4-no_replanning": ArchConfig(name="B4-no_replanning", replanning=False),
    "B4-no_debate": ArchConfig(name="B4-no_debate", debate=False),
    "B4-no_tier1": ArchConfig(name="B4-no_tier1", tier1_verifier=False),
    "B4-no_tier2": ArchConfig(name="B4-no_tier2", tier2_verifier=False),
}


class ResearchRequest(BaseModel):
    ticker: str
    tickers: list[str] = Field(default_factory=list)   # 2-4 companies: comparison run; `ticker` stays the primary
    as_of: datetime
    data_mode: DataMode = Field(default_factory=default_data_mode)   # offline_golden | hybrid_live | online_force_refresh
    question: str = "全面研究公司基本面、估值与风险"
    mode: Literal["fast", "deep"] = "deep"
    focus: list[str] = Field(default_factory=list)
    token_budget: int = 200_000
    hitl: bool = False          # pause for human approval instead of silently skipping over-budget re-planning
    hitl_always: bool = False   # with hitl: ask before every re-planning step, even within budget
    fault_injection: dict[str, Any] = Field(default_factory=dict)


class Ask(BaseModel):
    """A structured requirement parsed from the question."""
    ask_id: str
    kind: Literal["metric", "anomalies", "valuation", "events", "peer_compare", "factor_test", "unanswerable"]
    metric: Optional[str] = None
    period: Optional[str] = None
    scope: str = "consolidated"
    unit: Optional[str] = None
    topic: str = ""
    params: dict = Field(default_factory=dict)   # factor_test: factor, universe, horizon, lookback, quantiles, ...


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


class SubHypothesis(BaseModel):
    """One research sub-question the planner wants evidence for (v2.0 hypothesis-driven planning)."""
    sub_id: str
    target_ticker: str
    assigned_agent: Literal["fundamental", "market", "news"]
    hypothesis_statement: str          # what is to be verified
    custom_rag_queries: list[str] = Field(default_factory=list)   # 2-3 fine-grained search phrases for this question
    required_metrics: list[str] = Field(default_factory=list)     # deterministic metrics it depends on
    verification_criterion: str = ""   # what evidence would confirm or refute it


class ResearchPlan(BaseModel):
    plan_id: str
    subtasks: list[SubTask]
    asks: list[Ask]
    budget_tokens: int
    rationale: str = ""
    hypotheses: list[SubHypothesis] = Field(default_factory=list)


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
    section: Literal["overview", "financials", "valuation", "bull", "bear", "events", "risk", "answer", "falsification", "quant", "comparison"]
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
                        "SCOPE_MIX", "UNSUPPORTED", "CONTRADICTED",
                        "TREND_REVERSAL", "SCOPE_SWITCH", "CONDITION_OMISSION", "CAUSAL_INVERSION"]   # Tier-2 semantic fallacies
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
    approval: str
    report: dict
    shared_history: Annotated[list[str], operator.add]      # only used when context isolation is OFF
