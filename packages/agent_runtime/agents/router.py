"""ComplexityRouter: decide how much of the pipeline a question needs, before any LLM call.

fast      metric look-ups and refusals. Rule-parsed plan (no planner LLM), no re-planning / debate / risk /
          semantic probe / cross-check, Tier-2 LLM judge and the LLM executive summary are skipped.
          Tier-1 (numbers, PIT, evidence ids) and the deterministic fallacy checks still run on every claim.
standard  analysis of one topic (anomalies, notes, announcements). Keeps collection, re-planning, probe and the
          full verifier; drops the bull/bear debate and the risk matrix, which only a research-report ask needs.
full      everything: the question asks for a valuation view, bull/bear review, falsification conditions, a
          comparison with peers, or the caller declared nothing smaller.

The decision uses only the question text, the requested mode and the rule-parsed asks; it never looks at ticker,
task id or expected answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FULL_MARKERS = re.compile(r"多空|对抗|审议|证伪|估值|同行|可比|全面|投资研究|研究报告|支撑|牛熊|目标价|PE|PB")
OPEN_MARKERS = re.compile(r"比|如何|怎么|特点|位置|差异|能不能|成长|前景|趋势")   # open-ended research asks go the full way
STANDARD_MARKERS = re.compile(r"异常|背离|预警|警惕|附注|减值|账龄|回收|信号|风险点")
ANALYSIS_MARKERS = re.compile(r"为什么|原因|异常|背离|风险|预警|警惕|健康|附注|公告|结合|综合|分析|解读|趋势|信号|回收|账龄|减值")


@dataclass
class Route:
    name: str = "full"
    reasons: list[str] = field(default_factory=list)

    @property
    def fast(self) -> bool:
        return self.name == "fast"

    @property
    def lean(self) -> bool:  # no debate / risk matrix
        return self.name in ("fast", "standard")


def classify(question: str, mode: str, rule_asks: list, enabled: bool = True) -> Route:
    if not enabled:
        return Route("full", ["router disabled"])
    kinds = {a.kind for a in rule_asks}
    if FULL_MARKERS.search(question):
        return Route("full", ["question asks for valuation / debate / peers / full research"])
    if kinds & {"valuation", "peer_compare", "factor_test"}:
        return Route("full", [f"asks include {sorted(kinds & {'valuation', 'peer_compare', 'factor_test'})}"])
    only_lookup = bool(kinds) and kinds <= {"metric", "unanswerable"} and len(rule_asks) <= 3
    if only_lookup and not ANALYSIS_MARKERS.search(question):
        return Route("fast", [f"{len(rule_asks)} lookup ask(s), no analysis wording"])
    if mode == "fast" and not ANALYSIS_MARKERS.search(question) and only_lookup:
        return Route("fast", ["declared fast"])
    if STANDARD_MARKERS.search(question) and not OPEN_MARKERS.search(question):
        return Route("standard", ["single-topic anomaly / notes analysis"])
    return Route("full", ["unrecognised analysis: take the safe, complete path"])
