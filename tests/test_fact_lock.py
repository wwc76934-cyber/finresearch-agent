"""Fact Lock: the debate model may argue, but numbers reach the report only from MetricRecords.

A scripted 'online' model tries to write numbers into its arguments and to cite a made-up evidence id.
"""
import json
import re
from datetime import datetime

from agent_runtime import ResearchRequest, run_research
from agent_runtime.llm import BaseLLM, LLMResult

A = datetime(2025, 12, 31, 15)
DIGIT = re.compile(r"\d")


class ScriptedLLM(BaseLLM):
    online = True

    def chat(self, system, user, tier="strong", json_mode=False):
        try:
            d = json.loads(user)
        except (ValueError, TypeError):
            d = None
        if isinstance(d, dict) and "existing_bull_points" in d:
            ev = [e["id"] for e in d["evidence"] if e["kind"] == "metric"][:1]
            args = [{"thesis": "收入增长了20%，说明需求强劲", "evidence_ids": ev},        # digits: must be rejected
                    {"thesis": "现金流质量支撑盈利可持续", "evidence_ids": ev},            # clean
                    {"thesis": "品牌溢价稳固", "evidence_ids": ["E9999"]}]                # unknown evidence: rejected
            return LLMResult(json.dumps({"arguments": args}, ensure_ascii=False), 10, 10, 1.0, "scripted")
        return LLMResult(None, 10, 0, 0.0, "scripted")   # everything else falls back to the deterministic path


def test_debate_numbers_come_from_records_not_from_the_model():
    rep = run_research(ResearchRequest(ticker="990001.SZ", as_of=A), "B4", llm=ScriptedLLM(), persist=False)
    filtered = rep["debate"].get("filtered", [])
    reasons = {f["reason"] for f in filtered}
    assert "contains digits" in reasons and "no valid evidence id" in reasons
    bull = [c for c in rep["claims"].values() if c["produced_by"] == "bull" and c["claim_type"] == "model_inference"]
    assert bull, "the clean argument should be kept"
    for c in bull:
        text = re.sub(r"（依据：[^）]*）", "", c["text"])   # the basis is attached by code from the cited records
        assert not DIGIT.search(text), text
        assert "20%" not in c["text"]
