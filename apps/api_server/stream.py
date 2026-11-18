"""Stream protocol for POST /api/research/stream (v2.0 Epic 3).

Every event is one JSON object on an SSE `data:` line with a `type`. The eight protocol types:

  run_started          run id, resolved companies, as_of, data_mode (sent before any work, so the first byte is immediate)
  live_ingestion_status  a company is being fetched / stored (messages, counts, status)
  plan_generated       route (fast / standard / full) and its reasons, sub-tasks, sub-hypotheses, companies
  subgraph_progress    a sub-graph started / finished, or one of its tool calls (tool, query, result, elapsed)
  anomaly_detected     an anomaly flag, and the re-plan it triggered
  debate_round         one Bull / Bear round, the falsification conditions, the Fact Lock filter count
  verifier_audit       one verification pass: claims checked, failures by type, downgrades
  report_completed     the finished report (claims, evidence, charts, lineage), the token / time bill

Two helper types keep the topology view alive: node_start / node_end (graph nodes) and thesis_review (earlier
falsification conditions re-tested). An `error` event ends a failed stream; `approval_required` pauses it for a human.
"""
from __future__ import annotations

from typing import Any

PROTOCOL_TYPES = ("run_started", "live_ingestion_status", "plan_generated", "subgraph_progress", "anomaly_detected",
                  "debate_round", "verifier_audit", "report_completed")
PASS_THROUGH = set(PROTOCOL_TYPES) | {"thesis_review", "approval_required", "done", "error"}


def to_stream_events(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Map an event from the agent runtime to protocol events. Unknown raw events are dropped, never raised."""
    t = raw.get("type")
    node = str(raw.get("node") or "")
    if t in PASS_THROUGH:
        return [raw]
    if t == "node_start":
        out: list[dict[str, Any]] = [raw]
        if node.startswith("subgraph:"):
            out.append({"type": "subgraph_progress", "phase": "start", "node": node, "agent": node.split(":")[1]})
        return out
    if t == "node_end":
        out = [raw]
        if node.startswith("subgraph:"):
            out.append({"type": "subgraph_progress", "phase": "end", "node": node, "agent": node.split(":")[1],
                        "latency_ms": raw.get("latency_ms"), "tool_calls": raw.get("tool_calls"), "summary": raw.get("summary")})
        return out
    if t == "tool" and node.startswith("subgraph:"):
        return [{"type": "subgraph_progress", "phase": "tool", "node": node, "agent": node.split(":")[1],
                 "tool": raw.get("tool"), "args": raw.get("args"), "result": str(raw.get("result"))[:160],
                 "error": raw.get("error")}]
    return []


def sse(ev: dict[str, Any]) -> str:
    import json

    return f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"


def report_event(rep: dict[str, Any]) -> dict[str, Any]:
    """The final event: the report plus the bill. The node trace is kept (the timeline view needs it)."""
    t = rep.get("totals", {})
    return {"type": "report_completed", "run_id": rep.get("run_id"), "report": rep,
            "bill": {"tokens": t.get("input_tokens", 0) + t.get("output_tokens", 0), "cost_cny": t.get("cost_cny"),
                     "latency_ms": t.get("latency_ms"), "llm_calls": t.get("llm_calls"), "tool_calls": t.get("tool_calls")}}
