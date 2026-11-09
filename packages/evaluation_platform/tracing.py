"""Trace Collector: node spans, tool calls, LLM usage. Persists to agent_executions and
streams each event to an optional listener (the API's SSE channel)."""
from __future__ import annotations

import contextvars
import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Optional

from data_platform.db import AgentExecutionRow, get_session_factory


class Tracer:
    def __init__(self, run_id: str, listener: Optional[Callable[[dict], None]] = None):
        self.run_id = run_id
        self.listener = listener
        self.spans: list[dict] = []
        self._ctxvar: contextvars.ContextVar = contextvars.ContextVar(f"trace_{run_id}", default=())
        self._lock = threading.Lock()
        self.seq = 0

    def _get_stack(self) -> tuple:
        return self._ctxvar.get()

    def _emit(self, ev: dict) -> None:
        if self.listener:
            try:
                self.listener(ev)
            except Exception:  # noqa: BLE001 - a broken listener must not break a run
                pass

    @contextmanager
    def node(self, name: str):
        with self._lock:
            seq = self.seq
            self.seq += 1
        span = {"node": name, "seq": seq, "started_at": datetime.now(), "t0": time.perf_counter(),
                "tool_calls": [], "llm_calls": [], "input_tokens": 0, "output_tokens": 0, "anomaly_flags": [],
                "summary": "", "status": "ok"}
        token = self._ctxvar.set(self._get_stack() + (span,))
        self._emit({"type": "node_start", "node": name, "seq": span["seq"]})
        try:
            yield span
        except Exception as exc:
            span["status"] = "error"
            span["summary"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span["latency_ms"] = (time.perf_counter() - span.pop("t0")) * 1000
            self._ctxvar.reset(token)
            with self._lock:
                self.spans.append(span)
            self._emit({"type": "node_end", "node": name, "seq": span["seq"], "latency_ms": round(span["latency_ms"], 2),
                        "tool_calls": len(span["tool_calls"]), "tokens": span["input_tokens"] + span["output_tokens"],
                        "summary": span["summary"], "anomalies": span["anomaly_flags"], "status": span["status"]})

    def _cur(self) -> Optional[dict]:
        st = self._get_stack()
        return st[-1] if st else None

    def tool(self, name: str, args: dict, result: Any = None, error: Optional[str] = None) -> None:
        span = self._cur()
        rec = {"tool": name, "args": json.loads(json.dumps(args, default=str)), "error": error,
               "result": result if isinstance(result, (str, int, float)) or result is None else str(result)[:300]}
        if span is not None:
            span["tool_calls"].append(rec)
        self._emit({"type": "tool", "node": span["node"] if span else None, **rec})

    def llm(self, node: str, res, tier: str) -> None:
        span = self._cur()
        if span is not None:
            span["input_tokens"] += res.input_tokens
            span["output_tokens"] += res.output_tokens
            span["llm_calls"].append({"model": res.model, "tier": tier, "in": res.input_tokens,
                                      "out": res.output_tokens, "latency_ms": res.latency_ms,
                                      "online": res.text is not None})

    def totals(self) -> dict:
        tin = sum(s["input_tokens"] for s in self.spans)
        tout = sum(s["output_tokens"] for s in self.spans)
        tools = [t for s in self.spans for t in s["tool_calls"]]
        sigs = [json.dumps([t["tool"], t["args"]], sort_keys=True, default=str) for t in tools]
        return {"input_tokens": tin, "output_tokens": tout, "tool_calls": len(tools),
                "redundant_tool_calls": len(sigs) - len(set(sigs)), "nodes": len(self.spans)}

    def persist(self, db_url: Optional[str] = None) -> None:
        with get_session_factory(db_url)() as s:
            for sp in self.spans:
                s.add(AgentExecutionRow(run_id=self.run_id, seq=sp["seq"], node=sp["node"], started_at=sp["started_at"],
                                        latency_ms=sp["latency_ms"], input_tokens=sp["input_tokens"],
                                        output_tokens=sp["output_tokens"],
                                        tool_calls=sp["tool_calls"] + [{"llm": c} for c in sp["llm_calls"]],
                                        anomaly_flags=sp["anomaly_flags"], summary=sp["summary"][:2000],
                                        status=sp["status"]))
            s.commit()

    def as_dicts(self) -> list[dict]:
        return [{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in sp.items()} for sp in
                sorted(self.spans, key=lambda x: x["seq"])]
