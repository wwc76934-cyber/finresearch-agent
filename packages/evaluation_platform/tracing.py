"""Trace Collector: node spans, tool calls, LLM usage. Persists to agent_executions and
streams each event to an optional listener (the API's SSE channel)."""
from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Optional

from data_platform.db import AgentExecutionRow, get_session_factory


def tracing_backend() -> str:
    """TRACING_BACKEND=json|otel. json: the in-process trace only (always written). otel: also emit
    OpenTelemetry spans; they go to OTEL_EXPORTER_OTLP_ENDPOINT if set, otherwise they are kept in memory."""
    return os.environ.get("TRACING_BACKEND", "json").lower()


_OTEL: dict = {}
_OTEL_LOCK = threading.Lock()


def _otel_tracer():
    with _OTEL_LOCK:
        if "tracer" not in _OTEL:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

            provider = TracerProvider(resource=Resource.create({"service.name": "finresearch-agent"}))
            endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            if endpoint:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
                _OTEL["exporter"] = None
            else:
                _OTEL["exporter"] = InMemorySpanExporter()
                provider.add_span_processor(SimpleSpanProcessor(_OTEL["exporter"]))
            _OTEL["provider"] = provider
            _OTEL["tracer"] = provider.get_tracer("finresearch.evaluation_platform")
        return _OTEL["tracer"]


def finished_otel_spans() -> list:
    """Spans captured in memory (TRACING_BACKEND=otel without an OTLP endpoint). Used by tests and the UI."""
    _otel_tracer()
    ex = _OTEL.get("exporter")
    return list(ex.get_finished_spans()) if ex is not None else []


def clear_otel_spans() -> None:
    _otel_tracer()
    if _OTEL.get("exporter") is not None:
        _OTEL["exporter"].clear()


def flush_otel() -> None:
    if "provider" in _OTEL:
        _OTEL["provider"].force_flush()


def _attr(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)):
        return v
    return json.dumps(v, ensure_ascii=False, default=str)[:1000]


class Tracer:
    def __init__(self, run_id: str, listener: Optional[Callable[[dict], None]] = None):
        self.run_id = run_id
        self.listener = listener
        self._otel = None
        self._otel_spans: dict[int, Any] = {}
        self._root = None
        if tracing_backend() == "otel":
            self._otel = _otel_tracer()
            self._root = self._otel.start_span("research_run", attributes={"run_id": run_id})
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
        parent = self._otel_parent()
        if self._otel is not None:
            self._otel_spans[seq] = self._otel.start_span(f"langgraph.node.{name}", context=parent,
                                                          attributes={"run_id": self.run_id, "node": name})
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
            osp = self._otel_spans.pop(seq, None)
            if osp is not None:
                self._finish_otel_node(osp, span)
            with self._lock:
                self.spans.append(span)
            self._emit({"type": "node_end", "node": name, "seq": span["seq"], "latency_ms": round(span["latency_ms"], 2),
                        "tool_calls": len(span["tool_calls"]), "tokens": span["input_tokens"] + span["output_tokens"],
                        "summary": span["summary"], "anomalies": span["anomaly_flags"], "status": span["status"]})

    def _cur(self) -> Optional[dict]:
        st = self._get_stack()
        return st[-1] if st else None

    def _otel_parent(self):
        """Context whose current span is the innermost open node span (or the root span)."""
        if self._otel is None:
            return None
        from opentelemetry import trace

        cur = self._cur()
        parent = self._otel_spans.get(cur["seq"]) if cur is not None else None
        return trace.set_span_in_context(parent or self._root)

    def _finish_otel_node(self, osp, span: dict) -> None:
        from opentelemetry.trace import Status, StatusCode

        osp.set_attribute("latency_ms", round(span["latency_ms"], 2))
        osp.set_attribute("gen_ai.usage.prompt_tokens", int(span["input_tokens"]))
        osp.set_attribute("gen_ai.usage.completion_tokens", int(span["output_tokens"]))
        osp.set_attribute("tool_calls", len(span["tool_calls"]))
        osp.set_attribute("anomaly_flags", _attr(span["anomaly_flags"]))
        if span.get("summary"):
            osp.set_attribute("summary", span["summary"][:500])
        if span["status"] == "error":
            osp.set_status(Status(StatusCode.ERROR, span["summary"][:200]))
        osp.end()

    def _instant_span(self, name: str, attrs: dict, latency_ms: float = 0.0, error: Optional[str] = None) -> None:
        """A finished child span ending now and lasting latency_ms (the call has already returned)."""
        if self._otel is None:
            return
        from opentelemetry.trace import Status, StatusCode

        end = time.time_ns()
        sp = self._otel.start_span(name, context=self._otel_parent(), start_time=end - int(latency_ms * 1e6),
                                   attributes={k: _attr(v) for k, v in {**attrs, "latency_ms": round(latency_ms, 2)}.items()})
        if error:
            sp.set_status(Status(StatusCode.ERROR, error[:200]))
        sp.end(end_time=end)

    def span_event(self, name: str, attrs: dict, latency_ms: float = 0.0) -> None:
        """Extra span for something that is not a graph node, tool or LLM call (e.g. verifier.check)."""
        self._instant_span(name, attrs, latency_ms)

    def close(self, **attrs) -> None:
        """End the root research_run span with run-level attributes."""
        if self._root is not None:
            for k, v in attrs.items():
                self._root.set_attribute(k, _attr(v))
            self._root.end()
            self._root = None
            flush_otel()

    def annotate(self, **attrs) -> None:
        if self._root is not None:
            for k, v in attrs.items():
                self._root.set_attribute(k, _attr(v))

    def tool(self, name: str, args: dict, result: Any = None, error: Optional[str] = None,
             latency_ms: float = 0.0) -> None:
        span = self._cur()
        self._instant_span(f"tool.call.{name}", {"tool": name, "args": args, "result": result if result is not None else ""},
                           latency_ms, error)
        rec = {"tool": name, "args": json.loads(json.dumps(args, default=str)), "error": error,
               "result": result if isinstance(result, (str, int, float)) or result is None else str(result)[:300]}
        if span is not None:
            span["tool_calls"].append(rec)
        self._emit({"type": "tool", "node": span["node"] if span else None, **rec})

    def llm(self, node: str, res, tier: str) -> None:
        span = self._cur()
        if span is not None:
            with self._lock:  # LLM calls may run in parallel threads
                span["input_tokens"] += res.input_tokens
                span["output_tokens"] += res.output_tokens
                span["llm_calls"].append({"model": res.model, "tier": tier, "in": res.input_tokens,
                                          "out": res.output_tokens, "latency_ms": res.latency_ms,
                                          "online": res.text is not None})
        if res.model != "context":
            self._instant_span(f"llm.completion.{res.model}",
                               {"gen_ai.request.model": res.model, "tier": tier, "node": node,
                                "gen_ai.usage.prompt_tokens": res.input_tokens,
                                "gen_ai.usage.completion_tokens": res.output_tokens, "online": res.text is not None},
                               res.latency_ms, None if res.text is not None else "llm call failed or offline")

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
