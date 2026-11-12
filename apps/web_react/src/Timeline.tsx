import { useEffect, useMemo, useState } from "react";
import { api, type Span } from "./api";

interface Bar { name: string; start: number; dur: number; depth: number; note: string; err?: boolean }

const colour = (n: string) =>
  n.startsWith("subgraph") ? "bg-sky-500" : n === "verifier" || n === "rollback_rewrite" ? "bg-amber-500" :
  n === "debate" || n === "risk" ? "bg-violet-500" : n === "planner" ? "bg-emerald-500" : "bg-slate-500";

/** Gantt of one run. Source 1: the run's own node trace (always there). Source 2: OpenTelemetry spans, when the
 *  server runs with TRACING_BACKEND=otel (LLM calls and tool calls appear as child spans). */
export default function Timeline({ runId, trace }: { runId: string; trace: Span[] }) {
  const [otel, setOtel] = useState<{ enabled: boolean; spans: any[] } | null>(null);
  const [src, setSrc] = useState<"trace" | "otel">("trace");
  useEffect(() => { api(`/api/otel/spans?run_id=${runId}`).then(setOtel).catch(() => setOtel(null)); }, [runId]);

  const bars = useMemo<Bar[]>(() => {
    if (src === "otel" && otel?.spans.length) {
      const t0 = Math.min(...otel.spans.map((s) => s.start_ns));
      const depth = new Map<number, number>();
      const byId = new Map(otel.spans.map((s) => [s.span_id, s]));
      const d = (s: any): number => depth.get(s.span_id) ??
        (depth.set(s.span_id, s.parent && byId.get(s.parent) ? d(byId.get(s.parent)) + 1 : 0), depth.get(s.span_id)!);
      return otel.spans.sort((a, b) => a.start_ns - b.start_ns).map((s) => ({
        name: s.name.replace("langgraph.node.", ""), start: (s.start_ns - t0) / 1e6, dur: (s.end_ns - s.start_ns) / 1e6,
        depth: d(s), note: s.attributes?.["gen_ai.request.model"] ?? "", err: s.attributes?.error }));
    }
    const rows = trace.filter((s) => (s as any).started_at && s.latency_ms !== undefined);
    if (!rows.length) return [];
    const t0 = Math.min(...rows.map((s) => Date.parse((s as any).started_at)));
    return rows.map((s) => ({
      name: s.node, start: Date.parse((s as any).started_at) - t0, dur: s.latency_ms ?? 0, depth: 0,
      note: `${s.summary ?? ""}`.slice(0, 90), err: s.status !== undefined && s.status !== "ok"
    })).sort((a, b) => a.start - b.start);
  }, [src, otel, trace]);

  const total = Math.max(...bars.map((b) => b.start + b.dur), 1);
  return (
    <div className="card mt-3">
      <div className="flex items-center justify-between">
        <h3 className="h3">Trace 时间线（总 {(total / 1000).toFixed(1)}s）</h3>
        <span className="text-xs">
          <button className={src === "trace" ? "font-semibold text-blue-600" : "text-slate-500"} onClick={() => setSrc("trace")}>节点轨迹</button>
          {" · "}
          <button disabled={!otel?.spans.length} className={src === "otel" ? "font-semibold text-blue-600" : "text-slate-500 disabled:opacity-40"}
                  title={otel?.enabled ? "" : "服务端以 TRACING_BACKEND=otel 启动后可用"} onClick={() => setSrc("otel")}>OpenTelemetry</button>
        </span>
      </div>
      {bars.length === 0 && <p className="text-xs text-slate-500">这次运行没有时间戳（旧版本落库的报告）。</p>}
      <div className="space-y-0.5 text-[11px]">
        {bars.map((b, i) => (
          <div key={i} className="flex items-center gap-2" title={b.note}>
            <span className="w-40 shrink-0 truncate text-right text-slate-500" style={{ paddingRight: b.depth * 8 }}>{b.name}</span>
            <div className="relative h-3.5 flex-1 rounded bg-slate-100 dark:bg-slate-800">
              <div className={`absolute h-3.5 rounded ${b.err ? "bg-red-500" : colour(b.name)}`}
                   style={{ left: `${(b.start / total) * 100}%`, width: `${Math.max((b.dur / total) * 100, 0.4)}%` }} />
            </div>
            <span className="w-14 shrink-0 text-right tabular-nums">{b.dur >= 1000 ? (b.dur / 1000).toFixed(1) + "s" : b.dur.toFixed(0) + "ms"}</span>
          </div>))}
      </div>
    </div>
  );
}
