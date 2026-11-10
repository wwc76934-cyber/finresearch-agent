import { useEffect, useMemo, useState } from "react";
import { api, pct } from "./api";
import { Chart } from "./Chart";

const BASE = ["B0", "B1", "B2", "B3", "B4"];
const COLS = ["pass_rate", "numerical_accuracy", "citation_validity", "support_rate", "pit_violation", "anomaly_recall",
  "refusal", "conflict_handling", "recovery", "rubric", "redundant_tool_rate"];

export default function Evaluation({ onReplay }: { onReplay: (runId: string) => void }) {
  const [s, setS] = useState<any>(null);
  const [err, setErr] = useState("");
  const [arch, setArch] = useState("B3");
  const [fails, setFails] = useState<any[]>([]);
  useEffect(() => { api("/api/evaluation/summary").then(setS).catch(() => setErr("尚无评测结果：运行 python scripts/run_experiments.py")); }, []);
  useEffect(() => {
    api(`/api/evaluation/results?architecture=${encodeURIComponent(arch)}&failed_only=true`).then(setFails).catch(() => setFails([]));
  }, [arch]);

  const charts = useMemo(() => {
    if (!s) return null;
    const agg = s.comparison.aggregate, d = s.comparison.delta_vs_reference;
    const base = BASE.filter((a) => agg[a]);
    const ms = ["pass_rate", "numerical_accuracy", "pit_violation", "recovery", "rubric"];
    const abl = Object.keys(d).filter((a) => a.startsWith("B4-") && d[a].pass_rate !== null);
    const buckets = Object.keys(agg.B4.errors);
    return {
      base: { tooltip: {}, legend: { data: ms }, xAxis: { type: "category", data: base }, yAxis: { type: "value", max: 1 },
        series: ms.map((m) => ({ name: m, type: "bar", data: base.map((a) => agg[a][m]) })) },
      abl: { tooltip: {}, grid: { bottom: 80 }, xAxis: { type: "category", data: abl.map((a) => a.replace("B4-", "")), axisLabel: { rotate: 30 } },
        yAxis: { type: "value", name: "pp" }, series: [{ type: "bar", data: abl.map((a) => +(d[a].pass_rate * 100).toFixed(2)) }] },
      pareto: { tooltip: { formatter: (p: any) => p.data[2] }, xAxis: { type: "value", name: "tokens" }, yAxis: { type: "value", name: "pass", max: 1 },
        series: [{ type: "scatter", symbolSize: 14, label: { show: true, formatter: (p: any) => p.data[2] },
          data: Object.entries(agg).map(([a, r]: any) => [r.tokens_mean, r.pass_rate, a]) }] },
      err: { tooltip: {}, legend: { data: buckets }, grid: { bottom: 80 }, xAxis: { type: "category", data: Object.keys(agg), axisLabel: { rotate: 30 } },
        yAxis: { type: "value" }, series: buckets.map((b) => ({ name: b, type: "bar", stack: "e", data: Object.keys(agg).map((a) => agg[a].errors[b]) })) },
    };
  }, [s]);

  if (err) return <p className="p-6 text-sm text-slate-500">{err}</p>;
  if (!s || !charts) return <p className="p-6 text-sm text-slate-500">加载中…</p>;
  const agg = s.comparison.aggregate;
  return (
    <main className="space-y-3 p-3">
      <div className="grid grid-cols-2 gap-3">
        <div className="card"><h3 className="h3">B0–B4 基线对比</h3><Chart option={charts.base} /></div>
        <div className="card"><h3 className="h3">消融：关闭单一机制后的通过率变化 (B4 − X)</h3><Chart option={charts.abl} /></div>
        <div className="card"><h3 className="h3">质量 – 成本（tokens）</h3><Chart option={charts.pareto} /></div>
        <div className="card"><h3 className="h3">错误根因漏斗</h3><Chart option={charts.err} /></div>
      </div>
      <div className="card overflow-auto"><h3 className="h3">指标总表（离线实验 {s.experiment_id}）</h3>
        <table className="tbl"><thead><tr><th>架构</th>{COLS.map((c) => <th key={c}>{c}</th>)}<th>P50ms</th><th>P95ms</th><th>tokens</th></tr></thead>
          <tbody>{Object.entries(agg).map(([a, r]: any) => (
            <tr key={a}><td>{a}</td>{COLS.map((c) => <td key={c}>{pct(r[c])}</td>)}
              <td>{r.latency_p50_ms.toFixed(0)}</td><td>{r.latency_p95_ms.toFixed(0)}</td><td>{r.tokens_mean.toFixed(0)}</td></tr>))}
          </tbody></table></div>
      <div className="card"><h3 className="h3">失败样本（点击回放 Trace；仅持久化的 B3/B4 运行可回放）</h3>
        <select className="field w-60" value={arch} onChange={(e) => setArch(e.target.value)}>{Object.keys(agg).map((a) => <option key={a}>{a}</option>)}</select>
        {fails.length ? (
          <table className="tbl mt-2"><thead><tr><th>任务</th><th>题型</th><th>根因</th><th>分数</th></tr></thead><tbody>
            {fails.map((r) => (
              <tr key={r.task_id + r.run_id}>
                <td>{r.run_id ? <a href="#" className="text-blue-600 dark:text-blue-400" onClick={(e) => { e.preventDefault(); onReplay(r.run_id); }}>{r.task_id}</a> : r.task_id}</td>
                <td>{r.category}</td><td>{r.error_category}</td><td className="font-mono text-[11px]">{JSON.stringify(r.scores)}</td></tr>))}
          </tbody></table>) : <p className="mt-2 text-sm text-slate-500">该架构无失败样本。</p>}
      </div>
      <div className="card"><h3 className="h3">检索评测</h3>
        <table className="tbl"><thead><tr><th>配置</th><th>Recall@5</th><th>MRR</th></tr></thead><tbody>
          {Object.entries(s.rag).filter(([k]) => k !== "table_integrity").map(([k, v]: any) => (
            <tr key={k}><td>{k}</td><td>{pct(v["recall@5"])}</td><td>{v.mrr.toFixed(3)}</td></tr>))}
        </tbody></table>
        <p className="mt-2 text-xs">跨页表格完整率：结构感知 {pct(s.rag.table_integrity.structure_aware)}，固定长度切片 {pct(s.rag.table_integrity.naive_fixed_size)}</p>
      </div>
    </main>
  );
}
