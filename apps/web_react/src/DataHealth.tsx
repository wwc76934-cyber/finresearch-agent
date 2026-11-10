import { useEffect, useState } from "react";
import { api, pct } from "./api";

export default function DataHealth() {
  const [h, setH] = useState<any>(null);
  useEffect(() => { api("/api/data-health").then(setH); }, []);
  if (!h) return <p className="p-6 text-sm text-slate-500">加载中…</p>;
  return (
    <main className="space-y-3 p-3">
      <div className="card"><h3 className="h3">数据覆盖与重述</h3>
        <table className="tbl"><thead><tr><th>证券</th><th>事实数</th><th>最新报告期</th><th>重述记录</th><th>pit_unsafe</th><th>文档</th></tr></thead><tbody>
          {h.coverage.map((c: any) => (
            <tr key={c.ticker}><td>{c.ticker}</td><td>{c.facts}</td><td>{c.latest_period_end}</td><td>{c.restated_facts}</td><td>{c.pit_unsafe_facts}</td><td>{c.documents}</td></tr>))}
        </tbody></table></div>
      <div className="card"><h3 className="h3">数据质量校验</h3>
        <p className="text-sm">校验 {h.quality.facts_checked} 条事实，ERROR 级通过率 {pct(h.quality.pass_rate)}。规则：{h.quality.rules.join("、")}</p>
        <table className="tbl mt-2"><thead><tr><th>规则</th><th>命中</th></tr></thead><tbody>
          {Object.entries(h.quality.issues).map(([k, v]: any) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
        </tbody></table>
        <pre className="mt-2 whitespace-pre-wrap rounded bg-slate-100 p-2 text-xs dark:bg-slate-800">
          {h.quality.samples.slice(0, 5).map((s: any) => `${s.severity} ${s.rule} ${s.entity_id}: ${s.message}`).join("\n")}
        </pre></div>
      <div className="card"><h3 className="h3">DataSnapshotManifest</h3>
        <table className="tbl"><thead><tr><th>snapshot_id</th><th>as_of</th><th>行数</th><th>manifest_hash</th></tr></thead><tbody>
          {h.snapshots.map((s: any) => (
            <tr key={s.snapshot_id}><td>{s.snapshot_id}</td><td>{s.as_of}</td><td className="font-mono text-[11px]">{JSON.stringify(s.row_counts)}</td>
              <td className="font-mono">{s.manifest_hash.slice(0, 16)}</td></tr>))}
        </tbody></table></div>
    </main>
  );
}
