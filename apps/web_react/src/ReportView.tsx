import { useEffect, useState } from "react";
import { api, type Report } from "./api";
import { Chart } from "./Chart";

const TAG = { disclosed_fact: "事实", derived_metric: "算子", model_inference: "推断" } as const;

export default function ReportView({ rep }: { rep: Report }) {
  const [drawer, setDrawer] = useState<string | null>(null);
  const limits = [
    ...rep.open_questions.map((q) => `待解答：${q.topic}（${q.reason}）`),
    ...rep.uncertainties.map((u) => u.text),
    ...rep.conflicts.map((c) => `数据源冲突：${c.metric} ${c.period} 采用 ${c.chosen}`),
  ];
  return (
    <div className="text-sm">
      <div className="no-print flex items-center justify-between">
        <h3 className="h3">研究报告</h3>
        <span className="text-xs">
          <a className="text-blue-600 dark:text-blue-400" href={`/api/runs/${rep.run_id}/export.md`} target="_blank">导出 Markdown</a> ·{" "}
          <a className="text-blue-600 dark:text-blue-400" href={`/api/runs/${rep.run_id}/export.pdf`}>导出 PDF</a>
        </span>
      </div>
      <p>{rep.overview}</p>
      <p className="text-xs text-slate-500">快照 {rep.snapshot_id} · 架构 {rep.architecture} · {rep.totals?.tool_calls} 次工具调用 · {rep.totals?.latency_ms?.toFixed(0)}ms</p>
      {rep.executive_summary && <><h3 className="h3 mt-3">执行摘要</h3><p>{rep.executive_summary}</p></>}
      {rep.charts.quarterly && <Chart option={rep.charts.quarterly} />}
      {rep.sections.filter((s) => s.key !== "overview").map((s) => (
        <div key={s.key}>
          <h3 className="h3 mt-3">{s.title}</h3>
          {s.key === "falsification" && rep.debate?.disagreements?.length > 0 && (
            <table className="tbl mb-2"><thead><tr><th>看多</th><th>看空</th><th>分歧焦点</th></tr></thead><tbody>
              {rep.debate.disagreements.map((d: any, i: number) => (
                <tr key={i}><td>{rep.claims[d.bull]?.text}</td><td>{rep.claims[d.bear]?.text}</td><td>{d.issue}</td></tr>))}
            </tbody></table>)}
          {s.claim_ids.map((cid) => {
            const c = rep.claims[cid];
            return (
              <div key={cid} className="my-1">
                <span className="mr-1 rounded bg-slate-100 px-1.5 text-[11px] text-slate-500 dark:bg-slate-800">{TAG[c.claim_type]}</span>
                {c.text}
                {c.evidence_ids.map((e) => (
                  <button key={e} onClick={() => setDrawer(e)}
                    className="ml-1 rounded bg-blue-100 px-1.5 text-[11px] text-blue-700 dark:bg-blue-950 dark:text-blue-300">#{e}</button>))}
              </div>);
          })}
          {s.key === "valuation" && rep.charts.valuation && <Chart option={rep.charts.valuation} />}
          {s.key === "risk" && rep.charts.risk_radar && <Chart option={rep.charts.risk_radar} />}
        </div>))}
      <h3 className="h3 mt-3">数据局限与不确定性声明</h3>
      {limits.length ? limits.map((x, i) => <div key={i} className="my-1">{x}</div>) : <p className="text-slate-500">无</p>}
      {drawer && <EvidenceDrawer rep={rep} eid={drawer} onClose={() => setDrawer(null)} />}
    </div>
  );
}

function EvidenceDrawer({ rep, eid, onClose }: { rep: Report; eid: string; onClose: () => void }) {
  const [d, setD] = useState<any>(null);
  useEffect(() => {
    setD(null);
    api(`/api/runs/${rep.run_id}/lineage/${eid}`).then(setD);
  }, [rep.run_id, eid]);
  const ev = d?.evidence ?? {};
  const metric = (ev.metric_ids ?? []).map((m: string) => rep.metrics[m]).find(Boolean);
  const users = Object.values(rep.claims).filter((c) => c.evidence_ids.includes(eid)).map((c) => c.claim_id);
  return (
    <aside className="no-print fixed inset-y-0 right-0 z-10 w-[520px] overflow-auto border-l border-slate-200 bg-white p-4 shadow-2xl dark:border-slate-700 dark:bg-slate-900">
      <div className="flex items-center justify-between"><h3 className="h3">证据 #{eid}</h3><button className="btn-ghost" onClick={onClose}>关闭</button></div>
      {!d ? <p className="text-slate-500">加载中…</p> : <>
        <div className={`rounded px-2 py-1 text-xs ${d.pit_ok ? "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300" : "bg-red-100 text-red-700"}`}>
          PIT {d.pit_ok ? "合规" : "违规"}：available_at {ev.available_at} {d.pit_ok ? "≤" : ">"} as_of {d.as_of}
        </div>
        {metric && <p className="mt-2 text-xs"><b>算子</b> {metric.metric} = {String(metric.value)} ({metric.status})<br />
          <span className="text-slate-500">{metric.formula} · {metric.formula_version}</span><br />输入：{metric.input_ids.join(", ")}</p>}
        <h3 className="h3 mt-3">血缘链</h3>
        <div className="space-y-1 text-xs">
          <div className="rounded border border-slate-200 p-2 dark:border-slate-700">报告结论 Claim：{users.join(", ")}</div>
          <div className="text-center text-slate-400">↑</div>
          <div className="rounded border border-slate-200 p-2 dark:border-slate-700">证据卡 {eid}：{ev.summary}</div>
          {d.lineage.edges.map((e: any, i: number) => (
            <div key={i}><div className="text-center text-slate-400">↑</div>
              <div className="rounded border border-slate-200 p-2 dark:border-slate-700">{e.parent_type} <b>{e.parent_entity_id}</b> → {e.child_type} {e.child_entity_id}
                <span className="text-slate-500"> ({e.transform_tool} {e.formula_version})</span></div></div>))}
        </div>
        {ev.kind === "document" && <><h3 className="h3 mt-3">原文 {ev.page}</h3>
          <pre className="whitespace-pre-wrap rounded bg-slate-100 p-2 text-xs dark:bg-slate-800">{ev.excerpt}</pre></>}
      </>}
    </aside>
  );
}
