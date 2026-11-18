import { type StreamEvent } from "./stream";

const ROUTE_CN: Record<string, string> = { fast: "快速：查数或拒答", standard: "标准：单主题分析", full: "完整：多空辩论与风险" };
const STATUS_COLOUR: Record<string, string> = {
  CONFIRMED: "text-green-700 dark:text-green-400", WEAKENED: "text-amber-700 dark:text-amber-400",
  FALSIFIED: "text-red-600", UNKNOWN: "text-slate-500",
};
const STATUS_CN: Record<string, string> = { CONFIRMED: "得到强化", WEAKENED: "边际弱化", FALSIFIED: "已触发证伪", UNKNOWN: "数据不足" };

const of = (events: StreamEvent[], type: string) => events.filter((e) => e.type === type);

/** What the run is doing right now, built only from stream events. */
export default function LiveFeed({ events, running }: { events: StreamEvent[]; running: boolean }) {
  if (!events.length) return null;
  const started = of(events, "run_started")[0];
  const ingest = of(events, "live_ingestion_status");
  const plan = of(events, "plan_generated").at(-1);
  const tools = events.filter((e) => e.type === "subgraph_progress" && e.phase === "tool");
  const anomalies = of(events, "anomaly_detected");
  const debate = of(events, "debate_round");
  const audits = of(events, "verifier_audit");
  const review = of(events, "thesis_review").at(-1);
  const last = ingest.at(-1);
  const fetching = running && ingest.length > 0 && !ingest.some((e) => e.status) && !plan;

  return (
    <div className="mt-3 space-y-2 text-xs" aria-live="polite">
      {started && <p className="text-slate-500">运行 {started.run_id} · {started.tickers.join("、")} · {started.data_mode}</p>}

      {ingest.length > 0 && (
        <details open className="rounded border border-slate-200 p-2 dark:border-slate-700">
          <summary className="cursor-pointer font-semibold">在线数据同步</summary>
          <div className="mt-1 h-1.5 overflow-hidden rounded bg-slate-200 dark:bg-slate-700" role="progressbar" aria-label="数据同步">
            <div className={`h-full bg-blue-500 ${fetching ? "w-1/2 animate-pulse" : "w-full"}`} />
          </div>
          {ingest.slice(-5).map((e, i) => <div key={i} className="mt-0.5">{e.message}{e.status && <span className="ml-1 text-slate-500">[{e.status}]</span>}</div>)}
          {last?.counts && <div className="text-slate-500">{Object.entries(last.counts).filter(([, v]) => typeof v !== "object").map(([k, v]) => `${k} ${v}`).join(" · ")}</div>}
        </details>)}

      {plan && (
        <details open className="rounded border border-slate-200 p-2 dark:border-slate-700">
          <summary className="cursor-pointer font-semibold">
            研究计划 <span className="ml-1 rounded bg-slate-100 px-1.5 dark:bg-slate-800">{ROUTE_CN[plan.route] ?? plan.route}</span>
          </summary>
          <div className="text-slate-500">{(plan.route_reasons ?? []).join("；")}</div>
          {plan.hypotheses?.length > 0 && <ol className="mt-1 list-decimal pl-5">{plan.hypotheses.map((h: any) => (
            <li key={h.sub_id}><b>{h.sub_id}</b> {h.hypothesis_statement}<span className="text-slate-500">（{h.custom_rag_queries.join(" / ")}）</span></li>))}</ol>}
        </details>)}

      {tools.length > 0 && (
        <details className="rounded border border-slate-200 p-2 dark:border-slate-700">
          <summary className="cursor-pointer font-semibold">子图自主取证（{tools.length}）</summary>
          {tools.slice(-12).map((e, i) => (
            <div key={i} className="font-mono text-[11px]">{e.agent} · {e.tool}({e.args?.query ?? e.args?.sub_id ?? ""}) → {e.error ?? e.result}</div>))}
        </details>)}

      {anomalies.length > 0 && (
        <div className="rounded border border-amber-400 bg-amber-50 p-2 dark:bg-amber-950/40">
          <b>检出异常</b>{anomalies.map((a, i) => (
            <div key={i}>• {a.code} <span className={a.severity === "HIGH" ? "text-red-600" : "text-amber-700"}>{a.severity}</span>
              {a.action === "replan" && " → 追加调查"}<span className="text-slate-500"> {a.message}</span></div>))}
        </div>)}

      {debate.length > 0 && (
        <details className="rounded border border-slate-200 p-2 dark:border-slate-700">
          <summary className="cursor-pointer font-semibold">多空辩论（{debate.filter((d) => typeof d.round === "number").length} 轮）</summary>
          {debate.map((d, i) => typeof d.round === "number" ? (
            <div key={i} className="mt-1">第 {d.round} 轮：看多 {d.bull.length} 条 · 看空 {d.bear.length} 条</div>) : (
            <div key={i} className="mt-1">
              <div>证伪条件：{(d.falsification ?? []).length} 条 · Fact Lock 拒绝 {d.fact_lock?.rejected_arguments ?? 0} 条带数字或无证据的论点</div>
              {(d.falsification ?? []).map((f: string, j: number) => <div key={j} className="text-slate-500">• {f}</div>)}
            </div>))}
        </details>)}

      {audits.length > 0 && (
        <div className="rounded border border-slate-200 p-2 dark:border-slate-700">
          <b>校验</b>{audits.map((a, i) => (
            <div key={i}>第 {a.pass} 轮：{a.claims} 条论断，失败 {a.failures.length}
              {Object.keys(a.failures_by_type).length > 0 && `（${Object.entries(a.failures_by_type).map(([k, v]) => `${k}×${v}`).join("，")}）`}
              ，降级 {a.downgraded} {a.action === "pass" ? "✓" : "→ 局部重写"}</div>))}
        </div>)}

      {review && review.items?.length > 0 && <ThesisCard review={{ items: review.items, theses: review.theses, notes: review.notes }} />}
    </div>
  );
}

export function ThesisCard({ review }: { review: { items: any[]; theses?: any[]; notes?: string[] } }) {
  return (
    <div className="rounded border-2 border-violet-400 p-2" role="region" aria-label="历史投研观点复盘">
      <b>历史投研观点复盘</b>
      {(review.theses ?? []).map((t) => (
        <div key={t.thesis_id} className="mt-1">
          {t.as_of_date} 的研究（{t.stance === "bullish" ? "偏多" : t.stance === "bearish" ? "偏空" : "中性"}，{t.from_period} → {t.to_period}）：
          <b className={STATUS_COLOUR[t.overall]}> {STATUS_CN[t.overall] ?? t.overall}</b>
          <div className="text-slate-500">{t.summary}</div>
          {review.items.filter((i) => i.thesis_id === t.thesis_id).map((i) => (
            <div key={i.claim_id}>• {i.label}：<span className={STATUS_COLOUR[i.status]}>{STATUS_CN[i.status] ?? i.status}</span></div>))}
        </div>))}
      {(review.notes ?? []).map((n, i) => <div key={i} className="text-slate-500">{n}</div>)}
    </div>
  );
}
