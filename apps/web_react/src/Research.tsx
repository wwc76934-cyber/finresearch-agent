import { useEffect, useState } from "react";
import { api, post, type Approval, type Report, type Span } from "./api";
import ReportView from "./ReportView";

const NODES = ["planner", "subgraph:fundamental", "subgraph:market", "subgraph:news", "gather", "approval", "replan",
  "subgraph:investigate", "debate", "risk", "fault_injection", "verifier", "rollback_rewrite", "report"];
const nodeKey = (n: string) => NODES.find((k) => n === k || n.startsWith(k + ":")) ?? n;
type NodeState = Record<string, "running" | "done" | "error">;

export default function Research({ openRun }: { openRun: string | null }) {
  const [secs, setSecs] = useState<{ ticker: string; name: string; industry: string }[]>([]);
  const [archs, setArchs] = useState<string[]>([]);
  const [form, setForm] = useState({ ticker: "", as_of: "2025-12-31T15:00", mode: "deep", architecture: "B4",
    token_budget: 200000, question: "全面研究公司基本面、估值与风险，形成包含多空审议与证伪条件的研究报告", hitl_always: false });
  const [nodes, setNodes] = useState<NodeState>({});
  const [spans, setSpans] = useState<Span[]>([]);
  const [focus, setFocus] = useState<string | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [approval, setApproval] = useState<{ job: string; data: Approval } | null>(null);
  const [runs, setRuns] = useState<any[]>([]);

  const loadRuns = () => api("/api/runs?limit=15").then(setRuns);
  useEffect(() => {
    api("/api/securities").then((s) => { setSecs(s); setForm((f) => ({ ...f, ticker: f.ticker || s[0]?.ticker })); });
    api("/api/architectures").then(setArchs);
    loadRuns();
  }, []);
  useEffect(() => { if (openRun) open(openRun); }, [openRun]);

  async function open(runId: string) {
    try {
      const rep: Report = await api(`/api/runs/${runId}`);
      setReport(rep); setSpans(rep.trace ?? []);
      const st: NodeState = {};
      (rep.trace ?? []).forEach((s) => (st[nodeKey(s.node)] = s.status === "ok" ? "done" : "error"));
      setNodes(st); setError("");
    } catch { setError("该运行未持久化，无法回放"); }
  }

  async function start() {
    setRunning(true); setReport(null); setSpans([]); setNodes({}); setError(""); setApproval(null);
    try {
      const { job_id } = await post("/api/research", { ...form, as_of: form.as_of + ":00", hitl: true });
      const es = new EventSource(`/api/research/${job_id}/events`);
      es.onmessage = async (m) => {
        const ev = JSON.parse(m.data);
        if (ev.type === "node_start") setNodes((s) => ({ ...s, [nodeKey(ev.node)]: "running" }));
        if (ev.type === "node_end") {
          setNodes((s) => ({ ...s, [nodeKey(ev.node)]: ev.status === "ok" ? "done" : "error" }));
          setSpans((s) => [...s, ev]);
        }
        if (ev.type === "approval_required") setApproval({ job: job_id, data: ev });
        if (ev.type === "done") { es.close(); setRunning(false); setApproval(null); await open(ev.run_id); loadRuns(); }
        if (ev.type === "error") { es.close(); setRunning(false); setError(ev.message); }
      };
    } catch (e: any) { setRunning(false); setError(e.message); }
  }

  const shown = focus ? spans.filter((s) => nodeKey(s.node) === focus) : spans;
  const future = new Date(form.as_of) > new Date();
  const set = (k: string) => (e: any) => setForm({ ...form, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value });
  const replans = spans.filter((s) => s.node === "replan");

  return (
    <main className="grid h-[calc(100vh-45px)] grid-cols-[280px_1fr_1.4fr] gap-3 p-3">
      <section className="card no-print overflow-auto">
        <h3 className="h3">研究配置</h3>
        <label className="lbl">证券
          <select className="field" value={form.ticker} onChange={set("ticker")}>
            {secs.map((s) => <option key={s.ticker} value={s.ticker}>{s.ticker} {s.name} · {s.industry}</option>)}
          </select></label>
        <label className="lbl">研究时点 as_of<input className="field" type="datetime-local" value={form.as_of} onChange={set("as_of")} /></label>
        <div className={`mt-1 rounded px-2 py-1 text-xs ${future ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300" : "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300"}`}>
          {future ? "研究时点晚于当前时间：PIT 不可保证" : "PIT：仅使用 available_at ≤ as_of 的数据"}
        </div>
        <label className="lbl">模式<select className="field" value={form.mode} onChange={set("mode")}><option value="deep">深度研究</option><option value="fast">快速分析</option></select></label>
        <label className="lbl">架构<select className="field" value={form.architecture} onChange={set("architecture")}>{archs.map((a) => <option key={a}>{a}</option>)}</select></label>
        <label className="lbl">Token 预算<input className="field" type="number" step={10000} value={form.token_budget} onChange={set("token_budget")} /></label>
        <label className="lbl flex items-center gap-2"><input type="checkbox" checked={form.hitl_always} onChange={set("hitl_always")} />每次追加调查前都请我审批</label>
        <label className="lbl">研究问题<textarea className="field" rows={5} value={form.question} onChange={set("question")} /></label>
        <button className="btn mt-2 w-full" disabled={running || !form.ticker} onClick={start}>{running ? "运行中…" : "开始研究"}</button>
        <h3 className="h3 mt-4">Research Plan</h3>
        <ol className="list-decimal pl-5 text-xs">
          {report?.plan?.subtasks.map((t, i) => <li key={i}>{t.agent}：{t.objective}</li>)}
          {replans.map((s, i) => <li key={"r" + i} className="text-amber-700 dark:text-amber-400">动态追加：{s.summary}</li>)}
        </ol>
        <h3 className="h3 mt-4">历史运行</h3>
        <ul className="text-xs">{runs.map((r) => (
          <li key={r.run_id}><a href="#" className="text-blue-600 dark:text-blue-400" onClick={(e) => { e.preventDefault(); open(r.run_id); }}>
            {r.ticker} {String(r.as_of).slice(0, 10)} {r.architecture}</a></li>))}
        </ul>
      </section>

      <section className="card no-print overflow-auto">
        <h3 className="h3">Agent 执行拓扑</h3>
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          {NODES.map((n, i) => (
            <span key={n} className="flex items-center gap-1.5">
              {i > 0 && <span className="text-slate-400">→</span>}
              <button onClick={() => setFocus(focus === n ? null : n)}
                className={`rounded-full border px-2.5 py-1 ${nodes[n] === "running" ? "border-blue-500 text-blue-600" :
                  nodes[n] === "done" ? "border-green-600 text-green-700 dark:text-green-400" :
                  nodes[n] === "error" ? "border-red-500 text-red-600" : "border-slate-300 text-slate-400 dark:border-slate-600"} ${focus === n ? "ring-2 ring-blue-300" : ""}`}>
                {n.replace("subgraph:", "")}
              </button>
            </span>))}
        </div>
        {approval && <ApprovalCard job={approval.job} data={approval.data} onSent={() => setApproval(null)} />}
        <h3 className="h3 mt-4">节点轨迹{focus && <span className="ml-2 text-xs font-normal text-slate-500">（仅 {focus}，再点一次取消）</span>}</h3>
        <div className="text-xs">{shown.map((s, i) => <SpanRow key={i} s={s} />)}</div>
      </section>

      <section className="card overflow-auto">
        {error && <p className="text-sm text-red-600">{error}</p>}
        {report ? <ReportView rep={report} /> : !error && <p className="text-sm text-slate-500">{running ? "运行中…" : "选择证券并开始研究。"}</p>}
      </section>
    </main>
  );
}

function SpanRow({ s }: { s: Span }) {
  const calls = Array.isArray(s.tool_calls) ? s.tool_calls : [];
  const tokens = (s.input_tokens ?? 0) + (s.output_tokens ?? 0) || s.tokens || 0;
  return (
    <div className="my-1.5 border-l-[3px] border-slate-300 py-0.5 pl-2 dark:border-slate-600">
      <b>{s.node}</b> · {(s.latency_ms ?? 0).toFixed(1)}ms · tools {calls.length || (s.tool_calls as number) || 0} · tokens {tokens}
      <div className="text-slate-500">{s.summary}</div>
      {calls.map((t, i) => (
        <div key={i} className="font-mono text-[11px]">
          {t.tool ?? (t.llm ? "llm:" + t.llm.model : "")}({JSON.stringify(t.args ?? {})}){" "}
          {t.error ? <span className="text-red-600">{t.error}</span> : <span className="text-slate-500">{String(t.result ?? "")}</span>}
        </div>))}
    </div>
  );
}

function ApprovalCard({ job, data, onSent }: { job: string; data: Approval; onSent: () => void }) {
  const [query, setQuery] = useState("");
  const [extra, setExtra] = useState(20000);
  const [sent, setSent] = useState(false);
  const send = async (action: string) => {
    if (action === "modify" && !query) return;
    setSent(true);
    await post(`/api/research/${job}/approve`, { action, query: query || null, extra_tokens: extra });
    onSent();
  };
  const why = data.reason === "over_budget" ? `已用 ${data.used_tokens} tokens，超过预算 ${data.token_budget}` : "按设置，追加调查前需要审批";
  return (
    <div className="mt-3 rounded-lg border-2 border-amber-500 bg-amber-50 p-3 text-sm dark:bg-amber-950/40" role="dialog" aria-label="需要你的审批">
      <h3 className="h3">需要你的审批</h3>
      <p>{why}。系统想追查以下异常：</p>
      {data.proposed.map((p) => (
        <div key={p.code} className="my-1">• <b>{p.code}</b>：{p.message}<div className="text-xs text-slate-500">检索：{p.query}</div></div>))}
      <label className="lbl">改为检索（可选）<input className="field" maxLength={80} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="例如：应收账款 账龄 坏账准备" /></label>
      <label className="lbl">追加 token 预算<input className="field" type="number" step={10000} value={extra} onChange={(e) => setExtra(+e.target.value)} /></label>
      <div className="mt-2 flex gap-2">
        <button className="btn" disabled={sent} onClick={() => send("approve")}>批准</button>
        <button className="btn-ghost" disabled={sent || !query} onClick={() => send("modify")}>按新检索执行</button>
        <button className="btn-ghost" disabled={sent} onClick={() => send("skip")}>跳过，直接出报告</button>
      </div>
    </div>
  );
}
