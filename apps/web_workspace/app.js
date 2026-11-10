const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const api = async (p, opt) => { const r = await fetch(p, opt); if (!r.ok) throw new Error(await r.text()); return r.json(); };
const pct = (v) => (v === null || v === undefined ? "—" : (v * 100).toFixed(1) + "%");
const NODES = ["planner", "subgraph:fundamental", "subgraph:market", "subgraph:news", "gather", "replan", "subgraph:investigate",
  "debate", "risk", "fault_injection", "verifier", "rollback_rewrite", "report"];
let current = null, spans = [], charts = [];

function chart(el, opt) { if (!window.echarts) { el.innerHTML = '<p class="muted">ECharts 未加载（离线）。数据见下表。</p>'; return; }
  const c = echarts.init(el); c.setOption(opt); charts.push(c); }
window.addEventListener("resize", () => charts.forEach((c) => c.resize()));

// ---------------- navigation ----------------
document.querySelectorAll("header nav a").forEach((a) => a.onclick = () => show(a.dataset.view));
function show(v) {
  document.querySelectorAll(".view").forEach((x) => x.classList.add("hidden"));
  $("view-" + v).classList.remove("hidden");
  document.querySelectorAll("header nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === v));
  if (v === "evaluation") loadEval(); if (v === "data-health") loadHealth();
}

// ---------------- research ----------------
async function init() {
  const secs = await api("/api/securities");
  $("ticker").innerHTML = secs.map((s) => `<option value="${esc(s.ticker)}">${esc(s.ticker)} ${esc(s.name)} · ${esc(s.industry)}</option>`).join("");
  const archs = await api("/api/architectures");
  $("arch").innerHTML = archs.map((a) => `<option ${a === "B4" ? "selected" : ""}>${esc(a)}</option>`).join("");
  $("eval-arch").innerHTML = archs.map((a) => `<option ${a === "B4" ? "selected" : ""}>${esc(a)}</option>`).join("");
  renderTopology({});
  loadRuns();
  if (location.hash) show(location.hash.slice(1));
}
$("asof").onchange = () => {
  const d = new Date($("asof").value), now = new Date();
  $("pit-badge").className = "badge " + (d > now ? "bad" : "ok");
  $("pit-badge").textContent = d > now ? "研究时点晚于当前时间：PIT 不可保证" : "PIT：仅使用 available_at ≤ as_of 的数据";
};

function renderTopology(state) {
  $("topology").innerHTML = NODES.map((n) => `<span class="node ${state[n] || ""}" data-n="${n}">${n.replace("subgraph:", "")}</span>`).join(" → ");
  document.querySelectorAll(".node").forEach((el) => el.onclick = () => {
    const s = spans.filter((x) => x.node.startsWith(el.dataset.n)); renderTrace(s.length ? s : spans);
  });
}
const nodeKey = (n) => NODES.find((k) => n === k || n.startsWith(k + ":")) || n;

function renderTrace(list) {
  $("trace").innerHTML = list.map((s) => `<div class="span"><b>${esc(s.node)}</b> · ${(s.latency_ms ?? 0).toFixed(1)}ms · tools ${s.tool_calls?.length ?? s.tool_calls ?? 0} · tokens ${(s.input_tokens ?? 0) + (s.output_tokens ?? 0) || s.tokens || 0}
    <div class="muted">${esc(s.summary || "")}</div>
    ${(Array.isArray(s.tool_calls) ? s.tool_calls : []).map((t) => `<div><code>${esc(t.tool || (t.llm ? "llm:" + t.llm.model : ""))}(${esc(JSON.stringify(t.args || {}))})</code> ${t.error ? `<span style="color:var(--bad)">${esc(t.error)}</span>` : esc(t.result ?? "")}</div>`).join("")}
  </div>`).join("");
}

$("run").onclick = async () => {
  $("run").disabled = true; $("report").innerHTML = '<p class="muted">运行中…</p>'; $("plan").innerHTML = ""; spans = [];
  const st = {}; renderTopology(st); $("trace").innerHTML = "";
  const body = { ticker: $("ticker").value, as_of: $("asof").value + ":00", question: $("question").value, mode: $("mode").value,
    architecture: $("arch").value, token_budget: +$("budget").value, hitl: true, hitl_always: $("hitl-always").checked };
  try {
    const { job_id } = await api("/api/research", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const es = new EventSource(`/api/research/${job_id}/events`);
    es.onmessage = async (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === "node_start") { st[nodeKey(ev.node)] = "running"; renderTopology(st); }
      if (ev.type === "node_end") { st[nodeKey(ev.node)] = ev.status === "ok" ? "done" : "error"; renderTopology(st);
        spans.push(ev); renderTrace(spans); }
      if (ev.type === "approval_required") showApproval(job_id, ev);
      if (ev.type === "done") { $("approval").classList.add("hidden"); es.close(); $("run").disabled = false; await openRun(ev.run_id); loadRuns(); }
      if (ev.type === "error") { es.close(); $("run").disabled = false; $("report").innerHTML = `<p style="color:var(--bad)">${esc(ev.message)}</p>`; }
    };
  } catch (e) { $("run").disabled = false; $("report").innerHTML = `<p style="color:var(--bad)">${esc(e.message)}</p>`; }
};

function showApproval(jobId, ev) {
  const box = $("approval");
  const why = ev.reason === "over_budget" ? `已用 ${ev.used_tokens} tokens，超过预算 ${ev.token_budget}` : "按设置，追加调查前需要审批";
  box.innerHTML = `<h3>需要你的审批</h3><p>${esc(why)}。系统想追查以下异常：</p>` +
    ev.proposed.map((p) => `<div class="claim">• ${esc(p.code)}：${esc(p.message)}<br><span class="muted">检索：${esc(p.query)}</span></div>`).join("") +
    `<label>改为检索（可选）<input id="ap-query" maxlength="80" placeholder="例如：应收账款 账龄 坏账准备"></label>
     <label>追加 token 预算<input id="ap-extra" type="number" value="20000" step="10000"></label>
     <div class="row"><button data-a="approve">批准</button><button data-a="modify">按新检索执行</button><button data-a="skip">跳过，直接出报告</button></div>`;
  box.classList.remove("hidden");
  box.querySelectorAll("button").forEach((b) => b.onclick = async () => {
    const body = { action: b.dataset.a, query: $("ap-query").value || null, extra_tokens: +$("ap-extra").value || 0 };
    if (body.action === "modify" && !body.query) { $("ap-query").focus(); return; }
    box.innerHTML = '<p class="muted">已提交，继续运行…</p>';
    await api(`/api/research/${jobId}/approve`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  });
}

async function loadRuns() {
  const rs = await api("/api/runs?limit=15");
  $("runs").innerHTML = rs.map((r) => `<li><a href="#" data-r="${esc(r.run_id)}">${esc(r.ticker)} ${esc(String(r.as_of).slice(0, 10))} ${esc(r.architecture)}</a></li>`).join("");
  document.querySelectorAll("#runs a").forEach((a) => a.onclick = (e) => { e.preventDefault(); openRun(a.dataset.r); });
}

async function openRun(runId) {
  current = await api(`/api/runs/${runId}`);
  spans = current.trace || []; renderTrace(spans);
  const st = {}; spans.forEach((s) => st[nodeKey(s.node)] = s.status === "ok" ? "done" : "error"); renderTopology(st);
  $("plan").innerHTML = (current.plan?.subtasks || []).map((t) => `<li>${esc(t.agent)}：${esc(t.objective)}</li>`).join("") +
    spans.filter((s) => s.node === "replan").map((s) => `<li style="color:var(--warn)">动态追加：${esc(s.summary)}</li>`).join("");
  renderReport(current);
}

function renderReport(rep) {
  charts.forEach((c) => c.dispose()); charts = [];
  $("export").innerHTML = `<a href="/api/runs/${esc(rep.run_id)}/export.md" target="_blank">导出 Markdown</a> · <a href="/api/runs/${esc(rep.run_id)}/export.pdf">导出 PDF</a>`;
  const tag = { disclosed_fact: "事实", derived_metric: "算子", model_inference: "推断" };
  let h = `<p>${esc(rep.overview)}</p><p class="muted">快照 ${esc(rep.snapshot_id)} · 架构 ${esc(rep.architecture)} · ${rep.totals.tool_calls} 次工具调用 · ${rep.totals.latency_ms.toFixed(0)}ms</p>`;
  if (rep.executive_summary) h += `<h3>执行摘要</h3><p>${esc(rep.executive_summary)}</p>`;
  h += `<div id="c-q" class="chart"></div>`;
  for (const s of rep.sections) {
    if (s.key === "overview") continue;
    h += `<h3>${esc(s.title)}</h3>`;
    if (s.key === "falsification" && rep.debate?.disagreements?.length) {
      h += `<table class="small"><tr><th>看多</th><th>看空</th></tr>${rep.debate.disagreements.map((d) =>
        `<tr><td>${esc(rep.claims[d.bull]?.text)}</td><td>${esc(rep.claims[d.bear]?.text)}</td></tr>`).join("")}</table>`;
    }
    h += s.claim_ids.map((cid) => { const c = rep.claims[cid];
      return `<div class="claim"><span class="ctype">${tag[c.claim_type]}</span>${esc(c.text)}${c.evidence_ids.map((e) => `<span class="ev" data-e="${esc(e)}">#${esc(e)}</span>`).join("")}</div>`; }).join("");
    if (s.key === "valuation") h += `<div id="c-v" class="chart"></div>`;
    if (s.key === "risk") h += `<div id="c-r" class="chart"></div>`;
  }
  h += `<h3>数据局限与不确定性声明</h3>` + ([...rep.open_questions.map((q) => `待解答：${q.topic}（${q.reason}）`),
    ...rep.uncertainties.map((u) => u.text), ...rep.conflicts.map((c) => `数据源冲突：${c.metric} ${c.period} 采用 ${c.chosen}`)]
    .map((x) => `<div class="claim">${esc(x)}</div>`).join("") || '<p class="muted">无</p>');
  $("report").innerHTML = h;
  if (rep.charts.quarterly) chart($("c-q"), rep.charts.quarterly); else $("c-q").remove();
  if ($("c-v")) rep.charts.valuation ? chart($("c-v"), rep.charts.valuation) : $("c-v").remove();
  if ($("c-r")) rep.charts.risk_radar ? chart($("c-r"), rep.charts.risk_radar) : $("c-r").remove();
  document.querySelectorAll(".ev").forEach((el) => el.onclick = () => openEvidence(el.dataset.e));
}

async function openEvidence(eid) {
  const d = await api(`/api/runs/${current.run_id}/lineage/${eid}`);
  const ev = d.evidence || {};
  const chain = [];
  const claimsUsing = Object.values(current.claims).filter((c) => c.evidence_ids.includes(eid)).map((c) => c.claim_id);
  chain.push(`报告结论 Claim：${claimsUsing.join(", ")}`);
  chain.push(`证据卡 ${eid}：${ev.summary || ""}`);
  for (const e of d.lineage.edges) chain.push(`${e.parent_type} <b>${esc(e.parent_entity_id)}</b> → ${e.child_type} ${esc(e.child_entity_id)} <span class="muted">(${esc(e.transform_tool)} ${esc(e.formula_version)})</span>`);
  const metric = (ev.metric_ids || []).map((m) => current.metrics[m]).filter(Boolean)[0];
  let body = `<div class="badge ${d.pit_ok ? "ok" : "bad"}">PIT ${d.pit_ok ? "合规" : "违规"}：available_at ${esc(ev.available_at)} ${d.pit_ok ? "≤" : ">"} as_of ${esc(d.as_of)}</div>`;
  if (metric) body += `<p><b>算子</b> ${esc(metric.metric)} = ${esc(metric.value)} (${esc(metric.status)})<br><span class="muted">${esc(metric.formula)} · ${esc(metric.formula_version)}</span><br>输入：${metric.input_ids.map(esc).join(", ")}</p>`;
  body += `<h3>血缘链</h3><div class="chain">${chain.map((c, i) => (i ? '<div class="arrow">↑</div>' : "") + `<div>${i < 2 ? esc(c) : c}</div>`).join("")}</div>`;
  if (ev.kind === "document") body += `<h3>原文 ${esc(ev.page || "")}</h3><pre>${esc(ev.excerpt)}</pre>`;
  $("drawer-title").textContent = `证据 #${eid}`; $("drawer-body").innerHTML = body; $("drawer").classList.remove("hidden");
}
$("drawer-close").onclick = () => $("drawer").classList.add("hidden");

// ---------------- evaluation ----------------
async function loadEval() {
  let s; try { s = await api("/api/evaluation/summary"); } catch { $("eval-table").innerHTML = "尚无评测结果：运行 python scripts/run_experiments.py"; return; }
  const agg = s.comparison.aggregate, d = s.comparison.delta_vs_reference;
  const base = ["B0", "B1", "B2", "B3", "B4"].filter((a) => agg[a]);
  const ms = ["pass_rate", "numerical_accuracy", "pit_violation", "recovery", "rubric"];
  chart($("ch-base"), { tooltip: {}, legend: { data: ms }, xAxis: { type: "category", data: base }, yAxis: { type: "value", max: 1 },
    series: ms.map((m) => ({ name: m, type: "bar", data: base.map((a) => agg[a][m]) })) });
  const abl = Object.keys(d).filter((a) => a.startsWith("B4-"));
  chart($("ch-abl"), { tooltip: {}, xAxis: { type: "category", data: abl.map((a) => a.replace("B4-", "")), axisLabel: { rotate: 30 } },
    yAxis: { type: "value", name: "pp" }, series: [{ type: "bar", data: abl.map((a) => +(d[a].pass_rate * 100).toFixed(2)) }] });
  chart($("ch-pareto"), { tooltip: { formatter: (p) => p.data[2] }, xAxis: { type: "value", name: "tokens" }, yAxis: { type: "value", name: "pass", max: 1 },
    series: [{ type: "scatter", symbolSize: 14, label: { show: true, formatter: (p) => p.data[2] },
      data: Object.entries(agg).map(([a, r]) => [r.tokens_mean, r.pass_rate, a]) }] });
  const buckets = Object.keys(agg.B4.errors);
  chart($("ch-err"), { tooltip: {}, legend: { data: buckets }, xAxis: { type: "category", data: Object.keys(agg), axisLabel: { rotate: 30 } }, yAxis: { type: "value" },
    series: buckets.map((b) => ({ name: b, type: "bar", stack: "e", data: Object.keys(agg).map((a) => agg[a].errors[b]) })) });
  const cols = ["pass_rate", "numerical_accuracy", "citation_validity", "support_rate", "pit_violation", "anomaly_recall", "refusal", "conflict_handling", "recovery", "rubric", "redundant_tool_rate"];
  $("eval-table").innerHTML = `<table><tr><th>架构</th>${cols.map((c) => `<th>${c}</th>`).join("")}<th>P50ms</th><th>P95ms</th><th>tokens</th></tr>` +
    Object.entries(agg).map(([a, r]) => `<tr><td>${esc(a)}</td>${cols.map((c) => `<td>${pct(r[c])}</td>`).join("")}<td>${r.latency_p50_ms.toFixed(0)}</td><td>${r.latency_p95_ms.toFixed(0)}</td><td>${r.tokens_mean.toFixed(0)}</td></tr>`).join("") + "</table>";
  $("rag-table").innerHTML = `<table><tr><th>配置</th><th>Recall@5</th><th>MRR</th></tr>` + Object.entries(s.rag).filter(([k]) => k !== "table_integrity")
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${pct(v["recall@5"])}</td><td>${v.mrr.toFixed(3)}</td></tr>`).join("") +
    `</table><p>跨页表格完整率：结构感知 ${pct(s.rag.table_integrity.structure_aware)}，固定长度切片 ${pct(s.rag.table_integrity.naive_fixed_size)}</p>`;
  loadFails();
}
$("eval-arch").onchange = loadFails;
async function loadFails() {
  const rs = await api(`/api/evaluation/results?architecture=${encodeURIComponent($("eval-arch").value)}&failed_only=true`);
  $("eval-fails").innerHTML = rs.length ? `<table><tr><th>任务</th><th>题型</th><th>根因</th><th>分数</th></tr>` + rs.map((r) =>
    `<tr><td>${r.run_id ? `<a href="#" data-r="${esc(r.run_id)}">${esc(r.task_id)}</a>` : esc(r.task_id)}</td><td>${esc(r.category)}</td><td>${esc(r.error_category)}</td><td>${esc(JSON.stringify(r.scores))}</td></tr>`).join("") + "</table>"
    : '<p class="muted">该架构无失败样本。</p>';
  document.querySelectorAll("#eval-fails a").forEach((a) => a.onclick = async (e) => { e.preventDefault(); show("research"); try { await openRun(a.dataset.r); } catch (err) { $("report").textContent = "该运行未持久化"; } });
}

// ---------------- data health ----------------
async function loadHealth() {
  const h = await api("/api/data-health");
  $("cov").innerHTML = `<table><tr><th>证券</th><th>事实数</th><th>最新报告期</th><th>重述记录</th><th>pit_unsafe</th><th>文档</th></tr>` +
    h.coverage.map((c) => `<tr><td>${esc(c.ticker)}</td><td>${c.facts}</td><td>${esc(c.latest_period_end)}</td><td>${c.restated_facts}</td><td>${c.pit_unsafe_facts}</td><td>${c.documents}</td></tr>`).join("") + "</table>";
  $("dq").innerHTML = `<p>校验 ${h.quality.facts_checked} 条事实，ERROR 级通过率 ${pct(h.quality.pass_rate)}。规则：${h.quality.rules.map(esc).join("、")}</p>` +
    `<table><tr><th>规则</th><th>命中</th></tr>${Object.entries(h.quality.issues).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join("")}</table>` +
    `<pre>${esc(h.quality.samples.slice(0, 5).map((s) => `${s.severity} ${s.rule} ${s.entity_id}: ${s.message}`).join("\n"))}</pre>`;
  $("snaps").innerHTML = `<table><tr><th>snapshot_id</th><th>as_of</th><th>行数</th><th>manifest_hash</th></tr>` +
    h.snapshots.map((s) => `<tr><td>${esc(s.snapshot_id)}</td><td>${esc(s.as_of)}</td><td>${esc(JSON.stringify(s.row_counts))}</td><td><code>${esc(s.manifest_hash.slice(0, 16))}</code></td></tr>`).join("") + "</table>";
}

init();
