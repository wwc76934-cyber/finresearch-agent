import { useEffect, useMemo, useState } from "react";
import { api, pct, post } from "./api";
import { Chart } from "./Chart";

const UNI: Record<string, string> = { all: "全部 24 家", new_energy: "新能源（12）", consumer: "消费（12）" };
const num = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? "—" : v.toFixed(d));

export default function Factor() {
  const [meta, setMeta] = useState<any>(null);
  const [f, setF] = useState({ factor: "ocf_to_net_profit", universe: "all", as_of_date: "2025-12-31",
    lookback_quarters: 8, forward_horizon_days: 60, quantiles: 3, report_period: "" });
  const [rep, setRep] = useState<any>(null);
  const [err, setErr] = useState<{ error: string; message: string } | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { api("/api/factors").then(setMeta); }, []);

  const run = async (override?: Partial<typeof f>) => {
    const body = { ...f, ...override, report_period: (override?.report_period ?? f.report_period) || null };
    setBusy(true); setErr(null);
    try { setRep(await post("/api/factors/test", body)); }
    catch (e: any) {
      setRep(null);
      try { const d = JSON.parse(e.message).detail; setErr(typeof d === "object" ? d : { error: "ERROR", message: String(d) }); }
      catch { setErr({ error: "ERROR", message: e.message }); }
    } finally { setBusy(false); }
  };

  const nav = useMemo(() => rep && {
    tooltip: { trigger: "axis" }, legend: { top: 0 }, grid: { left: 48, right: 16, top: 32, bottom: 28 },
    xAxis: { type: "category", data: rep.nav.map((n: any) => n.date) }, yAxis: { type: "value", scale: true },
    series: [{ name: "多空（最高组−最低组）", type: "line", lineStyle: { width: 3 }, data: rep.nav.map((n: any) => n.long_short) },
      ...Array.from({ length: rep.quantiles }, (_, i) => ({ name: `G${i + 1}`, type: "line", data: rep.nav.map((n: any) => n[`G${i + 1}`]) }))],
  }, [rep]);
  const bars = useMemo(() => rep && {
    tooltip: {}, grid: { left: 48, right: 16, top: 16, bottom: 28 },
    xAxis: { type: "category", data: rep.group_mean_returns.map((_: any, i: number) => `G${i + 1}${i === 0 ? "（最低）" : i === rep.quantiles - 1 ? "（最高）" : ""}`) },
    yAxis: { type: "value", axisLabel: { formatter: (v: number) => (v * 100).toFixed(1) + "%" } },
    series: [{ type: "bar", data: rep.group_mean_returns }],
  }, [rep]);
  const ic = useMemo(() => rep && {
    tooltip: { trigger: "axis" }, legend: { top: 0 }, grid: { left: 48, right: 16, top: 32, bottom: 28 },
    xAxis: { type: "category", data: rep.cross_sections.map((c: any) => c.date) }, yAxis: { type: "value" },
    series: [{ name: "RankIC", type: "bar", data: rep.cross_sections.map((c: any) => c.rank_ic) },
      { name: "IC", type: "line", data: rep.cross_sections.map((c: any) => c.ic) }],
  }, [rep]);

  const set = (k: string) => (e: any) => setF({ ...f, [k]: e.target.type === "number" ? Number(e.target.value) : e.target.value });
  return (
    <main className="grid gap-3 p-3 lg:grid-cols-[300px_1fr]">
      <section className="card h-fit">
        <h3 className="h3">因子假设验证</h3>
        <label className="lbl">因子</label>
        <select className="field" value={f.factor} onChange={set("factor")}>
          {meta?.factors.map((x: any) => <option key={x.id} value={x.id}>{x.name}</option>)}</select>
        <label className="lbl">样本池</label>
        <select className="field" value={f.universe} onChange={set("universe")}>
          {Object.entries(UNI).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select>
        <label className="lbl">研究时点 as_of（面板价格到 {meta?.panel_last_date ?? "…"}）</label>
        <input className="field" type="date" value={f.as_of_date} onChange={set("as_of_date")} />
        <div className="grid grid-cols-3 gap-2">
          <div><label className="lbl">回看季度</label><input className="field" type="number" min={1} max={12} value={f.lookback_quarters} onChange={set("lookback_quarters")} /></div>
          <div><label className="lbl">持有天数</label><input className="field" type="number" min={5} max={250} value={f.forward_horizon_days} onChange={set("forward_horizon_days")} /></div>
          <div><label className="lbl">分组数</label><input className="field" type="number" min={2} max={5} value={f.quantiles} onChange={set("quantiles")} /></div>
        </div>
        <button className="btn mt-3 w-full" disabled={busy} onClick={() => run()}>{busy ? "计算中…" : "运行检验"}</button>
        <hr className="my-3 border-slate-200 dark:border-slate-700" />
        <p className="text-xs text-slate-500">看“拒绝”长什么样：用一份在截面日之后才发布的报告做因子。</p>
        <button className="btn-ghost mt-1 w-full" disabled={busy}
                onClick={() => run({ as_of_date: "2025-12-31", report_period: "2025Q4" })}>用 2025Q4 报告（2026 年才披露）</button>
        <p className="mt-3 text-xs text-slate-500">{meta ? `面板：${meta.panel_companies} 家公司，PIT 可用日 = 公告日次一交易日 09:30。` : ""}</p>
      </section>

      <section className="space-y-3">
        {err && <div className="card border-red-300 text-sm"><b className="text-red-600">{err.error === "PIT_VIOLATION" ? "已拒绝：时点穿越" : err.error}</b>
          <p className="mt-1">{err.message}</p></div>}
        {!rep && !err && <div className="card text-sm text-slate-500">选一个因子并运行。结果与 Agent 调用的是同一个 <code>evaluate_factor_hypothesis</code>。</div>}
        {rep && <>
          <div className="card text-sm"><p>{rep.summary}</p>
            {rep.warnings.length > 0 && <ul className="mt-2 space-y-0.5 text-xs text-amber-700 dark:text-amber-400">
              {rep.warnings.map((w: any) => <li key={w.code}>[{w.code}] {w.message}</li>)}</ul>}</div>
          <div className="card"><table className="tbl"><thead><tr><th>截面数</th><th>每截面公司数(中位)</th><th>RankIC 均值</th><th>t / p</th><th>IC 均值</th><th>ICIR</th>
            <th>RankIC&gt;0 占比</th><th>多空每期</th><th>多空累计</th><th>最大回撤</th><th>5% 显著</th></tr></thead><tbody><tr>
            <td>{rep.n_cross_sections}</td><td>{rep.n_stocks_median}</td><td>{num(rep.rank_ic_mean)}</td>
            <td>{num(rep.rank_ic_t_stat, 2)} / {num(rep.rank_ic_p_value)}</td><td>{num(rep.ic_mean)}</td><td>{num(rep.ic_ir, 2)}</td>
            <td>{pct(rep.rank_ic_positive_share)}</td><td>{pct(rep.long_short_mean_return)}</td><td>{pct(rep.long_short_cum_return)}</td>
            <td>{pct(rep.long_short_max_drawdown)}</td><td>{rep.significant_5pct ? "是" : "否"}</td></tr></tbody></table></div>
          <div className="grid gap-3 lg:grid-cols-2">
            <div className="card"><h3 className="h3">累计净值（起点 = 1）</h3>{nav && <Chart option={nav} />}</div>
            <div className="card"><h3 className="h3">各组平均持有期收益</h3>{bars && <Chart option={bars} />}</div>
          </div>
          <div className="card"><h3 className="h3">逐截面 IC / RankIC</h3>{ic && <Chart option={ic} height={240} />}
            <table className="tbl mt-2"><thead><tr><th>截面日</th><th>公司数</th><th>IC</th><th>RankIC</th><th>用到的最新报告期(中位)</th><th>全体平均收益</th></tr></thead><tbody>
              {rep.cross_sections.map((c: any) => <tr key={c.date}><td>{c.date}</td><td>{c.n}</td><td>{num(c.ic)}</td><td>{num(c.rank_ic)}</td>
                <td>{c.report_period}</td><td>{pct(c.universe_return)}</td></tr>)}</tbody></table></div>
        </>}
      </section>
    </main>
  );
}
