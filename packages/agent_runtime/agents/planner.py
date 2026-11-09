"""Planning Agent: question -> structured asks + subtask DAG with budgets."""
from __future__ import annotations

import json
import re
from typing import Optional

from ..context import RunContext
from ..llm import call_json
from ..state import Ask, ResearchPlan, SubTask
from .. import tools

METRIC_PATTERNS: list[tuple[str, str]] = [
    (r"单季.*(营业收入|营收)|(营业收入|营收).*单季", "single_quarter:operating_revenue"),
    (r"单季.*净利润|净利润.*单季", "single_quarter:net_profit"),
    (r"经营现金流.{0,4}净利润|OCF/", "ocf_to_net_profit"),
    (r"毛利率", "gross_margin"),
    (r"净利率", "net_margin"),
    (r"ROE|净资产收益率", "roe_annualized"),
    (r"应收账款周转天数", "receivable_days"),
    (r"存货周转天数", "inventory_days"),
    (r"有息负债率", "interest_debt_ratio"),
    (r"资产负债率", "debt_to_assets"),
    (r"营收同比|营业收入同比|收入增速", "revenue_yoy"),
    (r"净利润同比|利润增速", "net_profit_yoy"),
    (r"(PE|市盈率)", "pe_ttm"),
    (r"(PB|市净率)", "pb_lf"),
    (r"估值分位|历史分位", "pe_percentile_3y"),
    (r"波动率", "volatility_1y"),
    (r"回撤", "max_drawdown_1y"),
    (r"(?<!单季)(营业收入|营收)(?!同比)", "operating_revenue"),
    (r"(?<!单季)(?<!母公司)净利润(?!同比)", "net_profit"),
]
UNANSWERABLE = [
    (r"下(一)?季度.*(营收|收入|利润|订单)|未来.*(业绩|营收|利润)|明年.*(业绩|营收|利润)|预测", "未来业绩预测（快照内不存在）"),
    (r"(具体|真实)(名称|名字)|叫什么", "未披露的交易对手具体名称"),
    (r"在手订单|订单金额", "未披露的在手订单金额"),
    (r"未披露|内部|商业机密", "未公开信息"),
    (r"目标价", "目标股价（无DCF假设输入，不允许直接给出）"),
    (r"细分产品.*毛利率|分产品毛利率", "未披露的细分产品毛利率"),
]
PERIOD_RE = re.compile(r"(20\d{2})\s*(?:年)?\s*(?:Q([1-4])|第([一二三四])季度|(半年度|中报|上半年)|(年报|年度))")
CN_Q = {"一": 1, "二": 2, "三": 3, "四": 4}

SYSTEM = ("你是投研规划Agent。把研究问题解析为结构化需求 asks 与子任务。只输出JSON:"
          '{"asks":[{"kind":"metric|anomalies|valuation|events|peer_compare|unanswerable","metric":"...",'
          '"period":"2025Q3","scope":"consolidated|parent","topic":"..."}],"rationale":"..."}。'
          "metric 只能取给定的指标名；同一需求只列一次；问题未明确要求的指标不要列出；"
          "问题询问未来预测、未披露信息或目标价时，kind 用 unanswerable。rationale 用中文，不超过60字。"
          "数字由确定性算子计算，你不得给出任何数值。")


def parse_period(q: str) -> Optional[str]:
    m = PERIOD_RE.search(q)
    if not m:
        return None
    y = m.group(1)
    if m.group(2):
        return f"{y}Q{m.group(2)}"
    if m.group(3):
        return f"{y}Q{CN_Q[m.group(3)]}"
    if m.group(4):
        return f"{y}Q2"
    return f"{y}Q4"


def rule_asks(question: str, default_period: Optional[str]) -> list[Ask]:
    asks: list[Ask] = []
    period = parse_period(question) or default_period
    for pat, topic in UNANSWERABLE:
        if re.search(pat, question):
            asks.append(Ask(ask_id=f"A{len(asks) + 1}", kind="unanswerable", topic=topic))
    scope = "parent" if "母公司" in question else "consolidated"
    unit = "亿元" if "亿元" in question else ("万元" if "万元" in question else None)
    seen = set()
    if not any(a.kind == "unanswerable" for a in asks):
        for pat, metric in METRIC_PATTERNS:
            if re.search(pat, question) and metric not in seen:
                if metric in ("operating_revenue",) and "single_quarter:operating_revenue" in seen:
                    continue
                if metric == "net_profit" and ("single_quarter:net_profit" in seen or "ocf_to_net_profit" in seen):
                    continue
                if metric == "operating_revenue" and "revenue_yoy" in seen:
                    continue
                seen.add(metric)
                m_scope = scope if metric in ("operating_revenue", "net_profit") else "consolidated"
                asks.append(Ask(ask_id=f"A{len(asks) + 1}", kind="metric", metric=metric, period=period,
                                scope=m_scope, unit=unit))
    if re.search(r"异常|背离|风险信号", question):
        asks.append(Ask(ask_id=f"A{len(asks) + 1}", kind="anomalies", period=period))
    if re.search(r"公告|事件|重大事项", question):
        asks.append(Ask(ask_id=f"A{len(asks) + 1}", kind="events"))
    if re.search(r"同行|可比|对比", question):
        asks.append(Ask(ask_id=f"A{len(asks) + 1}", kind="peer_compare"))
    return asks


def plan(ctx: RunContext) -> ResearchPlan:
    req = ctx.request
    cov = tools.check_data_coverage(ctx)
    default_period = cov["latest_period"]
    asks = rule_asks(req.question, default_period)
    user = json.dumps({"question": req.question, "as_of": req.as_of.isoformat(), "coverage": cov,
                       "allowed_metrics": [m for _, m in METRIC_PATTERNS]}, ensure_ascii=False)
    data = call_json(ctx.llm, ctx.tracer, "planner", SYSTEM, user, "strong",
                     validate=lambda d: isinstance(d.get("asks"), list))
    rationale = "rule-based parse (offline)"
    if data:
        try:
            llm_asks = [Ask(ask_id=f"A{i + 1}", **{k: v for k, v in a.items() if k in Ask.model_fields and k != "ask_id"})
                        for i, a in enumerate(data["asks"])]
            # PIT/period guard: an LLM may not invent a period that is not yet visible
            for a in llm_asks:
                if a.period and a.period not in cov["periods"]:
                    a.period = default_period
            # dedupe: the model often repeats the same ask
            seen, uniq = set(), []
            for a in llm_asks:
                key = (a.kind, a.metric, a.period, a.scope, a.topic if a.kind == "unanswerable" else "")
                if key not in seen:
                    seen.add(key)
                    uniq.append(a.model_copy(update={"ask_id": f"A{len(uniq) + 1}"}))
            # guard: the model may refuse an answerable question (seen online: "2025年第三季度单季营收" marked
            # unanswerable). A refusal is accepted only if the rules also see one; otherwise rule asks win.
            rule_refuses = any(a.kind == "unanswerable" for a in asks)
            if not rule_refuses and any(u.kind == "unanswerable" for u in uniq):
                uniq = [u for u in uniq if u.kind != "unanswerable"]
                for a in asks:
                    if a.kind == "metric" and not any(u.metric == a.metric for u in uniq):
                        uniq.append(a.model_copy(update={"ask_id": f"A{len(uniq) + 1}"}))
            # refusals found by rules are kept even if the model missed them
            for a in asks:
                if a.kind == "unanswerable" and not any(u.kind == "unanswerable" for u in uniq):
                    uniq.append(a.model_copy(update={"ask_id": f"A{len(uniq) + 1}"}))
            asks = uniq or asks
            rationale = data.get("rationale", "llm plan")
        except Exception:  # noqa: BLE001 - invalid LLM output falls back to rules
            pass
    deep = req.mode == "deep"
    subtasks = [SubTask(task_id="T1", agent="fundamental", objective="财务三表、盈利与现金流质量、异常检测", tool_budget=12,
                        acceptance="核心比率与增长指标全部由算子计算并入证据库")]
    if deep or any(a.kind in ("valuation", "peer_compare") or (a.metric or "").startswith(("pe", "pb", "vol", "max"))
                   for a in asks):
        subtasks.append(SubTask(task_id="T2", agent="market", objective="估值分位、同行估值、波动率与回撤", tool_budget=8,
                                acceptance="PE/PB/PS 与3年分位、同行中位数"))
    if deep or any(a.kind == "events" for a in asks):
        subtasks.append(SubTask(task_id="T3", agent="news", objective="公告与重大事项时间线", tool_budget=6,
                                acceptance="所有事件带 published_at/available_at"))
    for a in asks:
        if a.kind == "unanswerable":
            ctx.open_questions.append({"ask_id": a.ask_id, "topic": a.topic,
                                       "reason": "当前 PIT 快照中无此数据，拒绝编造"})
    p = ResearchPlan(plan_id=f"{ctx.run_id}-plan", subtasks=subtasks, asks=asks, budget_tokens=req.token_budget,
                     rationale=rationale)
    ctx.tracer.tool("register_research_plan", {"subtasks": [s.agent for s in subtasks], "asks": len(asks)})
    return p
