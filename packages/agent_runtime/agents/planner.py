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
    (r"EV/?EBITDA|企业价值倍数", "ev_ebitda"),
    (r"营收环比|收入环比", "revenue_qoq"),
    (r"净利润环比|利润环比", "net_profit_qoq"),
    (r"5年分位|五年分位", "pe_percentile_5y"),
    (r"(?<!EV/)(PE|市盈率)", "pe_ttm"),
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
    (r"会不会|会否|是否会|走势|批价|涨到|跌破|股价.{0,6}(涨|跌)", "未来价格或走势判断（快照无法证实）"),
    (r"哪几家|是哪些公司|具体是哪|名单", "未披露的交易对手具体名称"),
]
# ---- scenario D: "does factor X predict next-quarter return?" -----------------------------------------
FACTOR_PATTERNS: list[tuple[str, str]] = [
    (r"经营现金流.{0,6}净利润|OCF\s*/\s*净利润|盈利现金保障", "ocf_to_net_profit"),
    (r"ROE|净资产收益率", "roe_ttm"),
    (r"毛利率.{0,3}(同比)?(变动|变化|改善)|毛利率同比", "gross_margin_yoy_delta"),
    (r"应计项目|应计占比", "accruals_to_assets"),
    (r"应收账款周转率", "receivable_turnover_yoy"),
    (r"盈利收益率|市盈率倒数|EP因子|1/PE", "pe_ttm_inverse"),
]
SCENE_D_RE = re.compile(r"因子|RankIC|\bIC\b|横截面|分组|多空组合|超额收益|回测|下一季度.{0,6}收益|持续(高于|低于)")
DATE_RE = re.compile(r"(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})日?")
CN_NUM = {"二": 2, "三": 3, "四": 4, "五": 5}


def factor_ask(question: str) -> Optional[Ask]:
    """Rule parse of a scenario-D question; None when the question is not a factor test."""
    if not SCENE_D_RE.search(question):
        return None
    factor = next((f for pat, f in FACTOR_PATTERNS if re.search(pat, question)), None)
    if factor is None:
        return None
    p: dict = {"factor_name": factor}
    ne, cons = bool(re.search(r"新能源|电池|光伏|锂", question)), bool(re.search(r"消费|白酒|家电|食品|饮料", question))
    p["universe"] = "new_energy" if ne and not cons else ("consumer" if cons and not ne else "all")
    # "31日" in "3月31日" is a calendar date, not a horizon: require 交易日 or a return word after the number
    m = re.search(r"(?<![月\d])(\d{1,3})\s*个?\s*交易日|(?<![月\d])(\d{1,3})\s*日(?:收益|超额|后续|持有)", question)
    p["forward_horizon_days"] = int(m.group(1) or m.group(2)) if m else (20 if re.search(r"下一月|一个月", question) else 60)
    m = re.search(r"过去\s*(\d+)\s*年", question)
    m2 = re.search(r"过去\s*(\d+)\s*个?季度", question)
    p["lookback_quarters"] = int(m.group(1)) * 4 if m else (int(m2.group(1)) if m2 else 8)
    m = re.search(r"([二三四五\d])\s*分组", question)
    if m:
        p["quantiles"] = CN_NUM.get(m.group(1)) or int(m.group(1))
    # PIT trap: "at 2024-03-31 use the 2024 Q1 report's factor"
    d = DATE_RE.search(question)
    rp = re.search(r"(20\d{2})\s*年?\s*(一季报|第一季度|Q1|半年报|中报|三季报|第三季度|Q3|年报)", question)
    if d and rp:
        q = {"一季报": 1, "第一季度": 1, "Q1": 1, "半年报": 2, "中报": 2, "三季报": 3, "第三季度": 3, "Q3": 3, "年报": 4}[rp.group(2)]
        p["report_period"] = f"{rp.group(1)}Q{q}"
        p["cross_section_dates_override"] = [f"{int(d.group(1)):04d}-{int(d.group(2)):02d}-{int(d.group(3)):02d}"]
    return Ask(ask_id="A1", kind="factor_test", topic=question[:80], params=p)


PERIOD_RE = re.compile(r"(20\d{2})\s*(?:年)?\s*(?:Q([1-4])|第?([一二三四])季度|(半年度|中报|上半年)|(年报|年度|全年))")
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
    fa = factor_ask(req.question)
    if fa is not None:
        # Scenario D is a universe-level statistical test. It has one deterministic tool, so no LLM parsing,
        # no company-level subgraphs, no debate: the plan is a single fundamental/quant task.
        st = [SubTask(task_id="T1", agent="fundamental", objective="横截面因子检验（IC/RankIC/分组/多空）", tool_budget=2,
                      acceptance="所有统计量由 evaluate_factor_hypothesis 算子计算并入证据库")]
        ctx.tracer.tool("register_research_plan", {"subtasks": ["fundamental"], "asks": 1, "scenario": "factor_test"})
        return ResearchPlan(plan_id=f"{ctx.run_id}-plan", subtasks=st, asks=[fa], budget_tokens=req.token_budget,
                            rationale="scenario D: factor test (rule parse)")
    cov = tools.check_data_coverage(ctx)
    default_period = cov["latest_period"]
    asks = rule_asks(req.question, default_period)
    from . import router
    ctx.route = router.classify(req.question, req.mode, asks, ctx.config.complexity_router)
    ctx.tracer.tool("complexity_router", {"route": ctx.route.name, "reasons": ctx.route.reasons})
    user = json.dumps({"question": req.question, "as_of": req.as_of.isoformat(), "coverage": cov,
                       "allowed_metrics": [m for _, m in METRIC_PATTERNS]}, ensure_ascii=False)
    # lean routes use the rule-parsed asks: the planner LLM call (15 s for a deep question) adds nothing for a lookup
    data = None if ctx.route.lean else call_json(ctx.llm, ctx.tracer, "planner", SYSTEM, user, "strong",
                                                 validate=lambda d: isinstance(d.get("asks"), list))
    rationale = "rule-based parse (offline)"
    if data:
        try:
            llm_asks = [Ask(ask_id=f"A{i + 1}", **{k: v for k, v in a.items() if k in Ask.model_fields and k != "ask_id"})
                        for i, a in enumerate(data["asks"])]
            # period guard: a period later than the latest visible one is NOT silently replaced by the latest
            # (that would answer a different question); it becomes a refusal below. Unknown older periods fall
            # back to the latest visible one.
            for a in llm_asks:
                if a.period and a.period not in cov["periods"] and not (default_period and a.period > default_period):
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
    # PIT refusal: the question names a report period that is not yet disclosed at as_of
    asked = parse_period(req.question)
    future = [a for a in asks if a.kind == "metric" and a.period and default_period and a.period > default_period]
    if (asked and default_period and asked > default_period) or future:
        p = asked if asked and default_period and asked > default_period else future[0].period
        asks = [a for a in asks if not (a.kind == "metric" and a.period and a.period > default_period)]
        asks.append(Ask(ask_id=f"A{len(asks) + 1}", kind="unanswerable",
                        topic=f"{p} 报告在研究时点尚未披露（最新可见 {default_period}）"))
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
