"""Hypothesis-driven planning (v2.0): turn the non-standard part of a question into sub-hypotheses with their own search
phrases, so the sub-graphs look for what was actually asked instead of only the fixed metric set.

Two sources, merged and de-duplicated:
* rules: a topic table (offshore expansion, capex conversion, customer concentration, rebates, ...). Deterministic and
  available offline, so the benchmark runs reproducibly.
* the planner model (online, full route only): it may add up to 4 more, validated against the same shape.
Questions that name nothing beyond the standard metrics get no hypotheses and run the fixed pipeline unchanged.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..state import SubHypothesis

MAX_HYPOTHESES = 4
AGENTS = ("fundamental", "market", "news")
# (pattern, agent, statement, queries, metrics, criterion)
TOPICS: list[tuple[str, str, str, list[str], list[str], str]] = [
    (r"海外|出海|境外|关税|地缘|建厂", "fundamental", "海外业务与海外建厂的规模、占比和政策风险",
     ["境外 收入 占比 关税", "海外 建厂 投资 子公司"], [], "年报分部/附注或公告给出海外收入占比、海外项目投资额或关税影响"),
    (r"资本开支|在建工程|转固|产能|扩产", "fundamental", "资本开支向产能与自由现金流的转化效率",
     ["资本开支 在建工程 转固", "购建固定资产 现金支出"], ["ocf_to_net_profit"], "在建工程转固进度、购建长期资产现金流出与经营现金流的对应"),
    (r"客户集中|大客户|前五名客户|依赖", "fundamental", "客户集中度与对大客户的依赖",
     ["前五名客户 销售额 占比", "客户集中 关联方"], [], "年报披露前五名客户销售额占营业收入比例"),
    (r"返利|折扣|补贴|价格补偿|价格调整", "fundamental", "返利、折扣或价格补偿安排对收入和利润的影响",
     ["销售返利 计提", "价格 调整 补偿 协议"], [], "附注或公告披露返利计划、计提方法与金额"),
    (r"存货跌价|跌价准备|计提", "fundamental", "存货减值计提的充足性",
     ["存货 跌价准备 计提比例", "存货 账面余额 库龄"], ["inventory_days"], "存货附注给出账面余额、跌价准备与计提比例"),
    (r"研发|专利|技术路线", "fundamental", "研发投入强度与资本化政策",
     ["研发投入 资本化 比例", "研发费用 变动 原因"], ["rd_ratio"], "年报披露研发投入总额、资本化比例及变动原因"),
    (r"原材料|成本端|上游价格|锂价|碳酸锂", "fundamental", "原材料价格波动对成本和毛利率的传导",
     ["原材料 价格 波动 成本", "营业成本 构成 变动"], ["gross_margin"], "营业成本构成和管理层对毛利率变动的解释"),
    (r"现金流|自由现金流|回款", "fundamental", "利润的现金含量与回款质量",
     ["经营活动现金流量净额 变动 原因", "应收账款 回款 周转"], ["ocf_to_net_profit", "receivable_days"], "现金流量表补充资料与管理层对现金流变动的解释"),
    (r"政策|监管|行业竞争|竞争格局|价格战", "news", "政策与行业竞争环境的近期变化",
     ["政策 监管 影响", "行业 竞争 价格"], [], "近期公告或事件中出现的政策、监管或竞争信息"),
    (r"回购|分红|股权激励|增持|减持", "news", "资本运作与股东回报安排",
     ["回购 股份 方案", "利润分配 现金分红"], [], "公告披露的回购、分红或增减持方案"),
]


def rule_hypotheses(question: str, ticker: str, start: int = 1) -> list[SubHypothesis]:
    out: list[SubHypothesis] = []
    for pat, agent, stmt, queries, metrics, crit in TOPICS:
        if re.search(pat, question):
            out.append(SubHypothesis(sub_id=f"H{start + len(out)}", target_ticker=ticker, assigned_agent=agent,  # type: ignore[arg-type]
                                     hypothesis_statement=stmt, custom_rag_queries=list(queries), required_metrics=list(metrics),
                                     verification_criterion=crit))
        if len(out) >= MAX_HYPOTHESES:
            break
    return out


def validate_model_hypotheses(items: Any, ticker: str, allowed_metrics: set[str], start: int) -> list[SubHypothesis]:
    """Keep only well-formed model proposals; anything else is dropped rather than repaired."""
    out: list[SubHypothesis] = []
    if not isinstance(items, list):
        return out
    for it in items[:MAX_HYPOTHESES]:
        if not isinstance(it, dict):
            continue
        agent = it.get("assigned_agent")
        stmt = str(it.get("hypothesis_statement") or "").strip()
        qs = [str(q).strip() for q in (it.get("custom_rag_queries") or []) if isinstance(q, str) and 2 <= len(q.strip()) <= 30][:3]
        if agent not in AGENTS or not stmt or not qs:
            continue
        mets = [m for m in (it.get("required_metrics") or []) if isinstance(m, str) and m in allowed_metrics]
        out.append(SubHypothesis(sub_id=f"H{start + len(out)}", target_ticker=ticker, assigned_agent=agent,
                                 hypothesis_statement=stmt[:80], custom_rag_queries=qs, required_metrics=mets,
                                 verification_criterion=str(it.get("verification_criterion") or "")[:100]))
    return out


def merge(rules: list[SubHypothesis], model: list[SubHypothesis]) -> list[SubHypothesis]:
    """Rule hypotheses first; a model one is skipped when it shares a search phrase with a kept one."""
    kept = list(rules)
    seen = {q for h in kept for q in h.custom_rag_queries}
    for h in model:
        if len(kept) >= MAX_HYPOTHESES:
            break
        if seen & set(h.custom_rag_queries):
            continue
        seen |= set(h.custom_rag_queries)
        kept.append(h.model_copy(update={"sub_id": f"H{len(kept) + 1}"}))
    return kept


def build(question: str, ticker: str, route_name: str, model_items: Optional[Any], allowed_metrics: set[str]) -> list[SubHypothesis]:
    """Fast route: none (a lookup has nothing to hypothesise). Otherwise rules, plus validated model proposals."""
    if route_name == "fast":
        return []
    rules = rule_hypotheses(question, ticker)
    model = validate_model_hypotheses(model_items, ticker, allowed_metrics, start=len(rules) + 1)
    return merge(rules, model)
