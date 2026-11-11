"""SemanticAnomalyProbe (PRD v1.2 1.2): qualitative red flags that the three statements cannot show.

Retrieval (PIT-filtered, the same tool every agent uses) pulls candidate passages for five categories; one batched
model call decides, per passage, whether it discloses something unusual or only routine boilerplate. The model must
quote the sentence it relies on, and the quote must be a verbatim substring of the passage, otherwise the flag is
dropped. Offline (no model) the probe does nothing; the title-based rule for related-party filings still runs.
"""
from __future__ import annotations

import json
import re

from .. import tools
from ..context import RunContext
from ..llm import call_json

# category -> (anomaly code, retrieval query). RELATED_PARTY_OR_POLICY_CHANGE is the PRD 2.2 rule-7 code; the other
# three are new codes for things rule 7 never covered.
PROBES = [
    ("accounting_change", "RELATED_PARTY_OR_POLICY_CHANGE", "会计政策变更 会计估计变更 追溯调整 前期差错更正"),
    ("non_operating_funds", "RELATED_PARTY_OR_POLICY_CHANGE", "非经营性资金占用 关联方资金往来 控股股东 资金拆借"),
    ("audit_opinion", "NON_STANDARD_AUDIT_OPINION", "保留意见 无法表示意见 强调事项段 持续经营 重大不确定性"),
    ("guarantee_litigation", "GUARANTEE_OR_LITIGATION_RISK", "违规担保 对外担保 未决诉讼 仲裁 或有负债 被立案"),
    ("project_delay", "PROJECT_DELAY_OR_CHANGE", "募集资金投资项目 延期 变更 终止 未达到预定可使用状态"),
]
SYSTEM = (
    "你是审计与财务风险分析师。下面每段是上市公司披露原文片段，附一个候选类别。判断该片段是否披露了【不寻常、需要投资者警惕】的事项，"
    "而不是例行披露或模板性表述。类别含义：accounting_change=会计政策/估计发生变更或前期差错更正；non_operating_funds="
    "关联方或控股股东非经营性占用资金/拆借；audit_opinion=非标准审计意见或强调事项段/持续经营不确定性；guarantee_litigation="
    "违规担保、重大未决诉讼仲裁、公司或高管被立案；project_delay=募投项目延期、变更或终止。"
    "对合并范围内子公司的担保额度、已履行审议程序的常规授信担保属于例行融资安排，判为非异常；只有对合并范围外主体的担保、"
    "未履行审议程序的担保、逾期或代偿，才算风险。"
    "例行表述（如“本年度无会计政策变更”“公司按规定履行审议程序”“关键审计事项”章节的常规描述、日常关联交易预计、"
    "财务公司存款说明、内部制度全文）一律判为非异常。"
    "只输出JSON {\"results\":[{\"id\":\"P1\",\"is_anomaly\":true或false,\"severity\":\"HIGH\"或\"MEDIUM\","
    "\"quote\":\"片段中支持判断的一句原文，必须逐字摘抄\"}]}。不确定时判为非异常。不得给出片段之外的信息。")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def probe(ctx: RunContext) -> list[dict]:
    """Return validated flags: {code, category, severity, quote, hit}."""
    if not ctx.llm.online or ctx.factor_only:
        return []
    cands, seen = [], set()
    for cat, code, query in PROBES:
        for h in tools.retrieve_document_evidence(ctx, query, None, top_k=2):
            c = h.parent or h.chunk
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            cands.append({"id": f"P{len(cands) + 1}", "category": cat, "code": code, "hit": h, "text": c.text[:1200]})
    if not cands:
        return []
    user = json.dumps([{"id": x["id"], "category": x["category"], "text": x["text"]} for x in cands], ensure_ascii=False)
    d = call_json(ctx.llm, ctx.tracer, "probe", SYSTEM, user, "strong",
                  validate=lambda d: isinstance(d.get("results"), list))
    if not d:
        return []
    by_id = {x["id"]: x for x in cands}
    out, dropped = [], 0
    for r in d["results"]:
        x = by_id.get(str(r.get("id")))
        quote = str(r.get("quote") or "")
        if not x or r.get("is_anomaly") is not True or len(quote) < 8:
            continue
        if _norm(quote) not in _norm(x["text"]):       # the model may only quote what is on the page
            dropped += 1
            continue
        out.append({"code": x["code"], "category": x["category"], "hit": x["hit"], "quote": quote.strip(),
                    # a qualitative flag has no number behind it, so it never drives the costly HIGH-severity
                  # re-planning loop; it is reported as MEDIUM with its verbatim quote
                  "severity": "MEDIUM"})
    ctx.tracer.tool("semantic_anomaly_probe", {"candidates": len(cands)},
                    result=f"{len(out)} flag(s), {dropped} dropped for a quote not found in the passage")
    return out
