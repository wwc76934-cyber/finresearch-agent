"""Two-tier independent Verifier + partial rollback + 2-strike circuit breaker.

Tier-1 (deterministic, no tokens): numbers, PIT, citation integrity, scope mixing.
Tier-2 (entailment judge): LLM when online; offline it uses a conservative
heuristic (direction consistency for inferences, lexical support for quotes).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ..context import RunContext
from ..llm import call_json
from ..state import Claim, UncertaintyStatement, VerificationFailure

STRIP = [r"\d{4}-\d{2}-\d{2}", r"\d{4}Q[1-4]", r"\d{4}年(\d{1,2}月(\d{1,2}日)?)?", r"\d{1,2}月\d{1,2}日",
         r"\b[CE]\d{3}\b", r"\d{6}\.(SZ|SH|BJ)", r"近\d+年", r"\d/5", r"\d+项", r"前[一二三四五]名", r"第\d+", r"\d+-\d+月",
         r"[A-Z_]{4,}", r"p\d+-\d+", r"\b[a-z]+_[0-9A-Za-z_]+"]
NUM = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(%|亿元|万元|元|天|倍)?")
UNIT_SCALE = {"%": ("ratio", 0.01), "亿元": ("CNY", 1e8), "万元": ("CNY", 1e4), "元": ("CNY", 1.0),
              "天": ("days", 1.0), "倍": ("x", 1.0)}


def extract_numbers(text: str) -> list[tuple[str, float, Optional[str], int]]:
    t = text
    for p in STRIP:
        t = re.sub(p, " ", t)
    out = []
    for m in NUM.finditer(t):
        raw = m.group(0).strip()
        num_s = re.sub(r"[^\d.\-+]", "", raw.replace(",", ""))
        try:
            val = float(num_s)
        except ValueError:
            continue
        decimals = len(num_s.split(".")[1]) if "." in num_s else 0
        out.append((raw, val, m.group(1), decimals))
    return out


def _match_metric(val: float, unit: Optional[str], decimals: int, recs) -> bool:
    for r in recs:
        if r is None or r.value is None:
            continue
        if unit:
            kind, scale = UNIT_SCALE[unit]
            if r.unit != kind:
                continue
            tol = 0.5 * 10 ** (-decimals) * scale + 1e-12
            if abs(val * scale - r.value) <= tol * 1.0001:
                return True
        else:
            for scale in (1.0, 0.01):
                if abs(val * scale - r.value) <= 0.5 * 10 ** (-decimals) * scale + 1e-12:
                    return True
    return False


def tier1(ctx: RunContext, c: Claim) -> list[VerificationFailure]:
    fails = []
    mk = lambda et, msg: VerificationFailure(claim_id=c.claim_id, tier=1, error_type=et,  # noqa: E731
                                             target_agent=c.produced_by, fix_instruction=msg)
    if not c.evidence_ids:
        return [mk("MISSING_CITATION", "claim has no evidence; cite or downgrade")]
    unknown = [e for e in c.evidence_ids if e not in ctx.evidence]
    if unknown:
        fails.append(mk("UNKNOWN_EVIDENCE", f"evidence not in this run's store: {unknown}"))
    evs = [ctx.evidence[e] for e in c.evidence_ids if e in ctx.evidence]
    late = [e.evidence_id for e in evs if e.available_at > ctx.as_of]
    if late:
        fails.append(mk("PIT_VIOLATION", f"evidence available after as_of: {late}"))
    recs = [ctx.metric(m) for e in evs for m in e.metric_ids]
    scopes = {r.statement_scope.value for r in recs if r is not None}
    if {"parent", "consolidated"} <= scopes and "母公司" not in c.text:
        fails.append(mk("SCOPE_MIX", "mixes parent and consolidated metrics without labelling"))
    doc_text = " ".join(e.excerpt for e in evs if e.kind == "document")
    for raw, val, unit, dec in extract_numbers(c.text):
        if raw.replace(" ", "") in doc_text.replace(",", "") or raw in doc_text:
            continue
        if _match_metric(val, unit, dec, recs):
            continue
        same_kind = unit and any(r is not None and r.unit == UNIT_SCALE[unit][0] for r in recs)
        fails.append(mk("NUMBER_MISMATCH" if same_kind else "WILD_NUMBER",
                        f"number '{raw}' does not match any cited MetricRecord or quoted source"))
    return fails


UP = ("增长", "提升", "改善", "上升", "增加")
DOWN = ("下降", "恶化", "下滑", "减少", "承压")


def tier2(ctx: RunContext, c: Claim) -> str:
    evs = [ctx.evidence[e] for e in c.evidence_ids if e in ctx.evidence]
    if ctx.llm.online:
        sys_p = ("你是金融事实核验裁判。判断证据是否支持论断。只输出JSON {\"label\":\"SUPPORTED|PARTIALLY_SUPPORTED|"
                 "CONTRADICTED|UNSUPPORTED\",\"reason\":\"...\"}")
        user = json.dumps({"claim": c.text, "evidence": [{"id": e.evidence_id, "summary": e.summary,
                                                          "excerpt": e.excerpt[:600]} for e in evs]}, ensure_ascii=False)
        d = call_json(ctx.llm, ctx.tracer, "verifier", sys_p, user, "strong",
                      validate=lambda d: d.get("label") in ("SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "UNSUPPORTED"))
        if d:
            return d["label"]
    # offline heuristic
    if c.claim_type == "model_inference":
        recs = [ctx.metric(m) for m in c.metric_ids]
        for r in recs:
            if r is None or r.value is None or not r.metric.endswith("_yoy"):
                continue
            if any(w in c.text for w in UP) and not any(w in c.text for w in DOWN) and r.value < 0:
                return "CONTRADICTED"
            if any(w in c.text for w in DOWN) and not any(w in c.text for w in UP) and r.value > 0:
                return "CONTRADICTED"
        if "置信度低" in c.text:
            return "PARTIALLY_SUPPORTED"
        return "SUPPORTED" if recs else "UNSUPPORTED"
    if c.claim_type == "disclosed_fact" and any(e.kind == "document" for e in evs):
        body = re.sub(r"^[^：]*：", "", c.text)
        grams = {body[i:i + 2] for i in range(len(body) - 1)}
        src = "".join(e.excerpt for e in evs)
        cov = sum(1 for g in grams if g in src) / max(len(grams), 1)
        return "SUPPORTED" if cov >= 0.5 else ("PARTIALLY_SUPPORTED" if cov >= 0.25 else "UNSUPPORTED")
    return "SUPPORTED"


def verify(ctx: RunContext) -> list[VerificationFailure]:
    cfg = ctx.config
    pending: list[VerificationFailure] = []
    for c in ctx.claims.values():
        if c.status != "DRAFT":
            continue
        fails = tier1(ctx, c) if cfg.tier1_verifier else []
        label = "SUPPORTED"
        if not fails and cfg.tier2_verifier:
            label = tier2(ctx, c)
            if label in ("CONTRADICTED", "UNSUPPORTED"):
                fails = [VerificationFailure(claim_id=c.claim_id, tier=2, error_type=label, target_agent=c.produced_by,
                                             fix_instruction="weaken or remove the inference")]
        for f in fails:
            ctx.verification_log.append({**f.model_dump(), "attempt": c.attempts, "passed": False, "circuit_broken": False})
        if not fails:
            c.status = "PARTIALLY_SUPPORTED" if label == "PARTIALLY_SUPPORTED" else "VERIFIED"
            ctx.verification_log.append({"claim_id": c.claim_id, "tier": 2 if cfg.tier2_verifier else 1, "passed": True,
                                         "error_type": None, "attempt": c.attempts, "circuit_broken": False})
        elif c.attempts >= cfg.max_claim_retries:
            c.status = "DOWNGRADED"
            reason = "; ".join(f"{f.error_type}: {f.fix_instruction}" for f in fails)
            ctx.uncertainties.append(UncertaintyStatement(
                claim_id=c.claim_id, reason=reason,
                text=f"以下论断未能通过核验，已降级为待核实事项：{_strip_numbers(c.text)}"))
            ctx.verification_log.append({"claim_id": c.claim_id, "tier": fails[0].tier, "passed": False,
                                         "error_type": fails[0].error_type, "attempt": c.attempts, "circuit_broken": True})
        else:
            pending.extend(fails)
    return pending


def _strip_numbers(text: str) -> str:
    return NUM.sub("[数值待核实]", text)


def rewrite(ctx: RunContext, failures: list[VerificationFailure]) -> int:
    """Partial rollback: only the failing claims are re-rendered by their owners."""
    from .faults import maybe_recorrupt

    touched = set()
    for f in failures:
        c = ctx.claims[f.claim_id]
        if f.error_type in ("PIT_VIOLATION", "UNKNOWN_EVIDENCE"):
            for e in c.evidence_ids:
                ev = ctx.evidence.get(e)
                if ev is not None and ev.available_at > ctx.as_of:
                    c.text = c.text.replace(f"；另据《{ev.summary}》", "").replace(f"《{ev.summary}》", "")
            c.evidence_ids = [e for e in c.evidence_ids if e in ctx.evidence and ctx.evidence[e].available_at <= ctx.as_of]
            c.metric_ids = [m for e in c.evidence_ids for m in ctx.evidence[e].metric_ids]
        if f.error_type == "SCOPE_MIX":
            c.evidence_ids = [e for e in c.evidence_ids
                              if not any((ctx.metric(m) and ctx.metric(m).statement_scope.value == "parent")
                                         for m in ctx.evidence[e].metric_ids)]
        if f.error_type in ("NUMBER_MISMATCH", "WILD_NUMBER", "PIT_VIOLATION", "SCOPE_MIX") and c.claim_id in ctx.renderers:
            c.text = ctx.renderers[c.claim_id]()
        if f.error_type in ("UNSUPPORTED",) and "置信度低" not in c.text:
            c.text = c.text.rstrip("。") + "（该推断证据支持有限，置信度低）。"
        touched.add(c.claim_id)
    for cid in touched:
        ctx.claims[cid].attempts += 1
        maybe_recorrupt(ctx, ctx.claims[cid])
    return len(touched)
