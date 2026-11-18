"""Episodic thesis memory (v2.0 Epic 4): what a run concluded and what would prove it wrong, kept across runs, and the
review that checks those conditions against disclosures that came out later.

Rules
* Off in `offline_golden` mode (neither read nor written): a benchmark must not change because an earlier run left a
  thesis behind. `ResearchRequest.use_memory` overrides either way (tests turn it on explicitly).
* The review is deterministic. A falsification condition is a metric, a direction and a threshold; the review recomputes
  the metric at the new as_of from the PIT data and applies the same rule. No model decides whether a thesis held.
* Every number in a review sentence is a MetricRecord (the old period recomputed, the new period, the threshold), so the
  verifier checks the review like any other claim.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Optional

from data_platform.db import ResearchThesisRow, get_session_factory
from data_platform.serving import label_to_end, period_label
from sqlalchemy import delete, select

from .context import RunContext, fmt
from .state import ResearchRequest

STATUS_CN = {"CONFIRMED": "得到新数据强化", "WEAKENED": "边际弱化", "FALSIFIED": "已触发证伪红线", "UNKNOWN": "数据不足，无法判定"}
SEVERITY = {"CONFIRMED": 0, "UNKNOWN": 1, "WEAKENED": 2, "FALSIFIED": 3}
SNAPSHOT_METRICS = [("ratios", k) for k in ("gross_margin", "net_margin", "roe_annualized", "ocf_to_net_profit", "debt_to_assets",
                                            "receivable_days", "inventory_days", "interest_debt_ratio")] + \
                   [("growth", "revenue_yoy"), ("growth", "net_profit_yoy"), ("valuation", "pe_ttm"), ("valuation", "pb_lf")]
WEAKEN_AT = 0.5        # moved this share of the way from the baseline to the threshold -> WEAKENED
STANCE_GAP = 2         # verified bull claims minus verified bear claims needed for a lean


def memory_enabled(request: ResearchRequest) -> bool:
    if request.use_memory is not None:
        return request.use_memory
    return request.data_mode != "offline_golden"


def stance_of(ctx: RunContext) -> str:
    """The balance of the verified arguments, not a rating: bullish / bearish when one side leads by STANCE_GAP."""
    ok = ("VERIFIED", "PARTIALLY_SUPPORTED")
    bull = sum(1 for c in ctx.claims.values() if c.section == "bull" and c.status in ok)
    bear = sum(1 for c in ctx.claims.values() if c.section == "bear" and c.status in ok)
    return "bullish" if bull - bear >= STANCE_GAP else "bearish" if bear - bull >= STANCE_GAP else "neutral"


def judge(spec: dict[str, Any], baseline: Optional[float], now: Optional[float]) -> str:
    """CONFIRMED / WEAKENED / FALSIFIED for one condition. comparator 'lt': falsified when the metric falls below the
    threshold; 'gt': when it rises above it."""
    if now is None or baseline is None:
        return "UNKNOWN"
    thr, lt = float(spec["threshold"]), spec["comparator"] == "lt"
    if (lt and now < thr) or (not lt and now > thr):
        return "FALSIFIED"
    gap = abs(baseline - thr)
    moved = (baseline - now) if lt else (now - baseline)          # progress toward the threshold
    return "WEAKENED" if gap > 0 and moved / gap >= WEAKEN_AT else "CONFIRMED"


class ThesisMemoryStore:
    def __init__(self, db_url: Optional[str] = None):
        self.Session = get_session_factory(db_url)

    # ------------------------------------------------------------ write
    def save(self, ctx: RunContext, rep: dict) -> Optional[str]:
        """Called after verification. Needs the falsification conditions of a full debate; anything else leaves no thesis."""
        specs = (rep.get("debate") or {}).get("falsification_specs") or []
        if not specs or ctx.request.mode != "deep":
            return None
        kept = {c for s in rep["sections"] for c in s["claim_ids"]}
        specs = [s for s in specs if s["claim_id"] in kept]           # a condition that was downgraded is not a commitment
        if not specs:
            return None
        pe = ctx.svc.latest_period(ctx.ticker)
        snap: dict[str, Any] = {"period": period_label(pe) if pe else None}
        if pe is not None:
            recs = {"ratios": ctx.calc.ratios(ctx.ticker, pe), "growth": ctx.calc.growth(ctx.ticker, pe),
                    "valuation": ctx.calc.valuation(ctx.ticker)}
            snap.update({k: recs[fam][k].value for fam, k in SNAPSHOT_METRICS if fam in recs and k in recs[fam]
                         and recs[fam][k].value is not None})
        summary = (rep.get("debate") or {}).get("manager_summary") or rep.get("executive_summary") or ""
        if not summary:
            top = [rep["claims"][c]["text"] for s in rep["sections"] if s["key"] in ("bull", "bear") for c in s["claim_ids"][:1]]
            summary = "；".join(top)[:300]
        tid = f"thesis_{ctx.ticker[:6]}_{ctx.as_of:%Y%m%d%H%M}_{uuid.uuid4().hex[:6]}"
        with self.Session() as s:
            s.add(ResearchThesisRow(thesis_id=tid, ticker=ctx.ticker, as_of_date=ctx.as_of.date().isoformat(), run_id=ctx.run_id,
                                    stance=stance_of(ctx), core_thesis_summary=summary[:500], falsification_criteria=specs,
                                    locked_metrics_snapshot=snap, created_at=datetime.now().isoformat(timespec="seconds"),
                                    scope=ctx.request.memory_scope))
            s.commit()
        return tid

    # ------------------------------------------------------------ read
    def purge(self, scope: str) -> int:
        """Delete a benchmark scope's theses (never the user's own, whose scope is empty)."""
        if not scope:
            return 0
        with self.Session() as s:
            n = s.execute(delete(ResearchThesisRow).where(ResearchThesisRow.scope == scope)).rowcount
            s.commit()
        return int(n or 0)

    def prior(self, ticker: str, before: datetime, limit: int = 2, scope: str = "") -> list[ResearchThesisRow]:
        with self.Session() as s:
            rows = list(s.scalars(select(ResearchThesisRow).where(ResearchThesisRow.ticker == ticker,
                                                                  ResearchThesisRow.scope == scope,
                                                                  ResearchThesisRow.as_of_date < before.date().isoformat())
                                  .order_by(ResearchThesisRow.as_of_date.desc(), ResearchThesisRow.created_at.desc())))
        seen: set[str] = set()
        out = []
        for r in rows:                      # one thesis per earlier as_of date (re-runs of the same day are duplicates)
            if r.as_of_date not in seen:
                seen.add(r.as_of_date)
                out.append(r)
        return out[:limit]

    def listing(self, ticker: Optional[str] = None, limit: int = 30) -> list[dict]:
        with self.Session() as s:
            stmt = select(ResearchThesisRow).where(ResearchThesisRow.scope == "").order_by(ResearchThesisRow.created_at.desc()).limit(limit)
            if ticker:
                stmt = stmt.where(ResearchThesisRow.ticker == ticker)
            return [{"thesis_id": r.thesis_id, "ticker": r.ticker, "as_of_date": r.as_of_date, "stance": r.stance,
                     "summary": r.core_thesis_summary, "criteria": len(r.falsification_criteria), "run_id": r.run_id}
                    for r in s.scalars(stmt)]


# ---------------------------------------------------------------------------------------------------- review
def review(ctx: RunContext, store: Optional[ThesisMemoryStore] = None) -> dict[str, Any]:
    """Check earlier theses of this company against what has been disclosed since. Adds one claim per condition to the
    `review` section and returns a summary for the report and the debate."""
    store = store or ThesisMemoryStore(ctx.db_url)
    out: dict[str, Any] = {"items": [], "notes": []}
    priors = store.prior(ctx.ticker, ctx.as_of, scope=ctx.request.memory_scope)
    if not priors:
        return out
    pe_new = ctx.svc.latest_period(ctx.ticker)
    if pe_new is None:
        return out
    new_recs = ctx.calc.ratios(ctx.ticker, pe_new)
    for th in priors:
        snap_period = (th.locked_metrics_snapshot or {}).get("period")
        if not snap_period or label_to_end(snap_period) >= pe_new:
            msg = f"{th.as_of_date} 的研究（{th.stance}）之后没有新披露的财报，无法复盘。"
            out["notes"].append(msg)
            ctx.open_questions.append({"topic": "历史观点复盘", "reason": msg})
            continue
        pe_old = label_to_end(snap_period)
        old_recs = ctx.calc.ratios(ctx.ticker, pe_old)
        statuses = []
        for spec in th.falsification_criteria:
            m = spec["metric"]
            old, new = old_recs.get(m), new_recs.get(m)
            if old is None or new is None:
                statuses.append(("UNKNOWN", spec))
                continue
            thr_rec = ctx.calc._record(ctx.ticker, f"thesis_threshold_{m}_{th.thesis_id[-6:]}", period_label(pe_new), float(spec["threshold"]), new.unit,
                                       [new], extra={"rule": "thesis_review", "thesis": th.thesis_id})
            status = judge(spec, old.value, new.value)
            statuses.append((status, spec))
            if abs((old.value or 0) - (spec.get("baseline") or 0)) > 1e-6 and spec.get("baseline") is not None:
                note = f"{spec['metric']} {snap_period} 的数值与当时记录不同：数据商后来修订过该期。"
                if note not in out["notes"]:
                    out["notes"].append(note)
                    ctx.open_questions.append({"topic": "历史观点复盘", "reason": note})

            def render(status: str = status, spec: dict = spec, old=old, new=new, thr_rec=thr_rec, th: ResearchThesisRow = th,
                       snap_period: str = snap_period) -> str:
                word = "低于" if spec["comparator"] == "lt" else "高于"
                return (f"[{status}] {th.as_of_date}研究（{STATUS_CN_STANCE.get(th.stance, th.stance)}）设定的证伪条件：{spec['label']}{word}{fmt(thr_rec)}。"
                        f"{snap_period}为{fmt(old)}，{period_label(pe_new)}为{fmt(new)}，{STATUS_CN[status]}。")
            evs = [ctx.evidence_from_metric(r, "thesis_review") for r in (old, new, thr_rec)]
            cl = ctx.add_claim("review", "derived_metric", render(), evs, "thesis_review")
            ctx.renderers[cl.claim_id] = render
            out["items"].append({"thesis_id": th.thesis_id, "as_of_date": th.as_of_date, "stance": th.stance, "metric": m,
                                 "label": spec["label"], "status": status, "claim_id": cl.claim_id})
        overall = max((s for s, _ in statuses), key=lambda s: SEVERITY[s], default="UNKNOWN")
        out.setdefault("theses", []).append({"thesis_id": th.thesis_id, "as_of_date": th.as_of_date, "stance": th.stance,
                                             "summary": th.core_thesis_summary, "overall": overall,
                                             "from_period": snap_period, "to_period": period_label(pe_new)})
    ctx.tracer.tool("thesis_review", {"ticker": ctx.ticker, "theses": len(priors)}, result=f"{len(out['items'])} conditions")
    return out


STATUS_CN_STANCE = {"bullish": "偏多", "bearish": "偏空", "neutral": "中性"}


def review_summary(rv: Optional[dict]) -> list[dict]:
    """What the debate model is told: earlier conditions and how they stand (names and statuses, no numbers)."""
    return [{"as_of": i["as_of_date"], "stance": i["stance"], "condition": i["label"], "status": i["status"]}
            for i in (rv or {}).get("items", [])]
