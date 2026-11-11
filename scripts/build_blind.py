"""Validate blind tasks written by an outside author and convert them to the benchmark task format.

  python scripts/build_blind.py [benchmarks/blind/blind_v1.jsonl]
Ground truth for `metric` tasks is computed by the independent pandas Oracle from the snapshot. If the
author also gave `expect_value`, a mismatch is reported (it usually means a mistake in the question).
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from data_platform.serving import label_to_end  # noqa: E402
from evaluation_platform.datasets import TASK_DIR, TICKERS, Oracle, _forbidden, label, num, real_securities  # noqa: E402

METRICS = {"operating_revenue", "net_profit", "revenue_yoy", "net_profit_yoy", "gross_margin", "ocf_to_net_profit",
           "single_quarter:operating_revenue", "single_quarter:net_profit", "pe_ttm"}
ANOMALIES = {"PROFIT_OCF_DIVERGENCE", "AR_OUTPACE_REVENUE", "INVENTORY_DAYS_DETERIORATION", "GROSS_MARGIN_SHIFT",
             "LARGE_IMPAIRMENT", "INTEREST_DEBT_JUMP", "RELATED_PARTY_OR_POLICY_CHANGE"}
TYPE_TO_CAT = {"metric": "blind_metric", "anomaly": "anomaly_attribution", "refuse": "refusal_trap", "open": "deep_research"}


def truth(o: Oracle, t: str, m: str, pe: date, as_of: datetime, scope: str):
    if m.startswith("single_quarter:"):
        return o.single_q(t, m.split(":")[1], pe, as_of)
    if m.endswith("_yoy"):
        return o.yoy(t, {"revenue_yoy": "operating_revenue", "net_profit_yoy": "net_profit"}[m], pe, as_of)
    if m == "gross_margin":
        rev, cost = o.val(t, "operating_revenue", pe, as_of), o.val(t, "operating_cost", pe, as_of)
        return ((rev - cost) / rev, "OK") if rev and cost is not None else (None, "INSUFFICIENT_DATA")
    if m == "ocf_to_net_profit":
        ocf, np_ = o.val(t, "operating_cash_flow", pe, as_of), o.val(t, "net_profit", pe, as_of)
        if ocf is None or np_ is None:
            return None, "INSUFFICIENT_DATA"
        return (ocf / np_, "OK") if np_ > 0 else (None, "NEGATIVE_EARNINGS_NA")
    if m == "pe_ttm":
        return o.pe_ttm(t, as_of)
    v = o.val(t, m, pe, as_of, scope)
    return (v, "OK") if v is not None else (None, "INSUFFICIENT_DATA")


def build(src: Path, dataset: str = "blind") -> list[dict]:
    o, out, errors = Oracle(), [], []
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        q = json.loads(line)
        where = f"line {i} ({q.get('id', '?')})"
        miss = [k for k in ("id", "author", "ticker", "as_of", "question", "type") if not q.get(k)]
        if miss:
            errors.append(f"{where}: missing {miss}")
            continue
        if q["ticker"] not in TICKERS + list(real_securities()) or q["type"] not in TYPE_TO_CAT:
            errors.append(f"{where}: bad ticker or type")
            continue
        as_of = datetime.fromisoformat(q["as_of"])
        expected = {"forbidden": _forbidden(o, q["ticker"], as_of)}
        if q["type"] == "metric":
            m = q.get("expect_metric")
            if m not in METRICS:
                errors.append(f"{where}: expect_metric must be one of {sorted(METRICS)}")
                continue
            scope = q.get("expect_scope", "consolidated")
            if m == "pe_ttm":
                v, st = truth(o, q["ticker"], m, as_of.date(), as_of, scope)
                period = as_of.date().isoformat()
            else:
                period = q.get("expect_period") or label(o.latest(q["ticker"], as_of))
                v, st = truth(o, q["ticker"], m, label_to_end(period), as_of, scope)
            if v is None and st == "INSUFFICIENT_DATA":
                errors.append(f"{where}: no data for {m} {period} at {as_of} (is the period visible yet?)")
                continue
            ev = q.get("expect_value")
            if ev is not None and v is not None and abs(ev - v) > abs(v) * 1e-3:
                errors.append(f"{where}: your expect_value {ev} != snapshot {v:.6g}")
            expected["numbers"] = [num(q["id"], m, period, v, st, scope=scope, tol_rel=1e-3 if m == "pe_ttm" else 1e-4)]
            expected["latest_period"] = label(o.latest(q["ticker"], as_of))
        elif q["type"] == "anomaly":
            an = q.get("expect_anomalies") or []
            if not set(an) <= ANOMALIES:
                errors.append(f"{where}: unknown anomaly codes {set(an) - ANOMALIES}")
                continue
            expected["anomalies"] = an
        elif q["type"] == "refuse":
            expected["must_refuse"] = True
        else:
            expected["rubric"] = True
        out.append({"task_id": q["id"], "dataset": dataset, "category": TYPE_TO_CAT[q["type"]], "trap_type": q["type"],
                    "ticker": q["ticker"], "as_of": as_of.isoformat(), "question": q["question"],
                    "mode": "deep" if q["type"] in ("open", "anomaly") else "fast", "fault_injection": {},
                    "expected": expected, "author": q["author"], "notes": q.get("notes", "")})
    if errors:
        print("\n".join(errors))
        sys.exit(f"{len(errors)} error(s); nothing written")
    return out


if __name__ == "__main__":
    # python scripts/build_blind.py [source.jsonl] [dataset]   dataset "blind" (default) or e.g. "blind2"
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmarks" / "blind" / "blind_v1.jsonl"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "blind"
    if not src.exists():
        sys.exit(f"{src} not found; copy benchmarks/blind/blind_template.jsonl and write your questions")
    tasks = build(src, dataset)
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    out = TASK_DIR / f"{dataset}_v1.jsonl"
    out.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n", encoding="utf-8")
    print(f"{len(tasks)} {dataset} task(s) -> {out}")
