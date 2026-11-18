"""One real run with the model on a company outside the snapshot. Prints how the v2.0 parts behaved.

  python scripts/demo_live_run.py 300308.SZ "分析海外业务与客户集中度风险" [--as-of 2026-11-18] [--arch B4]
Uses the real database, the real network and the real model (costs a few fen).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
from agent_runtime import ResearchRequest, run_research  # noqa: E402
from agent_runtime.llm import get_llm, load_dotenv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ticker")
ap.add_argument("question")
ap.add_argument("--as-of", default=datetime.now().strftime("%Y-%m-%d"))
ap.add_argument("--arch", default="B4")
ap.add_argument("--mode", default="deep")
a = ap.parse_args()
load_dotenv()
print("llm online:", get_llm().online)
events = []
t0 = time.time()
req = ResearchRequest(ticker=a.ticker, as_of=datetime.fromisoformat(a.as_of + "T15:00:00"), question=a.question, mode=a.mode, data_mode="hybrid_live")
rep = run_research(req, a.arch, listener=lambda e: events.append(e), persist=False)
print(f"total {time.time() - t0:.1f}s  tokens={rep['totals']['input_tokens'] + rep['totals']['output_tokens']}  cost=¥{rep['totals']['cost_cny']:.3f}  llm_failed={rep['totals'].get('llm_calls_failed', 0)}")
print("live_ingestion:", json.dumps(rep["live_ingestion"], ensure_ascii=False)[:500])
print("plan.hypotheses:", [(h["sub_id"], h["assigned_agent"], h["custom_rag_queries"]) for h in rep["plan"]["hypotheses"]])
for s in rep["trace"]:
    for t in s.get("tool_calls") or []:
        if isinstance(t, dict) and t.get("tool") in ("explore_hypothesis", "fetch_periodic_report", "build_peer_matrix"):
            print("  tool", t["tool"], json.dumps(t["args"], ensure_ascii=False), "->", str(t.get("result"))[:100])
print("hypothesis claims:", [c["text"][:150] for c in rep["claims"].values() if "对应假设[H" in c["text"]])
print("open questions:", [q["topic"][:70] for q in rep["open_questions"]])
if rep.get("comparison"):
    c = rep["comparison"]
    print("comparison:", c["period"], [t["name"] for t in c["tickers"]], c["warnings"])
    for r in c["rows"]:
        print("  ", r["dimension"], r["metric"], r["values"])
for u in rep["uncertainties"]:
    print("  DOWNGRADED", u["claim_id"], rep["claims"][u["claim_id"]]["text"][:90], "|", u["reason"][:80])
print("downgraded:", len(rep["uncertainties"]), " status:", {s: sum(1 for c in rep["claims"].values() if c["status"] == s) for s in {c["status"] for c in rep["claims"].values()}})
print("live ingest events:", [e.get("message") for e in events if e.get("type") == "live_ingestion_status"][:6])
