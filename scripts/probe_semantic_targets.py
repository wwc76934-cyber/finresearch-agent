"""Which (company, fallacy type) pairs have a claim we can corrupt? Prints B4 vs B4-no_tier2 outcome (offline).

  python scripts/probe_semantic_targets.py
"""
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["LLM_OFFLINE"] = "1"
os.environ.setdefault("CHECKPOINTER", "memory")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from agent_runtime.graph import run_research  # noqa: E402
from agent_runtime.state import ResearchRequest  # noqa: E402
from evaluation_platform.datasets import load_tasks  # noqa: E402

TYPES = ["TREND_REVERSAL", "SCOPE_SWITCH", "CONDITION_OMISSION", "CAUSAL_INVERSION"]

if __name__ == "__main__":
    golden = {t["ticker"]: t for t in load_tasks(("golden",)) if t["category"] == "deep_research"}
    for tk, t in golden.items():
        row = []
        for ty in TYPES:
            cell = []
            for arch in ("B4", "B4-no_tier2"):
                req = ResearchRequest(ticker=tk, as_of=datetime.fromisoformat(t["as_of"]), question=t["question"], mode="deep",
                                      fault_injection={"semantic": {"type": ty}})
                rep = run_research(req, arch, persist=False)
                sf = rep.get("semantic_fault") or {}
                if sf.get("applied") in (None, "none"):
                    cell.append("no_target")
                    break
                kept = any(c["text"] == sf["injected_text"] for s in rep["sections"] for c in [rep["claims"][i] for i in s["claim_ids"]])
                cell.append("KEPT" if kept else "stopped")
            row.append(f"{ty[:5]}:{'/'.join(cell)}")
        print(tk, t["task_id"], t["as_of"][:10], " ".join(row), flush=True)
