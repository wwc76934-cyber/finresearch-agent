"""Time per graph node and LLM calls for one task (online).  python scripts/profile_task.py TASK_ID [ARCH]"""
import json, sys, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
from agent_runtime.graph import run_research
from agent_runtime.state import ResearchRequest
from evaluation_platform.datasets import load_tasks

tid = sys.argv[1]; arch = sys.argv[2] if len(sys.argv) > 2 else "B4"
t = next(x for x in load_tasks(("golden", "blind", "chaos", "lab", "quant", "multidoc", "semantic")) if x["task_id"] == tid)
req = ResearchRequest(ticker=t["ticker"], as_of=datetime.fromisoformat(t["as_of"]), question=t["question"], mode=t.get("mode", "deep"),
                      fault_injection=t.get("fault_injection", {}))
t0 = time.perf_counter(); rep = run_research(req, arch, persist=False)
print(tid, t["category"], t["question"][:60], "total", round(time.perf_counter() - t0, 1), "s", rep["totals"])
for s in rep.get("trace", []):
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items() if k in ("node", "ms", "duration_ms", "latency_ms", "summary", "started", "calls")})

for s in rep.get('trace', []):
    for c in s.get('llm_calls', []):
        if c.get('online'):
            print('  LLM', s['node'], c['tier'], 'in', c['in'], 'out', c['out'], round(c['latency_ms'] / 1000, 1), 's')
