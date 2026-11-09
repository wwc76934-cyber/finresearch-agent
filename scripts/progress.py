"""Show how many research runs finished in the last N minutes (used to watch long online evals)."""
import sqlite3
import sys
from pathlib import Path

minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 60
db = Path(__file__).resolve().parents[1] / "data" / "finresearch.db"
c = sqlite3.connect(db)
q = "select count(*), sum(cost_cny), avg(latency_ms) from research_runs where started_at > datetime('now','localtime',?)"
n, cost, lat = c.execute(q, (f"-{minutes} minutes",)).fetchone()
print(f"runs={n} cost_cny={cost or 0:.3f} avg_latency_s={(lat or 0) / 1000:.1f}")
