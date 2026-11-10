import json, sys, collections, statistics
sys.path.insert(0, "packages")
from evaluation_platform.evaluators import task_passed
from evaluation_platform.datasets import load_tasks

tasks = {t["task_id"]: t for t in load_tasks(("golden", "blind", "chaos", "lab"))}
for tid in ("B118", "X015"):
    print(tid, tasks[tid]["dataset"], tasks[tid].get("category"))
for a in ("B4", "B3", "B2", "B0"):
    rows = [json.loads(l) for l in open(f"reports/eval/online_real_{a}.partial.jsonl", encoding="utf-8")]
    cat = collections.defaultdict(lambda: [0, 0])
    ok = 0
    for r in rows:
        p = task_passed(r["scores"], tasks[r["task_id"]]); ok += p
        k = f'{r["dataset"]}/{r["category"]}'; cat[k][0] += p; cat[k][1] += 1
    na = [r["scores"]["numerical_accuracy"]["score"] for r in rows if "score" in r["scores"].get("numerical_accuracy", {})]
    cv = [r["scores"]["citation_validity"]["score"] for r in rows if "score" in r["scores"].get("citation_validity", {})]
    pit = sum(1 for r in rows if r["scores"].get("pit_violation", {}).get("score", 0) > 0)
    rub = [r["scores"]["rubric"]["score"] for r in rows if "score" in r["scores"].get("rubric", {})]
    lat = [r["latency_ms"] / 1000 for r in rows]
    tok = [r["scores"].get("cost", {}).get("tokens", 0) for r in rows]
    cost = sum(r["scores"].get("cost", {}).get("cost_cny", 0) for r in rows)
    print(a, f"{ok}/60", {k: f"{v[0]}/{v[1]}" for k, v in sorted(cat.items())})
    print("   num_acc", round(statistics.mean(na), 3) if na else None, len(na), "cite", round(statistics.mean(cv), 3) if cv else None,
          "pit_tasks", pit, "rubric", round(statistics.mean(rub), 3) if rub else None,
          "p50", round(statistics.median(lat)), "p95", round(sorted(lat)[int(0.95 * len(lat)) - 1]),
          "tok_avg", round(statistics.mean(tok)), "cost", round(cost, 2))
