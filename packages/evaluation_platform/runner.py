"""Experiment Runner: run architectures x tasks (x repeats), evaluate, persist, compare."""
from __future__ import annotations

import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from agent_runtime import ResearchRequest, run_research
from data_platform.db import ROOT, EvalExperimentRow, EvalTaskResultRow, get_session_factory

from .datasets import load_tasks
from .error_analysis import classify
from .evaluators import evaluate, task_passed

OUT = ROOT / "reports" / "eval"


def run_task(task: dict, arch: str, persist: bool) -> dict:
    req = ResearchRequest(ticker=task["ticker"], as_of=datetime.fromisoformat(task["as_of"]), question=task["question"],
                          mode=task.get("mode", "deep"), fault_injection=task.get("fault_injection", {}))
    t0 = time.perf_counter()
    try:
        rep = run_research(req, arch, persist=persist)
    except Exception as exc:  # noqa: BLE001 - a crashed run is a scored failure, not a runner crash
        return {"task_id": task["task_id"], "architecture": arch, "run_id": None, "passed": False,
                "category": task["category"], "dataset": task["dataset"],
                "error_category": "TOOL_ERROR", "scores": {"crash": {"error": repr(exc), "tb": traceback.format_exc()[-800:]}},
                "latency_ms": (time.perf_counter() - t0) * 1000}
    scores = evaluate(task, rep)
    return {"task_id": task["task_id"], "architecture": arch, "run_id": rep["run_id"], "category": task["category"],
            "dataset": task["dataset"], "passed": task_passed(scores, task), "error_category": classify(task, rep, scores),
            "scores": scores, "latency_ms": rep["totals"]["latency_ms"],
            "llm_calls_failed": rep["totals"].get("llm_calls_failed", 0)}


def run_experiment(architectures: list[str], datasets=("golden", "chaos"), repeats: int = 1, workers: int = 1,
                   persist_archs: tuple = ("B4",), name: Optional[str] = None, task_filter=None,
                   write_latest: bool = True, resume_file: Optional[Path] = None) -> dict:
    """resume_file: append each finished result as one JSON line; on restart, finished tasks are skipped."""
    tasks = [t for t in load_tasks(datasets) if task_filter is None or task_filter(t)]
    exp_id = name or f"exp_{datetime.now():%Y%m%d_%H%M%S}"
    results = []
    Session = get_session_factory()
    done: dict[tuple, dict] = {}
    if resume_file and Path(resume_file).exists():
        for line in Path(resume_file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["architecture"], r["task_id"])] = r
    lock = threading.Lock()

    def job(t, arch, persist):
        if (arch, t["task_id"]) in done and repeats == 1:
            return done[(arch, t["task_id"])]
        r = run_task(t, arch, persist)
        if resume_file:
            with lock, open(resume_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            print(f"[{arch}] {t['task_id']} passed={r['passed']}", flush=True)
        return r

    for arch in architectures:
        started = datetime.now()
        jobs = [(t, arch, arch in persist_archs) for t in tasks for _ in range(repeats)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            arch_res = list(ex.map(lambda j: job(*j), jobs))
        results += arch_res
        with Session() as s:
            eid = f"{exp_id}:{arch}"
            s.merge(EvalExperimentRow(experiment_id=eid, architecture=arch, started_at=started,
                                      config={"datasets": list(datasets), "repeats": repeats, "n_tasks": len(tasks)},
                                      summary={"pass_rate": sum(r["passed"] for r in arch_res) / max(len(arch_res), 1)}))
            for r in arch_res:
                s.add(EvalTaskResultRow(experiment_id=eid, task_id=r["task_id"], run_id=r["run_id"] or "",
                                        scores=json.loads(json.dumps(r["scores"], default=str)),
                                        judge_notes=json.dumps(r["scores"].get("rubric", {}), ensure_ascii=False, default=str),
                                        error_category=r["error_category"]))
            s.commit()
        print(f"[{arch}] pass={sum(r['passed'] for r in arch_res)}/{len(arch_res)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"experiment_id": exp_id, "created_at": datetime.now().isoformat(), "architectures": architectures,
               "datasets": list(datasets), "repeats": repeats, "results": results}
    (OUT / f"{exp_id}.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    if write_latest:
        (OUT / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    return payload


def run_from_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return run_experiment(cfg["architectures"], tuple(cfg.get("datasets", ["golden", "chaos"])), cfg.get("repeats", 1),
                          cfg.get("workers", 1), tuple(cfg.get("persist_archs", ["B4"])), cfg.get("name"))
