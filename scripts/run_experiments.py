"""Run B0-B4 baselines + 7 ablations on the 80-task benchmark, RAG eval, and write EVALUATION.md.

  python scripts/run_experiments.py            # everything
  python scripts/run_experiments.py --quick    # B3/B4 only
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from evaluation_platform import rag_eval  # noqa: E402
from evaluation_platform.comparison import compare  # noqa: E402
from evaluation_platform.datasets import write_tasks  # noqa: E402
from evaluation_platform.runner import run_experiment  # noqa: E402

BASE = ["B0", "B1", "B2", "B3", "B4"]
ABL = ["B4-no_anchor_prefilter", "B4-no_hybrid_rerank", "B4-no_context_isolation", "B4-no_replanning",
       "B4-no_debate", "B4-no_tier1", "B4-no_tier2"]
LABEL = {"B4-no_anchor_prefilter": "-口径预过滤与表格锚点", "B4-no_hybrid_rerank": "-Hybrid RRF & Reranker",
         "B4-no_context_isolation": "-子图上下文隔离", "B4-no_replanning": "-Anomaly 动态重规划",
         "B4-no_debate": "-Bull/Bear 多空辩论", "B4-no_tier1": "-Tier-1 确定性校验器", "B4-no_tier2": "-Tier-2 语义 Verifier"}


def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def write_markdown(cmp: dict, rag: dict, exp_id: str) -> str:
    agg, d = cmp["aggregate"], cmp["delta_vs_reference"]
    L = ["# EVALUATION — FinResearch Agent 评测报告", "",
         f"> 自动生成于 {datetime.now():%Y-%m-%d %H:%M}，实验 `{exp_id}`，原始数据 `reports/eval/{exp_id}.json`。",
         "> 运行模式：**离线确定性模式**（强制 LLM_OFFLINE=1）。所有 Agent 使用规则/模板路径，Tier-2 与 Rubric 使用离线启发式裁判。真实模型结果见后文“在线”各节。",
         "> 数据：8 家真实 A 股（东方财富 F10 三表 + 巨潮原始财报/公告 + Yahoo 日线）与 6 家虚构“实验室”公司（故障注入用）。见 `DATA_CONTRACTS.md`。", "",
         "## 1. 五级基线对比 (B0–B4)", "",
         "| 架构 | 任务通过率 | 数值准确率 | 引用有效 | 语义支持 | PIT 泄漏率 | 异常召回 | 拒答正确 | 冲突识别 | 故障自愈 | Rubric | 冗余工具率 | P50 ms | P95 ms | 平均 tokens |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in BASE:
        r = agg.get(a)
        if not r:
            continue
        L.append(f"| {a} | {pct(r['pass_rate'])} | {pct(r['numerical_accuracy'])} | {pct(r['citation_validity'])} | "
                 f"{pct(r['support_rate'])} | {pct(r['pit_violation'])} | {pct(r['anomaly_recall'])} | {pct(r['refusal'])} | "
                 f"{pct(r['conflict_handling'])} | {pct(r['recovery'])} | {pct(r['rubric'])} | {pct(r['redundant_tool_rate'])} | "
                 f"{r['latency_p50_ms']:.0f} | {r['latency_p95_ms']:.0f} | {r['tokens_mean']:.0f} |")
    L += ["", "PIT 泄漏率 = 出现任一未来数据/未来公告的任务占比（目标 0%）。tokens 为离线模式下各节点实际构造的上下文的估算 token 数。", "",
          "### 按题型通过率", "", "| 架构 | " + " | ".join(sorted(agg["B4"]["pass_by_category"])) + " |",
          "|---|" + "---|" * len(agg["B4"]["pass_by_category"])]
    for a in BASE + ABL:
        if a in agg:
            L.append(f"| {a} | " + " | ".join(pct(agg[a]["pass_by_category"].get(k)) for k in sorted(agg["B4"]["pass_by_category"])) + " |")
    L += ["", "## 2. 七项消融 (在 B4 上单独关闭一个机制)", "",
          "| 消融 | 通过率 | Δ通过率 (B4−X) | 95% CI (paired bootstrap) | 数值准确率 | PIT 泄漏率 | 故障自愈 | Rubric | tokens 比 (B4/X) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in ABL:
        if a not in agg:
            continue
        r, dd = agg[a], d[a]
        bs = dd.get("bootstrap_pass_rate", {})
        ci = f"[{bs['ci95'][0] * 100:+.1f}, {bs['ci95'][1] * 100:+.1f}]" if bs else "—"
        L.append(f"| {LABEL[a]} | {pct(r['pass_rate'])} | {dd['pass_rate'] * 100:+.1f}pp | {ci} | {pct(r['numerical_accuracy'])} | "
                 f"{pct(r['pit_violation'])} | {pct(r['recovery'])} | {pct(r['rubric'])} | {dd['token_ratio']:.2f} |")
    real = rag.get("real", {})
    L += ["", f"## 3. 检索评测（实验室 {rag['bm25']['n']} 条 / 真实财报 {real.get('bm25', {}).get('n', 0)} 条查询）", "",
          "| 配置 | 实验室 Recall@5 | 实验室 MRR | 真实 Recall@5 | 真实 MRR |", "|---|---|---|---|---|"]
    for k, v in rag.items():
        if k in ("table_integrity", "real"):
            continue
        rv = real.get(k, {})
        L.append(f"| {k} | {pct(v['recall@5'])} | {v['mrr']:.3f} | {pct(rv.get('recall@5'))} | "
                 f"{rv.get('mrr', 0):.3f} |")
    ti = rag["table_integrity"]
    rti = real.get("table_integrity", {})
    L += ["", f"跨页表格完整率：实验室结构感知切片 {pct(ti['structure_aware'])}，固定长度切片 {pct(ti['naive_fixed_size'])}；"
          f"真实财报 {rti.get('statements_spanning_pages', 0)} 张跨页合并利润表，结构感知完整率 {pct(rti.get('structure_aware'))}。", "",
          "## 4. 错误根因漏斗", "", "| 架构 | " + " | ".join(agg["B4"]["errors"]) + " |", "|---|" + "---|" * len(agg["B4"]["errors"])]
    for a in BASE + ABL:
        if a in agg:
            L.append(f"| {a} | " + " | ".join(str(agg[a]["errors"][k]) for k in agg["B4"]["errors"]) + " |")
    return "\n".join(L) + "\n"


def by_dataset(results: list[dict], archs: list[str]) -> str:
    sets = ["golden", "blind", "chaos", "lab"]
    L = ["## 0. 按数据集通过率", "", "golden = 8 家真实 A 股（Tier-A 30 题）；blind = 独立出题人在真实数据上出的 30 题；"
         "chaos = 虚构实验室公司上的故障注入 50 题；lab = 虚构实验室公司上的旧版 30 题。", "",
         "| 架构 | " + " | ".join(sets) + " | 合计 |", "|---|" + "---|" * (len(sets) + 1)]
    for a in archs:
        rs = [r for r in results if r["architecture"] == a]
        cells = []
        for d in sets:
            x = [r["passed"] for r in rs if r.get("dataset") == d]
            cells.append(f"{sum(x)}/{len(x)}" if x else "—")
        L.append(f"| {a} | " + " | ".join(cells) + f" | {sum(r['passed'] for r in rs)}/{len(rs)} |")
    return "\n".join(L) + "\n\n"


if __name__ == "__main__":
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--datasets", default="golden,blind,chaos,lab")
    a = ap.parse_args()
    os.environ["LLM_OFFLINE"] = "1"   # this script is the offline matrix; online runs use run_online_eval.py
    print(write_tasks())
    archs = ["B3", "B4"] if a.quick else BASE + ABL
    exp_id = f"exp_{datetime.now():%Y%m%d_%H%M%S}"
    payload = run_experiment(archs, datasets=tuple(a.datasets.split(",")), repeats=a.repeats,
                             persist_archs=("B3", "B4"), name=exp_id)
    cmp = compare(payload["results"])
    rag = rag_eval.run()
    out = ROOT / "reports" / "eval"
    (out / f"{exp_id}_summary.json").write_text(json.dumps({"comparison": cmp, "rag": rag}, ensure_ascii=False, indent=1,
                                                           default=str), encoding="utf-8")
    (out / "summary_latest.json").write_text(json.dumps({"experiment_id": exp_id, "comparison": cmp, "rag": rag},
                                                        ensure_ascii=False, default=str), encoding="utf-8")
    if not a.quick:
        md = write_markdown(cmp, rag, exp_id)
        head, _, rest = md.partition("## 1.")
        md = head + by_dataset(payload["results"], archs) + "## 1." + rest
        (out / "EVALUATION_generated.md").write_text(md, encoding="utf-8")
        print(md)
