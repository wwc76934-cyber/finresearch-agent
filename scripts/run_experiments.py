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
         "> 运行模式：**离线确定性模式**（未配置 LLM_BASE_URL）。所有 Agent 使用规则/模板路径，Tier-2 与 Rubric 使用离线启发式裁判。",
         "> 数据：6 家**虚构**演示公司的 Golden Snapshot（见 `DATA_CONTRACTS.md`）。结论只说明本数据集上的机制差异，不代表真实市场表现。", "",
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
    L += ["", "## 3. 检索评测 (72 条查询, 6 公司 × 3 报告期 × 4 意图)", "", "| 配置 | Recall@5 | MRR |", "|---|---|---|"]
    for k, v in rag.items():
        if k != "table_integrity":
            L.append(f"| {k} | {pct(v['recall@5'])} | {v['mrr']:.3f} |")
    ti = rag["table_integrity"]
    L += ["", f"跨页表格完整率（合并利润表在第 3/4 页之间断开）：结构感知切片 {pct(ti['structure_aware'])}，固定长度切片 {pct(ti['naive_fixed_size'])}。", "",
          "## 4. 错误根因漏斗", "", "| 架构 | " + " | ".join(agg["B4"]["errors"]) + " |", "|---|" + "---|" * len(agg["B4"]["errors"])]
    for a in BASE + ABL:
        if a in agg:
            L.append(f"| {a} | " + " | ".join(str(agg[a]["errors"][k]) for k in agg["B4"]["errors"]) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    print(write_tasks())
    archs = ["B3", "B4"] if a.quick else BASE + ABL
    exp_id = f"exp_{datetime.now():%Y%m%d_%H%M%S}"
    payload = run_experiment(archs, repeats=a.repeats, persist_archs=("B3", "B4"), name=exp_id)
    cmp = compare(payload["results"])
    rag = rag_eval.run()
    out = ROOT / "reports" / "eval"
    (out / f"{exp_id}_summary.json").write_text(json.dumps({"comparison": cmp, "rag": rag}, ensure_ascii=False, indent=1,
                                                           default=str), encoding="utf-8")
    (out / "summary_latest.json").write_text(json.dumps({"experiment_id": exp_id, "comparison": cmp, "rag": rag},
                                                        ensure_ascii=False, default=str), encoding="utf-8")
    if not a.quick:
        md = write_markdown(cmp, rag, exp_id)
        (out / "EVALUATION_generated.md").write_text(md, encoding="utf-8")
        print(md)
