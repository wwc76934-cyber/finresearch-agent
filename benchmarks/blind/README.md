# 盲测题出题说明（给出题人）

请**不要**先读本仓库的代码、`benchmarks/tasks/` 或 `EVALUATION.md`。你只需要知道下面这些。

## 你能用的数据

8 家真实 A 股公司，数据都在 `benchmarks/real_snapshot/`（可以直接打开 CSV 和 `documents/*.md` 看定期报告与公告原文）：

| 代码 | 名称 |
|---|---|
| 300750.SZ | 宁德时代 |
| 300014.SZ | 亿纬锂能 |
| 002594.SZ | 比亚迪 |
| 601012.SH | 隆基绿能 |
| 600519.SH | 贵州茅台 |
| 000858.SZ | 五粮液 |
| 000333.SZ | 美的集团 |
| 000651.SZ | 格力电器 |

* 财报期：2021Q1–2026Q3（合并报表）；行情：2021-12 至 2026-11；文档：2024H1–2025 年报及部分公告。
* `available_at` 列是“最早什么时候能用这条数据”。研究时点 `as_of` 之前不可用的数据，系统不该用。
* 同一期数据可能有多个 `revision_id`（例如后来的重述）；`as_of` 时点只能看到当时已公开的那一版。
* `benchmarks/golden_snapshot/` 另有 6 家虚构“实验室”公司（990001–990006），也可以出题。

## 怎么写一道题

在 `benchmarks/blind/blind_v1.jsonl` 里每行写一道题（参考 `blind_template.jsonl`）：

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 自己起，如 `B001` |
| `author` | 是 | 你的名字或代号 |
| `ticker` | 是 | 上表中的代码 |
| `as_of` | 是 | 研究时点，如 `2025-11-03T15:00:00` |
| `question` | 是 | 自然语言问题，像真实投研人员那样问 |
| `type` | 是 | `metric`（问数字）/ `anomaly`（问异常）/ `refuse`（系统应拒答）/ `open`（开放研究） |
| `expect_metric` | type=metric | 见下表；系统会按快照自动算出标准答案 |
| `expect_period` | type=metric | 如 `2025Q3`；时点类指标（PE）可不填 |
| `expect_scope` | 否 | `consolidated`（默认，合并）或 `parent`（母公司） |
| `expect_value` | 否 | 你自己算的答案（元或小数比率）。填了会和自动算的对比，不一致会报错，帮你发现出题错误 |
| `expect_anomalies` | type=anomaly | 你认为应发现的异常，代码见下表 |
| `notes` | 否 | 出题意图、陷阱在哪 |

`expect_metric` 可选：`operating_revenue`、`net_profit`、`revenue_yoy`、`net_profit_yoy`、`gross_margin`、`ocf_to_net_profit`、`single_quarter:operating_revenue`、`single_quarter:net_profit`、`pe_ttm`。

`expect_anomalies` 可选：`PROFIT_OCF_DIVERGENCE`（利润与现金流背离）、`AR_OUTPACE_REVENUE`（应收增速远超营收）、`INVENTORY_DAYS_DETERIORATION`（存货周转恶化）、`GROSS_MARGIN_SHIFT`（毛利率突变）、`LARGE_IMPAIRMENT`（大额减值）、`INTEREST_DEBT_JUMP`（有息负债跃升）、`RELATED_PARTY_OR_POLICY_CHANGE`（关联交易或会计变更）。

## 写完之后

```bash
python scripts/build_blind.py          # 校验并生成 benchmarks/tasks/blind_v1.jsonl
python scripts/run_online_eval.py --datasets blind --name online_B4_blind
```

建议至少 20 题，其中至少 5 道 `refuse`、5 道带时间陷阱（`as_of` 卡在公告发布前后）。
