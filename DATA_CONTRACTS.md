# DATA_CONTRACTS

所有契约定义在 `packages/data_platform/schema.py`（Pydantic v2，`frozen=True, extra="forbid"`）与 `packages/agent_runtime/state.py`。存储表定义在 `packages/data_platform/db.py`。

## 1. 三时间戳 PIT 规范

| 字段 | 含义 | 规则 |
|---|---|---|
| `period_end` | 会计期末 | 绝不当作可用时间 |
| `published_at` | 法定渠道首次披露时间 | 契约校验：`published_at.date() ≥ period_end` |
| `available_at` | 研究/交易视角下最早可用时间 | 契约校验：`available_at ≥ published_at`。盘后披露 → 下一个交易日 09:30 |

**PIT 铁律**（`data_platform/serving.py`）：

1. 每个查询绑定 `(as_of, snapshot_id)`，SQL 层注入 `available_at <= as_of`。
2. 同一 `period_end` 有多个 `revision_id` 时，取 `as_of` 前已公开的最大 revision。
3. 多个来源时按 Registry 优先级取值，差异 > 0.5% 记入 `conflicts`。
4. `published_at` 不可靠的事实被质量引擎标记 `pit_unsafe=true`，默认不返回。
5. 任何返回记录若 `available_at > as_of`，抛出 `PITViolation`。

示例：990001 的 2025Q3 报告 `published_at=2025-10-28 20:00`、`available_at=2025-10-29 09:30`。`as_of=2025-10-28 21:00` 时最新可见期是 2025Q2（`tests/test_data_platform.py::test_pit_hides_report_published_after_close_until_next_open`）。

## 2. FinancialFact

```json
{"fact_id": "fact_990001_20250930_operating_revenue_con_filing_v1", "ticker": "990001.SZ",
 "metric": "operating_revenue", "statement_scope": "consolidated", "period_type": "cumulative_q3",
 "value": 350238051536.99, "unit": "CNY", "period_end": "2025-09-30",
 "published_at": "2025-10-28T20:00:00", "available_at": "2025-10-29T09:30:00",
 "retrieved_at": "2026-04-01T10:00:00", "source": "demo_filing",
 "source_document_id": "doc_990001_2025Q3", "revision_id": 1, "is_restated": false, "pit_unsafe": false}
```

* `statement_scope`：`consolidated` | `parent`，强制区分。
* `period_type`：`cumulative_q1|cumulative_h1|cumulative_q3|annual|single_quarter|instant`。利润表/现金流表存累计值，单季值由算子派生。
* `unit`：入库统一为元（`CNY`）；`CNY_10K`、`CNY_100M` 只出现在来源解析阶段。

## 3. MetricRecord（算子输出）

```json
{"metric_id": "m_990001_single_quarter_2025Q3_operating_revenue", "metric": "single_quarter", "value": 1.27e11,
 "unit": "CNY", "status": "OK", "period": "2025Q3", "statement_scope": "consolidated",
 "formula": "cum_q - cum_{q-1}; flagged if earlier cum restated after later cum was published", "formula_version": "v1",
 "input_ids": ["fact_..._20250930_...", "fact_..._20250630_..."], "available_at": "2025-10-29T09:30:00",
 "as_of": "2025-12-31T15:00:00", "extra": {"base_metric": "operating_revenue"}}
```

`status` 取值：`OK`、`NEGATIVE_EARNINGS_NA`、`ZERO_DENOMINATOR_NA`、`NEGATIVE_EQUITY_NA`、`INSUFFICIENT_DATA`、`RESTATEMENT_BREAKS_SPLIT`。非 OK 时 `value=null`，报告显示状态文字而不是数字。

单季拆分规则：若上期累计值的被选 revision 晚于本期累计值发布（即本期累计是在旧口径上披露的），返回 `RESTATEMENT_BREAKS_SPLIT`。例：990002 在 2024-12-10 更正了 2024H1 营收，因此 2025 年视角下 2024Q3 单季营收不可直接相减。

指标公式与行业模板见 `packages/data_platform/config/metrics.yaml`。

## 4. 证据与论断

| 契约 | 关键字段 |
|---|---|
| `EvidenceCard` | `evidence_id`、`kind`(fact/metric/document/event)、`source_ref`(metric_id 或 chunk_id)、`available_at`、`metric_ids`、`document_id`、`page` |
| `Claim` | `claim_id`、`section`、`claim_type`(disclosed_fact / derived_metric / model_inference)、`evidence_ids`、`metric_ids`、`status`、`attempts` |
| `VerificationFailure` | `claim_id`、`tier`、`error_type`、`target_agent`、`fix_instruction` |
| `UncertaintyStatement` | `claim_id`、`text`（数字替换为“[数值待核实]”）、`reason` |
| `AnomalyFlag` | `code`、`severity`、`metric_ids`、`message`、`suggested_query`、`section_type` |

## 5. 字段级血缘

`data_lineage_edges` 记录每一跳（`parent_entity_id → child_entity_id`、`transform_tool`、`formula_version`、`run_id`）：

```text
document(doc_990001_2025Q3) → fact(fact_..._operating_revenue_...) → metric(m_990001_revenue_yoy_2025Q3)
  → evidence(E002) → claim(C001)
document → chunk(doc_990001_2025Q3::s9) → evidence(E023) → claim(C014)
```

API：`GET /api/runs/{run_id}/lineage/{entity_id}` 返回上游 DAG；`LineageRecorder.upstream()` 是同一逻辑。

## 6. DataSnapshotManifest

每次运行对 `{as_of, tickers(目标+同行)}` 生成清单：各表 `max(available_at)` 水位线、可见行数、快照 CSV 的 SHA-256、数据源版本，以及以上内容的哈希 `manifest_hash`。`snapshot_id = snap_<as_of>_<hash前8位>`，相同输入得到相同 id（`test_snapshot_is_reproducible`）。

## 7. 质量规则（`data_platform/quality.py`）

| 规则 | 级别 | 处理 |
|---|---|---|
| uniqueness（同一 ticker/科目/口径/期间/来源/revision 唯一） | ERROR | 拒绝入库 |
| non_negative（收入、资产等非负） | ERROR | 拒绝入库 |
| balance_identity（资产 = 负债 + 权益，容差 0.1%） | ERROR | 拒绝入库 |
| unit_jump（相邻期变动 > 50 倍） | WARN | 记录，疑似元/万元混用 |
| pit_reliability（披露时间不可靠） | WARN | `pit_unsafe=true` |

## 8. 真实 A 股快照（`benchmarks/real_snapshot/`）

| 项 | 内容 |
|---|---|
| 公司 | 宁德时代 300750、亿纬锂能 300014、比亚迪 002594、隆基绿能 601012、贵州茅台 600519、五粮液 000858、美的集团 000333、格力电器 000651 |
| 三表 | 东方财富 F10 数据中心（合并报表，2021Q1–2026Q3），3411 条事实 |
| 文档 | 巨潮资讯原始 PDF：每家 2024H1–2025 年报共 7 期定期报告（含更新版）+ 最多 12 份关键公告，共 157 份，解析为 14,702 个切片 |
| 行情 | Yahoo Finance 不复权日线 2021-12 至 2026-11，9,560 行；总市值 = 收盘价 × PIT 可见的最新股本 |
| 构建 | `scripts/fetch_real.py --pdfs`（联网，原始响应缓存到 `data/raw_cache/`）→ `scripts/build_real_snapshot.py`（离线，可复现） |

**PIT 时间戳**：定期报告 `published_at` 取巨潮披露日（巨潮对定期报告只给日期，按披露日盘前视为已披露）；公告取巨潮毫秒时间戳；`available_at` = 不早于 `published_at` 的第一个交易日 09:30（交易日历取自真实行情）。

**重述识别（数据质量引擎的真实发现）**：东方财富会在后续期报发布后用重述后的比较数覆盖历史期的值（`UPDATE_DATE` 后移）。构建时用 PDF 解析出的原始合并报表逐项对数（11 个科目，428 次比对，399 次误差 ≤0.3%）：

| 公司 | 期间 | 科目 | 原因 | 处理 |
|---|---|---|---|---|
| 宁德时代、亿纬锂能、比亚迪、格力 | 2024H1 / 2024Q3 | 营业成本 | 2024 年起执行《解释第 18 号》，保证类质保费用由销售费用改列营业成本并追溯调整 | rev1 = 原始披露值（原披露日），rev2 = 重述值（`UPDATE_DATE` 起可见） |
| 五粮液 | 2025Q1–Q3 | 营收、营业成本、净利润、资产负债 | 东方财富在 2026 年的更新把 2025 年各期营收改为约原值的一半（原因未核实） | 同上；PDF 无法确认的其他科目推迟到 `UPDATE_DATE` 才可见，宁可缺失也不泄漏 |
| 隆基绿能 | 2025Q1 | 净利润 | 差异 0.39% | 同上 |

所以 `as_of=2025-12-31` 看到的是 2025 年 10 月原始季报口径；`as_of=2026-11` 看到的是重述口径。

**已知局限**：东方财富只给合并报表，母公司口径未接入；D&A 只在半年报与年报的现金流量表补充资料中有，EV/EBITDA 的 D&A 用最近一期半年/年度 TTM；文档只覆盖到 2025 年报，2026 年各期只有结构化数据没有原文；同行只有 8 家公司内的同申万二级行业（比亚迪、隆基无同组公司，退到一级行业）。

Baostock（登录无响应）与 GitHub `noblechang/a-share-data`（仅行情、约 1.9GB）评估后未采用。

## 9. 演示 Golden Snapshot（实验室公司）

`benchmarks/golden_snapshot/` 由 `scripts/build_golden_snapshot.py` 以固定随机种子生成，**公司与数字均为虚构**：

| 代码 | 名称 | 设计场景 |
|---|---|---|
| 990001.SZ | 演示动力电池 | 净利润 +44% 而经营现金流 −48%；应收账款激增；2026-01-20 立案公告（未来数据陷阱） |
| 990002.SZ | 演示新能源整车 | 2025Q2 毛利率突变；2024H1 营收更正（单季拆分陷阱）；第二数据源 2025Q1 营收 +3%（冲突陷阱） |
| 990003.SH | 演示锂电材料 | 大额减值、存货周转恶化；2025-11-20 会计差错更正 2025Q3 净利润（重述陷阱） |
| 990004.SH | 演示白酒 | 健康对照组；2022Q1 披露时间不可靠（pit_unsafe） |
| 990005.SZ | 演示光伏组件 | 2025 年亏损（PE 不适用）；有息负债率跃升 |
| 990006.SZ | 演示储能系统 | 重大关联交易公告 |

行情价格锚定于各日期已知的 TTM 净利润 × 目标 PE（亏损时为 0.6 倍净资产），叠加均值回复噪声，因此估值指标在经济上合理。
