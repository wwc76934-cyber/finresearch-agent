# 研究型因子检验报告

- 研究时点 as_of: `2025-12-31T15:00:00`  快照: `snap_202512311500_1c082b71`  架构: `B4`

研究型因子检验（场景D），研究时点 2025-12-31 15:00:00。样本池为 benchmarks/factor_panel 的24家A股，财务数据按公告次日可见，收益窗口在研究时点前结束。结果只描述样本期内的统计关系。

## 问题直接回答
- (算子指标) 盈利现金保障倍数(OCF/净利润,TTM)（越高越好）在全部样本池的横截面检验：8个季度截面，每个截面约20家公司，持有60个交易日；RankIC均值0.1687，IC均值-0.0140，RankIC_IR为0.4867。 [Evidence #E001] [Evidence #E002] [Evidence #E003] [Evidence #E004] [Evidence #E005] [Evidence #E006]
- (算子指标) RankIC序列的t统计量为1.3767，双侧p值为0.2110。 [Evidence #E007] [Evidence #E008]
- (算子指标) 分3组（因子值从低到高）平均后续收益：最低组-1.98%，中间组-0.66%，最高组1.90%；最高组减最低组每期3.88%，累计27.90%，最大回撤-22.43%。 [Evidence #E009] [Evidence #E010] [Evidence #E011] [Evidence #E012] [Evidence #E013] [Evidence #E014] [Evidence #E015]

## 因子检验结论
- (模型推断) 推断：RankIC的p值为0.2110，未达到常规显著性水平；在该样本池和样本期内，没有足够证据说明该因子能预测后续收益，RankIC均值0.1687可能只是随机波动。 [Evidence #E004] [Evidence #E008] [Evidence #E001]

## 数据局限与不确定性声明
- 待解答：样本局限[SURVIVORSHIP]：样本池全是今天仍在上市的公司，退市或被剔除的公司不在其中，质量类因子的结果偏乐观。（因子检验方法局限）

## 引用

- **E001** [metric] 2025-12-31 截面个数 = 8 — `m_300750_factor_n_cs_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E002** [metric] 2025-12-31 每截面公司数(中位数) = 20 — `m_300750_factor_n_stocks_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E003** [metric] 2025-12-31 持有交易日数 = 60 — `m_300750_factor_horizon_days_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E004** [metric] 2025-12-31 因子RankIC均值 = 0.1687 — `m_300750_factor_rank_ic_mean_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E005** [metric] 2025-12-31 因子IC均值 = -0.0140 — `m_300750_factor_ic_mean_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E006** [metric] 2025-12-31 RankIC_IR = 0.4867 — `m_300750_factor_rank_ic_ir_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E007** [metric] 2025-12-31 RankIC的t统计量 = 1.3767 — `m_300750_factor_rank_ic_t_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E008** [metric] 2025-12-31 RankIC的p值 = 0.2110 — `m_300750_factor_rank_ic_p_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E009** [metric] 2025-12-31 分组数 = 3 — `m_300750_factor_quantiles_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E010** [metric] 2025-12-31 分组平均后续收益 = -1.98% — `m_300750_factor_group_return_2025-12-31_ocf_to_net_profit_all_g1` available_at=2025-12-31T15:00:00
- **E011** [metric] 2025-12-31 分组平均后续收益 = -0.66% — `m_300750_factor_group_return_2025-12-31_ocf_to_net_profit_all_g2` available_at=2025-12-31T15:00:00
- **E012** [metric] 2025-12-31 分组平均后续收益 = 1.90% — `m_300750_factor_group_return_2025-12-31_ocf_to_net_profit_all_g3` available_at=2025-12-31T15:00:00
- **E013** [metric] 2025-12-31 最高组减最低组每期收益 = 3.88% — `m_300750_factor_ls_mean_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E014** [metric] 2025-12-31 多空组合累计收益 = 27.90% — `m_300750_factor_ls_cum_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
- **E015** [metric] 2025-12-31 多空组合最大回撤 = -22.43% — `m_300750_factor_ls_mdd_2025-12-31_ocf_to_net_profit_all` available_at=2025-12-31T15:00:00
