"""Eastmoney F10 statement rows -> the platform's metric names (shared by the offline snapshot builder's logic)."""
from __future__ import annotations

CUM = {3: "cumulative_q1", 6: "cumulative_h1", 9: "cumulative_q3", 12: "annual"}


def num(row: dict, *keys):
    for k in keys:
        v = row.get(k)
        if v is not None:
            return float(v)
    return None


def flows(inc: dict, cf: dict) -> dict[str, float]:
    imp = None
    if inc.get("ASSET_IMPAIRMENT_INCOME") is not None or inc.get("CREDIT_IMPAIRMENT_INCOME") is not None:
        imp = -(num(inc, "ASSET_IMPAIRMENT_INCOME") or 0) - (num(inc, "CREDIT_IMPAIRMENT_INCOME") or 0)
    elif inc.get("ASSET_IMPAIRMENT_LOSS") is not None:
        imp = num(inc, "ASSET_IMPAIRMENT_LOSS")
    da = [num(cf, k) for k in ("FA_IR_DEPR", "IA_AMORTIZE", "LPE_AMORTIZE", "USERIGHT_ASSET_AMORTIZE")]
    out = {"operating_revenue": num(inc, "OPERATE_INCOME", "TOTAL_OPERATE_INCOME"),
           "operating_cost": num(inc, "OPERATE_COST"), "rd_expense": num(inc, "RESEARCH_EXPENSE"),
           "asset_impairment_loss": imp, "net_profit": num(inc, "PARENT_NETPROFIT"),
           "operating_cash_flow": num(cf, "NETCASH_OPERATE"), "total_profit": num(inc, "TOTAL_PROFIT"),
           "interest_expense": num(inc, "FE_INTEREST_EXPENSE"),
           "depreciation_amortization": sum(d for d in da if d is not None) if any(d is not None for d in da) else None}
    return {k: v for k, v in out.items() if v is not None}


def instants(b: dict) -> dict[str, float]:
    debt = [num(b, k) for k in ("SHORT_LOAN", "LONG_LOAN", "BOND_PAYABLE", "NONCURRENT_LIAB_1YEAR")]
    out = {"accounts_receivable": num(b, "ACCOUNTS_RECE"), "inventory": num(b, "INVENTORY"),
           "total_assets": num(b, "TOTAL_ASSETS"), "total_liabilities": num(b, "TOTAL_LIABILITIES"),
           "total_equity": num(b, "TOTAL_EQUITY"), "parent_equity": num(b, "TOTAL_PARENT_EQUITY"),
           "interest_bearing_debt": sum(d for d in debt if d is not None) if any(d is not None for d in debt) else None,
           "goodwill": num(b, "GOODWILL") or 0.0, "cash": num(b, "MONETARYFUNDS"),
           "shares_outstanding": num(b, "SHARE_CAPITAL")}
    return {k: v for k, v in out.items() if v is not None}
