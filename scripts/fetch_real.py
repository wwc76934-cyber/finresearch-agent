"""Step 1 of the real snapshot: download and cache raw data (statements, prices, announcement lists, PDFs).

  python scripts/fetch_real.py            # statements + prices + announcement lists
  python scripts/fetch_real.py --pdfs     # also periodic-report and key-announcement PDFs
Everything lands in data/raw_cache/ (gitignored); scripts/build_real_snapshot.py turns it into a snapshot.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from data_platform.providers import Cninfo, EastmoneyStatements, HttpCache, YahooPrices  # noqa: E402

COMPANIES = [  # ticker, name, sw_l1, sw_l2, template
    ("300750.SZ", "宁德时代", "电力设备", "电池", "manufacturing"),
    ("300014.SZ", "亿纬锂能", "电力设备", "电池", "manufacturing"),
    ("002594.SZ", "比亚迪", "汽车", "乘用车", "manufacturing"),
    ("601012.SH", "隆基绿能", "电力设备", "光伏设备", "manufacturing"),
    ("600519.SH", "贵州茅台", "食品饮料", "白酒", "consumer"),
    ("000858.SZ", "五粮液", "食品饮料", "白酒", "consumer"),
    ("000333.SZ", "美的集团", "家用电器", "白色家电", "consumer"),
    ("000651.SZ", "格力电器", "家用电器", "白色家电", "consumer"),
]
PERIODIC_RE = re.compile(r"(20\d{2})年(年度|半年度|第?一季度|第?三季度)报告(（[^）]*）)?$")
KEY_WORDS = ("合同", "关联交易", "回购", "担保", "处罚", "立案", "减值", "更正", "业绩预告", "债券", "收购", "诉讼", "利润分配", "募集")
START, END = "2024-01-01", "2026-06-30"
REPORT_PERIODS = {"2024H1", "2024Q3", "2024A", "2025Q1", "2025H1", "2025Q3", "2025A"}


def periodic_kind(title: str):
    """'2025年三季度报告' -> ('2025', 'Q3'); None for summaries, audits, English versions etc."""
    t = re.sub(r"^.*?(20\d{2}年)", r"\1", title.strip().replace("<em>", "").replace("</em>", ""))
    m = PERIODIC_RE.match(t)
    if not m or "英文" in title:
        return None
    k = {"年度": "A", "半年度": "H1"}.get(m.group(2)) or ("Q1" if "一" in m.group(2) else "Q3")
    return m.group(1), k


def is_full_periodic(title: str) -> bool:
    return periodic_kind(title) is not None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", action="store_true")
    a = ap.parse_args()
    http = HttpCache()
    em, cn, yh = EastmoneyStatements(http), Cninfo(http), YahooPrices(http)
    index = {}
    for t, name, *_ in COMPANIES:
        code, mkt = t[:6], t[-2:]
        rec = {k: len(em.fetch(t, k)) for k in ("income", "balance", "cashflow")}
        sym = t.replace(".SH", ".SS")
        rec["prices"] = len(yh.daily(sym, datetime(2021, 12, 1), datetime(2026, 11, 9)))
        anns = cn.announcements(code, mkt, START, END)
        rec["announcements"] = len(anns)
        periodic = [x for x in anns if is_full_periodic(x["announcementTitle"])
                    and "".join(periodic_kind(x["announcementTitle"])) in REPORT_PERIODS]
        key = [x for x in anns if not is_full_periodic(x["announcementTitle"])
               and any(k in x["announcementTitle"] for k in KEY_WORDS)]
        rec["periodic"], rec["key"] = len(periodic), len(key)
        if a.pdfs:
            for x in periodic + key[:12]:
                cn.pdf(x["adjunctUrl"])
            rec["pdfs"] = len(periodic) + len(key[:12])
        index[t] = rec
        print(t, name, rec, flush=True)
    (ROOT / "data" / "raw_cache" / "fetch_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")
