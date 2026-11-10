"""Debug helper: parse cached periodic report PDFs and show what the parser keeps."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
from data_platform.providers import Cninfo, HttpCache  # noqa: E402
from rag_engine.pdf_parser import parse_pdf, statement_value  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_real import periodic_kind  # noqa: E402

REV = (r"其中营业收入", r"营业收入", r"营业总收入")
NP = (r"归属于母公司(股东|所有者)的净利润",)
cn = Cninfo(HttpCache())
t = sys.argv[1] if len(sys.argv) > 1 else "300750.SZ"
want = sys.argv[2:] or ["2025Q3", "2024A"]
for a in cn.announcements(t[:6], t[-2:], "2024-01-01", "2026-06-30"):
    k = periodic_kind(a["announcementTitle"])
    if not k or "".join(k) not in want:
        continue
    t0 = time.time()
    md, st = parse_pdf(cn.pdf(a["adjunctUrl"]), a["announcementTitle"])
    print("==", t, a["announcementTitle"], st, f"{time.time() - t0:.1f}s", len(md), "chars")
    print("headings:", [l for l in md.splitlines() if l.startswith("#")][:45])
    print("revenue:", statement_value(md, "合并利润表", REV), "np:", statement_value(md, "合并利润表", NP))
    i = md.find("### 合并利润表")
    seg = md[i:md.find("### ", i + 5)]
    print([l[:120] for l in seg.splitlines() if "归属于母公司" in l or "营业" in l][:8])
    j = md.find("### 应收账款")
    print(md[j:j + 500] if j >= 0 else "NO 应收账款 note")
