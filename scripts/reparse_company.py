"""Re-parse one company's periodic reports with the CURRENT parser, from the cached PDFs, into a scratch directory,
and report how the markdown changed. Nothing in benchmarks/ is overwritten.

  python scripts/reparse_company.py 000333 [out_dir]
Uses the same cninfo list query as build_real_snapshot.py (served from data/raw_cache, no network needed).
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from data_platform.providers import Cninfo, HttpCache  # noqa: E402
from fetch_real import COMPANIES, END, REPORT_PERIODS, START, periodic_kind  # noqa: E402
from rag_engine.pdf_parser import parse_pdf  # noqa: E402

code = sys.argv[1]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "reparse" / code
out.mkdir(parents=True, exist_ok=True)
t = next(c for c in COMPANIES if c[0].startswith(code))
cn = Cninfo(HttpCache())
anns = cn.announcements(code, t[0][-2:], START, END)
by = {}
for a in anns:
    k = periodic_kind(a["announcementTitle"])
    if k and "".join(k) in REPORT_PERIODS:
        by.setdefault("".join(k), []).append(a)
rows = []
for per, versions in sorted(by.items()):
    versions.sort(key=lambda a: a["announcementTime"])
    a = versions[0]
    y, kind = per[:4], per[4:]
    q = {"A": 4, "H1": 2, "Q1": 1, "Q3": 3}[kind]
    did = f"doc_{code}_{y}Q{q}"
    md, stats = parse_pdf(cn.pdf(a["adjunctUrl"]), f"{t[1]}（{t[0]}）{a['announcementTitle']}")
    (out / f"{did}.md").write_text(md, encoding="utf-8")
    old = ROOT / "benchmarks" / "real_snapshot" / "documents" / f"{did}.md"
    same = old.exists() and old.read_text(encoding="utf-8") == md
    rows.append((did, len(md.splitlines()), stats["tables"], same,
                 sum(1 for l in md.splitlines() if l.startswith("###")), sum(1 for l in md.splitlines() if "财务报表附注" in l and l.startswith("##"))))
    print(f"{did}: lines={rows[-1][1]} tables={rows[-1][2]} identical_to_stored={same} statement/note headings={rows[-1][4]} notes_head={rows[-1][5]}", flush=True)
