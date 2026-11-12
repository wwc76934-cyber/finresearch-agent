"""Audit the PDF -> markdown extraction: for every parsed periodic report, how much of the financial statements survived?

  python scripts/parse_audit.py
A report is flagged when it has no consolidated balance sheet, income statement or cash-flow table, or no notes
heading. Those are the documents where section pre-filtering and note retrieval cannot work.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
docs = pd.read_csv(ROOT / "benchmarks" / "real_snapshot" / "documents.csv")
rows = []
for d in docs[docs.doc_type != "announcement"].itertuples():
    p = ROOT / "benchmarks" / d.path
    md = p.read_text(encoding="utf-8")
    has = lambda pat: len(re.findall(pat, md, flags=re.M))  # noqa: E731
    rows.append({"doc": d.document_id, "type": d.doc_type, "pages": len(re.findall(r"<!-- page", md)),
                 "bs": has(r"^#+ .*合并资产负债表"), "is": has(r"^#+ .*合并利润表"), "cf": has(r"^#+ .*合并现金流量表"),
                 "notes": has(r"^#+ .*财务报表附注"), "tables": has(r"^\|")})
df = pd.DataFrame(rows)
df["ticker"] = df.doc.str.extract(r"doc_(\d{6})")
df["complete"] = (df.bs > 0) & (df["is"] > 0) & (df.cf > 0)
print(df.groupby("ticker").agg(n=("doc", "count"), complete=("complete", "sum"), with_notes=("notes", lambda s: int((s > 0).sum()))).to_string())
bad = df[(~df.complete) & (df.type.isin(["annual_report", "semi_annual_report"]))]
print("\nannual/half-year reports WITHOUT the three consolidated statements:", len(bad), "of",
      len(df[df.type.isin(["annual_report", "semi_annual_report"])]))
print(bad[["doc", "pages", "bs", "is", "cf", "notes", "tables"]].to_string(index=False))
