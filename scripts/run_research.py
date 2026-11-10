"""CLI: run one research task and write the report.

  python scripts/run_research.py 990001.SZ 2025-12-31 --arch B4 --question "..."
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from agent_runtime import ResearchRequest, run_research  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("as_of", help="YYYY-MM-DD or ISO datetime; date means 15:00 close")
    ap.add_argument("--arch", default="B4")
    ap.add_argument("--question", default="全面研究公司基本面、估值与风险，形成包含多空审议与证伪条件的研究报告")
    ap.add_argument("--mode", default="deep", choices=["fast", "deep"])
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--pdf", action="store_true", help="also write a PDF")
    a = ap.parse_args()
    as_of = datetime.fromisoformat(a.as_of if "T" in a.as_of else a.as_of + "T15:00:00")
    rep = run_research(ResearchRequest(ticker=a.ticker, as_of=as_of, question=a.question, mode=a.mode), a.arch)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{rep['ticker'][:6]}_{as_of:%Y%m%d}_{a.arch}"
    (out / f"{stem}.md").write_text(rep["markdown"], encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if a.pdf:
        from agent_runtime.pdf_export import markdown_to_pdf

        (out / f"{stem}.pdf").write_bytes(markdown_to_pdf(rep["markdown"]))
    print(rep["markdown"])
    print(json.dumps(rep["totals"], ensure_ascii=False))
    print(f"-> {out / stem}.md")
