"""Build a harder retrieval set: analyst-style questions that do NOT reuse the section's own wording.

The existing real set (126 queries) is built from section names ("合并利润表 营业收入"), so BM25 already finds
the target. Here an LLM reads one chunk and writes the question an analyst would ask whose answer is in it,
told not to copy phrases. Gold = same document and same section path. Needs LLM_API_KEY. Run once; cached.

  python scripts/build_hard_rag_queries.py [N]
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from agent_runtime.llm import get_llm, load_dotenv, parse_json  # noqa: E402
from rag_engine.index import get_retriever  # noqa: E402

OUT = ROOT / "benchmarks" / "real_snapshot" / "rag_eval_hard_v1.jsonl"
SYS = ("你是买方分析师。读下面一段上市公司披露原文,写一个分析师会问的中文问题,答案就在这段里。"
       "要求:1)不要照抄原文里的连续词组(不超过4个字相同);2)不要出现章节标题或表名;3)不要带具体数字;"
       "4)问题要能让人在整个公司披露库里只指向这一段(包含公司特有的事项或对象)。只输出JSON {\"question\":\"...\"}")

if __name__ == "__main__":
    load_dotenv()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    r = get_retriever(None, anchors=True)
    pool = [c for c in r.children if c.ticker[:2] != "99" and 180 <= len(c.text) <= 600 and not c.is_table
            and c.section_type in ("MDA", "NOTES", "ANNOUNCEMENT") and sum(ch.isdigit() for ch in c.text) / len(c.text) < 0.12]
    rng = random.Random(20260512)
    by_t: dict[str, list] = {}
    for c in pool:
        by_t.setdefault(c.ticker, []).append(c)
    picks = []
    tickers = sorted(by_t)
    while len(picks) < n and any(by_t.values()):
        for t in tickers:
            if by_t[t] and len(picks) < n:
                picks.append(by_t[t].pop(rng.randrange(len(by_t[t]))))
    llm = get_llm()
    assert llm.online, "needs the online model"
    rows = []
    for c in picks:
        d = parse_json(llm.chat(SYS, c.text, json_mode=True).text) or {}
        q = str(d.get("question", "")).strip()
        if len(q) < 8:
            continue
        rows.append({"query": q, "ticker": c.ticker, "as_of": c.available_at.isoformat(), "target_document_id": c.document_id,
                     "target_section_path": c.section_path, "section_type": c.section_type, "source_chunk": c.chunk_id})
    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    print(len(rows), "queries ->", OUT)
