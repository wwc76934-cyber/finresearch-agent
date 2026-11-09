"""Structure-aware parsing + parent/child chunking with context anchors.

Input documents are markdown with `<!-- page N -->` markers (the output format
of a MinerU/Docling/pdfplumber pipeline). Tables are never split mid-row and a
table that continues across a page marker is stitched back together.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

PAGE_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->")
UNIT_RE = re.compile(r"单位：(元|万元|亿元)")
CHILD_MAX_CHARS = 450  # ~300-500 tokens for Chinese text


@dataclass
class Block:
    kind: str  # "text" | "table"
    lines: list[str]
    page_start: int
    page_end: int


@dataclass
class Section:
    path: list[str]
    title: str
    blocks: list[Block] = field(default_factory=list)


def section_type_of(path: str, doc_type: str) -> str:
    if doc_type == "announcement":
        return "ANNOUNCEMENT"
    if "管理层讨论" in path:
        return "MDA"
    if "附注" in path:
        return "NOTES"
    if "风险" in path:
        return "RISK_FACTORS"
    if "财务报表" in path:
        return "STATEMENTS"
    return "OTHER"


def parse_markdown(text: str) -> list[Section]:
    sections: list[Section] = []
    path: list[str] = []
    page = 1
    cur: Optional[Section] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = PAGE_RE.search(line)
        if m:
            page = int(m.group(1))
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            path = path[: max(level - 1, 0)] + [title]
            cur = Section(path=list(path), title=title)
            sections.append(cur)
            continue
        if not line.strip():
            if cur and cur.blocks and cur.blocks[-1].kind == "text":
                cur.blocks.append(Block("text", [], page, page))  # paragraph break
            continue
        if cur is None:
            cur = Section(path=["(前言)"], title="(前言)")
            sections.append(cur)
        kind = "table" if line.startswith("|") else "text"
        last = cur.blocks[-1] if cur.blocks else None
        if last and last.kind == kind and (kind == "table" or last.lines):
            last.lines.append(line)
            last.page_end = page  # a table continuing after a page marker is stitched here
        else:
            cur.blocks.append(Block(kind, [line], page, page))
    for s in sections:
        s.blocks = [b for b in s.blocks if b.lines]
    return sections


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    ticker: str
    parent_id: Optional[str]
    level: str
    section_path: str
    section_type: str
    is_table: bool
    table_name: Optional[str]
    statement_scope: Optional[str]
    unit: Optional[str]
    page_start: int
    page_end: int
    text: str
    available_at: datetime


def _anchor(ticker, period, path, table, scope, unit, p0, p1) -> str:
    scope_cn = {"consolidated": "合并", "parent": "母公司"}.get(scope or "", "-")
    return f"[{ticker} | {period} | {path} | {table or '-'} | {scope_cn} | {unit or '-'} | p{p0}-{p1}]"


def chunk_document(document_id: str, ticker: str, period: str, doc_type: str, text: str,
                   available_at: datetime, anchors: bool = True) -> list[Chunk]:
    out: list[Chunk] = []
    title = text.splitlines()[0].lstrip("# ").strip() if text else document_id
    for si, sec in enumerate(parse_markdown(text)):
        if not sec.blocks:
            continue
        path = " > ".join([title] + sec.path[1:]) if len(sec.path) > 1 else f"{title} > {sec.path[0]}"
        stype = section_type_of(path, doc_type)
        is_table = any(b.kind == "table" for b in sec.blocks)
        scope = "parent" if "母公司" in sec.title else ("consolidated" if ("合并" in sec.title or stype == "NOTES") else None)
        um = UNIT_RE.search(sec.title)
        unit = um.group(1) if um else None
        p0 = min(b.page_start for b in sec.blocks)
        p1 = max(b.page_end for b in sec.blocks)
        anchor = _anchor(ticker, period, path, sec.title if is_table else None, scope, unit, p0, p1) if anchors else ""
        body = "\n".join("\n".join(b.lines) for b in sec.blocks)
        pid = f"{document_id}::s{si}"
        out.append(Chunk(pid, document_id, ticker, None, "parent", path, stype, is_table,
                         sec.title if is_table else None, scope, unit, p0, p1,
                         (anchor + "\n" + body).strip(), available_at))
        # children: text paragraphs individually; tables grouped by rows, header repeated
        ci = 0
        for b in sec.blocks:
            if b.kind == "text":
                pieces, buf = [], ""
                for line in b.lines:
                    if len(buf) + len(line) > CHILD_MAX_CHARS and buf:
                        pieces.append(buf)
                        buf = ""
                    buf += line
                if buf:
                    pieces.append(buf)
            else:
                header, rows = b.lines[:2], b.lines[2:]
                pieces, cur_rows = [], []
                for r in rows:
                    if sum(len(x) for x in cur_rows) + len(r) > CHILD_MAX_CHARS and cur_rows:
                        pieces.append("\n".join(header + cur_rows))
                        cur_rows = []
                    cur_rows.append(r)
                if cur_rows or not pieces:
                    pieces.append("\n".join(header + cur_rows))
            for piece in pieces:
                out.append(Chunk(f"{pid}::c{ci}", document_id, ticker, pid, "child", path, stype, b.kind == "table",
                                 sec.title if b.kind == "table" else None, scope, unit, b.page_start, b.page_end,
                                 (anchor + "\n" + piece).strip(), available_at))
                ci += 1
    return out


def naive_chunks(document_id: str, ticker: str, text: str, available_at: datetime, size: int = 200) -> list[Chunk]:
    """Fixed-size character windows with no structure or anchors (B1 / ablation baseline)."""
    clean = PAGE_RE.sub("", text)
    out = []
    for i in range(0, len(clean), size):
        out.append(Chunk(f"{document_id}::n{i // size}", document_id, ticker, None, "child", "-", "OTHER",
                         "|" in clean[i:i + size], None, None, None, 0, 0, clean[i:i + size], available_at))
    return out
