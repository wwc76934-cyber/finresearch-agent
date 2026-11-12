"""A-share filing PDF -> structured markdown with `<!-- page N -->` markers (the format chunking.py reads).

PyMuPDF text + table detection. Text lines and tables are interleaved by their vertical position on
the page, so a table always lands under the heading printed above it. Only the sections the agents
use are kept, which keeps a 250-page annual report to a few hundred chunks:
  主要会计数据 / 管理层讨论与分析 / 合并与母公司三表 / 附注中的应收账款、存货、商誉、借款、减值、关联交易、收入成本
A statement that continues on the next page stays one table because rows are appended under the
same heading until a new heading appears (chunking.py stitches across page markers).
"""
from __future__ import annotations

import re
from typing import Optional

SECTION_RE = re.compile(r"^第[一二三四五六七八九十]{1,3}节\s*(\S.{1,30})$")
STATEMENT_RE = re.compile(r"^(?:\d{1,2}[、.．]\s*)?(合并|母公司)(?:年初到报告期末|本报告期|本期)?(资产负债表|利润表|现金流量表)(（续）)?$")
EQUITY_STMT_RE = re.compile(r"^(?:\d{1,2}[、.．]\s*)?(合并|母公司)?(所有者|股东)权益变动表")
FIN_SKIP_RE = re.compile(r"^[一二三四五六七八九十]{1,3}、\s*(公司基本情况|财务报表的编制基础|重要会计政策|税项|企业合并及合并财务报表"
                         r"|合并范围的变更|在其他主体中的权益|政府补助|与金融工具相关的风险|公允价值的披露|关联方及关联交易|股份支付"
                         r"|承诺及或有事项|资产负债表日后事项|其他重要事项)")
NOTES_START_RE = re.compile(r"^[一二三四五六七八九十]{1,3}、\s*(合并)?财务报表(主要)?项目(注释|附注)$")
PARENT_NOTES_RE = re.compile(r"^[一二三四五六七八九十]{1,3}、\s*母公司财务报表(主要)?项目(注释|附注)$")
NOTE_HEAD_RE = re.compile(r"^(\d{1,2})[、.．]\s*(\S.{1,40})$")
# Second layout seen in the filings (e.g. 美的集团): the company name leads every title line, the consolidated and parent
# statements share one table ("合并及公司资产负债表"), and notes are numbered "(4)  应收账款". These patterns only
# fire on lines the strict patterns above reject, so reports already parsed correctly are not affected.
COMBINED_STATEMENT_RE = re.compile(r"^(?:\S{2,20}(?:股份)?有限公司)?(?:\d{4}\s*年(?:上半年度|半年度|度|第[一三]季度)?)?合并及公司(资产负债表|利润表|现金流量表)")
NOTES_TITLE_RE = re.compile(r"^(?:\S{2,20}(?:股份)?有限公司)?\s*\d{4}\s*年(?:上半年度|半年度|度|第[一三]季度)?财务报表附注")
NOTE_HEAD_PAREN_RE = re.compile(r"^\((\d{1,2})\)\s+(\S.{1,40})$")
UNIT_RE = re.compile(r"单位(?:为)?\s*[：:]?\s*(?:人民币)?(元|千元|万元|亿元)")      # "单位：万元", "单位为：千元" and "金额单位为人民币千元"; a strict superset of the original pattern
KEEP_SECTIONS = ("重要提示", "公司简介和主要财务指标", "主要财务指标", "管理层讨论与分析", "财务报告", "财务报表")
KEEP_NOTES = ("应收账款", "存货", "商誉", "短期借款", "长期借款", "应付债券", "一年内到期", "资产减值", "信用减值", "营业收入",
              "研发费用")


def _clean(s: Optional[str]) -> str:
    s = re.sub(r"\s+", "", s or "").replace("|", "/")
    return _fix_number(s)


def _fix_number(s: str) -> str:
    """Some filings come back with separators scrambled: '130,903,889,634.88' -> '13090388963488,,,.'.
    Digits keep their order, so rebuild the number: A-share statements print 2 decimals whenever a '.' is present."""
    paren = re.fullmatch(r"[（(]([\d,.]+)[)）]", s)  # accounting negatives: (1,234.56)
    if paren:
        s = "-" + paren.group(1)
    m = re.fullmatch(r"[\d,.\-−—－‐–]+", s)
    if not m or not any(ch.isdigit() for ch in s) or re.fullmatch(r"[\d,.]*[-−—－‐–]{2,}[\d,.]*", s):
        return s
    dash = re.compile(r"[-−—－‐–]")
    # a minus sign can come back anywhere in the scrambled string ('524334467795-,,,.')
    sign, body = ("-" if dash.search(s) else ""), dash.sub("", s)
    if not body or not any(ch.isdigit() for ch in body):
        return s
    digits = re.sub(r"\D", "", body)
    if re.fullmatch(r"\d{1,3}(,\d{3})*(\.\d+)?", body):
        return sign + body.replace(",", "")   # well-formed
    if "." in body and len(digits) > 2:
        return f"{sign}{digits[:-2]}.{digits[-2:]}"
    return sign + digits


def _table_md(rows: list[list]) -> list[str]:
    rows = [[_clean(c) for c in r] for r in rows if r and any(_clean(c) for c in r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep_cols = [j for j in range(width) if any(r[j] for r in rows)]  # drop columns that are empty everywhere
    rows = [[r[j] for j in keep_cols] for r in rows]
    width = len(keep_cols)
    out = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return out


def _page_items(page, want_tables: bool) -> list[tuple[float, str, object]]:
    # use_layout=False: PyMuPDF 2.x otherwise runs an ONNX layout model per page, which crashed natively
    # (access violation) on some filings and is far slower; ruling-line detection suits filing tables.
    tables = []
    if want_tables:
        try:
            tables = page.find_tables(use_layout=False).tables
        except TypeError:  # PyMuPDF < 2.0 has no use_layout argument
            tables = page.find_tables().tables
    boxes = [t.bbox for t in tables]
    items: list[tuple[float, str, object]] = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            x0, y0, x1, y1 = ln["bbox"]
            cy = (y0 + y1) / 2
            if any(bx[0] - 2 <= (x0 + x1) / 2 <= bx[2] + 2 and bx[1] - 2 <= cy <= bx[3] + 2 for bx in boxes):
                continue  # this line is part of a table
            text = "".join(s["text"] for s in ln["spans"]).strip()
            if text:
                items.append((y0, "text", text))
    for t, bx in zip(tables, boxes):
        md = _table_md(t.extract())
        if md:
            items.append((bx[1], "table", md))
    items.sort(key=lambda x: x[0])
    return items


def parse_pdf(data: bytes, title: str, max_pages: int = 400) -> tuple[str, dict]:
    import pymupdf

    doc = pymupdf.open(stream=data, filetype="pdf")
    lines: list[str] = [f"# {title}", ""]
    st = {"section": "", "keep": True, "in_notes": False, "note_keep": False, "fin_skip": False, "unit": None}
    stats = {"pages": doc.page_count, "kept_pages": 0, "tables": 0}

    def emitting() -> bool:
        return st["keep"] and not (st["fin_skip"] and not st["in_notes"]) and (not st["in_notes"] or st["note_keep"])

    try:
        for pno in range(min(doc.page_count, max_pages)):
            page = doc[pno]
            raw = page.get_text("text")
            raw_lines = [l.strip() for l in raw.splitlines()]
            heads_here = any((SECTION_RE.match(l) and any(k in l for k in KEEP_SECTIONS)) or STATEMENT_RE.match(l)
                             or NOTES_START_RE.match(l)
                             or ((m := NOTE_HEAD_RE.match(l)) and st["in_notes"] and any(k in m.group(2) for k in KEEP_NOTES))
                             for l in raw_lines)
            items = _page_items(page, want_tables=emitting() or heads_here)
            texts = [x[2] for x in items if x[1] == "text"]
            page_out: list[str] = []
            for idx, (_, kind, val) in enumerate(items):
                if kind == "table":
                    if emitting():
                        page_out += val
                        stats["tables"] += 1
                    continue
                l = val
                if um := UNIT_RE.search(l):
                    st["unit"] = um.group(1)
                if m := SECTION_RE.match(l):
                    st.update(section=m.group(1), keep=any(k in m.group(1) for k in KEEP_SECTIONS),
                              in_notes=False, note_keep=False, fin_skip=False)
                    if st["keep"]:
                        page_out.append(f"## {l}")
                    continue
                if not st["keep"]:
                    continue
                if PARENT_NOTES_RE.match(l):
                    st.update(in_notes=True, note_keep=False, fin_skip=True)
                    continue
                if NOTES_START_RE.match(l):
                    st.update(in_notes=True, note_keep=False, fin_skip=False)
                    page_out.append(f"## 财务报表附注 {l}")
                    continue
                if NOTES_TITLE_RE.match(l):
                    # this header repeats on every notes page glued to the page's first sentence: switch into notes
                    # mode once, emit the heading once, and keep the rest of the line only if it is real content
                    if not st["in_notes"]:
                        st.update(in_notes=True, note_keep=False, fin_skip=False)
                        page_out.append("## 财务报表附注 " + l[:30])
                    continue
                if not st["in_notes"] and (FIN_SKIP_RE.match(l) or EQUITY_STMT_RE.match(l)):
                    st["fin_skip"] = True
                    continue
                if (cm := COMBINED_STATEMENT_RE.match(l)) and not STATEMENT_RE.match(l) and not st["in_notes"] and len(l) <= 90:
                    st["fin_skip"] = False
                    # the unit is printed in the statement's own title line ("金额单位为人民币千元") or on the next lines of
                    # the page; the running unit from elsewhere in the report must NOT be inherited (a stray
                    # "单位：万元" in the MD&A would mislabel a 千元 statement by 10x)
                    k = texts.index(l) if l in texts else -1
                    um = UNIT_RE.search(" ".join([l] + texts[k + 1:k + 4])) if k >= 0 else UNIT_RE.search(l)
                    unit = um.group(1) if um else None
                    if unit is None:
                        page_out.append(f"### 合并{cm.group(1)}")      # unit unknown: say so instead of guessing
                        continue
                    st["unit"] = unit
                    page_out.append(f"### 合并{cm.group(1)}（单位：{unit}）")
                    continue
                if (sm := STATEMENT_RE.match(l)) and not st["in_notes"]:
                    st["fin_skip"] = False
                    k = texts.index(l) if l in texts else -1
                    um = UNIT_RE.search(" ".join(texts[k + 1:k + 6])) if k >= 0 else None
                    st["unit"] = um.group(1) if um else st["unit"]
                    page_out.append(f"### {sm.group(1)}{sm.group(2)}（单位：{st['unit'] or '元'}）")
                    continue
                # "(N) name" is a note heading only when the name is a note we keep; elsewhere "(1) (2) (3)" number
                # sub-items (e.g. 茅台's "(3).本期计提…") and must stay ordinary text
                if st["in_notes"] and (m := NOTE_HEAD_RE.match(l) or
                                       ((pm := NOTE_HEAD_PAREN_RE.match(l)) and any(k in pm.group(2) for k in KEEP_NOTES) and pm)):
                    st["note_keep"] = any(k in m.group(2) for k in KEEP_NOTES) and len(m.group(2)) <= 20
                    if st["note_keep"]:
                        page_out.append(f"### {m.group(2)}（单位：{st['unit'] or '元'}）")
                    continue
                if not emitting():
                    continue
                if re.fullmatch(r"[\d,.\-—%（）()\s]+", l) or len(l) < 2:
                    continue
                page_out.append(l)
            if page_out:
                stats["kept_pages"] += 1
                lines.append(f"<!-- page {pno + 1} -->")
                lines += _merge_paragraphs(page_out)
    finally:
        doc.close()
    return "\n".join(lines) + "\n", stats


def _merge_paragraphs(lines: list[str]) -> list[str]:
    """PDF text breaks lines mid-sentence; join consecutive prose lines into paragraphs."""
    out: list[str] = []
    for l in lines:
        struct = l.startswith(("#", "|", "<!--"))
        prev_struct = bool(out) and out[-1].startswith(("#", "|", "<!--"))
        if out and not struct and not prev_struct and out[-1] and not out[-1].endswith(("。", "：", ":", "；")):
            out[-1] += l
            continue
        if l.startswith("|") and out and not out[-1].startswith("|"):
            out.append("")
        if not l.startswith("|") and out and out[-1].startswith("|"):
            out.append("")
        out.append(l)
    return out


NUM = re.compile(r"-?[\d,]+(\.\d+)?")


def row_name(cell: str) -> str:
    """'1.归属于母公司股东的净利润（净亏损以“－”号填列）' / '营业总收入一、' -> core item name."""
    s = re.split(r"[（(]", cell)[0]
    s = re.sub(r"[\d.．、：:，,\s“”\"'－-]", "", s)
    return s.strip("一二三四五六七八九十")


def statement_value(md: str, statement: str, row_patterns: tuple[str, ...]) -> Optional[float]:
    """First number on the first matching row of a statement table, converted to 元."""
    for b in md.split("### ")[1:]:
        head = b.splitlines()[0]
        if not head.startswith(statement):
            continue
        um = re.search(r"单位：(元|千元|万元|亿元)", head)
        scale = {"元": 1, "千元": 1e3, "万元": 1e4, "亿元": 1e8}[um.group(1) if um else "元"]
        for line in b.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            name = row_name(cells[0]) if cells else ""
            if any(re.fullmatch(p, name) for p in row_patterns):
                for c in cells[1:]:
                    c = c.replace(" ", "")
                    if NUM.fullmatch(c) and any(ch.isdigit() for ch in c) and len(c.replace(",", "")) > 3:
                        return float(c.replace(",", "")) * scale
    return None
