"""Render a report's markdown to PDF with reportlab and a CJK system font (no browser needed)."""
from __future__ import annotations

import io
import os
import re
from functools import lru_cache
from xml.sax.saxutils import escape

FONT_CANDIDATES = [
    ("C:/Windows/Fonts/msyh.ttc", 0), ("C:/Windows/Fonts/simhei.ttf", None), ("C:/Windows/Fonts/simsun.ttc", 0),
    ("/System/Library/Fonts/PingFang.ttc", 0), ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
]


@lru_cache(maxsize=1)
def _font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    custom = os.environ.get("PDF_FONT")
    for path, idx in ([(custom, 0)] if custom else []) + FONT_CANDIDATES:
        if path and os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("CJK", path, subfontIndex=idx or 0))
                return "CJK"
            except Exception:  # noqa: BLE001 - try the next font
                continue
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))  # built-in CID font, no file needed
    return "STSong-Light"


def _inline(text: str) -> str:
    t = escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<font size=8>\1</font>", t)
    return t


def markdown_to_pdf(md: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    f = _font()
    body = ParagraphStyle("b", fontName=f, fontSize=9.5, leading=14, wordWrap="CJK")
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=11)
    h = {1: ParagraphStyle("h1", parent=body, fontSize=16, leading=22, spaceAfter=6),
         2: ParagraphStyle("h2", parent=body, fontSize=12.5, leading=18, spaceBefore=8, spaceAfter=4),
         3: ParagraphStyle("h3", parent=body, fontSize=11, leading=16, spaceBefore=6, spaceAfter=3)}
    story, table = [], []

    def flush_table():
        if not table:
            return
        rows = [[Paragraph(_inline(c), small) for c in r] for r in table]
        t = Table(rows, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.extend([t, Spacer(1, 4)])
        table.clear()

    for line in md.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                table.append(cells)
            continue
        flush_table()
        if not s:
            story.append(Spacer(1, 3))
        elif m := re.match(r"(#{1,3})\s+(.*)", s):
            story.append(Paragraph(_inline(m.group(2)), h[len(m.group(1))]))
        elif s.startswith(("- ", "* ")):
            story.append(Paragraph("• " + _inline(s[2:]), body))
        else:
            story.append(Paragraph(_inline(s), body))
    flush_table()
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                      bottomMargin=14 * mm, title="FinResearch Report").build(story)
    return buf.getvalue()
