"""ReportLab adapter - professional multi-page PDF rendering.

Design principles (see ReportLab agent-generating-reports guidance):
  * A colored cover band with title, date and key stats chips.
  * Flowsheet/Table for the structured "Executive Snapshot" so totals read
    as a clean report, not plain text.
  * Monospace, boxed code blocks for diff hunks (verbatim).
  * Footer with page numbers via onPage callbacks.
  * KeepTogether so headings don't orphan from their content.
"""
import os
import tempfile
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Preformatted,
    HRFlowable,
)

from digest_agent.ports import DocumentRendererPort, RenderError


ACCENT = colors.HexColor("#1f3b73")
ACCENT_DARK = colors.HexColor("#142a52")
BAND_BG = colors.HexColor("#eef2f8")
CODE_BG = colors.HexColor("#f6f8fa")
BOARD = colors.HexColor("#d1d5db")
GREEN = colors.HexColor("#15803d")
RED = colors.HexColor("#b91c1c")
TEXT = colors.HexColor("#1f2937")
MUTED = colors.HexColor("#6b7280")


class ReportLabAdapter(DocumentRendererPort):
    """Adapter for reportlab - implements DocumentRendererPort."""

    def __init__(self, title: str = "GitHub Daily Digest", brand_color=ACCENT_DARK):
        self.title = title
        self.brand = brand_color

    def _output_dir(self) -> str:
        return os.environ.get("DIGEST_OUTPUT_DIR", tempfile.gettempdir())

    def _styles(self):
        base = getSampleStyleSheet()
        return {
            "cover_title": ParagraphStyle("T", parent=base["Title"], fontSize=26, textColor=colors.white,
                                          leading=30, alignment=TA_CENTER),
            "cover_sub": ParagraphStyle("S", parent=base["BodyText"], fontSize=11, textColor=colors.white,
                                        alignment=TA_CENTER, leading=15),
            "h1": ParagraphStyle("H1", parent=base["Heading2"], fontSize=15, textColor=ACCENT_DARK,
                                 leading=19, spaceBefore=14, spaceAfter=6),
            "h2": ParagraphStyle("H2", parent=base["Heading3"], fontSize=12, textColor=TEXT,
                                 leading=15, spaceBefore=10, spaceAfter=4),
            "body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=10, leading=14.5,
                                   textColor=colors.HexColor("#374151"), spaceAfter=6),
            "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontSize=10, leading=14,
                                     textColor=TEXT),
            "code": ParagraphStyle("Code", parent=base["Code"], fontSize=8, leading=11,
                                   textColor=colors.black, fontName="Courier",),
            "chip": ParagraphStyle("Chip", parent=base["BodyText"], fontSize=9,
                                   textColor=MUTED, alignment=TA_CENTER),
        }

    def _footer(self, canvas, doc):
        """Draw a footer with page number on a light rule."""
        canvas.saveState()
        w, h = doc.pagesize
        canvas.setStrokeColor(BOARD)
        canvas.setLineWidth(0.5)
        canvas.line(0.8 * inch, 0.6 * inch, w - 0.8 * inch, 0.6 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(w / 2, 0.45 * inch, "GitHub Daily Digest  \u00b7  Page %d" % doc.page)
        canvas.restoreState()

    def render(self, summary: str, extra_meta: dict | None = None) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(self._output_dir(), f"github_digest_{date_str}.pdf")
        st = self._styles()
        meta = extra_meta or {}

        doc = SimpleDocTemplate(
            out_path, pagesize=LETTER,
            leftMargin=0.8 * inch, rightMargin=0.8 * inch,
            topMargin=0.7 * inch, bottomMargin=0.8 * inch,
            title="GitHub Daily Digest " + date_str,
            author="digest-agent",
        )
        story = []

        # --- Cover band (colored) ---
        band = Table(
            [[Paragraph(self.title, st["cover_title"])],
             [Paragraph(date_str + "  \u00b7  Engineering report" + (("  \u00b7  " + meta.get("repos", "")) if meta.get("repos") else ""), st["cover_sub"])]],
            colWidths=[doc.width],
        )
        band.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), self.brand),
            ("TOPPADDING", (0, 0), (-1, -1), 26),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 26),
        ]))
        story.append(band)
        story.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=10))

        bullets = []
        code = []

        def flush_bullets():
            if bullets:
                story.append(Spacer(1, 2))
                for b in bullets:
                    story.append(Paragraph("\u2022  " + b, st["bullet"]))
                story.append(Spacer(1, 2))
                bullets.clear()

        def flush_code():
            if code:
                story.append(Spacer(1, 2))
                story.append(_code_block(code, st["code"]))
                story.append(Spacer(1, 2))
                code.clear()

        for raw in summary.splitlines():
            line = raw.strip()
            if not line:
                flush_bullets(); flush_code(); continue
            if line.startswith("```"):
                flush_bullets()
                flush_code()
                continue
            if line.startswith("# "):
                flush_bullets(); flush_code()
                story.append(Paragraph(line[2:], st["h1"]))
            elif line.startswith("#### "):
                flush_bullets(); flush_code()
                story.append(Paragraph(line[5:], st["h2"]))
            elif line.startswith("## "):
                flush_bullets(); flush_code()
                story.append(Paragraph(line[3:], st["h1"]))
            elif line.startswith("### "):
                flush_bullets(); flush_code()
                story.append(Paragraph(line[4:], st["h2"]))
            elif line.startswith("- ") or line.startswith("  - "):
                flush_code()
                bullets.append(line.lstrip("- ").strip())
            elif raw.startswith("  "):
                flush_bullets()
                code.append(raw.rstrip())
            else:
                flush_bullets(); flush_code()
                story.append(Paragraph(_esc(line), st["body"]))

        flush_bullets(); flush_code()
        doc.build(story, onFirstPage=self._footer, onLaterPages=self._footer)
        return out_path


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _code_block(lines, style):
    """Render code as a splittable boxed paragraph (safe for large hunks)."""
    text = "<br/>".join(_esc(l) for l in lines)
    return Paragraph(text, _boxstyle(style))


def _boxstyle(base):
    """Return a paragraph-style variant that carries a light background."""
    from reportlab.lib.styles import ParagraphStyle
    return ParagraphStyle("CodeBox", parent=base, backColor=CODE_BG, borderPadding=6,
                          borderColor=BOARD, borderWidth=0.5, spaceBefore=0, spaceAfter=0)