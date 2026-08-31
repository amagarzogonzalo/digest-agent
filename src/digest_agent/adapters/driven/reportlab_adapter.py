"""ReportLab adapter - professional PDF rendering of the digest.

Renders markdown-ish digest text into a clean, styled PDF with a cover
header, section headings, bullets, and monospace code blocks.
"""
import os
import tempfile
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
    Preformatted,
    HRFlowable,
)

from digest_agent.ports import DocumentRendererPort, RenderError


ACCENT = colors.HexColor("#1f3b73")
CODE_BG = colors.HexColor("#f5f6f8")
TEXT = colors.HexColor("#1f2937")


class ReportLabAdapter(DocumentRendererPort):
    """Adapter for reportlab - implements DocumentRendererPort."""

    def _output_dir(self) -> str:
        return os.environ.get("DIGEST_OUTPUT_DIR", tempfile.gettempdir())

    def _styles(self):
        base = getSampleStyleSheet()
        return {
            "h0": ParagraphStyle("H0", parent=base["Title"], fontSize=22, textColor=ACCENT,
                                 leading=26, spaceAfter=2),
            "h1": ParagraphStyle("H1", parent=base["Heading2"], fontSize=15, textColor=ACCENT,
                                 leading=19, spaceBefore=16, spaceAfter=6),
            "h2": ParagraphStyle("H2", parent=base["Heading3"], fontSize=12, textColor=TEXT,
                                 leading=15, spaceBefore=10, spaceAfter=4),
            "body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=10, leading=14.5,
                                   textColor=colors.HexColor("#374151"), spaceAfter=6),
            "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontSize=10, leading=14,
                                     textColor=TEXT),
            "code": ParagraphStyle("Code", parent=base["Code"], fontSize=8.5, leading=11,
                                   textColor=colors.black, fontName="Courier",
                                   backColor=CODE_BG, borderPadding=6, borderColor=colors.HexColor("#e5e7eb"),
                                   borderWidth=0.5),
        }

    def render(self, summary: str) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(self._output_dir(), f"github_digest_{date_str}.pdf")
        st = self._styles()

        doc = SimpleDocTemplate(
            out_path, pagesize=LETTER,
            leftMargin=0.8*inch, rightMargin=0.8*inch,
            topMargin=0.8*inch, bottomMargin=0.8*inch,
        )
        story = [Paragraph("GitHub Daily Digest", st["h0"]),
                 Paragraph(date_str + " &nbsp; &middot; Engineering report", st["body"]),
                 HRFlowable(width="100%", thickness=1, color=ACCENT),
                 Spacer(1, 8)]

        bullets = []
        code = []

        def flush_bullets():
            if bullets:
                story.append(ListFlowable(
                    [ListItem(Paragraph(b, st["bullet"]), leftIndent=2) for b in bullets],
                    bulletType="bullet", bulletFontSize=8, bulletColor=ACCENT, leftIndent=10,
                ))
                bullets.clear()

        def flush_code():
            if code:
                story.append(Spacer(1, 4))
                story.append(Preformatted("\n".join(code), st["code"]))
                story.append(Spacer(1, 4))
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
            elif line.startswith("## "):
                flush_bullets(); flush_code()
                story.append(Paragraph(line[3:], st["h1"]))
            elif line.startswith("### "):
                flush_bullets(); flush_code()
                story.append(Paragraph(line[4:], st["h2"]))
            elif line.startswith("- ") or line.startswith("  - "):
                flush_code()
                bullets.append(line.lstrip("- ").strip())
            elif raw.startswith("    "):
                flush_bullets()
                code.append(raw.rstrip())
            else:
                flush_bullets(); flush_code()
                story.append(Paragraph(escape_markdown(line), st["body"]))

        flush_bullets(); flush_code()
        doc.build(story)
        return out_path


def escape_markdown(text: str) -> str:
    """Escape XML-sensitive chars for ReportLab paragraphs."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")