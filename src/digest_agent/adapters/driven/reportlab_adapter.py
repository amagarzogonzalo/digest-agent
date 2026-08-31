"""ReportLab adapter - implements DocumentRendererPort for the hexagonal architecture.

This adapter encapsulates all reportLab-specific logic: PDF document creation,
formatting, and file generation. The core depends only on the port interface,
not on reportlab directly."""
import os
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem

from digest_agent.ports import DocumentRendererPort, RenderError


class ReportLabAdapter(DocumentRendererPort):
    """Adapter for reportLab - implements DocumentRendererPort."""

    def _output_dir(self) -> str:
        """Return the directory PDFs are written to (cross-platform)."""
        return os.environ.get("DIGEST_OUTPUT_DIR", tempfile.gettempdir())

    def render(self, summary: str) -> str:
        """Render the summary to a PDF and return the file path."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(self._output_dir(), f"github_digest_{date_str}.pdf")

        styles = getSampleStyleSheet()
        h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=14)
        h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6)
        body = ParagraphStyle("Body", parent=styles["BodyText"], spaceAfter=6)

        doc = SimpleDocTemplate(out_path, pagesize=LETTER,
                                topMargin=0.75 * inch, bottomMargin=0.75 * inch)
        story = [Paragraph(f"GitHub Daily Digest — {date_str}", h1), Spacer(1, 6)]

        bullet_buffer = []

        def flush_bullets():
            if bullet_buffer:
                story.append(ListFlowable(
                    [ListItem(Paragraph(b, body)) for b in bullet_buffer],
                    bulletType="bullet",
                ))
                bullet_buffer.clear()

        for line in summary.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("## "):
                flush_bullets()
                story.append(Paragraph(line[3:], h2))
            elif line.startswith("# "):
                flush_bullets()
                story.append(Paragraph(line[2:], h1))
            elif line.startswith("- "):
                bullet_buffer.append(line[2:])
            else:
                flush_bullets()
                story.append(Paragraph(line, body))
        flush_bullets()

        doc.build(story)
        return out_path