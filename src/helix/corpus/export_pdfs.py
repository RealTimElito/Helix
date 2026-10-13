"""Write a PDF copy of each seed note, including one table fixture."""

from __future__ import annotations

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

from helix.config import SAMPLE_DIR, ensure_dirs
from helix.corpus.library import documents


def export_samples() -> None:
    """Export seed notes to data/samples for the upload path."""
    ensure_dirs()
    styles = getSampleStyleSheet()
    for doc in documents():
        path = SAMPLE_DIR / f"{doc.doc_id}.pdf"
        story = []
        for page in doc.pages:
            story.append(Paragraph(page.section, styles["Heading2"]))
            for paragraph in page.text.split(". "):
                if paragraph.strip():
                    story.append(Paragraph(paragraph.strip() + ".", styles["BodyText"]))
                    story.append(Spacer(1, 8))
        if doc.doc_id == "HX-005":
            story.append(Paragraph("Transcribed grid", styles["Heading2"]))
            story.append(Table([
                ["Subset", "Cell accuracy"],
                ["Digital PDF", "0.96"],
                ["Phone photo", "0.71"],
            ]))
        document = SimpleDocTemplate(str(path), pagesize=letter)
        document.build(story)
    print(f"wrote {len(documents())} pdfs to {SAMPLE_DIR}")


if __name__ == "__main__":
    export_samples()
