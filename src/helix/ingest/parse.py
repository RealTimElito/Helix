"""Parse PDFs, text files, and images into page-sized records."""

from __future__ import annotations

from pathlib import Path


def parse_path(path: Path) -> list[dict]:
    """Return page records with 1-based page numbers and plain text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [{"page": 1, "section": "Document", "text": text.strip()}]
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return [{
            "page": 1,
            "section": "Image",
            "text": (
                f"Image file {path.name} was stored. Helix does not OCR images "
                "in this build, so there is no extractable text to cite."
            ),
        }]
    raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(path: Path) -> list[dict]:
    import pymupdf

    pages: list[dict] = []
    tables = _tables_by_page(path)
    with pymupdf.open(path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            extra = tables.get(index)
            if extra:
                text = f"{text}\n\nTables:\n{extra}".strip()
            pages.append({"page": index, "section": f"Page {index}", "text": text})
    return [page for page in pages if page["text"]]


def _tables_by_page(path: Path) -> dict[int, str]:
    try:
        import pdfplumber
    except ImportError:
        return {}
    found: dict[int, str] = {}
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            rendered = []
            for table in page.extract_tables() or []:
                rows = []
                for row in table:
                    cells = [(cell or "").strip().replace("\n", " ") for cell in row]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    rendered.append("\n".join(rows))
            if rendered:
                found[index] = "\n\n".join(rendered)
    return found


def chunk_pages(pages: list[dict], max_chars: int = 1600, overlap: int = 160) -> list[dict]:
    """Split long pages into section-bounded chunks near the configured length."""
    chunks: list[dict] = []
    for page in pages:
        text = " ".join(page["text"].split())
        if not text:
            continue
        if len(text) <= max_chars:
            chunks.append({**page, "text": text})
            continue
        start = 0
        part = 1
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):
                break_at = text.rfind(". ", start, end)
                if break_at > start + max_chars // 2:
                    end = break_at + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "page": page["page"],
                    "section": f"{page['section']} part {part}",
                    "text": piece,
                })
                part += 1
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks
