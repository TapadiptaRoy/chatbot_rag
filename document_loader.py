"""
document_loader.py
-------------------
Extracts plain text from uploaded files (PDF, DOCX, TXT, MD) and splits
that text into overlapping chunks suitable for embedding.
"""

import os
from pypdf import PdfReader
import docx2txt


def extract_text(filepath):
    """Extract raw text from a file based on its extension."""
    root, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext == ".pdf":
        return _extract_pdf(filepath)
    elif ext == ".docx":
        return _extract_docx(filepath)
    elif ext in (".txt", ".md"):
        return _extract_plain(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)

    return "\n".join(full_text)


def _extract_docx(docx_path):
    text = docx2txt.process(docx_path)
    return text


def _extract_plain(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text, chunk_size=800, overlap=120):
    """Split text into overlapping chunks, breaking near word/sentence boundaries."""
    text = " ".join(text.split())  # collapse whitespace/newlines
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # try to break at a sentence or word boundary instead of mid-word
        if end < text_len:
            boundary = text.rfind(". ", start, end)
            if boundary == -1 or boundary < start + chunk_size * 0.5:
                boundary = text.rfind(" ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


if __name__ == "__main__":
    # Quick manual test — replace with a real file path to try it
    test_file = "fsnd-interview-questions.pdf"
    text = extract_text(test_file)
    chunks = chunk_text(text)
    print(f"Extracted {len(text)} characters, split into {len(chunks)} chunks")