"""
ingestion.py

Extracts plain text from an uploaded file, regardless of its original
format. Everything downstream of this module (detection, masking,
logging) only ever deals with plain text — this is the one place in
the whole pipeline that needs to know PDF, DOCX, and TXT are different.
"""

import PyPDF2
import pdfplumber
from docx import Document


def extract_text(uploaded_file) -> str:
    """`uploaded_file` is a Streamlit UploadedFile object (file-like,
    supports .read() and .name)."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    if filename.endswith(".pdf"):
        return _extract_pdf(uploaded_file)

    if filename.endswith(".docx"):
        return _extract_docx(uploaded_file)

    raise ValueError(f"Unsupported file type: {filename}. Supported: .txt, .pdf, .docx")


def _extract_pdf(uploaded_file) -> str:
    """Tries pdfplumber first (better text-extraction fidelity). Falls
    back to PyPDF2 if pdfplumber can't parse the file at all — some
    malformed or unusually encoded PDFs trip up one library but not
    the other."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass  # fall through to PyPDF2

    uploaded_file.seek(0)  # reset the file pointer before the second attempt
    reader = PyPDF2.PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _extract_docx(uploaded_file) -> str:
    """python-docx does not flatten tables into paragraph text
    automatically, so paragraphs and table cells are extracted
    separately and concatenated.

    Two-column rows are rendered as "Label: Value" — matching the
    colon-separated format several training templates already use
    (e.g. "Branch: Karimnagar") — rather than a pipe-delimited format
    the NER model has never seen and handles unreliably. Tables with
    more than two columns fall back to a pipe join, since a colon
    format doesn't make sense there anyway."""
    doc = Document(uploaded_file)

    parts = [p.text for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) == 2 and cells[0] and cells[1]:
                row_text = f"{cells[0]}: {cells[1]}"
            else:
                row_text = " | ".join(cells)
            if row_text.strip(" |:"):
                parts.append(row_text)

    return "\n".join(parts)