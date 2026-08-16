"""
ingestion.py

Extracts plain text from an uploaded file, regardless of its original
format. Everything downstream of this module (detection, masking,
logging) only ever deals with plain text — this is the one place in
the whole pipeline that needs to know PDF, DOCX, TXT, and images are
different.
"""

import cv2
import numpy as np
import PyPDF2
import pdfplumber
import pymupdf
import pytesseract
from docx import Document
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")

# Below this many extracted characters, OCR is flagged as low-confidence
# in the UI — not because this threshold is scientifically derived, but
# because near-empty OCR output is a reliable practical signal that
# something (resolution, blur, lighting) went wrong upstream.
LOW_CONFIDENCE_CHAR_THRESHOLD = 20


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Light, empirically-tested preprocessing: grayscale + 2x upscale.
    Testing against deliberately degraded sample scans found this
    combination never made OCR output worse, while more aggressive
    techniques (adaptive thresholding, denoising, sharpening) actively
    hurt results on blurred input by amplifying artifacts into false
    edges. This is intentionally conservative rather than maximal."""
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(gray)


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

    if filename.endswith(IMAGE_EXTENSIONS):
        return _extract_image(uploaded_file)

    raise ValueError(
        f"Unsupported file type: {filename}. "
        f"Supported: .txt, .pdf, .docx, {', '.join(IMAGE_EXTENSIONS)}"
    )


def _extract_image(uploaded_file) -> str:
    """Runs Tesseract OCR on an uploaded image file, after light
    preprocessing (see _preprocess_for_ocr)."""
    image = Image.open(uploaded_file)
    processed = _preprocess_for_ocr(image)
    return pytesseract.image_to_string(processed).strip()


def _extract_pdf(uploaded_file) -> str:
    """Tries pdfplumber first (better text-extraction fidelity). Falls
    back to PyPDF2 if pdfplumber can't parse the file at all. If BOTH
    come back empty — a scanned/image-only PDF with no real text
    layer — falls back to OCR: rasterize each page and run Tesseract
    on it, using the same engine as a standalone image upload."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass  # fall through to PyPDF2

    uploaded_file.seek(0)
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception:
        pass  # fall through to OCR

    uploaded_file.seek(0)
    return _extract_pdf_via_ocr(uploaded_file)


def _extract_pdf_via_ocr(uploaded_file) -> str:
    """Rasterizes each PDF page to an image at 200 DPI and runs
    Tesseract on it. Used only when normal text extraction found
    nothing — i.e. the PDF is genuinely scanned/image-only."""
    pdf_bytes = uploaded_file.read()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page_texts = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        processed = _preprocess_for_ocr(image)
        page_texts.append(pytesseract.image_to_string(processed))
    doc.close()
    return "\n".join(page_texts).strip()


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


def is_low_confidence_extraction(text: str, filename: str) -> bool:
    """Flags extractions worth warning the user about — currently only
    meaningful for OCR-derived text (images and scanned PDFs), where a
    very short result is a practical signal that the scan quality was
    likely too poor for reliable extraction, per the degraded-image
    testing that motivated this threshold."""
    is_ocr_source = filename.lower().endswith(IMAGE_EXTENSIONS)
    return is_ocr_source and len(text.strip()) < LOW_CONFIDENCE_CHAR_THRESHOLD