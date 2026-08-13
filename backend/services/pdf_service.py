"""
PDF validation, text extraction, and chunking for the temporary document
Q&A feature - upload -> extract -> chunk, nothing more. Storage (where the
chunks live afterward) is document_store.py's job, retrieval is
pdf_retrieval.py's job; this module never touches either.

The PDF bytes handed in here are never written to disk and never kept
around beyond this module's own function calls - only the resulting text
chunks leave this module, which is what routes/documents.py stores.

PDF content is UNTRUSTED DATA. This module only ever treats it as text to
extract and split - it never evaluates, executes, or otherwise interprets
anything found inside a PDF.
"""

import io
import os

import pymupdf
from pypdf import PdfReader

from services.gemini_client import generate_from_image

MAX_PDF_BYTES = int(os.getenv("PDF_MAX_BYTES", str(15 * 1024 * 1024)))

# Same chunk size/overlap this project's README documents for the
# (since-removed) FAISS pipeline - kept as the default here for
# continuity, still overridable per-deployment like the rest of this
# project's tunables (e.g. DB_QUERY_ROW_LIMIT).
CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "80"))

# Bounds cost/latency of the OCR fallback (see extract_text_via_ocr) -
# one Gemini vision call per page, so an unbounded scanned PDF could
# otherwise mean dozens of calls for a single upload. Same rationale as
# MAX_PDF_BYTES/DB_QUERY_ROW_LIMIT: a sane default that's still
# overridable per-deployment.
OCR_MAX_PAGES = int(os.getenv("PDF_OCR_MAX_PAGES", "10"))

OCR_PROMPT = (
    "Read and transcribe all text visible in this document page, verbatim. "
    "Preserve the original wording exactly - do not summarize, correct, or "
    "add anything. If the page has no readable text, respond with exactly: "
    "(no text found)"
)


class PdfProcessingError(Exception):
    """Raised with a message safe to show the user as-is."""


class NoExtractableTextError(PdfProcessingError):
    """A more specific PdfProcessingError: the PDF opened and parsed
    fine, it just has no text layer (e.g. a scanned/image-only PDF).
    Still caught by any existing `except PdfProcessingError` unchanged -
    this only exists so routes/documents.py can distinguish "try OCR,
    this might be scannable" from "give up, this file is actually
    broken" (corrupted/password-protected) without parsing error text.
    """


def validate_pdf(filename, content_type, data):
    """Raise PdfProcessingError if this doesn't look like a usable PDF
    upload. Cheap, structural checks only - real corruption is caught
    later by extract_text() actually trying to parse it."""

    if not data:
        raise PdfProcessingError("The uploaded file is empty.")

    if len(data) > MAX_PDF_BYTES:
        max_mb = MAX_PDF_BYTES / (1024 * 1024)
        raise PdfProcessingError(f"PDF is too large (max {max_mb:.0f} MB).")

    looks_like_pdf_name = (filename or "").lower().endswith(".pdf")
    looks_like_pdf_type = (content_type or "").lower() == "application/pdf"

    if not (looks_like_pdf_name or looks_like_pdf_type):
        raise PdfProcessingError("Only PDF files are supported.")

    if not data.startswith(b"%PDF-"):
        raise PdfProcessingError("This doesn't look like a valid PDF file.")


def extract_text(data):
    """Return the full extracted text of a validated PDF's bytes.

    Raises PdfProcessingError for anything that keeps this from
    producing usable text: a corrupted/unreadable file, a
    password-protected file that can't be opened with an empty
    password, or a file that opens fine but has no extractable text
    (e.g. a scanned/image-only PDF - see the module docstring in
    pdf_retrieval.py for why OCR isn't in scope here).
    """

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as error:
        raise PdfProcessingError("This PDF could not be read - it may be corrupted.") from error

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            pass
        if reader.is_encrypted:
            raise PdfProcessingError("This PDF is password-protected and cannot be read.")

    try:
        page_texts = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:
        raise PdfProcessingError("This PDF could not be read - it may be corrupted.") from error

    text = "\n".join(page_texts).strip()

    if not text:
        raise NoExtractableTextError(
            "This PDF does not contain extractable text. OCR support is "
            "required for scanned PDFs."
        )

    return text


async def extract_text_via_ocr(data, max_pages=None):
    """OCR fallback for a PDF with no text layer (see
    NoExtractableTextError) - renders each page to an image with
    PyMuPDF (no system-level dependency, unlike poppler/Tesseract) and
    asks Gemini's vision model to transcribe it, page by page.

    Capped at max_pages (OCR_MAX_PAGES by default) - remaining pages are
    silently skipped, not an error, so a long scanned document still
    returns whatever its first pages say rather than failing outright.

    Returns the concatenated transcribed text (empty string if no page
    had readable content - the caller treats that the same as
    NoExtractableTextError). Only raises PdfProcessingError if the file
    can't even be opened for rendering (rare, since pypdf already parsed
    it successfully in extract_text() before this is ever called) - a
    single page's Gemini call failing propagates as GeminiError, left to
    the caller (routes/documents.py) rather than silently skipped, since
    that's a real failure worth surfacing, unlike a page that's just
    genuinely blank.
    """

    max_pages = max_pages or OCR_MAX_PAGES

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as error:
        raise PdfProcessingError("This PDF could not be read - it may be corrupted.") from error

    page_texts = []

    try:
        for page in doc[:max_pages]:
            pixmap = page.get_pixmap(dpi=150)
            page_image = pixmap.tobytes("png")

            transcribed = await generate_from_image(page_image, "image/png", OCR_PROMPT)
            transcribed = transcribed.strip()

            if transcribed and transcribed != "(no text found)":
                page_texts.append(transcribed)
    finally:
        doc.close()

    return "\n\n".join(page_texts).strip()


def chunk_text(text, chunk_size=None, overlap=None):
    """Split text into overlapping character chunks.

    Overlap keeps a sentence that straddles a chunk boundary from losing
    context in whichever chunk retrieval picks - same rationale as the
    project's since-removed FAISS pipeline used (see README).
    """

    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = end - overlap

    return chunks
