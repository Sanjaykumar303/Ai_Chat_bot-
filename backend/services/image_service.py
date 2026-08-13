"""
Image validation and text extraction for the temporary document Q&A
feature - the image counterpart to pdf_service.py. A directly-uploaded
image (a photo of a page, a screenshot, a scanned receipt, ...) is read
via Gemini's vision model instead of pypdf, but otherwise joins the exact
same pipeline as a PDF from here on: routes/documents.py chunks whatever
text comes back with pdf_service.chunk_text() and stores it in
document_store.py the same way, so PDF_QUERY/HYBRID_QUERY answering never
needs to know or care which format the original upload was.

Image bytes are never written to disk and never kept around beyond this
module's own function calls - only the resulting text leaves it.

Image content is UNTRUSTED DATA. This module only ever treats a
transcribed image as text to extract - it never evaluates, executes, or
otherwise interprets anything found inside one (see routes/chat.py's
PDF_ANSWER_PROMPT, which already treats all document context this way).
"""

import io
import os

from PIL import Image

from services.gemini_client import generate_from_image

MAX_IMAGE_BYTES = int(os.getenv("IMAGE_MAX_BYTES", str(10 * 1024 * 1024)))

# Gemini natively accepts these; anything else gets rejected before an
# API call is even attempted rather than failing less clearly downstream.
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

OCR_PROMPT = (
    "Read and transcribe all text visible in this image, verbatim. If the "
    "image contains a document, receipt, invoice, or similar, transcribe "
    "its full content preserving structure (labels, amounts, dates) as "
    "plain text. If the image has no readable text at all, briefly "
    "describe what the image shows instead."
)


class ImageProcessingError(Exception):
    """Raised with a message safe to show the user as-is."""


def validate_image(filename, content_type, data):
    """Raise ImageProcessingError if this doesn't look like a usable
    image upload. Cheap, structural checks only - real corruption is
    caught later by extract_text() actually decoding it."""

    if not data:
        raise ImageProcessingError("The uploaded file is empty.")

    if len(data) > MAX_IMAGE_BYTES:
        max_mb = MAX_IMAGE_BYTES / (1024 * 1024)
        raise ImageProcessingError(f"Image is too large (max {max_mb:.0f} MB).")

    looks_like_supported_name = (filename or "").lower().endswith(SUPPORTED_EXTENSIONS)
    looks_like_supported_type = (content_type or "").lower() in SUPPORTED_MIME_TYPES

    if not (looks_like_supported_name or looks_like_supported_type):
        raise ImageProcessingError("Only JPG, PNG, or WEBP images are supported.")


def _detected_mime_type(data):
    """Confirm data is actually a decodable image (not just a file with
    an image-y name/extension) and return its real MIME type - Gemini
    needs an accurate mime_type, and PIL's own format detection is more
    trustworthy than trusting the upload's declared content_type."""

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            format_name = (image.format or "").upper()
    except Exception as error:
        raise ImageProcessingError("This image could not be read - it may be corrupted.") from error

    format_to_mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
    mime_type = format_to_mime.get(format_name)

    if mime_type is None:
        raise ImageProcessingError("Only JPG, PNG, or WEBP images are supported.")

    return mime_type


async def extract_text(data):
    """Return Gemini's transcription of a validated image's bytes.

    Raises ImageProcessingError for a file that isn't actually a
    decodable image despite passing validate_image()'s name/type check.
    A GeminiError from generate_from_image() (bad key, rate limit, ...)
    propagates as-is - routes/documents.py already knows how to surface
    that the same way every other Gemini call in this project does.
    """

    mime_type = _detected_mime_type(data)

    text = await generate_from_image(data, mime_type, OCR_PROMPT)

    return text.strip()
