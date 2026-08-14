import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from services import document_store, image_service, pdf_retrieval, pdf_service
from services.gemini_client import GeminiError
from services.image_service import ImageProcessingError
from services.pdf_service import NoExtractableTextError, PdfProcessingError
from services.rate_limiter import enforce_rate_limit

router = APIRouter()

logger = logging.getLogger("uvicorn")


def _is_pdf(filename, content_type):
    return (filename or "").lower().endswith(".pdf") or (content_type or "").lower() == "application/pdf"


def _is_image(filename, content_type):
    return (
        (filename or "").lower().endswith(image_service.SUPPORTED_EXTENSIONS)
        or (content_type or "").lower() in image_service.SUPPORTED_MIME_TYPES
    )


@router.post("/documents/upload", dependencies=[Depends(enforce_rate_limit)])
async def upload_document(file: UploadFile = File(...)):

    data = await file.read()

    try:
        if _is_pdf(file.filename, file.content_type):
            pdf_service.validate_pdf(file.filename, file.content_type, data)

            try:
                text = pdf_service.extract_text(data)
            except NoExtractableTextError:
                # No text layer (e.g. scanned) - fall back to rendering
                # each page as an image and reading it via Gemini vision
                # instead of rejecting outright.
                logger.info(f"'{file.filename}' has no PDF text layer, trying OCR")
                text = await pdf_service.extract_text_via_ocr(data)

        elif _is_image(file.filename, file.content_type):
            image_service.validate_image(file.filename, file.content_type, data)
            text = await image_service.extract_text(data)

        else:
            raise HTTPException(
                status_code=400,
                detail="Only PDF, JPG, PNG, or WEBP files are supported.",
            )

    except (PdfProcessingError, ImageProcessingError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    except GeminiError as error:
        raise HTTPException(status_code=502, detail=str(error))

    if not text:
        raise HTTPException(
            status_code=400,
            detail="No readable text or content was found in this file, even with OCR.",
        )

    chunks = pdf_service.chunk_text(text)

    if not chunks:
        raise HTTPException(status_code=400, detail="No readable content was found in this file.")

    try:
        vectorizer, matrix = pdf_retrieval.build_index(chunks)
    except Exception as error:
        logger.warning(f"Failed to build retrieval index for uploaded file: {error}")
        raise HTTPException(status_code=502, detail="Could not process this file. Please try again.")

    document_id = document_store.create_document(file.filename, chunks, vectorizer, matrix)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "status": "ready",
        "chunks": len(chunks),
    }


@router.delete("/documents/{document_id}")
async def remove_document(document_id: str):

    removed = document_store.delete_document(document_id)

    if not removed:
        raise HTTPException(status_code=404, detail="Document not found or already expired.")

    return {"status": "removed", "document_id": document_id}
