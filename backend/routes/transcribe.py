import os
import tempfile

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from config import DEBUG_VOICE_PIPELINE
from services.transcription import transcribe_audio, TranscriptionError
from services.rate_limiter import enforce_rate_limit

router = APIRouter()

# Generous enough for a spoken question (well over a minute of compressed
# voice audio), small enough to keep memory use bounded on an 8 GB machine.
MAX_AUDIO_BYTES = 15 * 1024 * 1024


@router.post("/transcribe", dependencies=[Depends(enforce_rate_limit)])
async def transcribe(audio: UploadFile = File(...), language_hint: str | None = Form(None)):

    # "auto" (the UI's default option) means "no hint" - only a real
    # language code should reach transcribe_audio() as an override.
    if language_hint == "auto":
        language_hint = None

    data = await audio.read()

    if not data:
        raise HTTPException(status_code=400, detail="No audio was recorded.")

    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Recording is too long. Please ask a shorter question."
        )

    # PyAV (services/transcription.py's _convert_to_wav) decodes from a
    # real file, so the upload is written to a temp file, transcribed,
    # then always removed - nothing from the recording is kept on disk
    # afterwards.
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)

        raw_text, clean_text, language = await transcribe_audio(temp_path, language_hint)
    except TranscriptionError as error:
        raise HTTPException(status_code=502, detail=str(error))
    finally:
        os.remove(temp_path)

    if not raw_text:
        raise HTTPException(
            status_code=400,
            detail="No speech was detected. Please try again."
        )

    # clean_text is already the code-mixed/business-domain-aware rewrite
    # (see transcription.py's TRANSCRIBE_PROMPT) - this is what actually
    # enters the chat input, so the existing intent router and RAG
    # pipeline see well-formed text just like typed input.
    response = {"transcript": clean_text, "language": language}

    if DEBUG_VOICE_PIPELINE:
        response["debug"] = {"raw_transcript": raw_text}

    return response
