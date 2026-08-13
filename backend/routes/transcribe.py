import os
import tempfile

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from starlette.concurrency import run_in_threadpool

from config import DEBUG_VOICE_PIPELINE
from services.transcription import transcribe_audio, TranscriptionError
from services.query_normalizer import normalize_query

router = APIRouter()

# Generous enough for a spoken question (well over a minute of compressed
# voice audio), small enough to keep memory use bounded on an 8 GB machine.
MAX_AUDIO_BYTES = 15 * 1024 * 1024


@router.post("/transcribe")
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

    # Whisper decodes from a real file (via PyAV), so the upload is written
    # to a temp file, transcribed, then always removed - nothing from the
    # recording is kept on disk afterwards.
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)

        raw_text, language, language_probability = await run_in_threadpool(
            transcribe_audio, temp_path, language_hint
        )
    except TranscriptionError as error:
        raise HTTPException(status_code=502, detail=str(error))
    finally:
        os.remove(temp_path)

    if not raw_text:
        raise HTTPException(
            status_code=400,
            detail="No speech was detected. Please try again."
        )

    # Turns a messy/code-mixed raw transcript into a clean question - this
    # is what actually enters the chat input, so the existing intent
    # router and RAG pipeline see well-formed text just like typed input.
    normalized_text = await normalize_query(raw_text, language)

    response = {"transcript": normalized_text, "language": language}

    if DEBUG_VOICE_PIPELINE:
        response["debug"] = {
            "raw_transcript": raw_text,
            "language_probability": round(language_probability, 3),
        }

    return response
