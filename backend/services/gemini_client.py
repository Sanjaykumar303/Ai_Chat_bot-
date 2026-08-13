"""
The one place that calls Gemini to generate text.

Question answering, document summaries, and document analysis all need to
turn a prompt into generated text and handle the same handful of Gemini
failure modes (bad key, retired model, rate limit). This module is that
shared call, so none of the three processing pipelines duplicate it.
"""

import os

from google import genai
from google.genai import types

import config

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")


class GeminiError(Exception):
    """Raised with a user-facing message already attached."""


def _require_api_key():
    api_key = config.GEMINI_API_KEY

    if not api_key:
        raise GeminiError("GEMINI_API_KEY is missing. Add it to backend/.env")

    return api_key


async def generate(prompt):
    """Send one prompt to Gemini and return the generated text.

    Raises GeminiError with a friendly message on any failure.
    """

    api_key = _require_api_key()

    try:
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        text = (response.text or "").strip()
    except Exception as error:
        message = str(error)

        # The most common mistake is a missing or wrong key,
        # so we show a clear message for it.
        if "API key not valid" in message or "API_KEY_INVALID" in message:
            raise GeminiError(
                "Your Gemini API key is not valid. Check GEMINI_API_KEY in backend/.env"
            ) from error

        if "NOT_FOUND" in message or "no longer available" in message:
            raise GeminiError(
                f"The model '{MODEL_NAME}' is not available for your key. "
                f"Try a different GEMINI_MODEL in backend/.env"
            ) from error

        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            raise GeminiError(
                "Gemini rate limit reached. Please wait a moment and ask again."
            ) from error

        raise GeminiError(f"Gemini request failed: {message}") from error

    return text or "Sorry, no answer was returned. Please try asking again."


async def generate_from_image(image_bytes, mime_type, prompt):
    """Send one image + text prompt to Gemini (multimodal) and return the
    generated text. Used for OCR/transcription: a directly-uploaded image
    (see services/image_service.py) or a scanned PDF page rendered to an
    image (see services/pdf_service.py's OCR fallback) - both need "read
    what's in this image", which is exactly what this call does.

    Deliberately a separate function from generate() rather than a
    shared helper with an optional image param: keeps generate()'s own
    body untouched (same error handling, duplicated here rather than
    factored out, so nothing about the existing text-only path changes).

    Raises GeminiError with a friendly message on any failure, same
    error categories as generate().
    """

    api_key = _require_api_key()

    try:
        client = genai.Client(api_key=api_key)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part],
        )
        text = (response.text or "").strip()
    except Exception as error:
        message = str(error)

        if "API key not valid" in message or "API_KEY_INVALID" in message:
            raise GeminiError(
                "Your Gemini API key is not valid. Check GEMINI_API_KEY in backend/.env"
            ) from error

        if "NOT_FOUND" in message or "no longer available" in message:
            raise GeminiError(
                f"The model '{MODEL_NAME}' is not available for your key. "
                f"Try a different GEMINI_MODEL in backend/.env"
            ) from error

        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            raise GeminiError(
                "Gemini rate limit reached. Please wait a moment and ask again."
            ) from error

        raise GeminiError(f"Gemini image request failed: {message}") from error

    return text


async def generate_from_audio(audio_bytes, mime_type, prompt):
    """Send one audio clip + text prompt to Gemini (multimodal) and return
    the generated text. Used for voice input transcription (see
    services/transcription.py) - replaces the local Whisper model that
    used to do this.

    Same shape as generate_from_image (separate function, same error
    handling duplicated rather than factored out) for the same reason:
    keeps generate()'s own body untouched.

    Raises GeminiError with a friendly message on any failure, same
    error categories as generate().
    """

    api_key = _require_api_key()

    try:
        client = genai.Client(api_key=api_key)
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, audio_part],
        )
        text = (response.text or "").strip()
    except Exception as error:
        message = str(error)

        if "API key not valid" in message or "API_KEY_INVALID" in message:
            raise GeminiError(
                "Your Gemini API key is not valid. Check GEMINI_API_KEY in backend/.env"
            ) from error

        if "NOT_FOUND" in message or "no longer available" in message:
            raise GeminiError(
                f"The model '{MODEL_NAME}' is not available for your key. "
                f"Try a different GEMINI_MODEL in backend/.env"
            ) from error

        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            raise GeminiError(
                "Gemini rate limit reached. Please wait a moment and ask again."
            ) from error

        raise GeminiError(f"Gemini audio request failed: {message}") from error

    return text
