"""
The one place that calls Gemini.

Question answering, document summaries, and document analysis all need to
turn a prompt into generated text and handle the same handful of Gemini
failure modes (bad key, retired model, rate limit). This module is that
shared call, so none of the three processing pipelines duplicate it.

open_live_session() below is the same idea applied to the Gemini Live API
(real-time, bidirectional audio) instead of a single request/response
call - see services/voice_live_service.py, the only caller.
"""

import os

from google import genai
from google.genai import types

import config

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# The Gemini Live API (real-time, bidirectional audio) is a separate
# model family from MODEL_NAME above. The name below was confirmed
# directly against this key's own client.models.list() output (filtered
# to models whose supported_actions include "bidiGenerateContent") - see
# services/voice_live_service.py for how to re-run that check if this
# ever starts failing again, since preview/latest Live model names move
# between SDK releases and accounts.
#
# Switched from the native-audio-dialog model ('gemini-2.5-flash-native-
# audio-latest') to this half-cascade "flash-live" model after live
# side-by-side testing against a real recorded sample
# (backend/scripts/samples/sample1_profit.wav) found the native-audio
# model mishearing domain words ("What is the profit today?" transcribed
# as "What is the prophet today?") and taking ~4.8s to start replying,
# while this model transcribed the same clip correctly and started
# replying in ~0.2-0.3s - a cascade model's separate, dedicated ASR
# stage is simply more literal/accurate than a native-audio model's
# end-to-end audio understanding, and its dedicated TTS stage starts
# speaking sooner than one long native-audio generation does.
#
# This model does NOT reliably auto-detect end-of-turn from silence the
# way the native-audio model did in testing (a full clip with trailing
# silence and no explicit signal got zero response even after 15s) - it
# needs an explicit audio_stream_end=True sent once the user stops
# talking, confirmed live to then respond in ~0.2-0.3s and to handle
# multiple back-to-back turns correctly in the same session. See
# services/voice_live_service.py's "end_turn" control message for where
# that signal now comes from.
LIVE_MODEL_NAME = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")

# The SDK's own default (no retry_options passed) is "never retry" - a
# single transient network blip or 5xx/429 spike fails the whole
# request, whether that's a chat answer, OCR, or voice transcription.
# 3 attempts (1 initial + 2 retries) with a capped delay is deliberately
# more conservative than the SDK's own out-of-the-box retry default (5
# attempts, up to 60s max delay each) - that's tuned for batch/background
# work, but this app's calls are all in the critical path of a
# synchronous user-facing request, so a bounded few seconds of added
# latency is worth it, not a multi-minute hang before finally failing.
# Which errors get retried (408/429/5xx, plus connection/timeout errors)
# is the SDK's own default (see google.genai._api_client.retry_args) -
# not overridden here, just given a tighter attempt/delay budget.
_RETRY_OPTIONS = types.HttpRetryOptions(attempts=3, initial_delay=1.0, max_delay=8.0)


class GeminiError(Exception):
    """Raised with a user-facing message already attached."""


def _require_api_key():
    api_key = config.GEMINI_API_KEY

    if not api_key:
        raise GeminiError("GEMINI_API_KEY is missing. Add it to backend/.env")

    return api_key


def _client(api_key):
    """The one place a genai.Client is constructed, so the retry policy
    above can't drift between generate()/generate_from_image()/
    generate_from_audio() - unlike their error-handling blocks (kept
    duplicated on purpose, see each function's docstring), a retry
    policy is a single cross-cutting setting, not per-endpoint
    behavior, so there's no reason for three copies of it to exist."""

    return genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=_RETRY_OPTIONS))


def open_live_session(live_config):
    """Return the async context manager for one Gemini Live (real-time
    audio) session: `async with open_live_session(config) as session: ...`.

    Not itself async - client.aio.live.connect() is decorated as an
    async context manager, so calling it only builds that object; the
    actual WebSocket connection to Gemini opens on `async with`, same as
    every other call in this module only reaches the network once
    awaited/entered, not at construction time.

    Raises GeminiError immediately, before any connection attempt, if
    GEMINI_API_KEY is missing - same precondition check as every other
    function here, just surfaced synchronously instead of on first
    await, since there's no request to fail yet at this point.

    live_config is the caller's own policy (response modality,
    transcription, system instruction, ...) - this function only owns
    *how* to reach Gemini Live, not *what* to ask it for, matching how
    every other prompt/config in this app lives in its own service
    module rather than here.
    """

    api_key = _require_api_key()
    client = _client(api_key)

    return client.aio.live.connect(model=LIVE_MODEL_NAME, config=live_config)


async def generate(prompt):
    """Send one prompt to Gemini and return the generated text.

    Raises GeminiError with a friendly message on any failure.
    """

    api_key = _require_api_key()

    try:
        client = _client(api_key)
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


async def generate_stream(prompt):
    """Send one prompt to Gemini and yield the generated text incrementally,
    chunk by chunk, as Gemini produces it - the streaming counterpart to
    generate() above, used by routes/chat.py to start showing an answer
    to the user before the full response has finished generating.

    Raises GeminiError with a friendly message on any failure - same
    error categories as generate(). Can raise EITHER before the first
    chunk is yielded (e.g. a bad API key, caught immediately, same as a
    single-shot generate() failure) OR partway through iteration (e.g. a
    transient network/rate-limit error after some text has already been
    yielded and, once a caller is actively relaying it onward, already
    shown to the user). Callers that stream chunks onward as they arrive
    need to handle the second case explicitly - there is no way to
    "un-send" text the user has already seen - not just wrap the whole
    call in one try/except the way a single-shot generate() caller safely
    can.
    """

    api_key = _require_api_key()

    try:
        # `client` MUST be held in a local variable for the whole
        # streaming iteration below, not just the initial await - a real
        # bug, caught live: collapsing this into one chained expression
        # (`await _client(api_key).aio.models.generate_content_stream(...)`)
        # left nothing referencing the genai.Client once that first
        # await returned, so it (and its underlying aiohttp session) was
        # eligible for garbage collection before the `async for` below
        # ever pulled a chunk - observed as an AssertionError deep in
        # aiohttp ("self._connector is not None") the moment iteration
        # actually started.
        client = _client(api_key)
        stream = await client.aio.models.generate_content_stream(
            model=MODEL_NAME,
            contents=prompt,
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
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

        raise GeminiError(f"Gemini request failed: {message}") from error


# Cap on how many web sources one grounded answer reports back. Google
# Search grounding routinely attaches a dozen-plus chunks for a single
# query; the frontend renders each one as a badge under the answer, so
# past a handful this stops being provenance and starts being clutter.
_MAX_GROUNDING_SOURCES = 5


def _grounding_sources(response):
    """Pull the web pages a grounded response actually cited out of its
    grounding metadata, as {"type": "web", "title", "url"} dicts.

    Defensive throughout (getattr with defaults, empty-list fallbacks):
    grounding_metadata is absent entirely when the model chose not to
    search, and individual chunks can carry a retrieval source rather
    than a web one - neither is an error, both just mean "no web source
    here", and neither should be able to fail a request that otherwise
    has a perfectly good answer in hand.
    """

    sources = []
    seen = set()

    for candidate in getattr(response, "candidates", None) or []:
        metadata = getattr(candidate, "grounding_metadata", None)

        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None)

            if not uri or uri in seen:
                continue

            seen.add(uri)
            sources.append({
                "type": "web",
                "title": getattr(web, "title", None) or getattr(web, "domain", None) or uri,
                "url": uri,
            })

            if len(sources) >= _MAX_GROUNDING_SOURCES:
                return sources

    return sources


async def generate_with_search(prompt):
    """Send one prompt to Gemini with Google Search grounding enabled and
    return (text, sources).

    Used by services/entity_resolution.py to check what a named entity in
    a question actually refers to before an answer is written about it -
    the one thing the model's own training data cannot be trusted for,
    since an entity it has never seen is exactly the case where it
    substitutes a similarly-named one it has.

    Google Search grounding is a built-in tool of the Gemini API, so this
    needs no second search provider, no extra API key, and no new
    dependency - just a tools= config on the same client every other call
    in this module already builds.

    Same shape as generate_from_image/generate_from_audio (a separate
    function with its own copy of the error handling, rather than an
    optional flag on generate()) for the same reason those are: it leaves
    generate()'s own body, and therefore every existing caller of it,
    completely untouched.

    Raises GeminiError with a friendly message on any failure, same error
    categories as generate().
    """

    api_key = _require_api_key()

    try:
        client = _client(api_key)
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        text = (response.text or "").strip()
        sources = _grounding_sources(response)
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

        raise GeminiError(f"Gemini web search request failed: {message}") from error

    return text, sources


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
        client = _client(api_key)
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
        client = _client(api_key)
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
