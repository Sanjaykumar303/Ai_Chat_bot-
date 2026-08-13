"""
Turns a raw (possibly code-mixed, broken-English, repeated-word) Whisper
transcript into a clean, well-formed question - using the existing Gemini
model, not a new one. This is what actually fixes code-mixed voice input:
Whisper cannot cleanly transcribe two interleaved languages in one pass
(see transcription.py's docstring), but Gemini can still understand what
a messy transcript of one meant and restate it clearly.

This stage only rephrases. It never answers the question - that still
happens later, in routes/chat.py, through the exact same pipeline typed
questions already use.
"""

import logging

from config import DEBUG_VOICE_PIPELINE
from services.gemini_client import generate, GeminiError

logger = logging.getLogger("uvicorn")

NORMALIZE_PROMPT = """You are cleaning up a speech-to-text transcript so it can be understood correctly by a downstream system.

The speaker's speech was transcribed by a small local speech-to-text model, and may contain:
- Code-mixed Tamil-English (or Kannada-English, Bengali-English) speech
- Broken or grammatically incorrect English
- Transcription mistakes, stuttered or repeated words, filler sounds
- An informal Indian conversational speaking style ("...na enna", "...pannunga", "bro", etc.)

Your ONLY job is to rewrite this into a single, clear, well-formed question or instruction in English that preserves the speaker's exact intended meaning.

Rules:
- Do NOT answer the question. Only rewrite it.
- Do NOT add information that was not implied by the transcript.
- Do NOT drop meaningful content - keep references to "this document", specific topics, or requests to summarize/analyze/compare.
- Collapse repeated or stuttered phrases into one clean statement instead of listing them twice.
- If the transcript is already a clear, well-formed English question, return it unchanged (only fix obvious transcription typos).
- Output ONLY the rewritten question or instruction. No explanation, no quotes, no preamble, no answer.

TRANSCRIPT (speech-to-text detected language: {language}):
"{transcript}"

REWRITTEN QUESTION:"""

# A guard against Gemini "helpfully" answering instead of rephrasing: a
# real rephrased question is rarely much longer than the original, messy
# transcript. If it comes back drastically longer, that's a sign Gemini
# answered rather than restated, so the raw transcript is used instead.
MAX_LENGTH_MULTIPLIER = 3


async def normalize_query(raw_transcript, language):
    """Return a clean, semantically-equivalent question for raw_transcript.

    Falls back to the raw transcript unchanged (never raises) if Gemini is
    unavailable or its output looks unusable - a normalization failure
    should never block the user from at least getting their raw transcript
    into the chat box.
    """

    if not raw_transcript:
        return raw_transcript

    prompt = NORMALIZE_PROMPT.format(language=language or "unknown", transcript=raw_transcript)

    try:
        normalized = await generate(prompt)
    except GeminiError as error:
        logger.warning(f"Query normalization skipped (Gemini unavailable): {error}")
        return raw_transcript

    normalized = normalized.strip().strip('"')

    if not normalized or len(normalized) > len(raw_transcript) * MAX_LENGTH_MULTIPLIER:
        logger.warning("Query normalization produced an unusable result, using raw transcript instead.")
        return raw_transcript

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] NORMALIZED QUERY: {normalized!r}")

    return normalized
