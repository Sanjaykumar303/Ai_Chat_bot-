"""
Speech-to-text for voice input, via Gemini's native audio understanding
(services/gemini_client.py's generate_from_audio) - replaces the local
Whisper model (faster-whisper/ctranslate2) this app used previously.
That model was CPU-only, limited to "tiny"/"base" accuracy on an 8GB
machine, and hit a genuine upstream ctranslate2 packaging bug under
Python 3.13. Gemini is already a hard dependency for every other part of
this app (chat answers, OCR), is far stronger on accented/code-mixed
Indian English, and needs no local model file at all.

One Gemini call does transcription, language detection, AND the
code-mixed/business-domain cleanup a separate query_normalizer.py stage
used to do (now removed) - Gemini can reason about *meaning* while it
transcribes, which a pure acoustic model like Whisper cannot, so
collapsing this into one call is both simpler and more accurate, not
just fewer round trips.
"""

import io
import logging
import re

import av
import numpy as np

from config import DEBUG_VOICE_PIPELINE
from services.gemini_client import generate_from_audio, GeminiError

logger = logging.getLogger("uvicorn")


class TranscriptionError(Exception):
    """Raised with a user-facing message already attached."""


# Maps an optional language_hint code (routes/transcribe.py's
# language_hint form field - no current UI sends one, transcription is
# always auto-detected in practice) to a name Gemini can follow a "the
# speaker is speaking ___" instruction with.
LANGUAGE_HINT_NAMES = {
    "en": "English",
    "ta": "Tamil",
    "kn": "Kannada",
    "bn": "Bengali",
}

# PDF_CONTEXT-style framing isn't needed here (this prompt only ever sees
# this app's own recorded audio, not third-party untrusted content), but
# the business-domain steer below is deliberately the same one
# query_normalizer.py used to apply as a separate, later stage: doing it
# in the same pass the model already listens to the audio in means it
# can weigh the actual acoustics against domain plausibility together,
# rather than correcting a transcript it can no longer double-check
# against the sound. Found via real testing: "what is the profit" was
# coming back as "what do you prefer" - two ordinary, similar-sounding
# phrases, but a domain hint fixes it now that Gemini knows the assistant
# is business-focused.
TRANSCRIBE_PROMPT = """You are transcribing a spoken question for a business/accounting assistant that answers questions about revenue, profit, loss, expenses, income, sales, payments, invoices, vouchers, and ledger balances.

Listen to the attached audio and do three things:
1. Detect the primary language spoken. Output its ISO 639-1 code (e.g. "en", "ta", "kn", "bn").
2. Transcribe literally what was said.
3. Rewrite the transcript as a single, clear, well-formed English question or instruction that preserves the speaker's exact intended meaning. The speaker may mix languages (Tamil-English, Kannada-English, Bengali-English) or speak in an informal Indian conversational style ("...na enna", "...pannunga", "bro", etc.) - untangle that into plain English. If a word is ambiguous or unclear but closely resembles one of the business terms listed above, prefer that business-domain interpretation - but never invent a business meaning from an otherwise clear, unrelated question.{language_hint_instruction}

If the audio contains no intelligible speech (silence, noise, an unrelated sound), output NO_SPEECH for all three fields below instead.

Output EXACTLY these three lines and nothing else - no markdown, no explanation:
LANGUAGE: <iso 639-1 code, or NO_SPEECH>
RAW: <literal transcript, or NO_SPEECH>
QUESTION: <rewritten question, or NO_SPEECH>"""

_RESPONSE_RE = re.compile(
    r"LANGUAGE:\s*(?P<language>\S+)\s*"
    r"RAW:\s*(?P<raw>.*?)\s*"
    r"QUESTION:\s*(?P<question>.*)",
    re.DOTALL,
)


# Root-mean-square amplitude (of int16 samples, max possible 32767) below
# which a clip counts as silence/no-speech. Calibrated against real
# clips: true digital silence measures 0, actual recorded speech (even a
# quiet test clip) measured 1800-2700 - 150 sits with wide margin below
# any real speech while still well above pure silence, so it won't
# false-reject quiet-but-real speech.
SILENCE_RMS_THRESHOLD = 150


def _convert_to_wav(file_path):
    """Decode whatever container/codec PyAV understands (webm/opus from
    Chrome/Edge's MediaRecorder, mp4/aac from Safari, ...) into mono
    16kHz WAV bytes - a format every Gemini deployment accepts, so this
    app never has to track which of the many browser recording formats
    Gemini's audio input does or doesn't support. faster-whisper already
    depended on PyAV transitively for exactly this kind of decoding, so
    it's kept as a direct dependency now that it's called explicitly.

    Also returns whether the clip has any audible signal at all (see
    SILENCE_RMS_THRESHOLD) - a real, observed failure mode is Gemini
    confidently "transcribing" a plausible-sounding business question
    out of pure silence when just asked to say NO_SPEECH instead. The
    old Whisper path's vad_filter handled this deterministically at the
    acoustic level; this is the same idea, computed directly from the
    decoded samples so a silent clip never even reaches the Gemini call
    that would otherwise hallucinate on it - matching this project's own
    established preference for a deterministic guard over trusting a
    prompt instruction alone (see chat_service.py's zero-row guards)."""

    try:
        input_container = av.open(file_path)
        input_stream = input_container.streams.audio[0]

        output_buffer = io.BytesIO()
        output_container = av.open(output_buffer, mode="w", format="wav")
        output_stream = output_container.add_stream("pcm_s16le", rate=16000)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        sample_chunks = []
        for frame in input_container.decode(input_stream):
            for resampled_frame in resampler.resample(frame):
                sample_chunks.append(resampled_frame.to_ndarray())
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)
        for packet in output_stream.encode(None):
            output_container.mux(packet)

        output_container.close()
        input_container.close()
    except Exception as error:
        raise TranscriptionError(f"Could not read the recorded audio: {error}") from error

    if sample_chunks:
        samples = np.concatenate(sample_chunks, axis=None).astype(np.float64)
        rms = float(np.sqrt(np.mean(samples ** 2)))
    else:
        rms = 0.0

    return output_buffer.getvalue(), rms >= SILENCE_RMS_THRESHOLD


def _parse_response(text):
    match = _RESPONSE_RE.search(text)

    if not match:
        raise TranscriptionError(f"Unexpected transcription response: {text!r}")

    language = match.group("language").strip()
    raw = match.group("raw").strip()
    question = match.group("question").strip()

    if not raw or language.upper() == "NO_SPEECH" or raw.upper() == "NO_SPEECH":
        return "", "", None

    return raw, question, language


async def transcribe_audio(file_path, language_hint=None):
    """Transcribe and clean up one recorded audio file.

    Returns (raw_transcript, clean_question, language) - clean_question
    is already normalized (code-mixed/business-domain aware, see
    TRANSCRIBE_PROMPT), ready to drop straight into the chat input, same
    as what a separate query_normalizer.py pass used to produce.
    raw_transcript is "" and language is None when no speech was
    detected - callers should treat that as "please try again", same as
    the old Whisper-empty-string case.

    language_hint: an explicit language code (e.g. "ta") naming the
    language being spoken, same manual-override purpose the old Whisper
    path's language_hint had - pass None for full auto-detect.
    """

    wav_bytes, has_audible_signal = _convert_to_wav(file_path)

    if DEBUG_VOICE_PIPELINE:
        logger.info(
            f"[voice-pipeline] RAW AUDIO: {len(wav_bytes)} bytes (converted), "
            f"has_audible_signal={has_audible_signal}, language_hint={language_hint or 'auto'}"
        )

    if not has_audible_signal:
        return "", "", None

    language_hint_instruction = ""
    if language_hint:
        language_name = LANGUAGE_HINT_NAMES.get(language_hint, language_hint)
        language_hint_instruction = f" The speaker is speaking {language_name}."

    prompt = TRANSCRIBE_PROMPT.format(language_hint_instruction=language_hint_instruction)

    try:
        response_text = await generate_from_audio(wav_bytes, "audio/wav", prompt)
    except GeminiError as error:
        raise TranscriptionError(str(error)) from error

    raw, question, language = _parse_response(response_text)

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] GEMINI RAW TRANSCRIPT: {raw!r}")
        logger.info(f"[voice-pipeline] GEMINI CLEAN QUESTION: {question!r}")
        logger.info(f"[voice-pipeline] DETECTED LANGUAGE: {language}")

    return raw, question, language
