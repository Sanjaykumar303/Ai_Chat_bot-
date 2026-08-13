"""
Local speech-to-text: turns a recorded audio clip into text using a small
Whisper model that runs entirely on the CPU, no GPU or external API.

The model is loaded once per process (see main.py's startup hook) since
loading it is the expensive part - transcribe_audio() only ever reuses the
already-loaded instance.

Whisper decodes one language per clip. It cannot "switch" languages mid
utterance, so genuinely code-mixed Tamil-English speech will still come
out imperfect here - that's a real limitation of the tiny model, not a
config bug. What this module does control, and what was actually hurting
accuracy, is condition_on_previous_text (see transcribe_audio below). The
bigger fix for code-mixed speech - turning a messy raw transcript into a
clean semantic query - lives in query_normalizer.py, one stage downstream.
"""

import logging
import os
import threading

logger = logging.getLogger("uvicorn")

# "tiny" is the smallest multilingual Whisper model (~75 MB on disk in
# int8), chosen to fit comfortably on an 8 GB machine with no GPU. int8
# quantization trades a little accuracy for much lower memory use and
# faster CPU inference than float32. Configurable so "base" can be tried
# later (meaningfully better multilingual/code-switch accuracy, still
# CPU-friendly - see the note in transcribe_audio) without a code change.
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

DEBUG_VOICE_PIPELINE = os.getenv("DEBUG_VOICE_PIPELINE", "false").lower() == "true"

# One model instance is shared by every request, and faster-whisper's CPU
# inference isn't guaranteed safe for concurrent calls on one instance, so
# calls are serialized - the same pattern rag_service.py uses to guard its
# shared FAISS index.
_lock = threading.Lock()
_model = None


class TranscriptionError(Exception):
    """Raised with a user-facing message already attached."""


def load_model():
    """Load the Whisper model into memory. Safe to call more than once.

    Imports faster_whisper here rather than at module level so a broken
    install (e.g. no working ctranslate2 build for the running Python
    version) doesn't crash the whole app at import time - main.py's
    startup hook already treats this function raising as non-fatal
    ("Failure here doesn't crash startup"), and every other caller is
    /transcribe, which should surface a clear per-request error instead
    of the whole backend refusing to start.
    """

    global _model

    if _model is not None:
        return _model

    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model '{MODEL_SIZE}' ({DEVICE}/{COMPUTE_TYPE})...")
            _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            logger.info("Whisper model loaded.")

    return _model


def _clean_transcript(text):
    """Mechanical cleanup only - collapses whitespace/newlines from segment
    joins. Semantic cleanup (fixing meaning, not just spacing) is a
    separate, deliberately heavier step - see query_normalizer.py."""

    return " ".join(text.split())


def transcribe_audio(file_path, language_hint=None):
    """Transcribe one audio file on disk. Returns (text, language, language_probability).

    language_hint: an explicit Whisper language code (e.g. "ta") to force,
    bypassing auto-detection - pass None for full auto-detect (the
    default for anyone who doesn't set it). Added because real-world
    testing showed auto-detection under-uses Tamil on short/code-mixed
    clips (small Whisper models are known to be English-biased under
    auto-detect) - a manual override is the pragmatic fix for someone who
    already knows which language they're about to speak, rather than
    trying to out-tune detection further.

    Configuration notes (audited per the project's own checklist):

    - language=None (auto-detect) by default: still the right default for
      anyone who leaves it on Auto. Whisper decodes a whole clip in ONE
      language regardless; forcing a language for every user would make
      genuinely English speech worse. language_hint above is the targeted
      fix, not a change to the default.
    - task="transcribe" (default): correct as-is. task="translate" would
      convert everything to English text, destroying Tamil/Kannada/Bengali
      content instead of preserving it.
    - beam_size=5 (default): left alone. Tiny's bottleneck is model
      capacity, not search width - a wider beam has little accuracy payoff
      here and costs real CPU time.
    - temperature (default fallback ladder [0, 0.2, ... 1.0]): left alone.
      This is faster-whisper's built-in anti-repetition mechanism (it
      retries a segment at a higher temperature if the deterministic pass
      looks like a loop) - already doing useful work, no reason to override.
    - vad_filter=True (already set): correct. Silero VAD trims silence so
      it isn't transcribed as (often hallucinated) text.
    - vad_parameters (default): left alone. Default min_silence_duration_ms
      is 2000ms, well above a natural pause between code-switched clauses,
      so it won't cut a real utterance in the middle.
    - condition_on_previous_text=False (CHANGED from the default True):
      this was the one real bug. With it True, each segment is decoded
      conditioned on the previous segment's text - so once the model
      mistranscribes an early code-switched phrase, later segments keep
      drifting further from what was actually said instead of resetting.
      Recordings here are short standalone questions, not a long lecture
      that benefits from cross-segment context, so turning this off has
      no downside and directly reduces this failure mode.
    - word_timestamps: left at default (False) - not used anywhere.

    A move from "tiny" to "base" is a real, meaningful step up specifically
    for multilingual/code-switched accuracy (base has ~2x tiny's
    parameters and noticeably better non-English WER in OpenAI's own
    benchmarks), while still being CPU-only and comfortably light on 8GB
    RAM. Set WHISPER_MODEL_SIZE=base in backend/.env to try it - no code
    change needed.
    """

    try:
        model = load_model()
    except Exception as error:
        raise TranscriptionError(f"Could not load the Whisper model: {error}") from error

    if DEBUG_VOICE_PIPELINE:
        logger.info(
            f"[voice-pipeline] RAW AUDIO: {os.path.getsize(file_path)} bytes, "
            f"model={MODEL_SIZE}, language_hint={language_hint or 'auto'}"
        )

    try:
        with _lock:
            segments, info = model.transcribe(
                file_path,
                language=language_hint,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = "".join(segment.text for segment in segments)
    except Exception as error:
        raise TranscriptionError(f"Could not transcribe audio: {error}") from error

    text = _clean_transcript(text)

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] WHISPER RAW TRANSCRIPT: {text!r}")
        logger.info(f"[voice-pipeline] DETECTED LANGUAGE: {info.language} (p={info.language_probability:.2f})")

    return text, info.language, info.language_probability
