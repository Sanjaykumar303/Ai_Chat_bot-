# Covers _parse_response(), the part of the Gemini-based transcription
# pipeline that doesn't need a real audio file or a live API call - the
# text-parsing contract between TRANSCRIBE_PROMPT's exact three-line
# format and what routes/transcribe.py ends up returning to the client.

from services.transcription import _parse_response, SILENCE_RMS_THRESHOLD


def test_parses_a_well_formed_response():
    text = "LANGUAGE: en\nRAW: what to you prefer\nQUESTION: What is the profit?"
    raw, question, language = _parse_response(text)
    assert raw == "what to you prefer"
    assert question == "What is the profit?"
    assert language == "en"


def test_no_speech_response_returns_empty_triple():
    text = "LANGUAGE: NO_SPEECH\nRAW: NO_SPEECH\nQUESTION: NO_SPEECH"
    assert _parse_response(text) == ("", "", None)


def test_question_spanning_multiple_lines_is_still_captured():
    text = (
        "LANGUAGE: en\n"
        "RAW: what is the profit today and also\nwhat about yesterday\n"
        "QUESTION: What is the profit today, and what about yesterday?"
    )
    raw, question, language = _parse_response(text)
    assert "yesterday" in question
    assert language == "en"


def test_unparseable_response_raises():
    from services.transcription import TranscriptionError
    import pytest

    with pytest.raises(TranscriptionError):
        _parse_response("I couldn't understand the audio, sorry.")


def test_silence_threshold_is_well_below_typical_speech_levels():
    # Calibrated in services/transcription.py's own comment against real
    # clips (silence measured 0, real speech measured 1800-2700) - this
    # just guards against someone accidentally setting it to something
    # that would reject quiet-but-real speech.
    assert 0 < SILENCE_RMS_THRESHOLD < 1000
