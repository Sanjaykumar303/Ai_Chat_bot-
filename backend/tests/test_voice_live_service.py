# Real-time Voice Chat's actual bidirectional relay (audio in, audio +
# transcripts + interruption events out) needs a genuine Gemini Live
# WebSocket connection to exercise meaningfully - deeply mocking the SDK's
# typed message classes to fake that would test the mock, not this code
# (matches this project's own established practice: verify anything
# touching a real Gemini/DB call live, not by mocking it into a test).
#
# What IS deterministic and worth covering here: the concurrent-session
# capacity guard (services/voice_live_service.py's own protection against
# a runaway client holding many expensive streaming connections open at
# once - see its module comment), and the session counter's own
# increment/release bookkeeping around it - both pure logic, no network.
#
# _handle_gemini_message/_handle_db_tool_call (the database tool-call
# routing) are a different case, despite taking a real
# types.LiveServerMessage - that message is a plain, already-constructed
# data object (no network involved in building one), and the function
# under test only ever reads it and calls chat_service.answer_database_
# query() (monkeypatched below) - genuinely deterministic Python this
# project's own testing practice says IS worth a real test, unlike
# session.receive()'s own async generator behavior, which stays
# live-tested only.

import asyncio
import json

from google.genai import types

import config
from services import voice_live_service


class _FakeWebSocket:
    """Just enough of FastAPI's WebSocket surface for run_voice_session's
    error/capacity paths - never a real connection."""

    def __init__(self):
        self.sent_json = []
        self.closed_with = None

    async def send_json(self, payload):
        self.sent_json.append(payload)

    async def close(self, code=1000):
        self.closed_with = code


class _ImmediateFailContextManager:
    """Stands in for open_live_session()'s return value - raises the
    instant it's entered, so run_voice_session's session-count guard and
    surrounding error handling are exercised without ever reaching the
    network."""

    def __init__(self, error):
        self._error = error

    async def __aenter__(self):
        raise self._error

    async def __aexit__(self, *exc_info):
        return False


def _run(coroutine):
    return asyncio.run(coroutine)


def test_session_rejected_at_capacity(monkeypatch):
    monkeypatch.setattr(voice_live_service, "_active_session_count", voice_live_service.VOICE_LIVE_MAX_CONCURRENT_SESSIONS)

    ws = _FakeWebSocket()
    _run(voice_live_service.run_voice_session(ws, "session-1"))

    assert ws.sent_json == [{
        "type": "error",
        "message": "Voice chat is busy right now. Please try again in a moment.",
    }]
    assert ws.closed_with == 1013  # "Try Again Later"
    # A rejected connection never counted against capacity in the first
    # place - the count must be left exactly as it was.
    assert voice_live_service._active_session_count == voice_live_service.VOICE_LIVE_MAX_CONCURRENT_SESSIONS


def test_session_below_capacity_is_let_through_and_count_is_released(monkeypatch):
    monkeypatch.setattr(voice_live_service, "_active_session_count", 0)
    monkeypatch.setattr(
        voice_live_service, "open_live_session",
        lambda live_config: _ImmediateFailContextManager(RuntimeError("simulated connection failure")),
    )

    ws = _FakeWebSocket()
    _run(voice_live_service.run_voice_session(ws, "session-1"))

    # The guard let it through (no capacity-rejection message - a
    # generic failure was reported instead), and the counter was
    # correctly released afterward rather than left incremented forever.
    assert ws.sent_json == [{"type": "error", "message": "Voice session failed. Please try again."}]
    assert voice_live_service._active_session_count == 0


def test_websocket_disconnect_while_connecting_is_not_reported_as_an_error(monkeypatch):
    from fastapi import WebSocketDisconnect

    monkeypatch.setattr(voice_live_service, "_active_session_count", 0)
    monkeypatch.setattr(
        voice_live_service, "open_live_session",
        lambda live_config: _ImmediateFailContextManager(WebSocketDisconnect()),
    )

    ws = _FakeWebSocket()
    _run(voice_live_service.run_voice_session(ws, "session-1"))

    # An ordinary disconnect is not a failure - nothing should be sent
    # back over a socket that's already gone.
    assert ws.sent_json == []
    assert voice_live_service._active_session_count == 0


def test_missing_api_key_reports_a_clear_error(monkeypatch):
    # Exercises the REAL open_live_session() (services/gemini_client.py),
    # not a fake - _require_api_key() raises synchronously before any
    # connection is attempted, so this still never touches the network.
    monkeypatch.setattr(voice_live_service, "_active_session_count", 0)
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)

    ws = _FakeWebSocket()
    _run(voice_live_service.run_voice_session(ws, "session-1"))

    assert ws.sent_json == [{"type": "error", "message": "GEMINI_API_KEY is missing. Add it to backend/.env"}]
    assert voice_live_service._active_session_count == 0


class _FakeToolSession:
    """Stands in for the real Gemini Live AsyncSession for
    _handle_gemini_message/_handle_db_tool_call tests - just enough
    surface (send_tool_response) to capture what got sent back to
    Gemini, never a real connection."""

    def __init__(self):
        self.tool_responses = []  # list of function_responses lists, one per call

    async def send_tool_response(self, *, function_responses):
        self.tool_responses.append(list(function_responses))


def _db_tool_call_message(call_id, question):
    """One LiveServerMessage carrying a single query_business_database
    call - the exact shape Gemini Live sends when it decides (per
    _SYSTEM_INSTRUCTION) that a spoken question needs real data."""

    return types.LiveServerMessage(
        tool_call=types.LiveServerToolCall(function_calls=[
            types.FunctionCall(
                id=call_id,
                name=voice_live_service._DB_QUERY_FUNCTION_NAME,
                args={"question": question},
            ),
        ]),
    )


async def _fail_if_called(*args, **kwargs):
    raise AssertionError("chat_service.answer_database_query_full should not be called for a general question")


def test_general_question_relays_normally_and_never_touches_the_database(monkeypatch):
    # A general-knowledge reply has no tool_call at all - proves the new
    # dispatch branch in _handle_gemini_message doesn't disturb the
    # existing content relay, and that Gemini answering conversationally
    # (the normal case) never reaches the database pipeline.
    monkeypatch.setattr(voice_live_service.chat_service, "answer_database_query_full", _fail_if_called)

    ws = _FakeWebSocket()
    session = _FakeToolSession()
    message = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            output_transcription=types.Transcription(text="Paris is the capital of France."),
            turn_complete=True,
        ),
    )

    _run(voice_live_service._handle_gemini_message(message, ws, session))

    assert ws.sent_json == [
        {"type": "output_transcript", "text": "Paris is the capital of France."},
        {"type": "turn_complete"},
    ]
    assert session.tool_responses == []


def test_db_tool_call_answers_using_the_real_database_pipeline(monkeypatch):
    seen = {}

    async def fake_answer_database_query(question, language=None):
        seen["question"] = question
        # Same {"answer", "sources"} shape chat_service.answer_database_
        # query_full() really returns - see chat_service.py.
        return {"answer": "The total profit today is ₹42,500.", "sources": []}

    monkeypatch.setattr(voice_live_service.chat_service, "answer_database_query_full", fake_answer_database_query)

    session = _FakeToolSession()
    ws = _FakeWebSocket()
    message = _db_tool_call_message("call-1", "What is the profit today?")

    _run(voice_live_service._handle_gemini_message(message, ws, session))

    assert seen["question"] == "What is the profit today?"
    [response] = session.tool_responses[0]
    assert response.id == "call-1"
    assert response.name == voice_live_service._DB_QUERY_FUNCTION_NAME
    assert response.response == {"answer": "The total profit today is ₹42,500."}
    # Sent to the BROWSER (not part of the Gemini tool-response protocol
    # above) so the frontend can show something more specific than
    # generic "Thinking..." during the slow round trip - see
    # components/VoiceChat.jsx's thinkingReason state.
    assert {"type": "tool_call_started", "name": voice_live_service._DB_QUERY_FUNCTION_NAME} in ws.sent_json


def test_db_tool_call_no_data_case_reports_the_guarded_fallback_verbatim(monkeypatch):
    # The guard text this now exercises lives in db_query_service.py,
    # not chat_service.py (moved so it can pre-empt generation rather
    # than replace already-generated text - see that module's own
    # comment). chat_service.answer_database_query_full() is still what
    # the voice tool call actually calls, so it's still what's stubbed
    # here.
    from services.db_query_service import _INSUFFICIENT_DATA_ANSWER

    async def fake_answer_database_query(question, language=None):
        # The exact guarded text the DB pipeline substitutes when a
        # query's rows are all zero/null - the tool call must pass this
        # through unchanged, never paraphrase away the "this may not be
        # real zero" caveat.
        return {"answer": _INSUFFICIENT_DATA_ANSWER, "sources": []}

    monkeypatch.setattr(voice_live_service.chat_service, "answer_database_query_full", fake_answer_database_query)

    session = _FakeToolSession()
    message = _db_tool_call_message("call-2", "What did we earn last February?")

    _run(voice_live_service._handle_gemini_message(message, _FakeWebSocket(), session))

    [response] = session.tool_responses[0]
    assert response.response == {"answer": _INSUFFICIENT_DATA_ANSWER}


def test_multi_turn_db_conversation_answers_each_turn_independently(monkeypatch):
    questions_seen = []

    async def fake_answer_database_query(question, language=None):
        questions_seen.append(question)
        if question == "What is the profit today?":
            return {"answer": "Today's profit is ₹10,000.", "sources": []}
        return {"answer": "Yesterday's profit was ₹8,500.", "sources": []}

    monkeypatch.setattr(voice_live_service.chat_service, "answer_database_query_full", fake_answer_database_query)

    session = _FakeToolSession()
    ws = _FakeWebSocket()

    _run(voice_live_service._handle_gemini_message(
        _db_tool_call_message("call-a", "What is the profit today?"), ws, session,
    ))
    _run(voice_live_service._handle_gemini_message(
        _db_tool_call_message("call-b", "What about yesterday?"), ws, session,
    ))

    # Each turn got its own DB answer, correctly correlated by call id -
    # nothing from the first turn leaked into or was reused by the
    # second, and the session stayed open across both (no reconnect).
    assert questions_seen == ["What is the profit today?", "What about yesterday?"]
    assert len(session.tool_responses) == 2
    assert session.tool_responses[0][0].id == "call-a"
    assert session.tool_responses[0][0].response == {"answer": "Today's profit is ₹10,000."}
    assert session.tool_responses[1][0].id == "call-b"
    assert session.tool_responses[1][0].response == {"answer": "Yesterday's profit was ₹8,500."}


# ---------------------------------------------------------------------------
# _describe_gemini_message - the unconditional (not DEBUG_VOICE_PIPELINE-
# gated) per-message log tag added to diagnose a real, reported bug: a
# voice turn hanging on "Thinking..." with literally nothing in the
# backend terminal to show whether Gemini ever replied at all, because
# the user's own .env didn't happen to have DEBUG_VOICE_PIPELINE set.
# Getting the field names wrong here would make the new logging silently
# useless (always "empty/unrecognized message") rather than loudly
# broken, so this is worth locking down precisely rather than trusting
# it by inspection alone.


def test_describe_tool_call_message():
    message = _db_tool_call_message("call-1", "What is the profit today?")
    assert voice_live_service._describe_gemini_message(message) == "tool_call"


def test_describe_turn_complete():
    message = types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))
    assert voice_live_service._describe_gemini_message(message) == "turn_complete"


def test_describe_interrupted():
    message = types.LiveServerMessage(server_content=types.LiveServerContent(interrupted=True))
    assert voice_live_service._describe_gemini_message(message) == "interrupted"


def test_describe_transcript_and_audio_chunks():
    input_interim = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            interim_input_transcription=types.Transcription(text="what is the"),
        ),
    )
    assert voice_live_service._describe_gemini_message(input_interim) == "interim_input_transcript"

    input_final = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            input_transcription=types.Transcription(text="what is the profit today"),
        ),
    )
    assert voice_live_service._describe_gemini_message(input_final) == "final_input_transcript"

    output_chunk = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            output_transcription=types.Transcription(text="Let me check"),
        ),
    )
    assert voice_live_service._describe_gemini_message(output_chunk) == "output_transcript_chunk"

    audio_chunk = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(parts=[types.Part(inline_data=types.Blob(data=b"\x00\x01", mime_type="audio/pcm"))]),
        ),
    )
    assert voice_live_service._describe_gemini_message(audio_chunk) == "audio_chunk"


def test_describe_combines_multiple_fields_set_on_one_message():
    # A single message CAN carry more than one of these at once (e.g. the
    # final chunk of audio arriving alongside turn_complete) - the log tag
    # has to show all of them, not just the first one found.
    message = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            output_transcription=types.Transcription(text="...today."),
            model_turn=types.Content(parts=[types.Part(inline_data=types.Blob(data=b"\x00", mime_type="audio/pcm"))]),
            turn_complete=True,
        ),
    )
    assert voice_live_service._describe_gemini_message(message) == "output_transcript_chunk+audio_chunk+turn_complete"


def test_describe_empty_message_does_not_crash():
    # No tool_call, no server_content at all - a shape this code has to
    # tolerate even if the real API is not expected to send it.
    message = types.LiveServerMessage()
    assert voice_live_service._describe_gemini_message(message) == "empty/unrecognized message"


# ---------------------------------------------------------------------------
# _relay_browser_audio_to_gemini's control-message routing - specifically
# the "activity_start"/"activity_end" dispatch to session.send_realtime_
# input(), added when _LIVE_CONFIG's automatic_activity_detection was
# disabled in favor of manual, client-driven turn boundaries (see that
# config's own long comment for the real, reported bug this replaced).
# The function's OWN module-level comment says session.receive()'s async
# generator behavior stays live-tested only - this is different: it's
# plain routing logic (a message dict in, a specific keyword argument to
# a fake session out), the same category this file's own testing
# practice already covers for _handle_db_tool_call's dispatch.


class _FakeRealtimeSession:
    """Stands in for the real Gemini Live AsyncSession's
    send_realtime_input() - records every call's kwargs, never a real
    connection."""

    def __init__(self):
        self.calls = []

    async def send_realtime_input(self, **kwargs):
        self.calls.append(kwargs)


def _receiving_websocket(*messages):
    """A fake FastAPI WebSocket whose receive() yields each of `messages`
    in order, then a websocket.disconnect - exactly what
    _relay_browser_audio_to_gemini's own `while True: message = await
    websocket.receive()` loop needs to run through a fixed script and
    then return on its own."""

    queue = list(messages) + [{"type": "websocket.disconnect"}]

    class _FakeReceivingWebSocket:
        async def receive(self):
            return queue.pop(0)

    return _FakeReceivingWebSocket()


def _control_message(payload):
    return {"type": "websocket.receive", "text": json.dumps(payload)}


def test_activity_start_relays_to_gemini_as_activity_start():
    session = _FakeRealtimeSession()
    ws = _receiving_websocket(_control_message({"type": "activity_start"}))

    _run(voice_live_service._relay_browser_audio_to_gemini(ws, session, "session-1"))

    assert len(session.calls) == 1
    assert "activity_start" in session.calls[0]
    assert isinstance(session.calls[0]["activity_start"], types.ActivityStart)


def test_activity_end_relays_to_gemini_as_activity_end_not_audio_stream_end():
    # The specific regression this guards: audio_stream_end was the OLD
    # signal (end_turn), no longer valid once automatic_activity_detection
    # is disabled - Google's own docs say it's unused in that mode. A
    # revert that brings back "audio_stream_end=True" here would silently
    # stop turn-ending from working at all under the new config.
    session = _FakeRealtimeSession()
    ws = _receiving_websocket(_control_message({"type": "activity_end"}))

    _run(voice_live_service._relay_browser_audio_to_gemini(ws, session, "session-1"))

    assert len(session.calls) == 1
    assert "activity_end" in session.calls[0]
    assert isinstance(session.calls[0]["activity_end"], types.ActivityEnd)
    assert "audio_stream_end" not in session.calls[0]


def test_audio_bytes_still_relayed_continuously_regardless_of_activity_boundaries():
    # Streaming does NOT gate on activity_start/activity_end - the app
    # still forwards every captured chunk continuously (see
    # VoiceChat.jsx's own onaudioprocess), the same as before this
    # change; only how Gemini is told which parts of that stream count
    # as "real activity" changed.
    session = _FakeRealtimeSession()
    ws = _receiving_websocket(
        {"type": "websocket.receive", "bytes": b"\x00\x01"},
        _control_message({"type": "activity_start"}),
        {"type": "websocket.receive", "bytes": b"\x02\x03"},
        _control_message({"type": "activity_end"}),
    )

    _run(voice_live_service._relay_browser_audio_to_gemini(ws, session, "session-1"))

    kinds = [tuple(call.keys())[0] for call in session.calls]
    assert kinds == ["audio", "activity_start", "audio", "activity_end"]


def test_unknown_control_message_is_ignored_not_an_error():
    session = _FakeRealtimeSession()
    ws = _receiving_websocket(_control_message({"type": "something_future_and_unrecognized"}))

    _run(voice_live_service._relay_browser_audio_to_gemini(ws, session, "session-1"))

    assert session.calls == []


# ---------------------------------------------------------------------------
# _VoiceTurnRecorder - persists settled voice turns to chat_memory and
# tells the browser via the new "user_turn_saved"/"assistant_turn_saved"
# events. chat_memory.save_message itself is monkeypatched (a real DB
# round trip is covered by backend/tests/test_chat_memory.py and this
# project's own live-verification practice, not duplicated here) - what's
# deterministic and worth locking down here is the recorder's OWN control
# flow: exactly one save per settled turn, no save for a partial/
# in-progress one, and the consecutive-duplicate guard.
# ---------------------------------------------------------------------------


def _stub_save_message(monkeypatch, calls):
    def fake_save_message(session_id, role, content):
        calls.append((session_id, role, content))

    monkeypatch.setattr(voice_live_service.chat_memory, "save_message", fake_save_message)


def test_user_final_transcript_saves_and_reports_once(monkeypatch):
    calls = []
    _stub_save_message(monkeypatch, calls)
    recorder = voice_live_service._VoiceTurnRecorder("session-1")
    ws = _FakeWebSocket()

    _run(recorder.handle_user_final(ws, "What is the profit today?"))

    assert calls == [("session-1", "user", "What is the profit today?")]
    assert ws.sent_json == [{"type": "user_turn_saved", "text": "What is the profit today?"}]


def test_consecutive_identical_user_final_is_not_saved_twice(monkeypatch):
    # A real, observed-possible Gemini ASR quirk (re-sending an identical
    # final for the same utterance) must not create a duplicate saved
    # message - the exact "prevent duplicate messages" requirement.
    calls = []
    _stub_save_message(monkeypatch, calls)
    recorder = voice_live_service._VoiceTurnRecorder("session-1")
    ws = _FakeWebSocket()

    _run(recorder.handle_user_final(ws, "Hello"))
    _run(recorder.handle_user_final(ws, "Hello"))

    assert calls == [("session-1", "user", "Hello")]
    assert ws.sent_json == [{"type": "user_turn_saved", "text": "Hello"}]


def test_same_text_after_an_assistant_reply_is_saved_again(monkeypatch):
    # The dedupe guard is scoped to ONE turn, not forever - a user
    # genuinely repeating the same question in a later turn (after
    # Gemini has already replied once) must still be saved.
    calls = []
    _stub_save_message(monkeypatch, calls)
    recorder = voice_live_service._VoiceTurnRecorder("session-1")
    ws = _FakeWebSocket()

    _run(recorder.handle_user_final(ws, "Hello"))
    recorder.on_output_chunk("Hi there!")
    _run(recorder.flush_assistant_turn(ws))
    _run(recorder.handle_user_final(ws, "Hello"))

    assert calls == [
        ("session-1", "user", "Hello"),
        ("session-1", "assistant", "Hi there!"),
        ("session-1", "user", "Hello"),
    ]


def test_assistant_turn_flushes_accumulated_chunks_once(monkeypatch):
    calls = []
    _stub_save_message(monkeypatch, calls)
    recorder = voice_live_service._VoiceTurnRecorder("session-1")
    ws = _FakeWebSocket()

    recorder.on_output_chunk("The profit ")
    recorder.on_output_chunk("today is ₹5,000.")
    _run(recorder.flush_assistant_turn(ws))

    assert calls == [("session-1", "assistant", "The profit today is ₹5,000.")]
    assert ws.sent_json == [{"type": "assistant_turn_saved", "text": "The profit today is ₹5,000."}]


def test_flushing_an_empty_assistant_buffer_saves_nothing(monkeypatch):
    # No output_transcription chunks arrived at all (e.g. a turn that
    # ended with nothing spoken) - must not save/report an empty message.
    calls = []
    _stub_save_message(monkeypatch, calls)
    recorder = voice_live_service._VoiceTurnRecorder("session-1")
    ws = _FakeWebSocket()

    _run(recorder.flush_assistant_turn(ws))

    assert calls == []
    assert ws.sent_json == []


def test_flushing_twice_only_saves_once(monkeypatch):
    # The buffer is emptied the instant it's flushed - a stray second
    # flush call for the same turn (e.g. both an interrupted and a
    # turn_complete somehow firing) must be a safe no-op, never a
    # duplicate save.
    calls = []
    _stub_save_message(monkeypatch, calls)
    recorder = voice_live_service._VoiceTurnRecorder("session-1")
    ws = _FakeWebSocket()

    recorder.on_output_chunk("Done.")
    _run(recorder.flush_assistant_turn(ws))
    _run(recorder.flush_assistant_turn(ws))

    assert calls == [("session-1", "assistant", "Done.")]


def test_interrupted_reply_is_saved_via_the_same_flush(monkeypatch):
    # What Gemini actually said before being cut off is still real,
    # settled speech - it must be saved, not discarded just because the
    # user interrupted it.
    calls = []
    _stub_save_message(monkeypatch, calls)
    recorder = voice_live_service._VoiceTurnRecorder("session-1")
    ws = _FakeWebSocket()

    recorder.on_output_chunk("Let me check tha")
    _run(recorder.flush_assistant_turn(ws))

    assert calls == [("session-1", "assistant", "Let me check tha")]


def test_no_session_id_skips_the_database_write_but_still_reports_to_the_browser(monkeypatch):
    # A defensive fallback (in practice the frontend always sends a real
    # session_id now) - persistence to chat_memory is skipped, but the
    # browser still gets its event so the frontend's own local session
    # history (which the browser fully owns) is never silently starved.
    calls = []
    _stub_save_message(monkeypatch, calls)
    recorder = voice_live_service._VoiceTurnRecorder(None)
    ws = _FakeWebSocket()

    _run(recorder.handle_user_final(ws, "Hello"))

    assert calls == []
    assert ws.sent_json == [{"type": "user_turn_saved", "text": "Hello"}]


def test_handle_gemini_message_threads_recorder_for_a_full_general_turn(monkeypatch):
    # End-to-end through _handle_gemini_message (not just the recorder in
    # isolation): a general-knowledge reply's output_transcription chunks
    # accumulate and flush on turn_complete, alongside the existing,
    # completely unchanged output_transcript/turn_complete events.
    calls = []
    _stub_save_message(monkeypatch, calls)
    recorder = voice_live_service._VoiceTurnRecorder("session-1")
    ws = _FakeWebSocket()
    session = _FakeToolSession()

    message = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            input_transcription=types.Transcription(text="What is the capital of France?"),
            output_transcription=types.Transcription(text="Paris is the capital of France."),
            turn_complete=True,
        ),
    )

    _run(voice_live_service._handle_gemini_message(message, ws, session, recorder))

    assert calls == [
        ("session-1", "user", "What is the capital of France?"),
        ("session-1", "assistant", "Paris is the capital of France."),
    ]
    # Existing events are completely unchanged, just interleaved with the
    # two new ones.
    assert ws.sent_json == [
        {"type": "input_transcript", "text": "What is the capital of France?", "final": True},
        {"type": "user_turn_saved", "text": "What is the capital of France?"},
        {"type": "output_transcript", "text": "Paris is the capital of France."},
        {"type": "assistant_turn_saved", "text": "Paris is the capital of France."},
        {"type": "turn_complete"},
    ]


def test_handle_gemini_message_without_a_recorder_behaves_exactly_as_before(monkeypatch):
    # recorder defaults to None - every existing caller that doesn't pass
    # one (see the other tests in this file, all still passing unmodified)
    # must see byte-identical behavior to before this feature existed.
    ws = _FakeWebSocket()
    session = _FakeToolSession()
    message = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            input_transcription=types.Transcription(text="Hello"),
            output_transcription=types.Transcription(text="Hi!"),
            turn_complete=True,
        ),
    )

    _run(voice_live_service._handle_gemini_message(message, ws, session))

    assert ws.sent_json == [
        {"type": "input_transcript", "text": "Hello", "final": True},
        {"type": "output_transcript", "text": "Hi!"},
        {"type": "turn_complete"},
    ]
