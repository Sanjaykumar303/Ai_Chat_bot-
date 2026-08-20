"""
Real-time Voice Chat - bridges a browser WebSocket to a Gemini Live API
session, so the user can speak naturally and hear Gemini's streamed
audio response, with barge-in (interrupting Gemini mid-reply) working
the same way a real phone call does.

This is a genuinely separate pipeline from the existing voice INPUT
feature (routes/transcribe.py -> services/transcription.py): that one
records a clip, uploads it once fully recorded, gets back one block of
text, and feeds it through the normal /chat text pipeline. This one
keeps a live, bidirectional audio stream open with Gemini for as long as
Voice Chat is active - continuous mic audio in, continuous spoken audio
out, both directions overlapping in real time. Nothing about the
existing /transcribe or /chat pipelines is touched by this module.

DATABASE QUESTIONS ("what is the profit today?") are answered with real
data, not Gemini's own guess. _LIVE_CONFIG declares one function/tool,
query_business_database, and _SYSTEM_INSTRUCTION tells Gemini to call it
- never answer from memory - whenever the question asks for a real
figure or record. When Gemini calls it (a LiveServerMessage.tool_call,
handled by _handle_db_tool_call below), the question text is answered by
services/chat_service.answer_database_query_full() - the non-streaming
collect wrapper around answer_database_query() (an async generator, so
real-time text chat can stream it), used here unmodified for the exact
same SQL generation, sql_guard validation, read-only Postgres execution,
and pre-emptive zero/null + exact-day guards routes/chat.py's own
DATABASE_QUERY intent already uses. The text answer is sent back via
session.send_tool_response(), and Gemini speaks it naturally in its own
voice, the same as any other reply - Gemini itself never runs SQL, opens
a Postgres connection, or sees a row; it only ever receives this
function's already-verified text, the same as reading a report.

This round trip is genuinely slow - a real SQL-generation call, a
Postgres query, and a natural-language-answer call, all nested inside
the already-open Live session, measured live at 13-20s end to end. Since
silence for that long reads as the app having frozen, _SYSTEM_INSTRUCTION
also tells Gemini to speak a brief acknowledgment ("Let me check that
for you...") BEFORE calling the tool - confirmed live that Gemini's own
turn CAN stream that audio first (arriving in well under a second) and
only then emit the tool_call message, needing no protocol change here:
the existing model_turn audio relay in _handle_gemini_message already
carries it to the browser exactly like any other spoken reply. This is
NOT fully reliable, though (see below) - the acknowledgment is a
best-effort prompt, not a guarantee, and components/VoiceChat.jsx's own
turnActiveRef exists specifically so a turn that skips it (going
straight to the tool call) still shows "Thinking..." correctly instead
of relying on the acknowledgment's audio to drive that state.

A COMPOUND question asking for more than one figure at once (observed:
"Can you show me the profit and revenue? And what is the expense
cost?") is a genuinely harder case for this model to call the tool
reliably for - live-tested 5x, only 1/5 correctly called the tool once
with real data; the rest either fabricated plausible-but-fake numbers
outright, or literally spoke the function-call syntax out loud
("calling query_business_database(...)") instead of invoking it. Both
_DB_QUERY_FUNCTION's description and _SYSTEM_INSTRUCTION now explicitly
cover this shape (one combined call for every figure asked for, never
partial answers from memory, never narrating the call itself) - this is
prompt-level mitigation, not a guarantee, since the underlying model
behavior is measurably non-deterministic for this specific question
shape even with identical config.

Still deliberately NOT wired into the PDF/RAG pipeline
(services/pdf_retrieval.py) - Voice Chat has no attached-document
context to answer from, and _SYSTEM_INSTRUCTION says so plainly rather
than silently failing to answer a document question.

PROTOCOL between the browser and routes/voice.py's WebSocket endpoint:

  Browser -> backend:
    binary frames - raw 16-bit PCM mic audio, mono, 16 kHz (the exact
      format Gemini Live's input side requires - see INPUT_AUDIO_MIME_TYPE).
    text frames - a small JSON control message:
      {"type": "end"} ends the session from the browser side (closing
        the WebSocket outright does the same thing; this is just an
        explicit alternative).
      {"type": "activity_start"} tells Gemini the user has just started
        talking - required because _LIVE_CONFIG disables Gemini's own
        automatic_activity_detection (see that config's own long
        comment for why: its server-side noise/echo false-positive rate
        was too high even at its most conservative setting). Relayed as
        session.send_realtime_input(activity_start=...) - see
        _relay_browser_audio_to_gemini. The frontend sends this the
        moment its own client-side detector (utils/turnDetector.js)
        confirms real, sustained speech, whether that's the start of an
        ordinary turn (mic status "listening") or a genuine barge-in
        over Gemini's own reply (mic status "speaking") - Google's own
        Live API defaults to activity_start interrupting an in-progress
        reply (activity_handling: START_OF_ACTIVITY_INTERRUPTS), so
        barge-in still works exactly as before, just triggered by this
        app's own detector instead of Gemini's.
      {"type": "activity_end"} tells Gemini the user has stopped talking
        for now (relayed as session.send_realtime_input(activity_end=...) -
        NOT audio_stream_end, which Google's own docs say is unused once
        automatic_activity_detection is disabled). The frontend sends
        this after its own detector confirms a real pause following
        confirmed speech - confirmed live that LIVE_MODEL_NAME (see
        gemini_client.py) does not reliably start responding on its own
        from silence alone, so this explicit signal is what actually
        triggers a timely reply.

  Backend -> browser:
    binary frames - raw 16-bit PCM audio from Gemini (mono, at the
      sample rate given once in the "ready" event below - Gemini Live's
      documented, fixed output rate for this model family, see
      DEFAULT_OUTPUT_AUDIO_SAMPLE_RATE).
    text frames - JSON events:
      {"type": "ready", "sampleRate": 24000}
        The connection to Gemini is open; safe to start streaming mic
        audio and move the UI from "connecting" to "listening". Sent as
        soon as the connection itself opens, not once Gemini has said
        anything - this model sends nothing back at all, not even its
        own setup acknowledgement, until it has received something from
        the client first (confirmed live), so this event is what
        actually breaks that deadlock.
      {"type": "input_transcript", "text": "...", "final": bool}
        Live speech-to-text of what the USER said. final=false entries
        are interim (still being recognized) and each one replaces the
        last interim entry, exactly like a live captioning display;
        final=true is the settled transcript for that utterance.
      {"type": "output_transcript", "text": "..."}
        A chunk of Gemini's own spoken reply, as text - streamed
        incrementally as it talks, meant to be appended, not replaced.
      {"type": "interrupted"}
        The user started talking while Gemini's audio was still
        playing (barge-in) - the frontend must stop playback of
        whatever's queued immediately, matching how a real phone call
        cuts off, since Gemini itself has already abandoned that reply.
      {"type": "turn_complete"}
        Gemini finished its spoken turn.
      {"type": "tool_call_started", "name": "query_business_database"}
        Gemini has decided to call a tool (currently only ever the
        database one - see _DB_QUERY_FUNCTION_NAME) and the (13-20s live)
        round trip described above is about to start. Sent once, right
        before _handle_db_tool_call runs - purely a hint so the frontend
        can show something more specific than generic "Thinking..." while
        it waits (e.g. "Checking the database...").
      {"type": "user_turn_saved", "text": "..."}
        The user's just-settled utterance (one final input_transcription
        event - the same "whole utterance" granularity the browser's own
        transcript already displays) has been persisted to this chat
        session's shared memory (see _VoiceTurnRecorder below). Purely
        additive - sent alongside, never instead of, the existing
        "input_transcript" event above. The frontend uses this to append
        the exact same text to the chat session's own persisted message
        history, so a voice conversation shows up in that session's
        history exactly like a typed one does.
      {"type": "assistant_turn_saved", "text": "..."}
        Gemini's just-finished spoken reply for one turn (every
        output_transcription chunk since the last save, concatenated) has
        been persisted the same way - sent once per turn, right alongside
        whichever of turn_complete/interrupted actually ended it, never
        for a still-in-progress reply.
      {"type": "error", "message": "..."}
        Something went wrong (missing API key, Gemini connection
        failure, session capacity) - always followed by the backend
        closing the WebSocket.
"""

import asyncio
import json
import logging
import os
import threading

from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types
from starlette.concurrency import run_in_threadpool

from services import chat_memory, chat_service
from services.gemini_client import open_live_session, GeminiError

logger = logging.getLogger("uvicorn")

# Gemini Live's own documented input format for the Developer API - the
# SDK's own bundled example (google/genai/live.py) sends audio this same
# way. 16-bit signed little-endian PCM, mono, is implied by "audio/pcm"
# with no other parameters; the sample rate is the one thing that has to
# be stated explicitly.
INPUT_AUDIO_SAMPLE_RATE = 16000
INPUT_AUDIO_MIME_TYPE = f"audio/pcm;rate={INPUT_AUDIO_SAMPLE_RATE}"

# Gemini Live's documented, fixed output rate for this model family -
# unlike the input rate (which this app controls, see
# INPUT_AUDIO_SAMPLE_RATE above), this isn't something the caller
# requests or that varies response to response, so there's nothing to
# read off individual messages. Still sent to the browser explicitly in
# the "ready" event below rather than hardcoded on the frontend too, so
# a future Gemini model change only needs updating here, not in both
# places.
DEFAULT_OUTPUT_AUDIO_SAMPLE_RATE = 24000

# Bidirectional audio streaming holds a real connection to Gemini open
# for as long as Voice Chat is active, unlike a single request/response
# call - a runaway client (or several) leaving sessions open could rack
# up cost/exhaust quota in a way RATE_LIMIT_MAX_REQUESTS (requests per
# minute, see services/rate_limiter.py) doesn't actually bound. This is
# the equivalent guard sized for this endpoint's own shape: how many
# concurrent sessions this process will hold open at once, not how fast
# new ones can be started - a plain HTTPException-based per-request
# limiter like rate_limiter.py's doesn't fit a WebSocket's lifecycle
# anyway (there's no per-request response to attach a 429 to once the
# connection is already open), so this is a small, purpose-built
# equivalent rather than a forced reuse of that one.
VOICE_LIVE_MAX_CONCURRENT_SESSIONS = int(os.getenv("VOICE_LIVE_MAX_CONCURRENT_SESSIONS", "10"))

_session_count_lock = threading.Lock()
_active_session_count = 0

# The one seam between Voice Chat and the database - see the module
# docstring's DATABASE QUESTIONS paragraph. Gemini never runs SQL or
# touches Postgres itself; it only ever gets this function's
# already-safety-checked text back (see _handle_db_tool_call), the same
# as routes/chat.py's DATABASE_QUERY intent already does for text chat.
_DB_QUERY_FUNCTION_NAME = "query_business_database"

_DB_QUERY_FUNCTION = types.FunctionDeclaration(
    name=_DB_QUERY_FUNCTION_NAME,
    description=(
        "Answer a question about this business's own live data - profit, revenue, "
        "expenses, invoices, customers, students, or any other specific number or "
        "record the connected database might hold. Always call this instead of "
        "answering from memory whenever the question asks for a real figure or "
        "record; never guess a number yourself. If the person asks for several "
        "figures at once - even across more than one sentence, e.g. 'show me the "
        "profit and revenue, and what's the expense?' - that is still ONE data "
        "request: call this exactly once with a single question covering every "
        "figure asked for (e.g. 'profit, revenue, and expenses'), never once per "
        "figure and never answering some figures from memory while calling this "
        "for the rest."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "question": types.Schema(
                type=types.Type.STRING,
                description="The user's question about the database, restated clearly in English - combine every figure asked for into this one string if the person asked for more than one.",
            ),
        },
        required=["question"],
    ),
)

# Sets expectations honestly about documents (still out of scope - see
# the module docstring) rather than let Gemini attempt and fail at
# something it has no way to do in this mode; database questions are
# routed to _DB_QUERY_FUNCTION above instead of being declined.
_SYSTEM_INSTRUCTION = (
    "You are the voice mode of an AI Document Assistant used mainly for "
    "business/accounting questions (profit, revenue, expenses, invoices, "
    "customers) - when the audio is ambiguous, resolve it toward that "
    "domain's vocabulary rather than an unrelated same-sounding word "
    "(for example, hearing something that could be 'profit' or 'prophet' "
    "should resolve to 'profit'). "
    "The person you're talking to is speaking to you out loud and will "
    "hear your reply read aloud, so answer conversationally and "
    "concisely - short, natural sentences, not a long written-style "
    "answer. "
    "You are connected to this business's live database. Whenever the "
    "question asks about specific data - a number, a total, or a record, "
    "such as profit, revenue, expenses, or a named customer/student/"
    "invoice - call query_business_database with the question instead of "
    "answering from memory. This applies exactly the same way when several "
    "figures are asked for at once, even split across more than one "
    "sentence - for example 'Can you show me the profit and revenue? And "
    "what is the expense cost?' is ONE data request for three figures, not "
    "three separate questions and not a request to clarify anything: "
    "restate it as one combined question ('profit, revenue, and expense') "
    "and call query_business_database exactly ONCE, then report every "
    "figure it returns. Never call it more than once for a single "
    "question, never answer even one part of a multi-part data question "
    "from memory while calling it for the rest, and never invent a plausible-"
    "sounding number for any part of it - if the question asks for real "
    "data, every figure in your answer has to come from what the function "
    "actually returns. "
    "Call the function using the real function-calling mechanism only - "
    "never say the word 'calling' or speak anything that looks like "
    "function syntax (such as 'query_business_database(...)') out loud; "
    "the person should only ever hear your brief acknowledgment and then "
    "your natural spoken answer, never the mechanics of how you got it. "
    "Looking up real data can take several seconds, so ALWAYS say a brief, "
    "natural spoken acknowledgment out loud first - like 'Let me check "
    "that for you' or 'One moment, checking the records' - so the person "
    "hears something right away instead of silence, THEN call the "
    "function; vary the phrase rather than repeating the exact same one "
    "every time. Once it returns, speak its answer back naturally and "
    "accurately: never change a number it gives you, never add a figure "
    "it didn't give you, and if it says the data isn't available or "
    "couldn't be verified, say that honestly instead of guessing. "
    "You do not have access to any uploaded document in this voice mode. "
    "If asked about one, say plainly that voice mode can't see it yet and "
    "suggest switching to text chat for it - never guess or make up an "
    "answer about a document you cannot see."
)

_LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=[types.Modality.AUDIO],
    system_instruction=_SYSTEM_INSTRUCTION,
    tools=[types.Tool(function_declarations=[_DB_QUERY_FUNCTION])],
    # Empty configs, not omitted fields - AudioTranscriptionConfig() with
    # no options turns transcription on with Gemini's own defaults; this
    # is what populates the input_transcript/output_transcript events
    # described in the module docstring. Leaving these unset entirely
    # would silently drop live transcripts, not just default them.
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    # Real, observed bug (real enough to be reported live twice now, most
    # recently with a spoken reply cut off mid-sentence right after a
    # question that misheard "hi there" as Japanese - a sign of a genuinely
    # noisy/echoey mic environment): mid-reply, Gemini would abruptly
    # self-interrupt (an "interrupted" event, see the module docstring)
    # because its SERVER-SIDE voice-activity detector mistook faint
    # background noise or the AI's own voice bleeding back into the mic
    # (browser echoCancellation is not fully reliable for audio played via
    # raw Web Audio API buffer nodes rather than an <audio> element - see
    # components/VoiceChat.jsx's playAudioChunk) for the start of new user
    # speech. A first attempt at fixing this (start_of_speech_sensitivity=
    # START_SENSITIVITY_LOW, the most conservative setting the automatic
    # detector offers) was NOT enough to stop it recurring - there is no
    # lower setting to fall back to within that mechanism.
    #
    # Disabling automatic_activity_detection entirely and driving turn
    # boundaries MANUALLY instead (see _relay_browser_audio_to_gemini's
    # "activity_start"/"activity_end" handling below) replaces Gemini's
    # own imperfect server-side audio analysis with THIS app's own
    # client-side one (utils/turnDetector.js's MIN_SPEECH_DURATION_MS
    # debounce - added earlier the same day this was found, specifically
    # to reject exactly this kind of noise/echo blip for the ordinary
    # end-of-turn case; VoiceChat.jsx now runs that same, already-hardened
    # detector during "speaking" too, not just "listening", so the SAME
    # false-positive protection now also covers barge-in). Confirmed via
    # Google's own Live API docs (https://ai.google.dev/gemini-api/docs/live-api/capabilities)
    # that activity_handling defaults to START_OF_ACTIVITY_INTERRUPTS - so
    # genuine barge-in (a real, wanted feature - see the module docstring)
    # still works exactly as before; only WHO decides "activity started"
    # changes, from Gemini's own detector to this app's.
    #
    # Caveat, not swept under the rug: a mid-sentence stop has also been
    # reported by other Gemini Live users even with interruption fully
    # disabled (google-gemini/live-api-web-console#117) - so this is not
    # guaranteed to be the ONLY cause of every such report, possibly
    # including this one. It closes a real, previously-unverified gap in
    # THIS app regardless of whether it turns out to be the full story.
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
    ),
)


async def _relay_browser_audio_to_gemini(websocket, session, chat_session_id=None):
    """Forwards mic audio from the browser to Gemini for as long as the
    WebSocket stays open. Returns (does not raise) on a normal
    disconnect or an explicit {"type": "end"} control message - either
    one is this session ending on the browser's own terms, not a
    failure.
    """

    while True:
        message = await websocket.receive()

        if message["type"] == "websocket.disconnect":
            return

        audio_chunk = message.get("bytes")
        if audio_chunk:
            await session.send_realtime_input(
                audio=types.Blob(data=audio_chunk, mime_type=INPUT_AUDIO_MIME_TYPE)
            )
            continue

        text = message.get("text")
        if not text:
            continue

        try:
            control = json.loads(text)
        except ValueError:
            continue

        control_type = control.get("type")

        if control_type == "end":
            return

        # "activity_start"/"activity_end" replace the older single
        # "end_turn" signal now that _LIVE_CONFIG disables Gemini's own
        # automatic_activity_detection (see that config's own long
        # comment for why) - manual mode requires the client to mark
        # BOTH boundaries of user speech explicitly, not just the end.
        # Per Google's own docs (ai.google.dev/gemini-api/docs/live-api/
        # capabilities), audio_stream_end is specifically NOT used in
        # this mode - activity_end is what now marks a completed turn.
        # Sent by VoiceChat.jsx's turnDetector (utils/turnDetector.js):
        # onSpeechStart -> activity_start, onEndTurn -> activity_end -
        # the exact same already-hardened (debounced against a single
        # noise/echo blip - see that module's own comment) detector that
        # already ran during "listening", now ALSO run during "speaking"
        # so the same false-positive protection covers barge-in too, not
        # just ordinary turn-taking.
        if control_type == "activity_start":
            logger.info(f"[voice-live] chat session {chat_session_id!r}: activity_start sent")
            await session.send_realtime_input(activity_start=types.ActivityStart())
            continue

        if control_type == "activity_end":
            # Logged UNCONDITIONALLY (not gated behind DEBUG_VOICE_PIPELINE)
            # - real, reported bug this is for: a turn that hangs with NO
            # response at all, from a user whose own .env didn't happen to
            # have DEBUG_VOICE_PIPELINE set, so nothing was visible to
            # diagnose it with. This one line establishes the exact moment
            # Gemini was told the user's turn was over, so a hang after it
            # (see _relay_gemini_to_browser's own matching log below) is
            # unambiguous - not "did we ever even ask", but "we asked at
            # this time and nothing came back".
            logger.info(f"[voice-live] chat session {chat_session_id!r}: activity_end sent, awaiting Gemini's reply")
            await session.send_realtime_input(activity_end=types.ActivityEnd())


async def _handle_db_tool_call(session, tool_call):
    """Answers every query_business_database call in one tool_call
    message, then reports the result(s) back via
    session.send_tool_response() so Gemini can speak them naturally in
    its own voice, as part of its normal turn - not a separate audio
    path, no change to the model_turn/turn_complete relaying below.

    Reuses services/chat_service.answer_database_query_full() - the
    non-streaming collect-to-a-dict wrapper around answer_database_
    query() (now an async generator, so real-time text chat can stream
    it - see chat_service.py's own module docstring). This tool call has
    nothing to stream token-by-token to (the text goes to Gemini Live's
    own function-response mechanism, then Gemini speaks it in its own
    separate TTS generation), so it needs the complete answer as one
    value, same as before this feature existed. Never raises: a Gemini
    failure while answering still gets a (apologetic) text response
    back, so Gemini's turn isn't left hanging on a tool call that will
    never resolve.
    """

    responses = []

    for call in tool_call.function_calls or []:
        if call.name != _DB_QUERY_FUNCTION_NAME:
            # Unreachable in practice - _LIVE_CONFIG declares exactly one
            # function - but answered rather than dropped, so a future
            # second tool can't leave a call silently unanswered.
            responses.append(types.FunctionResponse(
                id=call.id, name=call.name, response={"error": "unknown function"},
            ))
            continue

        question = (call.args or {}).get("question") or ""

        # Unconditional (not gated behind DEBUG_VOICE_PIPELINE) - see
        # _relay_gemini_to_browser's own comment. This is specifically the
        # slow (13-20s live) path a "stuck on Thinking..." report needs
        # visibility into: whether the tool call started at all, and
        # whether it finished (the next log line) or is what's hanging.
        logger.info(f"[voice-live] DB tool call started: {question!r}")

        try:
            result = await chat_service.answer_database_query_full(question)
            answer = result["answer"]
            logger.info("[voice-live] DB tool call finished")
        except GeminiError as error:
            answer = "Sorry, I couldn't reach the database just now. Please try again."
            logger.warning(f"[voice-live] DB tool call failed: {error}")

        responses.append(types.FunctionResponse(
            id=call.id, name=call.name, response={"answer": answer},
        ))

    await session.send_tool_response(function_responses=responses)


class _VoiceTurnRecorder:
    """Persists each voice turn's FINAL text into this app's shared,
    persistent chat memory (services/chat_memory.py) - the SAME store and
    session_id text chat already writes to via routes/chat.py's
    /chat endpoint - and tells the browser once a turn's exact final text
    is settled, via two new, purely additive control messages
    ("user_turn_saved"/"assistant_turn_saved" - see the module docstring)
    so pages/Chat.jsx can append the identical text to that chat
    session's own persisted (localStorage) message history. Every
    existing event this class is threaded alongside (input_transcript/
    output_transcript/interrupted/turn_complete) is completely
    unchanged - this only adds two new events on top of them, and touches
    no other voice/DB/memory/text-chat logic.

    One user "turn" is one settled (content.input_transcription, i.e.
    final=True) event - the exact same "whole utterance" granularity
    VoiceChat.jsx's own on-screen transcript already uses (see its
    updateUserTranscript), so what gets persisted is exactly what the
    user saw settle on screen. Saved and reported immediately, with a
    simple consecutive-duplicate guard (_last_user_text) in case Gemini's
    ASR ever re-sends an identical final for the same utterance - the
    single most direct way this class avoids a duplicate message.

    One assistant "turn" is every output_transcription chunk streamed
    since the last flush, concatenated and flushed exactly once - on
    whichever of turn_complete/interrupted ends that turn (never both:
    the buffer is emptied the instant it's flushed, so a stray second
    call for the same turn is a safe no-op that saves nothing). This is
    what prevents saving a partial mid-reply fragment: nothing is ever
    persisted until the turn genuinely ends, and nothing is persisted at
    all if the connection just drops mid-turn (run_voice_session's own
    teardown path never calls flush) - an incomplete turn is silently
    dropped, never saved as a fragment.

    Persisting to chat_memory is best-effort (same contract as routes/
    chat.py's own _load_conversation_context/_remember_answer): a DB
    hiccup is logged and swallowed, never raised into the live session.
    The "*_turn_saved" browser event is still sent even if that DB write
    failed - the browser's own local session history is the primary
    record of what was said; chat_memory is supplementary long-term
    memory for future context, not the only copy.
    """

    def __init__(self, session_id):
        self.session_id = session_id
        self._assistant_chunks = []
        self._last_user_text = None

    async def _save(self, role, text):
        if not self.session_id:
            return
        try:
            await run_in_threadpool(chat_memory.save_message, self.session_id, role, text)
        except Exception as error:
            logger.warning(f"[voice-live] could not save {role} turn to chat memory: {error}")

    async def handle_user_final(self, websocket, text):
        text = text.strip()
        if not text or text == self._last_user_text:
            return
        self._last_user_text = text
        await self._save(chat_memory.USER_ROLE, text)
        await websocket.send_json({"type": "user_turn_saved", "text": text})

    def on_output_chunk(self, text):
        self._assistant_chunks.append(text)

    async def flush_assistant_turn(self, websocket):
        text = "".join(self._assistant_chunks).strip()
        self._assistant_chunks = []
        # A settled reply closes the dedupe window for whatever user
        # utterance preceded it - a later, textually-identical utterance
        # (the user genuinely repeating themselves in a new turn) must be
        # saved, not treated as a stale duplicate of the earlier one.
        self._last_user_text = None
        if not text:
            return
        await self._save(chat_memory.ASSISTANT_ROLE, text)
        await websocket.send_json({"type": "assistant_turn_saved", "text": text})


async def _handle_gemini_message(message, websocket, session, recorder=None):
    """Dispatches one message from Gemini Live: a tool call is answered
    via _handle_db_tool_call above; anything else is the existing
    audio/transcript/turn-taking content, relayed to the browser exactly
    as before this function existed - pulled out of
    _relay_gemini_to_browser's loop only so each branch can be exercised
    with a single constructed message in a test, not to change what
    either branch does.

    recorder (a _VoiceTurnRecorder, or None) is threaded through purely
    to persist settled turns - see that class's own docstring. Defaults
    to None (skips persistence entirely) so every existing caller/test
    that doesn't pass one reproduces today's exact relay behavior
    unchanged.
    """

    if message.tool_call is not None:
        # Purely additive, sent before the (13-20s live, see the module
        # docstring) tool call actually runs - lets the browser show
        # something more specific than generic "Thinking..." while it
        # waits (see components/VoiceChat.jsx's thinkingReason state).
        # `name` rather than a fixed string so a future second tool
        # doesn't need a new event type, just a frontend label for it.
        await websocket.send_json({"type": "tool_call_started", "name": _DB_QUERY_FUNCTION_NAME})
        await _handle_db_tool_call(session, message.tool_call)
        return

    content = message.server_content
    if content is None:
        return

    if content.interrupted:
        if recorder is not None:
            # Gemini has already abandoned this reply - whatever it said
            # before being cut off is still real, settled speech, so it's
            # flushed (saved) here too, not discarded.
            await recorder.flush_assistant_turn(websocket)
        await websocket.send_json({"type": "interrupted"})

    if content.interim_input_transcription and content.interim_input_transcription.text:
        await websocket.send_json({
            "type": "input_transcript",
            "text": content.interim_input_transcription.text,
            "final": False,
        })

    if content.input_transcription and content.input_transcription.text:
        await websocket.send_json({
            "type": "input_transcript",
            "text": content.input_transcription.text,
            "final": True,
        })
        if recorder is not None:
            await recorder.handle_user_final(websocket, content.input_transcription.text)

    if content.output_transcription and content.output_transcription.text:
        await websocket.send_json({
            "type": "output_transcript",
            "text": content.output_transcription.text,
        })
        if recorder is not None:
            recorder.on_output_chunk(content.output_transcription.text)

    if content.model_turn:
        for part in content.model_turn.parts or []:
            inline = part.inline_data
            if inline and inline.data:
                await websocket.send_bytes(inline.data)

    if content.turn_complete:
        if recorder is not None:
            await recorder.flush_assistant_turn(websocket)
        await websocket.send_json({"type": "turn_complete"})


def _describe_gemini_message(message):
    """One short, log-safe tag per message from Gemini Live - never the
    actual transcript/audio content (that's real user data, and audio
    bytes aren't printable anyway), just enough to see the SHAPE of what
    (if anything) is coming back. Used by _relay_gemini_to_browser's own
    unconditional per-message log - see that function's own comment for
    why this exists unconditionally rather than behind DEBUG_VOICE_PIPELINE."""

    if message.tool_call is not None:
        return "tool_call"

    content = message.server_content
    if content is None:
        return "empty/unrecognized message"

    tags = []
    if content.interrupted:
        tags.append("interrupted")
    if content.interim_input_transcription and content.interim_input_transcription.text:
        tags.append("interim_input_transcript")
    if content.input_transcription and content.input_transcription.text:
        tags.append("final_input_transcript")
    if content.output_transcription and content.output_transcription.text:
        tags.append("output_transcript_chunk")
    if content.model_turn:
        tags.append("audio_chunk")
    if content.turn_complete:
        tags.append("turn_complete")

    return "+".join(tags) if tags else "server_content (no recognized field set)"


async def _relay_gemini_to_browser(websocket, session, recorder=None, chat_session_id=None):
    """Forwards Gemini's streamed events/audio to the browser for as
    long as the session stays open. Returns (does not raise) only once
    Gemini's underlying connection itself actually ends (an exception
    propagating out of session.receive(), or the caller cancelling this
    task from the outside) - never on its own.

    The outer `while True` around `session.receive()` is load-bearing,
    not defensive styling: confirmed live, against the real API, that
    LIVE_MODEL_NAME's (see gemini_client.py) receive() generator ends on
    its own right after each turn_complete rather than staying open
    across multiple turns the way the previous native-audio model's did.
    A single `async for message in session.receive(): ...` (the original
    shape here) would make this task finish after the FIRST turn, which
    run_voice_session's asyncio.wait(FIRST_COMPLETED) then reads as "the
    session is over" and tears the whole WebSocket down - observed live
    as the connection dying right after one exchange, with no error,
    before a second turn's audio could even be sent. Re-entering
    receive() in a loop is what lets a real multi-turn conversation work
    at all with this model - including a multi-turn conversation that
    mixes ordinary replies with database tool calls.

    Does NOT wait for a setup_complete message before anything else
    happens - confirmed live, against the real API, that this model
    sends nothing at all, not even setup_complete, until the client has
    sent its own first input. The "ready" event is sent by
    run_voice_session the instant the Gemini connection itself opens
    (before this function's loop even starts), which is what lets the
    browser start streaming mic audio right away - the thing that
    actually unblocks Gemini into responding at all.

    Every message actually received is logged UNCONDITIONALLY (not
    gated behind DEBUG_VOICE_PIPELINE - see _describe_gemini_message's
    own comment for why): real, reported bug this is for - a turn that
    hangs on "Thinking..." with nothing ever coming back, reported by a
    user whose own .env didn't have DEBUG_VOICE_PIPELINE set, so there
    was no way to tell "Gemini went silent after end_turn" apart from
    "Gemini replied but something else ate the reply" without asking
    them to change config and reproduce it again. This makes that
    distinction visible in the default log output on the very first
    reproduction.
    """

    while True:
        async for message in session.receive():
            logger.info(f"[voice-live] chat session {chat_session_id!r}: received {_describe_gemini_message(message)}")
            await _handle_gemini_message(message, websocket, session, recorder)


async def run_voice_session(websocket, chat_session_id):
    """Owns one Voice Chat session end-to-end: opens the Gemini Live
    connection, relays audio/events in both directions until either side
    ends it, then closes cleanly either way. The caller (routes/voice.py)
    has already accept()-ed the WebSocket; this function is the entire
    lifetime of the session from there.

    chat_session_id is this app's own chat-session id (see
    utils/chatStorage.js on the frontend, and pages/Chat.jsx's
    handleOpenVoiceChat - reused as-is from whatever chat session was
    active when Voice Chat was opened, or a freshly created one if none
    was, never a new id minted here). It's used for two things: log
    correlation (as before), and now also as the session_id every settled
    turn is saved under in services/chat_memory.py - the SAME shared
    store and session text chat already writes to, via the
    _VoiceTurnRecorder created below. Real isolation between chat
    sessions still doesn't come from this id alone: it comes from Voice
    Chat being a foreground-only, one-at-a-time feature (like the
    existing mic-recording feature) with its own dedicated WebSocket
    connection and Gemini Live session per use, so there is structurally
    no shared state for one chat session's voice conversation to leak
    into another's.
    """

    global _active_session_count

    with _session_count_lock:
        if _active_session_count >= VOICE_LIVE_MAX_CONCURRENT_SESSIONS:
            at_capacity = True
        else:
            _active_session_count += 1
            at_capacity = False

    if at_capacity:
        await websocket.send_json({
            "type": "error",
            "message": "Voice chat is busy right now. Please try again in a moment.",
        })
        await websocket.close(code=1013)  # 1013 = Try Again Later
        return

    try:
        try:
            # open_live_session() raises GeminiError synchronously, right
            # here, if GEMINI_API_KEY is missing - before any connection
            # is attempted - so that's caught by the same except
            # GeminiError below as a genuine mid-session failure would be.
            async with open_live_session(_LIVE_CONFIG) as session:
                # Unconditional, not gated behind DEBUG_VOICE_PIPELINE -
                # see _relay_gemini_to_browser's own comment on why this
                # whole file's lifecycle logging was moved out from
                # behind that flag.
                logger.info(f"[voice-live] session opened for chat session {chat_session_id!r}")

                # Sent as soon as the Gemini connection itself is open,
                # not once Gemini has said anything back - see
                # _relay_gemini_to_browser's own docstring for why
                # waiting on a message from Gemini first would deadlock
                # (confirmed live: this model sends nothing at all,
                # including its own setup acknowledgement, until the
                # client has sent something first). This IS that first
                # something: it's what tells the browser to start
                # streaming mic audio.
                await websocket.send_json({"type": "ready", "sampleRate": DEFAULT_OUTPUT_AUDIO_SAMPLE_RATE})

                recorder = _VoiceTurnRecorder(chat_session_id)

                browser_to_gemini = asyncio.create_task(
                    _relay_browser_audio_to_gemini(websocket, session, chat_session_id)
                )
                gemini_to_browser = asyncio.create_task(
                    _relay_gemini_to_browser(websocket, session, recorder, chat_session_id)
                )

                # Whichever direction ends first (browser disconnected,
                # or Gemini's own stream ended) is this session over -
                # the other direction is cancelled rather than left
                # running against a session that's about to close, same
                # "first to finish wins, tear the rest down cleanly"
                # shape as any two-directional relay.
                done, pending = await asyncio.wait(
                    {browser_to_gemini, gemini_to_browser},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                for task in done:
                    error = task.exception()
                    if error is not None and not isinstance(error, WebSocketDisconnect):
                        raise error

        except GeminiError as error:
            # Previously logged nowhere at all, gated or not - a genuine
            # Gemini-side failure (missing key, connection refused, a
            # mid-session error) was visible to the user (the message
            # sent below) but left zero server-side trace to diagnose it
            # from afterward.
            logger.warning(f"[voice-live] Gemini error for chat session {chat_session_id!r}: {error}")
            try:
                await websocket.send_json({"type": "error", "message": str(error)})
            except Exception:
                pass
        except WebSocketDisconnect:
            pass
        except Exception as error:
            logger.warning(f"[voice-live] session error for chat session {chat_session_id!r}: {error}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": "Voice session failed. Please try again.",
                })
            except Exception:
                pass

    finally:
        with _session_count_lock:
            _active_session_count -= 1
        logger.info(f"[voice-live] session closed for chat session {chat_session_id!r}")
