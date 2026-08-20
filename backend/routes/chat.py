import json
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from config import DEBUG_VOICE_PIPELINE, GEMINI_API_KEY
from services import chat_memory, chat_service, document_store, export_intent, pdf_retrieval
from services.gemini_client import GeminiError
from services.intent_router import (
    resolve_intent,
    GENERAL_KNOWLEDGE,
    DATABASE_QUERY,
    PDF_QUERY,
    HYBRID_QUERY,
)
from services.db_query_service import get_routing_terms
from services.followup_context import rewrite_with_context
from services.rate_limiter import enforce_rate_limit

router = APIRouter()

logger = logging.getLogger("uvicorn")

# How relevant a question has to score (pdf_retrieval.top_score, 0-1)
# against the attached PDF's chunks before intent_router treats it as a
# PDF question even without the word "document"/"pdf" in it. 0.2 was
# picked empirically: an actually-relevant question against this
# project's test fixture scored ~0.20-0.44, while unrelated questions
# (including ones sharing only stopword-like filler) scored 0.0.
PDF_RELEVANCE_THRESHOLD = float(os.getenv("PDF_RELEVANCE_THRESHOLD", "0.15"))


class ChatRequest(BaseModel):
    question: str
    # Spoken language detected by /transcribe (e.g. "ta", "kn", "bn"), or
    # omitted/None for typed questions - unchanged, English-default
    # behavior for every existing (typed) caller.
    language: str | None = None
    # document_id from POST /documents/upload's response, or None/omitted
    # for every existing caller - the PDF feature is entirely opt-in per
    # request, so omitting it reproduces today's exact DATABASE_QUERY/
    # GENERAL_KNOWLEDGE-only behavior.
    document_id: str | None = None
    # The *resolved* text of this chat's last question (see this
    # endpoint's response - "resolved_question"), not necessarily what
    # the user literally typed last time. Lets a short follow-up like
    # "yesterday" be interpreted in context (see services/
    # followup_context.py) - omitted/None reproduces today's exact
    # behavior (every question judged standalone).
    previous_question: str | None = None
    # The exact text of this chat's last AI message (session.messages,
    # Chat.jsx) - only read for a DOCX export request ("convert the
    # summary into a document", see services/export_intent.py). Omitted/
    # None reproduces today's exact behavior for every other question;
    # a DOCX export request with nothing here is answered with a plain
    # "ask a question first" error rather than fabricating a document.
    previous_answer: str | None = None
    # The frontend's OWN chat session id (utils/chatStorage.js's
    # session.id, a crypto.randomUUID() string) - reused as-is to persist
    # this conversation's memory (see services/chat_memory.py). Omitted/
    # None reproduces today's exact behavior: no message is saved, no
    # prior context is loaded, and every prompt is byte-identical to
    # before this feature (voice, which never sends it, is unaffected).
    session_id: str | None = None
    # The user's manual Web Search toggle (frontend/src/components/
    # ChatBox.jsx) - False/omitted reproduces today's exact behavior for
    # every existing caller (voice, and every typed question before this
    # feature). Only ever read for the GENERAL_KNOWLEDGE intent (see
    # _stream_chat_response below) - DATABASE_QUERY/PDF_QUERY/HYBRID_QUERY
    # never receive it, so this flag cannot change SQL generation, SQL
    # Guard, or PDF/RAG behavior no matter its value.
    web_search: bool = False


async def _load_conversation_context(session_id, question):
    """Persist the user's raw message and return the bounded prior-
    conversation block for this session (summary + recent window - never
    the whole transcript). Chat memory is strictly additive: any failure
    here (DB unreachable, schema not creatable) degrades to no memory and
    an empty context string rather than failing the /chat request, so the
    core answer path keeps working exactly as it did before this feature
    even if the memory store is down."""

    if not session_id:
        return ""

    try:
        await run_in_threadpool(chat_memory.save_message, session_id, chat_memory.USER_ROLE, question)
        return await run_in_threadpool(chat_memory.build_context_block, session_id)
    except Exception as error:
        logger.warning(f"Chat memory unavailable, answering without it: {error}")
        return ""


async def _remember_answer(session_id, answer):
    """Persist the assistant's answer and fold any newly-aged-out older
    messages into the rolling summary (maybe_summarize is itself throttled
    - see its docstring). Same best-effort contract as
    _load_conversation_context: a memory failure is logged and swallowed,
    never surfaced as a request error, since the answer has already been
    produced and returned to the user regardless."""

    if not session_id:
        return

    try:
        await run_in_threadpool(chat_memory.save_message, session_id, chat_memory.ASSISTANT_ROLE, answer)
        await chat_memory.maybe_summarize(session_id)
    except Exception as error:
        logger.warning(f"Could not persist assistant message / update summary: {error}")


def _ndjson_line(payload):
    return json.dumps(payload) + "\n"


async def _stream_chat_response(request, question):
    """The async generator StreamingResponse drives for every non-export
    /chat request - everything that used to run after building a single
    JSON response body, restructured to yield newline-delimited JSON
    (NDJSON - one compact JSON object per line, NOT textbook SSE framing)
    as the answer is produced instead of waiting for the whole thing.
    Both ends of this wire protocol are this project's own code (backend
    here, frontend services/api.js/pages/Chat.jsx), not a generic SSE
    client library, so NDJSON was picked over SSE's data:/event: framing
    specifically to sidestep needing to escape embedded newlines inside a
    chunk of streamed prose - a JSON string already escapes its own
    newlines, so "one JSON object per line" is simple and unambiguous
    without any extra framing rules.

    Wire protocol, one JSON object per line:
      {"type": "chunk", "text": "..."}
        A piece of the answer, in order - the frontend appends each one
        to what it's already shown, exactly as it arrives.
      {"type": "done", "sources": [...], "resolved_question": "..."}
        Sent exactly once, after every chunk. resolved_question is used
        the same way the old single-response body's top-level field was
        used - handed back as the next request's previous_question, so a
        chain of follow-ups ("today" -> "yesterday" -> "and last week?")
        each build on the fully-expanded form of the one before it.
      {"type": "error", "detail": "..."}
        A GeminiError raised mid-stream (see chat_service.py's own
        generator docstrings for why this can happen after some chunks
        were already sent, not just before the first one). The HTTP
        response has already started with a 200 status by the time this
        can happen - unlike the old single-response version's
        HTTPException(502, ...), a streamed answer can never retroactively
        become an HTTP error status once headers are already sent, so
        this event is the only way the frontend finds out - it has to
        treat this as "the partial answer shown so far is all there is;
        something went wrong after that", not silently ignore it.
    """

    # Chat memory is wired in AFTER the export short-circuit (in the
    # caller, chat()) on purpose: an export request never reaches this
    # generator at all, so export flows stay completely untouched and no
    # export request is ever saved as a conversation turn. From here on,
    # if a session_id was sent, this saves the user's raw message and
    # loads the bounded prior-conversation context (summary + recent
    # window) to hand to whichever answer path runs below. No session_id
    # -> "" -> byte-identical prompts.
    conversation_context = await _load_conversation_context(request.session_id, question)

    # A bare follow-up ("yesterday", "and last week?") gets expanded
    # against the previous *resolved* question before anything else -
    # routing, PDF relevance scoring, and every answer path below all
    # see the resolved text, exactly as if the user had typed it out in
    # full. Returns `question` unchanged for every other case (no
    # previous_question, or question isn't just a bare time reference),
    # so this is a no-op for every existing caller.
    resolved_question = rewrite_with_context(question, request.previous_question)

    # A document_id only counts as "a document is attached" if it still
    # resolves to a live (non-expired) entry - a stale/unknown id quietly
    # falls back to DATABASE_QUERY/GENERAL_KNOWLEDGE routing instead of
    # forcing a PDF_QUERY/HYBRID_QUERY path that would just answer
    # NO_DOCUMENT_ANSWER anyway.
    active_document = document_store.get_document(request.document_id) if request.document_id else None
    has_document = active_document is not None

    # Real content-overlap check, not just a fixed phrase list - lets a
    # question that never says "document"/"pdf" (e.g. "What is the
    # expected payment amount?") still route to PDF_QUERY when it's
    # actually about the attached PDF's content. See intent_router's
    # pdf_relevant parameter and pdf_retrieval.top_score's docstring.
    pdf_relevance_score = 0.0
    if active_document is not None:
        pdf_relevance_score = pdf_retrieval.top_score(
            active_document["chunks"], active_document["vectorizer"], active_document["matrix"], resolved_question
        )

    # Live table/column names, so classify_intent() can recognize a
    # database question (e.g. "show students older than 20") without any
    # schema-specific keyword hardcoded here - falls back to keyword-only
    # routing (None) if the database isn't configured/reachable.
    database_terms = await get_routing_terms()
    # resolved_question is classified independently first; only when
    # that comes back GENERAL_KNOWLEDGE and resolved_question is itself
    # an ambiguous, topic-less follow-up ("What about it?") does this
    # fall back to classifying request.previous_question instead - see
    # intent_router.resolve_intent's own docstring for why that's a
    # separate, later step rather than something baked into the
    # classification of resolved_question itself.
    intent = resolve_intent(
        resolved_question,
        request.previous_question,
        database_terms,
        has_document=has_document,
        pdf_relevant=pdf_relevance_score >= PDF_RELEVANCE_THRESHOLD,
    )

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] RAW USER QUERY: {question!r} (language={request.language})")
        logger.info(f"[voice-pipeline] PREVIOUS QUESTION: {request.previous_question!r}")
        logger.info(f"[voice-pipeline] RESOLVED QUERY: {resolved_question!r}")
        logger.info(f"[voice-pipeline] DATABASE TERMS: {database_terms}")
        logger.info(f"[voice-pipeline] DOCUMENT_ID: {request.document_id!r} (has_document={has_document})")
        logger.info(f"[voice-pipeline] PDF RELEVANCE SCORE: {pdf_relevance_score:.4f}")
        logger.info(f"[voice-pipeline] INTENT: {intent}")

    if intent == DATABASE_QUERY:
        answer_stream = chat_service.answer_database_query(resolved_question, request.language, conversation_context)
    elif intent == PDF_QUERY:
        answer_stream = chat_service.answer_pdf_query(
            resolved_question, request.document_id, request.language, conversation_context
        )
    elif intent == HYBRID_QUERY:
        answer_stream = chat_service.answer_hybrid_query(
            resolved_question, request.document_id, request.language, conversation_context
        )
    else:  # GENERAL_KNOWLEDGE - the fallback for everything else
        answer_stream = chat_service.answer_general_knowledge(
            resolved_question, request.language, conversation_context, request.web_search
        )

    chunks = []
    sources = []

    try:
        async for event in answer_stream:
            if event["type"] == "chunk":
                chunks.append(event["text"])
                yield _ndjson_line({"type": "chunk", "text": event["text"]})
            else:  # "done"
                sources = event.get("sources", [])
    except GeminiError as error:
        # See this function's own docstring on why this can't become an
        # HTTP 502 the way the old single-response version's
        # HTTPException(502, ...) did - the response already started.
        yield _ndjson_line({"type": "error", "detail": str(error)})
        return

    full_answer = "".join(chunks)

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] FINAL ANSWER: {full_answer[:200]!r}")

    # Persist the assistant's answer (and, if the session just grew long
    # enough, roll older messages into its summary) - best-effort, after
    # the full answer is in hand. Skipped entirely when no session_id was
    # sent. See _remember_answer.
    await _remember_answer(request.session_id, full_answer)

    yield _ndjson_line({"type": "done", "sources": sources, "resolved_question": resolved_question})


@router.post("/chat", dependencies=[Depends(enforce_rate_limit)])
async def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Please type a question.")

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing. Add it to backend/.env"
        )

    # Export requests are checked FIRST, on the raw (un-rewritten) text,
    # and short-circuit entirely - a single, ordinary JSON response,
    # never streamed (see chat_service.answer_docx_export's own docstring
    # for why an export's own "answer" is never worth streaming). Every
    # one of the four main intents below (handled by _stream_chat_
    # response) is only ever reached when this comes back None, so none
    # of that logic changes for any question that isn't an export
    # request. See services/export_intent.py.
    export_format = export_intent.detect_export_format(question)

    if export_format == export_intent.DOCX:
        result = chat_service.answer_docx_export(request.previous_answer)
    elif export_format == export_intent.XLSX:
        result = await chat_service.answer_xlsx_export(question)
    else:
        result = None

    if result is not None:
        if DEBUG_VOICE_PIPELINE:
            logger.info(f"[voice-pipeline] EXPORT ({export_format}): {question!r} -> {result.get('export')}")
        result["resolved_question"] = question
        return result

    return StreamingResponse(_stream_chat_response(request, question), media_type="application/x-ndjson")
