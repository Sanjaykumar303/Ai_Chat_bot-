import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config import DEBUG_VOICE_PIPELINE, GEMINI_API_KEY
from services import chat_service, document_store, pdf_retrieval
from services.gemini_client import GeminiError
from services.intent_router import (
    classify_intent,
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
    intent = classify_intent(
        resolved_question,
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

    try:
        if intent == DATABASE_QUERY:
            result = await chat_service.answer_database_query(resolved_question, request.language)
        elif intent == PDF_QUERY:
            result = await chat_service.answer_pdf_query(resolved_question, request.document_id, request.language)
        elif intent == HYBRID_QUERY:
            result = await chat_service.answer_hybrid_query(resolved_question, request.document_id, request.language)
        else:  # GENERAL_KNOWLEDGE - the fallback for everything else
            result = await chat_service.answer_general_knowledge(resolved_question, request.language)

    except GeminiError as error:
        raise HTTPException(status_code=502, detail=str(error))

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] FINAL ANSWER: {result['answer'][:200]!r}")

    # Handed back so the frontend can pass it in as the next
    # previous_question - lets a chain of follow-ups ("today" ->
    # "yesterday" -> "and last week?") each build on the fully-expanded
    # form of the one before it, not just the raw "yesterday" text.
    result["resolved_question"] = resolved_question

    return result
