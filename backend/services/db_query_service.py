"""
Text-to-SQL orchestration: turns a natural-language question about the
database into an answer. Same context -> Gemini -> answer shape
routes/chat.py already uses for the document RAG path, except the
"context" here is a live SQL query and its results instead of retrieved
document chunks.

db_client.py only knows how to talk to Postgres. sql_guard.py only knows
how to validate SQL. This module is the only place that calls Gemini to
write and then explain a query - everything else stays decoupled.
"""

import logging
import os
import re

from starlette.concurrency import run_in_threadpool

from services.gemini_client import generate, GeminiError
from services.db_client import (
    get_schema_description,
    get_table_allowlist,
    get_schema_terms,
    execute_readonly_query,
    DatabaseError,
    DB_QUERY_ROW_LIMIT,
)
from services.sql_guard import validate_and_limit, SqlValidationError

logger = logging.getLogger("uvicorn")

DEBUG_VOICE_PIPELINE = os.getenv("DEBUG_VOICE_PIPELINE", "false").lower() == "true"

FALLBACK_ANSWER = "I couldn't answer that from the database."

SQL_PROMPT = """You are a PostgreSQL expert. Given the database schema below, write ONE read-only SQL SELECT query that answers the question.

The question may ask for a business metric - such as revenue, cost, profit, or loss - that is NOT a literal column anywhere in the schema. That does not mean it can't be answered: derive it with arithmetic, JOINs, and CTEs across the relevant tables. Do not output NO_QUERY just because no single column has that exact name - only use NO_QUERY when the schema truly has no table/column related to the question at all. These are illustrative PATTERNS, not literal names to copy - map them onto whatever the real schema below actually calls things:
- "Revenue" is typically quantity x a unit price, often reduced by a discount percentage or amount.
- "Cost" is typically quantity x a unit cost price, usually from a related product-style table joined in.
- "Profit" or "loss" is typically revenue minus cost, further reduced by any expenses and any returns/refunds for the same period. A negative result is a loss - state it as a loss, not a negative profit.
- For "today", "yesterday", "this week", "this month", etc., use CURRENT_DATE and interval arithmetic (e.g. CURRENT_DATE, CURRENT_DATE - INTERVAL '1 day', DATE_TRUNC('week', CURRENT_DATE)) - never a hardcoded literal date.

Financial/accounting questions (revenue, profit, expenses, income, P&L) specifically:
- If the schema contains a general-ledger-style pair of tables - one holding transaction records where a JSONB column lists line items (each with an account/ledger name and an amount), and another table that classifies each named account into a group (e.g. a column distinguishing income/sales accounts from expense accounts) - prefer this pair as the authoritative source over any single-purpose table that only covers one payment channel or one expense category. The ledger pair reflects the closed books; a narrower table may miss entire income/expense sources, or double-count what the ledger already includes under a different label.
- To use such a table, unnest its JSONB line-item column with jsonb_array_elements(column), then JOIN the unnested entry's account/ledger name to the classification table to filter or group by which side of the books (income vs. expense) each entry belongs to.
- Do not sum two different tables that could represent the same underlying transactions (similar category names, overlapping date ranges, similar totals) without being sure they're additive rather than overlapping - this double-counts. When unsure, use the single most complete, authoritative source instead of combining tables.

Each table below may be annotated with the actual date range its data currently covers. Before filtering by a date range, check whether the table you're about to use actually covers the FULL period the question asks about:
- If a table that fully covers the requested period exists for the same concept, use it.
- If nothing fully covers the requested period, still answer with the best available data, but also SELECT the actual MIN/MAX of the date column you filtered on alongside the result, so the answer can disclose exactly what period was covered - never let a partial-period sum look like a complete one.

Rules:
- Output ONLY the SQL query. No markdown fences, no explanation, no semicolon.
- Only SELECT statements (JOINs, subqueries, CTEs, and aggregate functions are all fine - they're still a single read-only SELECT). Never write, update, delete, or modify anything.
- Only use the tables and columns shown in the schema below - never invent one.
- All monetary amounts in this database are stored in Indian Rupees (INR) already - never convert them to another currency.
- If the question truly cannot be answered from this schema, output exactly: NO_QUERY

SCHEMA:
{schema}

QUESTION: {question}

SQL:"""

# One corrective retry: the rejection reason is appended so Gemini can see
# exactly what was wrong (e.g. "table X not allowed") rather than repeating
# the same mistake blind.
SQL_RETRY_PROMPT = """Your previous SQL was rejected: {reason}

Write a corrected read-only SELECT query for the same question, following the same rules as before - remember that a business metric like revenue/cost/profit/loss not being a literal column doesn't mean it can't be computed; derive it with JOINs/CTEs/arithmetic across the schema below instead of giving up, prefer a general-ledger-style table pair for financial questions if one exists, and avoid double-counting overlapping tables.

SCHEMA:
{schema}

QUESTION: {question}

SQL:"""

ANSWER_PROMPT = """You are an AI Document Assistant answering a question using live data from a database.

Answer the question in plain, natural language based ONLY on the query results below. Do not mention SQL, tables, or column names unless the question specifically asks about the schema. If the results are empty, say the data wasn't found.

All monetary amounts in this database are in Indian Rupees. Format every amount with the ₹ symbol (e.g. ₹15,836,835.46) - never $ or USD, and never convert the amount to another currency.

If the query results include a date range narrower than what the question asked about (for example the question asked about a full month but the data only covers part of it), say so plainly in your answer instead of presenting the number as if it covers the full requested period.

QUESTION: {question}

QUERY RESULTS (as JSON):
{results}{language_instruction}

ANSWER:"""

_SQL_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_sql(raw_text):
    """Strip markdown code fences / stray labels Gemini sometimes adds
    despite being told not to - generate() returns freeform text, not
    structured output."""

    text = _SQL_FENCE_RE.sub("", raw_text).strip()
    text = text.rstrip(";").strip()
    return text


async def _generate_sql(question, schema_description):
    raw = await generate(SQL_PROMPT.format(schema=schema_description, question=question))
    return _extract_sql(raw)


async def get_routing_terms():
    """Schema vocabulary for intent_router.classify_intent(), or None if
    the database isn't configured/reachable - callers should fall back to
    keyword-only routing in that case, not fail the whole /chat request.
    Cheap on the common path: get_schema_terms() only does a real DB round
    trip once per DB_SCHEMA_CACHE_TTL, otherwise it's an in-memory lookup.
    """

    try:
        return await run_in_threadpool(get_schema_terms)
    except DatabaseError as error:
        logger.info(f"Database routing terms unavailable, falling back to keyword-only routing: {error}")
        return None


async def answer_database_question(question, language_instruction=""):
    """Answer a question about the connected database.

    Returns {"answer": str, "sources": [], "sql": str | None}. Never raises
    DatabaseError/SqlValidationError/GeminiError - failures degrade to a
    plain-language fallback answer, matching how _answer_question in
    chat.py already treats "nothing relevant found" as a normal outcome,
    not a hard error.
    """

    try:
        schema_description = await run_in_threadpool(get_schema_description)
        allowed_tables = await run_in_threadpool(get_table_allowlist)
    except DatabaseError as error:
        logger.warning(f"Database schema unavailable: {error}")
        return {"answer": FALLBACK_ANSWER, "sources": [], "sql": None}

    sql = await _generate_sql(question, schema_description)

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] GENERATED SQL: {sql!r}")

    if sql.upper() == "NO_QUERY":
        return {"answer": FALLBACK_ANSWER, "sources": [], "sql": None}

    try:
        safe_sql = validate_and_limit(sql, allowed_tables, DB_QUERY_ROW_LIMIT)
    except SqlValidationError as first_error:
        # One corrective retry, then give up cleanly rather than looping.
        retry_raw = await generate(
            SQL_RETRY_PROMPT.format(reason=first_error, schema=schema_description, question=question)
        )
        sql = _extract_sql(retry_raw)

        if sql.upper() == "NO_QUERY":
            return {"answer": FALLBACK_ANSWER, "sources": [], "sql": None}

        try:
            safe_sql = validate_and_limit(sql, allowed_tables, DB_QUERY_ROW_LIMIT)
        except SqlValidationError as second_error:
            logger.warning(f"Generated SQL rejected twice, giving up: {second_error}")
            return {"answer": FALLBACK_ANSWER, "sources": [], "sql": None}

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] VALIDATED SQL: {safe_sql!r}")

    try:
        rows = await run_in_threadpool(execute_readonly_query, safe_sql)
    except DatabaseError as error:
        logger.warning(f"Database query execution failed: {error}")
        return {"answer": FALLBACK_ANSWER, "sources": [], "sql": safe_sql}

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] QUERY RESULT: {len(rows)} row(s)")

    answer_prompt = ANSWER_PROMPT.format(
        question=question,
        results=rows,
        language_instruction=language_instruction,
    )

    try:
        answer = await generate(answer_prompt)
    except GeminiError as error:
        raise error  # matches chat.py's existing GeminiError -> 502 handling

    return {"answer": answer, "sources": [], "sql": safe_sql}
