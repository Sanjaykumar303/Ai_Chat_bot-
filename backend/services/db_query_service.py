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
import time

from starlette.concurrency import run_in_threadpool

from config import DEBUG_VOICE_PIPELINE
from services.gemini_client import generate, generate_stream, GeminiError
from services.db_client import (
    get_schema_description,
    get_table_allowlist,
    get_schema_terms,
    execute_readonly_query,
    DatabaseError,
    DB_QUERY_ROW_LIMIT,
)
from services.intent_router import (
    classify_database_question,
    DATABASE_QUESTION_CAPABILITY,
    DATABASE_QUESTION_SCHEMA,
    DATABASE_QUESTION_WHY,
)
from services.sql_guard import validate_and_limit, SqlValidationError

logger = logging.getLogger("uvicorn")

FALLBACK_ANSWER = "I couldn't answer that from the database."

# Attached to the "done" event's sources list only where a real answer was
# actually derived from live database content (a query that returned real
# rows, or the introspected schema itself) - see components/ChatBox.jsx's
# SourceBadge, which already renders a "Database" pill for this shape but,
# until now, never actually received one (every path here always sent
# sources=[]). Deliberately NOT attached to the capability answer (a fixed
# yes/no about connectivity, not database content) or any fallback/guard
# path (no_query, rejected SQL, a failed query, zero rows, the all-zero/
# unverified-single-day guards) - those aren't a real answer to attribute
# to a source, and badging them would overstate how much was actually
# found.
_DATABASE_SOURCE = [{"type": "database"}]

# ---------------------------------------------------------------------
# Deterministic anti-fabrication guards, checked on the RAW rows/SQL
# BEFORE any natural-language prose is generated from them.
#
# Moved here from chat_service.py (which used to generate the full
# answer first, THEN check these same conditions and REPLACE the answer
# text if triggered) specifically to make streaming the final answer
# safe: once tokens are being streamed to the browser as Gemini produces
# them, there is no way to "un-send" ones the user already saw on
# screen, so a guard that only decides AFTER generation is too late to
# do its job. Checking here, before answer_database_question() ever
# calls generate() for the prose, means a triggered guard skips
# generation entirely - the user never sees anything to un-see, and (a
# free side benefit) no Gemini call is wasted producing text that would
# just be thrown away.
#
# A useful side effect of moving these here rather than duplicating
# them: services/chat_service.py's answer_hybrid_query() also calls this
# module's answer_database_question() directly for its own DB leg, so
# its db_context now gets the SAME single-day protection the plain
# DATABASE_QUERY path already had - it didn't before this move, since
# only answer_database_query()'s own post-hoc check used to apply it.
# ---------------------------------------------------------------------

_INSUFFICIENT_DATA_ANSWER = (
    "I don't have enough recorded data to answer that reliably. The query for this returned only "
    "zero/empty values, which most likely means the relevant data hasn't been recorded for that "
    "period yet - not that the true figure is actually zero."
)

_UNVERIFIED_SINGLE_DAY_ANSWER = (
    "I can't reliably answer that for that specific day. The generated query didn't actually filter "
    "to that exact date, so its result may reflect a running total or a different period instead of "
    "just that day's figure - I'd rather say so than present a number that might not really answer "
    "what was asked."
)

# "today"/"yesterday" ask about one specific calendar day - a SQL query
# claiming to answer that has to filter on it with an EXACT match
# (date_column = CURRENT_DATE, or = CURRENT_DATE - INTERVAL '1 day'; see
# SQL_PROMPT's matching instruction below). Deliberately just these two
# words, not every relative-date phrase ("this week"/"last month"
# legitimately use a range comparison instead of equality, which this
# check isn't meant to second-guess).
_SINGLE_DAY_QUESTION_RE = re.compile(r"\b(today|yesterday)\b", re.IGNORECASE)
# The negative lookbehind is load-bearing, not decorative: a bare "="
# search matches the "=" *inside* "<=" and ">=" too, since both contain
# a literal equals sign - found via this project's own test suite
# treating "voucher_date <= CURRENT_DATE" as a real exact-day filter,
# exactly the running-total-passed-off-as-a-single-day case this guard
# exists to catch (see _sql_has_exact_day_filter's docstring).
_EXACT_DATE_FILTER_RE = re.compile(r"(?<![<>!])=\s*CURRENT_DATE\b", re.IGNORECASE)


def _is_single_day_question(text):
    return bool(_SINGLE_DAY_QUESTION_RE.search(text))


def _sql_has_exact_day_filter(sql):
    return bool(_EXACT_DATE_FILTER_RE.search(sql))


def _all_values_zero_or_null(rows):
    """True if every column of every row is 0/0.0/Decimal(0)/None - the
    exact shape a SUM()/COALESCE(...,0) aggregate produces when its WHERE
    clause matched nothing. Used to catch a specific, observed failure
    mode: a database answer confidently stating an entity "owes nothing"/
    "has a $0 balance" purely because the generated SQL's filter (e.g.
    matching a marketing batch code against a voucher_number column)
    happened to match zero rows. A prompt-level caveat alone wasn't
    reliable against how persuasive an already-generated sentence reads,
    so this pre-empts generation outright instead - see
    answer_database_question and chat_service.answer_hybrid_query (which
    also uses this directly on its own db_rows for the same reason).
    """

    if not rows:
        return False

    for row in rows:
        for value in row.values():
            if value not in (0, None):
                return False

    return True

SQL_PROMPT = """You are a PostgreSQL expert. Given the database schema below, write ONE read-only SQL SELECT query that answers the question.

The schema and its actual data determine what domain this is - do not assume any particular subject area (e.g. finance, healthcare, retail, logistics). Base every judgment call below purely on what the schema below actually contains, never on an assumed domain.

Some questions ask for a metric's plain VALUE ("what is the total X?"); others ask WHY a metric is at its current level, WHAT CAUSED a change, or HOW something happened or changed - an EXPLANATION, not just a value ("why is X low?", "how did Y change?", "what's behind the increase in Z?"). Recognize which one this is from the actual MEANING of the question, not fixed trigger words - "why"/"how"/"caused" are common but not the only ways to ask for an explanation, and a question can ask for a cause without using any of those exact words. For an explanation-type question, a single overall total almost never actually explains anything, so write SQL that retrieves what's needed to explain it instead:
- A COMPARISON: the current period's value for the metric alongside a prior period's (of the same length), so the change itself is visible in the results.
- A BREAKDOWN: the metric grouped by its components (whatever categorical/grouping column the schema actually has - a category, type, region, product, customer, account, or similar), so which part(s) actually moved is visible, not just the overall total.
- Both together, if the question calls for it.
For a plain value question, a single direct query for that value is correct and preferred - do not manufacture an unnecessary comparison or breakdown when the question only asked for the number itself.

The question may ask for a metric that is NOT a literal column anywhere in the schema - a value that has to be derived from other columns (arithmetic, an aggregate, a ratio, a JOIN across related tables) rather than read directly. That does not mean it can't be answered: derive it from whatever the schema actually models. Do not output NO_QUERY just because no single column has that exact name - only use NO_QUERY when the schema truly has no table/column related to the question at all.
- For "today", "yesterday", "this week", "this month", etc., use CURRENT_DATE and interval arithmetic (e.g. CURRENT_DATE, CURRENT_DATE - INTERVAL '1 day', DATE_TRUNC('week', CURRENT_DATE)) - never a hardcoded literal date.
- "today" and "yesterday" mean exactly that one day - filter with an EXACT match (date_column = CURRENT_DATE, or date_column = CURRENT_DATE - INTERVAL '1 day'), never date_column <= CURRENT_DATE or any other open-ended/cumulative comparison. A "<=" filter computes an all-time running total, not that single day's figure, and would silently misrepresent one as the other.
- If a column's schema annotation below shows its actual stored values (e.g. `status VARCHAR(20) [values: 'active', 'completed', 'cancelled']`), and the question asks about one of those states, filter on that column with the value copied EXACTLY as shown - same spelling and case, never a guessed variant like 'Active' or 'ACTIVE'. Do not re-derive the same state from a different column instead (for example, computing a status from a date comparison) when an authoritative status column already records it directly - the stored status can genuinely differ from what other columns alone would suggest, so a derived guess can silently overcount or undercount real rows. Only fall back to deriving the state yourself when no column in the schema actually stores it.
- Do not sum two different tables that could represent the same underlying records (similar column names, overlapping date ranges, similar totals) without being sure they're additive rather than overlapping - this double-counts. When unsure, use the single most complete, authoritative source instead of combining tables.
- The same "most complete, authoritative source" judgment applies to CHOOSING between two tables, not just to summing them. When more than one table could plausibly answer a general metric question - for example, one table is a comprehensive ledger covering every category/source of the metric via a categorization column, while a second table is a narrower operational table covering only ONE of those categories/sources (sometimes linked to the first by a foreign key, e.g. a column that names or references the other table) - prefer the comprehensive table for a question about the metric in general. Only use the narrower table when the question is specifically scoped to that one category/channel, not to the metric as a whole. This choice must not depend on how the time period happens to be phrased: an equivalent relative period ("last month") and absolute period ("July 2026") that describe the exact same dates must resolve to the SAME table and produce the SAME figure - never a different, smaller or larger number purely because of which phrasing was used to ask.

Each table below may be annotated with the actual date range its data currently covers. Before filtering by a date range, check whether the table you're about to use actually covers the FULL period the question asks about:
- If a table that fully covers the requested period exists for the same concept, use it.
- If nothing fully covers the requested period, still answer with the best available data, but also SELECT the actual MIN/MAX of the date column you filtered on alongside the result, so the answer can disclose exactly what period was covered - never let a partial-period sum look like a complete one.

Rules:
- Output ONLY the SQL query. No markdown fences, no explanation, no semicolon.
- Only SELECT statements (JOINs, subqueries, CTEs, and aggregate functions are all fine - they're still a single read-only SELECT). Never write, update, delete, or modify anything.
- Only use the tables and columns shown in the schema below - never invent one.
- If the question truly cannot be answered from this schema, output exactly: NO_QUERY

SCHEMA:
{schema}

QUESTION: {question}

SQL:"""

# One corrective retry: the rejection reason is appended so Gemini can see
# exactly what was wrong (e.g. "table X not allowed") rather than repeating
# the same mistake blind.
SQL_RETRY_PROMPT = """Your previous SQL was rejected: {reason}

Write a corrected read-only SELECT query for the same question, following the same rules as before - remember that a metric not being a literal column doesn't mean it can't be computed; derive it with JOINs/CTEs/arithmetic across the schema below instead of giving up, and avoid double-counting overlapping tables. If the question is asking WHY/HOW something happened or changed rather than for a plain value, the corrected query must still retrieve a COMPARISON against a prior period and/or a BREAKDOWN by component, never just one overall total.

SCHEMA:
{schema}

QUESTION: {question}

SQL:"""

# Used instead of SQL_PROMPT/ANSWER_PROMPT when classify_database_question
# says this question is about the database's STRUCTURE (what tables/
# columns/data exist), not about specific records - "what are tables in
# my db?", "summary the db", "what data do we have?". The schema
# description already has the full, accurate answer (it's the same text
# SQL_PROMPT above is given), so this skips SQL generation entirely -
# faster, and correct even though information_schema-style questions
# aren't answerable through sql_guard's table allowlist anyway.
SCHEMA_SUMMARY_PROMPT = """You are an AI Document Assistant describing what is available in a connected database.

Answer the question in plain, natural language using ONLY the SCHEMA below. Mention the table names and, if relevant to the question, what each table holds - do not invent a table or column that isn't listed, and do not write or suggest any SQL.

{conversation_context}QUESTION: {question}

SCHEMA:
{schema}{language_instruction}

ANSWER:"""

# Used instead of the whole SQL/answer pipeline when
# classify_database_question says this question is about the assistant's
# own ABILITY to reach the database ("Can you connect to our DB?"), not
# about any content in it. Fixed text, not a Gemini call: by the time
# this is used, get_schema_description() above has already either
# succeeded (so the database is, factually, reachable right now) or
# raised DatabaseError (handled before this point ever runs) - there's
# nothing left for a model to reason about, and a fixed answer can't
# misstate a yes/no fact a free-form one might.
DATABASE_CAPABILITY_ANSWER = (
    "Yes - I'm connected to the live database right now, and I can answer questions using its data "
    "(for example \"how many records are in a table?\" or \"why did a particular number change recently?\")."
)

ANSWER_PROMPT = """You are an AI Document Assistant answering a question using live data from a database.

Answer the question in plain, natural language based ONLY on the query results below. Do not mention SQL, tables, or column names unless the question specifically asks about the schema. If the results are empty, say the data wasn't found.

If the question is asking WHY a metric is at its current level, WHAT CAUSED a change, or HOW something happened or changed - an explanation, not just a value - recognize this from the actual meaning of the question, not fixed trigger words, and explain the cause using ONLY what the results show: if they include a comparison against a prior period or a breakdown by category, identify what actually moved and by how much, citing the real numbers from the results - never outside knowledge, industry assumptions, or a plausible-sounding guess about what "usually" causes this kind of change. If the results do NOT actually contain enough information to explain the cause (for example only a single total with nothing to compare it against, or a breakdown that doesn't reveal a clear driver), say so plainly instead of guessing: state what the data does show, and that it isn't enough to pinpoint the specific cause.

Format numbers clearly (e.g. thousands separators for large values). If the data or schema indicates a specific currency, unit, or measurement - a symbol or code present in the results, or a column name/annotation that implies one - reflect that in your answer and never convert it to a different currency or unit; otherwise present the numbers plainly without inventing a currency or unit the data doesn't actually indicate.

If the query results include a date range narrower than what the question asked about (for example the question asked about a full month but the data only covers part of it), say so plainly in your answer instead of presenting the number as if it covers the full requested period.

{conversation_context}QUESTION: {question}

QUERY RESULTS (as JSON):
{results}{language_instruction}

ANSWER:"""

# Zero ROWS (not the zero-VALUE case _INSUFFICIENT_DATA_ANSWER above
# covers) - a case a plain single-aggregate value question rarely hits
# (SUM/COUNT always return exactly one row even over no matching data),
# but a non-aggregate lookup or a comparison/breakdown query (see
# SQL_PROMPT's WHY/HOW guidance above) genuinely can return nothing at
# all to answer from.
_NO_ROWS_ANSWER = (
    "I don't have enough recorded data to answer that - the query I ran didn't return any matching "
    "records, so I don't have anything real to base an answer on rather than guessing."
)

_SQL_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_sql(raw_text):
    """Strip markdown code fences / stray labels Gemini sometimes adds
    despite being told not to - generate() returns freeform text, not
    structured output."""

    text = _SQL_FENCE_RE.sub("", raw_text).strip()
    text = text.rstrip(";").strip()
    return text


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


# ---------------------------------------------------------------------
# Short-TTL cache for a repeated identical DATA question - see
# answer_database_question's own docstring for what this does and does
# NOT cache (only (sql, rows), never the final natural-language answer).
# ---------------------------------------------------------------------

# Deliberately short compared to db_client.DB_SCHEMA_CACHE_TTL (600s,
# structural schema that barely changes) - this caches actual row DATA,
# which can change at any moment, so the bound on staleness has to be
# much tighter. 45s is long enough to absorb the common case this exists
# for (a user re-asking the same question, or two different users asking
# the same thing close together) while being short enough that a stale
# answer is a non-issue for a business-reporting chat assistant - the
# same "some staleness is an acceptable trade-off for real DB/Gemini
# round trips saved" reasoning this project already accepts for the
# schema cache, just tuned much tighter here.
DB_QUERY_CACHE_TTL = int(os.getenv("DB_QUERY_CACHE_TTL", "45"))

# Each entry is tiny (SQL text + a handful of result rows), but without
# SOME bound a long-lived process fielding many distinct questions over
# days/weeks would grow this dict forever - entries are only invalidated
# lazily (checked at lookup time), never proactively swept. The oldest
# entry is evicted once full, same simple shape as this being a small,
# short-lived cache rather than anything needing document_store.py's own
# periodic sweep (that store holds much heavier per-entry state - full
# text, a TF-IDF matrix - and accumulates from user uploads, not from a
# bounded set of English questions about a fixed schema).
_DB_QUERY_CACHE_MAX_ENTRIES = 200

_query_cache = {}  # {normalized_question: {"sql", "rows", "cached_at"}}


def _normalize_question_for_cache(question):
    return " ".join(question.lower().split())


def _get_cached_query(question):
    """Return (sql, rows) for a not-yet-expired cache hit, or None."""

    entry = _query_cache.get(_normalize_question_for_cache(question))
    if entry is None:
        return None
    if time.time() - entry["cached_at"] > DB_QUERY_CACHE_TTL:
        return None
    return entry["sql"], entry["rows"]


def _cache_query(question, sql, rows):
    """Store a SUCCESSFUL (sql, rows) pair. Deliberately never called for
    NO_QUERY or a validation failure - those are the model's own
    uncertainty about this specific attempt, not a stable fact about the
    question worth remembering; a retry (by this user or the next one
    asking the same thing) should always get a fresh shot at it."""

    key = _normalize_question_for_cache(question)
    if key not in _query_cache and len(_query_cache) >= _DB_QUERY_CACHE_MAX_ENTRIES:
        oldest_key = min(_query_cache, key=lambda existing: _query_cache[existing]["cached_at"])
        del _query_cache[oldest_key]
    _query_cache[key] = {"sql": sql, "rows": rows, "cached_at": time.time()}


def _ms_since(start):
    return round((time.monotonic() - start) * 1000)


async def _resolve_sql_and_rows(question, schema_description, allowed_tables, timings):
    """A cache lookup, then (on a miss) generate -> validate through SQL
    Guard (sql_guard.validate_and_limit) -> one corrective retry on
    rejection -> execute via db_client.execute_readonly_query -> cache on
    success. The one pipeline every SQL-backed database question goes
    through, regardless of whether it's asking for a plain value or an
    explanation (see SQL_PROMPT's own WHY/HOW guidance - Gemini decides
    that from the question's actual wording, not a separate code path
    here) - extracted out of answer_database_question mainly so this
    function's own body stays readable, not because there's more than
    one caller of it.

    timings is mutated in place with cache/sql_gen_ms/sql_retry_ms/
    db_query_ms, matching answer_database_question's own docstring - this
    helper never calls log_timing()/yields anything itself, so every
    existing log line's exact kind string/event shape stays under the
    caller's own control, unchanged.

    Returns ("ok", safe_sql, rows) on success, or ("fallback", kind, sql)
    when the caller should fall back to FALLBACK_ANSWER - kind is one of
    "no_query"/"no_query_after_retry"/"sql_rejected_twice"/
    "db_query_failed", and sql is the safe_sql actually attempted (only
    non-None for db_query_failed, matching the exact {"sql": ...} shape
    every existing done event for each of these outcomes already yields).
    """

    cached = _get_cached_query(question)

    if cached is not None:
        timings["cache"] = "hit"
        safe_sql, rows = cached
        return "ok", safe_sql, rows

    timings["cache"] = "miss"

    sql_gen_start = time.monotonic()
    raw = await generate(SQL_PROMPT.format(schema=schema_description, question=question))
    sql = _extract_sql(raw)
    timings["sql_gen_ms"] = _ms_since(sql_gen_start)

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] GENERATED SQL: {sql!r}")

    if sql.upper() == "NO_QUERY":
        return "fallback", "no_query", None

    try:
        safe_sql = validate_and_limit(sql, allowed_tables, DB_QUERY_ROW_LIMIT)
    except SqlValidationError as first_error:
        # One corrective retry, then give up cleanly rather than looping.
        sql_retry_start = time.monotonic()
        retry_raw = await generate(
            SQL_RETRY_PROMPT.format(reason=first_error, schema=schema_description, question=question)
        )
        sql = _extract_sql(retry_raw)
        timings["sql_retry_ms"] = _ms_since(sql_retry_start)

        if sql.upper() == "NO_QUERY":
            return "fallback", "no_query_after_retry", None

        try:
            safe_sql = validate_and_limit(sql, allowed_tables, DB_QUERY_ROW_LIMIT)
        except SqlValidationError as second_error:
            logger.warning(f"Generated SQL rejected twice, giving up: {second_error}")
            return "fallback", "sql_rejected_twice", None

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] VALIDATED SQL: {safe_sql!r}")

    db_query_start = time.monotonic()
    try:
        rows = await run_in_threadpool(execute_readonly_query, safe_sql)
    except DatabaseError as error:
        logger.warning(f"Database query execution failed: {error}")
        timings["db_query_ms"] = _ms_since(db_query_start)
        return "fallback", "db_query_failed", safe_sql
    timings["db_query_ms"] = _ms_since(db_query_start)

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] QUERY RESULT: {len(rows)} row(s)")

    _cache_query(question, safe_sql, rows)
    return "ok", safe_sql, rows


async def answer_database_question(question, language_instruction="", conversation_context=""):
    """Answer a question about the connected database.

    An ASYNC GENERATOR: yields {"type": "chunk", "text": str} events as
    the natural-language answer is produced (see gemini_client.
    generate_stream), followed by exactly one final {"type": "done",
    "sources": [], "sql": str | None, "rows": list | None} event. A
    fixed-text answer (capability question, FALLBACK_ANSWER, a triggered
    guard) that was never actually streamed from Gemini still yields
    ONE chunk event carrying the whole string before its done event, so
    every caller can consume this the same way regardless of whether the
    text was truly streamed - see chat_service.py's own answer_* functions
    for the same shape, and db_query_service... callers that want the
    complete answer as a single value rather than incrementally (there is
    no separate "non-streaming" version of this function) collect the
    chunks themselves - see e.g. chat_service.answer_hybrid_query's own
    small collect loop.

    "rows" (on the done event) is the exact result set the SQL path
    already fetched to write its own answer from (None for the two
    non-SQL question kinds, or on any failure before a query ran) -
    callers that need to double-check those same rows should use this
    instead of re-running safe_sql themselves, which would just be a
    second identical round trip to the same database for the exact same
    rows.

    Never raises DatabaseError/SqlValidationError - those degrade to a
    plain-language fallback answer, matching how _answer_question in
    chat.py already treats "nothing relevant found" as a normal outcome,
    not a hard error. CAN raise GeminiError, at any point during
    iteration (not just before the first chunk) - see generate_stream's
    own docstring for why a streaming caller has to handle that
    differently from a single-shot one.

    For any SQL-backed question (DATA or WHY - not the capability/schema
    kinds), a repeated question within DB_QUERY_CACHE_TTL seconds skips
    SQL generation and the DB round trip entirely, reusing the (sql,
    rows) from the first ask (see _resolve_sql_and_rows/
    _get_cached_query/_cache_query) - keyed purely on the question text,
    shared across every caller/session, since the underlying business
    database is the same for everyone. The natural-language ANSWER is
    always freshly generated regardless of a cache hit - conversation_
    context and language_instruction are per-request and never part of
    the cache key, so a cache hit can never serve stale personalized
    prose or the wrong language to a different session asking the
    identical thing.

    The deterministic anti-fabrication guards (all-zero/null rows; an
    unverified "today"/"yesterday" question; WHY-only: zero rows at all)
    are checked BEFORE any prose is generated - on a trigger, generation
    is skipped entirely and the guard's fixed text is yielded as the one
    chunk instead. This is what makes streaming safe here: once tokens
    are being streamed to a browser as Gemini produces them, there's no
    way to un-send ones the user already saw, so a guard that only
    decided AFTER generation (this function's own pre-refactor shape)
    would be too late.

    Logs one "[db-timing]" line per call, unconditionally (not gated
    behind DEBUG_VOICE_PIPELINE like the rest of this module's debug
    logging) - the whole point is to see where time actually goes on a
    real, possibly-slow production request without needing to first
    flip a debug flag and reproduce it. Each stage that actually ran gets
    its own *_ms field (schema/sql_gen/sql_retry/db_query/answer_gen, as
    applicable - a question that short-circuits, e.g. NO_QUERY or a
    capability question, simply has fewer fields), plus total_ms and
    which branch (`kind`) was taken. Covers every return/raise point, not
    just the success path - a slow failure is exactly the kind of thing
    this exists to catch.
    """

    request_start = time.monotonic()
    timings = {}

    def log_timing(kind):
        timings["total_ms"] = _ms_since(request_start)
        detail = " ".join(f"{key}={value}" for key, value in timings.items())
        logger.info(f"[db-timing] kind={kind} {detail}")

    schema_start = time.monotonic()
    try:
        schema_description = await run_in_threadpool(get_schema_description)
        allowed_tables = await run_in_threadpool(get_table_allowlist)
    except DatabaseError as error:
        logger.warning(f"Database schema unavailable: {error}")
        timings["schema_ms"] = _ms_since(schema_start)
        log_timing("schema_unavailable")
        yield {"type": "chunk", "text": FALLBACK_ANSWER}
        yield {"type": "done", "sources": [], "sql": None, "rows": None}
        return
    timings["schema_ms"] = _ms_since(schema_start)

    # Which of four genuinely different things this question is asking
    # for - see intent_router.classify_database_question's docstring.
    # CAPABILITY/SCHEMA return before any SQL is generated. WHY and DATA
    # are NOT separate code paths below (they used to be - see
    # [[project_db_why_analysis_intent_2026-08-20]] for why that changed)
    # - question_kind is kept only as a `[db-timing] kind=...` log LABEL
    # from here on, never a gate on which prompt Gemini receives.
    # SQL_PROMPT/ANSWER_PROMPT above already teach Gemini to recognize,
    # from the question's own actual wording, whether it's asking for a
    # plain value or an explanation (why/how/what caused something) and
    # shape the SQL/answer accordingly - a real, live model judgment
    # call, not a fixed keyword match. This matters because
    # _WHY_ANALYSIS_RE is necessarily incomplete (it can't enumerate
    # every possible phrasing of "explain this to me") - a question it
    # mislabels DATA (e.g. "How did we end up with less revenue this
    # quarter?") still gets full explanation-shaped treatment below,
    # because Gemini itself decided that from reading the question,
    # independent of the label.
    question_kind = classify_database_question(question)

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[voice-pipeline] DATABASE QUESTION KIND: {question_kind}")

    if question_kind == DATABASE_QUESTION_CAPABILITY:
        # schema_description above already proved the database is
        # reachable right now, or this line would never be reached.
        log_timing(question_kind)
        yield {"type": "chunk", "text": DATABASE_CAPABILITY_ANSWER}
        yield {"type": "done", "sources": [], "sql": None, "rows": None}
        return

    if question_kind == DATABASE_QUESTION_SCHEMA:
        prompt = SCHEMA_SUMMARY_PROMPT.format(
            question=question,
            schema=schema_description,
            conversation_context=conversation_context,
            language_instruction=language_instruction,
        )
        answer_gen_start = time.monotonic()
        try:
            async for chunk in generate_stream(prompt):
                yield {"type": "chunk", "text": chunk}
        except GeminiError as error:
            timings["answer_gen_ms"] = _ms_since(answer_gen_start)
            log_timing(f"{question_kind}_failed")
            raise error  # matches chat.py's existing GeminiError -> 502 handling
        timings["answer_gen_ms"] = _ms_since(answer_gen_start)
        log_timing(question_kind)
        yield {"type": "done", "sources": _DATABASE_SOURCE, "sql": None, "rows": None}
        return

    # question_kind is DATABASE_QUESTION_WHY or DATABASE_QUESTION_DATA
    # from here on - both take the exact same SQL-backed path; _log_kind
    # just keeps WHY-labeled log lines distinguishable (kind=why_no_query
    # vs kind=no_query) without branching any actual behavior on it.
    def _log_kind(suffix):
        return f"{question_kind}_{suffix}" if question_kind == DATABASE_QUESTION_WHY else suffix

    outcome = await _resolve_sql_and_rows(question, schema_description, allowed_tables, timings)

    if outcome[0] == "fallback":
        _, log_kind, sql = outcome
        log_timing(_log_kind(log_kind))
        yield {"type": "chunk", "text": FALLBACK_ANSWER}
        yield {"type": "done", "sources": [], "sql": sql, "rows": None}
        return

    _, safe_sql, rows = outcome

    # Zero rows at all - rare for a plain single-aggregate value question
    # (SUM/COUNT always return exactly one row, even over no matching
    # data - that shape is what _all_values_zero_or_null below catches
    # instead), but real for a non-aggregate lookup or an explanation-
    # type comparison/breakdown query (see SQL_PROMPT's WHY/HOW guidance)
    # that found nothing at all to compare or break down.
    if not rows:
        log_timing(_log_kind("no_rows"))
        yield {"type": "chunk", "text": _NO_ROWS_ANSWER}
        yield {"type": "done", "sources": [], "sql": safe_sql, "rows": rows}
        return

    # Pre-emptive anti-fabrication guards - see this function's own
    # docstring for why these run HERE, before generation, rather than
    # checking the generated text afterward. Applies equally to a cache
    # hit or a fresh query (rows came from either path above) - a cached
    # all-zero result must still be caught every time it's served, not
    # just the first time.
    if _all_values_zero_or_null(rows):
        log_timing(_log_kind("insufficient_data"))
        yield {"type": "chunk", "text": _INSUFFICIENT_DATA_ANSWER}
        yield {"type": "done", "sources": [], "sql": safe_sql, "rows": rows}
        return

    if _is_single_day_question(question) and not _sql_has_exact_day_filter(safe_sql):
        log_timing(_log_kind("unverified_single_day"))
        yield {"type": "chunk", "text": _UNVERIFIED_SINGLE_DAY_ANSWER}
        yield {"type": "done", "sources": [], "sql": safe_sql, "rows": rows}
        return

    # conversation_context/language_instruction are per-request and
    # never part of the cache key (see _get_cached_query/_cache_query's
    # own docstrings) - the answer below is ALWAYS freshly generated,
    # cache hit or not, so a cached (sql, rows) pair never serves a
    # stale/wrong-session/wrong-language piece of PROSE, only saves the
    # SQL-generation Gemini call and the DB round trip that produced it.
    answer_prompt = ANSWER_PROMPT.format(
        question=question,
        results=rows,
        conversation_context=conversation_context,
        language_instruction=language_instruction,
    )

    answer_gen_start = time.monotonic()
    try:
        async for chunk in generate_stream(answer_prompt):
            yield {"type": "chunk", "text": chunk}
    except GeminiError as error:
        timings["answer_gen_ms"] = _ms_since(answer_gen_start)
        log_timing(_log_kind("answer_gen_failed"))
        raise error  # matches chat.py's existing GeminiError -> 502 handling
    timings["answer_gen_ms"] = _ms_since(answer_gen_start)

    log_timing(question_kind)
    yield {"type": "done", "sources": _DATABASE_SOURCE, "sql": safe_sql, "rows": rows}


async def answer_database_question_full(question, language_instruction="", conversation_context=""):
    """Non-streaming convenience wrapper around answer_database_question()
    - collects every chunk into one string before returning the same
    {"answer", "sources", "sql", "rows"} dict shape this module returned
    before it became a generator. For callers that need the complete
    answer as a single value rather than incrementally: chat_service.
    answer_hybrid_query() (its own downstream HYBRID_ANSWER_PROMPT call
    is what actually streams to the browser, so the DB leg's own text is
    just material fed into that prompt, never shown on its own) and,
    transitively via chat_service.answer_database_query_full(),
    services/voice_live_service.py's DB tool call (hands the text to
    Gemini Live's own function-response mechanism, not a browser SSE
    stream, so there's nothing to stream token-by-token there either).
    """

    chunks = []
    done = {"sources": [], "sql": None, "rows": None}

    async for event in answer_database_question(question, language_instruction, conversation_context):
        if event["type"] == "chunk":
            chunks.append(event["text"])
        else:
            done = event

    return {"answer": "".join(chunks), "sources": done["sources"], "sql": done["sql"], "rows": done["rows"]}
