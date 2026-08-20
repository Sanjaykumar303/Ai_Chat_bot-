# Regression tests for the schema/capability branches added to
# db_query_service.answer_database_question_full() - see intent_router.py's
# classify_database_question for the three-way split these exercise.
#
# Same scope rule as the rest of this suite (see
# test_chat_service_guards.py / test_entity_resolution.py): the real
# Gemini/DB calls stay unmocked in normal use, but the CONTROL FLOW that
# decides whether a call happens at all is deterministic and worth
# locking down directly - these monkeypatch db_client/gemini_client at
# the db_query_service module boundary to assert which path each kind of
# question takes, not what any live call would return.
#
# answer_database_question() itself is a streaming async generator (see
# its own docstring) - these tests mostly go through
# answer_database_question_full(), the non-streaming collect-to-a-dict
# wrapper, since they're testing CONTROL FLOW (which calls happened, what
# the final answer/sql/rows were), not the chunking itself (see
# test_answer_is_actually_streamed_in_multiple_chunks at the bottom for
# that). SQL generation still calls generate() (unchanged, never
# streamed - SQL is consumed programmatically, never shown to a user);
# the natural-language ANSWER now calls generate_stream() instead, so
# every fake here patches BOTH, routing both through the same
# question-shape-sensing logic via _stream_from().

import asyncio

import pytest

from services import db_query_service
from services.db_client import DatabaseConnectionError


@pytest.fixture(autouse=True)
def _reset_query_cache():
    # answer_database_question()'s short-TTL (sql, rows) cache is
    # module-level state, shared across every call - without this reset,
    # a test could see a cache hit populated by an EARLIER test's own
    # monkeypatched generate()/execute_readonly_query() for the same
    # question text, silently returning stale/wrong-test data instead of
    # exercising this test's own stubs. Same "clear shared module state
    # before and after each test" pattern test_document_store.py already
    # uses for document_store._documents.
    db_query_service._query_cache.clear()
    yield
    db_query_service._query_cache.clear()


def _run(coroutine):
    return asyncio.run(coroutine)


def _stub_schema(monkeypatch, description="Table students(id int, name text)", tables=None):
    monkeypatch.setattr(db_query_service, "get_schema_description", lambda: description)
    monkeypatch.setattr(db_query_service, "get_table_allowlist", lambda: tables or {"students"})


def _stream_from(fake_generate):
    """Wrap a coroutine `async def f(prompt) -> str` (the shape every
    test's own fake_generate already takes) into an async-generator
    matching generate_stream()'s real shape - yields the whole string as
    ONE chunk, which is all these control-flow tests need (real
    incremental chunking is covered separately, see
    test_answer_is_actually_streamed_in_multiple_chunks below)."""

    async def fake_stream(prompt):
        text = await fake_generate(prompt)
        if text:
            yield text

    return fake_stream


def _stub_both(monkeypatch, fake_generate):
    """Patch both generate() (still used for SQL generation, which is
    never streamed) and generate_stream() (used for the answer prompt)
    to route through the SAME fake - one place that decides what to
    return per prompt, used by whichever call site actually fires."""

    monkeypatch.setattr(db_query_service, "generate", fake_generate)
    monkeypatch.setattr(db_query_service, "generate_stream", _stream_from(fake_generate))


def _stub_generate(monkeypatch, record, response="stub answer"):
    async def fake_generate(prompt):
        record.append(prompt)
        return response

    _stub_both(monkeypatch, fake_generate)


def test_capability_question_answers_without_calling_gemini_at_all(monkeypatch):
    # The whole point of a fixed capability answer: there's nothing left
    # for a model to reason about once the schema fetch above already
    # proved the database is reachable, so no generate() call should
    # happen at all for this branch.
    _stub_schema(monkeypatch)
    prompts = []
    _stub_generate(monkeypatch, prompts)

    result = _run(db_query_service.answer_database_question_full("Can you connect to our DB?"))

    assert prompts == [], "capability answers must not call Gemini"
    assert "connected to the live database" in result["answer"].lower()
    assert result["sql"] is None
    assert result["sources"] == []


def test_schema_question_answers_from_schema_description_with_no_sql(monkeypatch):
    _stub_schema(monkeypatch, description="Table students(id int, name text, age int)")
    prompts = []
    _stub_generate(monkeypatch, prompts, response="You have a students table.")

    result = _run(db_query_service.answer_database_question_full("what are tables in my db?"))

    assert result["answer"] == "You have a students table."
    assert result["sql"] is None
    # Genuinely sourced from live introspected schema - see ChatBox.jsx's
    # SourceBadge, which renders a "Database" pill for this exact shape.
    assert result["sources"] == [{"type": "database"}]
    # The schema text has to actually reach the prompt - the point of
    # this branch is answering from real introspected structure, not a
    # generic "I don't know" or an invented table - and it has to be the
    # dedicated schema-summary prompt, not SQL_PROMPT (which asks Gemini
    # to write a query, the exact machinery this branch exists to skip).
    assert "Table students(id int, name text, age int)" in prompts[0]
    assert "describing what is available in a connected database" in prompts[0]


def test_summary_the_db_takes_the_schema_branch_not_the_sql_branch(monkeypatch):
    # The literal bug report case - previously misrouted to
    # GENERAL_KNOWLEDGE upstream; here we confirm that once it does reach
    # this function, it takes the schema branch (no SQL) rather than
    # falling through to text-to-SQL generation, which would either
    # invent a query or come back NO_QUERY for a question this vague.
    _stub_schema(monkeypatch)
    prompts = []
    _stub_generate(monkeypatch, prompts, response="Here's what's in the database.")

    result = _run(db_query_service.answer_database_question_full("summary the db"))

    assert result["sql"] is None
    assert result["answer"] == "Here's what's in the database."


def test_data_question_still_goes_through_the_unchanged_sql_pipeline(monkeypatch):
    # The pre-existing, unmodified behavior: a real data question calls
    # Gemini twice (SQL, then the natural-language answer) and runs the
    # query. Confirms the new branches only intercept schema/capability
    # questions and leave this path untouched.
    _stub_schema(monkeypatch)
    calls = []

    async def fake_generate(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "SELECT COUNT(*) AS count FROM students"
        return "There are 5 students."

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"count": 5}])

    result = _run(db_query_service.answer_database_question_full("how many students are there?"))

    assert len(calls) == 2, "a genuine data question must still generate SQL and then an answer"
    # sql_guard.validate_and_limit re-renders the query (adds a LIMIT), so
    # this checks the query was actually validated and run, not for a
    # byte-exact string.
    assert result["sql"] is not None
    assert "FROM students" in result["sql"]
    assert result["answer"] == "There are 5 students."
    # A real answer derived from live database rows - must be attributed
    # (see ChatBox.jsx's SourceBadge/"Database" pill; this was previously
    # always sources=[] for every DB-backed answer, a real reported gap).
    assert result["sources"] == [{"type": "database"}]


def test_schema_unavailable_falls_back_before_any_kind_classification(monkeypatch):
    # If the database itself can't be reached, every kind of question
    # (including a capability one) has to degrade to the existing
    # fallback - unchanged pre-existing behavior, not a new code path.
    def broken_schema():
        raise DatabaseConnectionError("no connection configured")

    monkeypatch.setattr(db_query_service, "get_schema_description", broken_schema)

    result = _run(db_query_service.answer_database_question_full("Can you connect to our DB?"))

    assert result["answer"] == db_query_service.FALLBACK_ANSWER
    assert result["sql"] is None


# --- the short-TTL (sql, rows) cache for a repeated DATA question ------
#
# Deliberately never caches the final natural-language answer (see
# answer_database_question's own docstring) - only what SQL generation
# and the DB round trip produced, so per-request conversation_context/
# language_instruction can never be served stale from a different
# session/language. These monkeypatch generate()/generate_stream()/
# execute_readonly_query() with call-counting fakes to prove exactly how
# many times each was actually invoked, which is the only thing that
# distinguishes a cache hit from a miss from the caller's side.


def test_repeated_data_question_skips_sql_generation_and_the_db_round_trip(monkeypatch):
    _stub_schema(monkeypatch)
    sql_calls = []
    db_calls = []

    async def fake_generate(prompt):
        if "SQL:" in prompt and "QUERY RESULTS" not in prompt:
            sql_calls.append(prompt)
            return "SELECT COUNT(*) AS count FROM students"
        return "There are 5 students."

    def fake_execute(sql):
        db_calls.append(sql)
        return [{"count": 5}]

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", fake_execute)

    first = _run(db_query_service.answer_database_question_full("how many students are there?"))
    second = _run(db_query_service.answer_database_question_full("how many students are there?"))

    assert len(sql_calls) == 1, "SQL generation must not be repeated for a cache hit"
    assert len(db_calls) == 1, "the DB round trip must not be repeated for a cache hit"
    assert first["sql"] == second["sql"]
    assert first["rows"] == second["rows"] == [{"count": 5}]


def test_cache_key_is_case_and_whitespace_insensitive(monkeypatch):
    _stub_schema(monkeypatch)
    sql_calls = []

    async def fake_generate(prompt):
        if "SQL:" in prompt and "QUERY RESULTS" not in prompt:
            sql_calls.append(prompt)
            return "SELECT COUNT(*) AS count FROM students"
        return "answer"

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"count": 5}])

    _run(db_query_service.answer_database_question_full("How many students are there?"))
    _run(db_query_service.answer_database_question_full("  how many students are there?  "))

    assert len(sql_calls) == 1, "differing only by case/whitespace must still hit the same cache entry"


def test_answer_is_always_freshly_generated_even_on_a_cache_hit(monkeypatch):
    # The one thing that must NEVER be cached: conversation_context is
    # per-request, so a second, different session asking the identical
    # question must get an answer reflecting ITS OWN context, not
    # whatever the first session's context happened to produce.
    _stub_schema(monkeypatch)
    seen_contexts = []

    async def fake_generate(prompt):
        if "SQL:" in prompt and "QUERY RESULTS" not in prompt:
            return "SELECT COUNT(*) AS count FROM students"
        seen_contexts.append("Alex" in prompt)
        return "answer mentioning context" if "Alex" in prompt else "plain answer"

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"count": 5}])

    _run(db_query_service.answer_database_question_full("how many students are there?", conversation_context=""))
    _run(
        db_query_service.answer_database_question_full(
            "how many students are there?", conversation_context="User: My name is Alex\n\n"
        )
    )

    assert seen_contexts == [False, True], "the second call's own context must reach the answer prompt"


def test_no_query_result_is_not_cached(monkeypatch):
    # A NO_QUERY verdict is the model's own uncertainty about this one
    # attempt, not a stable fact about the question - a retry (by this
    # user or the next one) must always get a fresh shot at it.
    _stub_schema(monkeypatch)
    calls = []

    async def fake_generate(prompt):
        calls.append(prompt)
        return "NO_QUERY"

    _stub_both(monkeypatch, fake_generate)

    _run(db_query_service.answer_database_question_full("what is the meaning of life?"))
    _run(db_query_service.answer_database_question_full("what is the meaning of life?"))

    assert len(calls) == 2, "a NO_QUERY outcome must not be cached"


def test_cache_expires_after_its_ttl(monkeypatch):
    _stub_schema(monkeypatch)
    sql_calls = []

    async def fake_generate(prompt):
        if "SQL:" in prompt and "QUERY RESULTS" not in prompt:
            sql_calls.append(prompt)
            return "SELECT COUNT(*) AS count FROM students"
        return "answer"

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"count": 5}])

    _run(db_query_service.answer_database_question_full("how many students are there?"))
    # Backdate the cached entry past its TTL instead of sleeping - same
    # "manipulate the timestamp, don't wait in real time" pattern
    # test_document_store.py already uses for its own TTL tests.
    for entry in db_query_service._query_cache.values():
        entry["cached_at"] -= db_query_service.DB_QUERY_CACHE_TTL + 1
    _run(db_query_service.answer_database_question_full("how many students are there?"))

    assert len(sql_calls) == 2, "an expired cache entry must not be reused"


# --- pre-emptive anti-fabrication guards: skip generation, not replace -
#
# The actual point of moving these guards into this module (see its own
# comment above _INSUFFICIENT_DATA_ANSWER): a triggered guard must skip
# the answer-generation Gemini call ENTIRELY, not call it and then
# discard/replace the result - the latter is what made streaming the
# answer unsafe (tokens already shown to the user can't be un-sent).
# These assert the answer-gen call never happens at all when a guard
# fires, using a fake generate()/generate_stream() that blows up if
# reached past SQL generation - not just checking the returned text.


def test_all_zero_rows_skips_answer_generation_entirely(monkeypatch):
    _stub_schema(monkeypatch)

    async def fake_generate(prompt):
        if "SQL:" in prompt and "QUERY RESULTS" not in prompt:
            return "SELECT SUM(amount) AS total FROM students"
        raise AssertionError("answer generation must be skipped when the guard fires")

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"total": 0}])

    result = _run(db_query_service.answer_database_question_full("What is our total revenue?"))

    assert result["answer"] == db_query_service._INSUFFICIENT_DATA_ANSWER
    assert result["sql"] is not None
    assert result["rows"] == [{"total": 0}]
    # Deliberately NOT sourced - a guard saying "don't trust this" isn't a
    # real answer to attribute to the database (see _DATABASE_SOURCE's own
    # comment in db_query_service.py for the full list of unbadged paths).
    assert result["sources"] == []


def test_unverified_single_day_question_skips_answer_generation_entirely(monkeypatch):
    _stub_schema(monkeypatch)

    async def fake_generate(prompt):
        if "SQL:" in prompt and "QUERY RESULTS" not in prompt:
            # No exact CURRENT_DATE filter - a running total, not today's figure.
            return "SELECT SUM(amount) AS total FROM students WHERE created_at <= CURRENT_DATE"
        raise AssertionError("answer generation must be skipped when the guard fires")

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"total": 15000}])

    result = _run(db_query_service.answer_database_question_full("What is our total revenue today?"))

    assert result["answer"] == db_query_service._UNVERIFIED_SINGLE_DAY_ANSWER
    assert result["rows"] == [{"total": 15000}]


def test_a_real_result_still_reaches_answer_generation_normally(monkeypatch):
    # Sanity check the guards aren't over-firing: a genuine, exact-day-
    # filtered, non-zero result must still go through normal generation.
    _stub_schema(monkeypatch)
    answer_calls = []

    async def fake_generate(prompt):
        if "SQL:" in prompt and "QUERY RESULTS" not in prompt:
            return "SELECT SUM(amount) AS total FROM students WHERE created_at = CURRENT_DATE"
        answer_calls.append(prompt)
        return "Revenue today is 15,000."

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"total": 15000}])

    result = _run(db_query_service.answer_database_question_full("What is our total revenue today?"))

    assert len(answer_calls) == 1
    assert result["answer"] == "Revenue today is 15,000."


def test_guard_still_fires_on_a_cache_hit(monkeypatch):
    # The guard has to re-check every time, not just on the first,
    # cache-populating call - a cached all-zero result must be caught on
    # every subsequent lookup too.
    _stub_schema(monkeypatch)
    answer_gen_attempts = []

    async def fake_generate(prompt):
        if "SQL:" in prompt and "QUERY RESULTS" not in prompt:
            return "SELECT SUM(amount) AS total FROM students"
        answer_gen_attempts.append(prompt)
        raise AssertionError("answer generation must be skipped when the guard fires")

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"total": 0}])

    first = _run(db_query_service.answer_database_question_full("What is our total revenue?"))
    second = _run(db_query_service.answer_database_question_full("What is our total revenue?"))

    assert first["answer"] == db_query_service._INSUFFICIENT_DATA_ANSWER
    assert second["answer"] == db_query_service._INSUFFICIENT_DATA_ANSWER
    assert answer_gen_attempts == []


# --- the generator itself actually streams multiple chunks -------------


def test_answer_is_actually_streamed_in_multiple_chunks(monkeypatch):
    # Unlike the tests above (which go through answer_database_question_
    # full()'s collect-to-a-dict convenience wrapper, appropriate for
    # testing control flow), this drives the raw generator directly to
    # prove the answer genuinely arrives as separate chunk events - the
    # whole point of this feature - not as one single chunk that merely
    # LOOKS the same after collection.
    _stub_schema(monkeypatch)

    async def fake_generate(prompt):
        return "SELECT COUNT(*) AS count FROM students"

    async def fake_generate_stream(prompt):
        for piece in ["There ", "are ", "5 ", "students."]:
            yield piece

    monkeypatch.setattr(db_query_service, "generate", fake_generate)
    monkeypatch.setattr(db_query_service, "generate_stream", fake_generate_stream)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"count": 5}])

    events = []

    async def collect():
        async for event in db_query_service.answer_database_question("how many students are there?"):
            events.append(event)

    _run(collect())

    chunk_events = [event for event in events if event["type"] == "chunk"]
    done_events = [event for event in events if event["type"] == "done"]

    assert len(chunk_events) == 4, "the four pieces must arrive as four separate chunk events, not one"
    assert "".join(event["text"] for event in chunk_events) == "There are 5 students."
    assert len(done_events) == 1
    assert done_events[0]["sql"] is not None
    assert done_events[0]["rows"] == [{"count": 5}]
    # The done event must be LAST - a consumer relaying chunks onward as
    # they arrive relies on this order to know when the stream is over.
    assert events[-1]["type"] == "done"
