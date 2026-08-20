# Regression tests for the DB Analysis / WHY capability in
# db_query_service.answer_database_question() - see intent_router.py's
# classify_database_question for the DATABASE_QUESTION_WHY kind.
#
# IMPORTANT architecture note (see [[project_db_why_analysis_intent_2026-08-20]]
# for the full history): WHY and DATA are NOT separate prompts/code paths
# here. SQL_PROMPT/ANSWER_PROMPT (used for EVERY SQL-backed database
# question) teach Gemini to recognize, from the question's own actual
# wording, whether it's asking for a plain value or an explanation
# (why/how/what caused something) and shape the SQL/answer accordingly -
# a real model judgment call, not a regex match. classify_database_
# question's _WHY_ANALYSIS_RE still runs and still labels a question WHY
# or DATA, but that label is now used ONLY for the `[db-timing] kind=...`
# log line - it does not gate which prompt Gemini receives. This is
# deliberate: a fixed keyword list can never enumerate every way a person
# might ask "explain this to me", so gating behavior on it would leave
# real questions (e.g. "How did we end up with less revenue?") stuck with
# a flat number instead of an explanation just because they didn't happen
# to say "why".
#
# Same scope/style as test_db_query_service_branching.py, which this file
# deliberately mirrors rather than shares fixtures with (this project's
# own established pattern - each test file keeps its own small local
# stubs rather than a shared conftest.py): SQL Guard
# (sql_guard.validate_and_limit) is NEVER mocked - a WHY question's
# generated SQL goes through the exact same real validation a DATA
# question's does, so a test asserting SQL Guard actually rejects unsafe
# WHY-generated SQL is proof the guard wasn't bypassed, not just a claim
# about it. Only generate()/generate_stream() (Gemini) and
# execute_readonly_query() (the DB round trip) are ever monkeypatched.

import asyncio

import pytest

from services import db_query_service
from services.db_client import DatabaseConnectionError


@pytest.fixture(autouse=True)
def _reset_query_cache():
    # The short-TTL (sql, rows) cache is module-level state shared across
    # every call, including between DATA- and WHY-labeled questions (they
    # share the exact same cache - see _resolve_sql_and_rows) - without
    # this reset, a test could see a cache hit populated by an earlier
    # test's own monkeypatched stubs for the same question text.
    db_query_service._query_cache.clear()
    yield
    db_query_service._query_cache.clear()


def _run(coroutine):
    return asyncio.run(coroutine)


def _stub_schema(monkeypatch, description="Table sales(id int, amount int, sale_date date, category text)", tables=None):
    monkeypatch.setattr(db_query_service, "get_schema_description", lambda: description)
    monkeypatch.setattr(db_query_service, "get_table_allowlist", lambda: tables or {"sales"})


def _stream_from(fake_generate):
    async def fake_stream(prompt):
        text = await fake_generate(prompt)
        if text:
            yield text

    return fake_stream


def _stub_both(monkeypatch, fake_generate):
    monkeypatch.setattr(db_query_service, "generate", fake_generate)
    monkeypatch.setattr(db_query_service, "generate_stream", _stream_from(fake_generate))


def _is_sql_gen_prompt(prompt):
    # SQL_PROMPT/SQL_RETRY_PROMPT both end in "SQL:"; ANSWER_PROMPT ends
    # in "ANSWER:" and additionally contains "QUERY RESULTS" - that's
    # what disambiguates the two (both now also contain "WHY"/"HOW"
    # internally, since the guidance is unified - see the module comment
    # above).
    return "SQL:" in prompt and "QUERY RESULTS" not in prompt


# --- the unified prompt actually teaches Gemini to recognize BOTH shapes --


def test_the_one_sql_prompt_used_for_every_question_teaches_both_value_and_explanation_shapes(monkeypatch):
    # The whole point of the merge: there is no separate "WHY prompt" any
    # more for classify_database_question to gate access to - EVERY
    # SQL-generation call (regardless of question_kind) gets the same
    # instructions to recognize an explanation-type question from its
    # actual meaning and write a comparison/breakdown query for it, or a
    # plain query otherwise.
    _stub_schema(monkeypatch)
    sql_prompts = []

    async def fake_generate(prompt):
        if _is_sql_gen_prompt(prompt):
            sql_prompts.append(prompt)
            return "SELECT COUNT(*) AS count FROM sales"
        return "5"

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"count": 5}])

    _run(db_query_service.answer_database_question_full("how many sales are there?"))

    assert "WHY" in sql_prompts[0] and "HOW" in sql_prompts[0]
    assert "COMPARISON" in sql_prompts[0]
    assert "BREAKDOWN" in sql_prompts[0]
    assert "not fixed trigger words" in sql_prompts[0]


# --- a plain value question is unaffected ---------------------------------


def test_normal_data_question_still_gets_a_plain_direct_query_and_answer(monkeypatch):
    _stub_schema(monkeypatch)
    calls = []

    async def fake_generate(prompt):
        calls.append(prompt)
        if _is_sql_gen_prompt(prompt):
            return "SELECT COUNT(*) AS count FROM sales"
        return "There are 5 sales."

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [{"count": 5}])

    result = _run(db_query_service.answer_database_question_full("how many sales are there?"))

    assert len(calls) == 2, "a plain data question must still generate SQL once and then an answer once"
    assert "FROM sales" in result["sql"]
    assert result["answer"] == "There are 5 sales."


# --- a WHY-phrased question: targeted SQL, real rows only, causal explanation


def test_why_question_generates_comparison_sql_and_explains_the_cause_from_only_the_rows(monkeypatch):
    _stub_schema(monkeypatch)
    sql_prompts = []
    answer_prompts = []

    async def fake_generate(prompt):
        if _is_sql_gen_prompt(prompt):
            sql_prompts.append(prompt)
            return (
                "SELECT DATE_TRUNC('month', sale_date) AS month, SUM(amount) AS total "
                "FROM sales GROUP BY 1 ORDER BY 1"
            )
        answer_prompts.append(prompt)
        return "Revenue dropped because October's total (₹40,000) is well below September's (₹90,000)."

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(
        db_query_service,
        "execute_readonly_query",
        lambda sql: [{"month": "2026-09-01", "total": 90000}, {"month": "2026-10-01", "total": 40000}],
    )

    result = _run(db_query_service.answer_database_question_full("Why did revenue drop?"))

    # The answer prompt must carry ONLY the actually-retrieved rows -
    # never the raw schema or "the whole dataset".
    assert "based ONLY on the query results" in answer_prompts[0]
    assert "90000" in answer_prompts[0] and "40000" in answer_prompts[0]
    assert "Table sales(id int" not in answer_prompts[0]  # the schema itself must never be sent as "data"

    assert result["sql"] is not None
    assert "GROUP BY" in result["sql"] or "group by" in result["sql"].lower()
    assert result["rows"] == [{"month": "2026-09-01", "total": 90000}, {"month": "2026-10-01", "total": 40000}]
    assert "40,000" in result["answer"] or "40000" in result["answer"]


def test_a_how_phrased_question_the_why_regex_would_miss_still_gets_full_explanation_treatment(monkeypatch):
    # THE key regression this file exists to prove: intent_router's
    # _WHY_ANALYSIS_RE does not match every possible phrasing of a causal
    # question (it's a fixed regex, necessarily incomplete). Before the
    # prompts were unified, a question like this one would have been
    # labeled DATABASE_QUESTION_DATA and sent through the plain
    # single-value SQL_PROMPT/ANSWER_PROMPT, missing the comparison this
    # scenario needs entirely. Now, because the SAME prompt is used
    # either way, Gemini can still choose to write comparison SQL and
    # explain from it - proven here by having the fake `generate()` do
    # exactly that regardless of which label this question got.
    _stub_schema(monkeypatch)

    # Confirm the premise: this phrasing is NOT recognized by the regex.
    from services.intent_router import classify_database_question, DATABASE_QUESTION_DATA

    assert classify_database_question("How did we end up with less revenue this quarter?") == DATABASE_QUESTION_DATA

    async def fake_generate(prompt):
        if _is_sql_gen_prompt(prompt):
            # Simulates Gemini choosing a comparison shape on its own,
            # from reading the question - not because a label told it to.
            return (
                "SELECT DATE_TRUNC('month', sale_date) AS month, SUM(amount) AS total "
                "FROM sales GROUP BY 1 ORDER BY 1"
            )
        return "Revenue ended up lower this quarter because September's total (₹90,000) fell to ₹40,000 in October."

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(
        db_query_service,
        "execute_readonly_query",
        lambda sql: [{"month": "2026-09-01", "total": 90000}, {"month": "2026-10-01", "total": 40000}],
    )

    result = _run(db_query_service.answer_database_question_full("How did we end up with less revenue this quarter?"))

    assert "40,000" in result["answer"] or "40000" in result["answer"]
    assert result["rows"] == [{"month": "2026-09-01", "total": 90000}, {"month": "2026-10-01", "total": 40000}]


def test_why_question_reuses_the_real_sql_guard_and_rejects_a_disallowed_table(monkeypatch):
    # SQL Guard (sql_guard.validate_and_limit) is NOT mocked in this test
    # - a WHY question naming a table outside the allowlist must be
    # rejected by the exact same real guard a DATA question's SQL would
    # be, proving the guard isn't bypassed for a causal question.
    _stub_schema(monkeypatch, tables={"sales"})
    sql_attempts = []

    async def fake_generate(prompt):
        if _is_sql_gen_prompt(prompt):
            sql_attempts.append(prompt)
            # Every attempt (including the one corrective retry) names a
            # table that was never in the allowlist above.
            return "SELECT * FROM secret_admin_table"
        raise AssertionError("answer generation must never run when every SQL attempt was rejected")

    _stub_both(monkeypatch, fake_generate)

    def fail_if_queried(sql):
        raise AssertionError("execute_readonly_query must never run for SQL that SQL Guard rejected")

    monkeypatch.setattr(db_query_service, "execute_readonly_query", fail_if_queried)

    result = _run(db_query_service.answer_database_question_full("Why did revenue drop?"))

    assert len(sql_attempts) == 2, "one corrective retry, then give up - same as any other SQL-backed question"
    assert result["answer"] == db_query_service.FALLBACK_ANSWER
    assert result["sql"] is None
    assert result["rows"] is None


def test_why_question_no_query_falls_back_without_calling_the_database(monkeypatch):
    _stub_schema(monkeypatch)

    async def fake_generate(prompt):
        if _is_sql_gen_prompt(prompt):
            return "NO_QUERY"
        raise AssertionError("answer generation must not run for NO_QUERY")

    _stub_both(monkeypatch, fake_generate)

    def fail_if_queried(sql):
        raise AssertionError("the database must never be queried for NO_QUERY")

    monkeypatch.setattr(db_query_service, "execute_readonly_query", fail_if_queried)

    result = _run(db_query_service.answer_database_question_full("Why did revenue drop?"))

    assert result["answer"] == db_query_service.FALLBACK_ANSWER


# --- insufficient data: say so, never guess ---------------------------


def test_why_question_with_no_rows_says_so_instead_of_guessing(monkeypatch):
    # A comparison/breakdown query can genuinely return zero rows (e.g.
    # nothing recorded in either period) - a plain single-aggregate value
    # question rarely hits this (SUM/COUNT always return exactly one
    # row), but a causal question's own comparison/breakdown query with
    # nothing to compare must say so, not fabricate a cause.
    _stub_schema(monkeypatch)

    async def fake_generate(prompt):
        if _is_sql_gen_prompt(prompt):
            return "SELECT category, SUM(amount) AS total FROM sales WHERE sale_date >= CURRENT_DATE - INTERVAL '60 days' GROUP BY category"
        raise AssertionError("answer generation must be skipped when there is nothing to explain from")

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", lambda sql: [])

    result = _run(db_query_service.answer_database_question_full("Why did revenue drop?"))

    assert result["answer"] == db_query_service._NO_ROWS_ANSWER
    assert result["rows"] == []


def test_why_question_with_all_zero_values_says_so_instead_of_guessing(monkeypatch):
    # Reuses the exact same _all_values_zero_or_null guard every SQL-
    # backed question already uses - a real, non-empty comparison where
    # every value came back zero/null is just as much "no real signal to
    # explain" as the plain-value path's own zero-aggregate case. A
    # COMPARISON shape (numbers only, no text label column) is what
    # actually exercises this guard - see _all_values_zero_or_null's own
    # definition: it checks every column of every row, so a BREAKDOWN row
    # that also carries a text category label (e.g. {"category": "A",
    # "total": 0}) never counts as "all zero", since the label itself
    # isn't 0/None either. That's pre-existing, unchanged behavior - not
    # something this test's own scenario needs to work around by
    # avoiding a label column.
    _stub_schema(monkeypatch)

    async def fake_generate(prompt):
        if _is_sql_gen_prompt(prompt):
            return (
                "SELECT (SELECT SUM(amount) FROM sales WHERE sale_date >= CURRENT_DATE - INTERVAL '30 days') AS current_total, "
                "(SELECT SUM(amount) FROM sales WHERE sale_date >= CURRENT_DATE - INTERVAL '60 days' AND sale_date < CURRENT_DATE - INTERVAL '30 days') AS previous_total"
            )
        raise AssertionError("answer generation must be skipped when the guard fires")

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(
        db_query_service, "execute_readonly_query", lambda sql: [{"current_total": 0, "previous_total": None}]
    )

    result = _run(db_query_service.answer_database_question_full("Why did revenue drop?"))

    assert result["answer"] == db_query_service._INSUFFICIENT_DATA_ANSWER


def test_why_question_database_failure_falls_back_cleanly(monkeypatch):
    _stub_schema(monkeypatch)

    async def fake_generate(prompt):
        if _is_sql_gen_prompt(prompt):
            return "SELECT category, SUM(amount) AS total FROM sales GROUP BY category"
        raise AssertionError("answer generation must not run when the query itself failed")

    _stub_both(monkeypatch, fake_generate)

    def broken_query(sql):
        raise DatabaseConnectionError("connection reset")

    monkeypatch.setattr(db_query_service, "execute_readonly_query", broken_query)

    result = _run(db_query_service.answer_database_question_full("Why did revenue drop?"))

    assert result["answer"] == db_query_service.FALLBACK_ANSWER


# --- follow-up questions: fresh conversation_context, shared cache -------


def test_why_question_as_a_follow_up_gets_its_own_conversation_context(monkeypatch):
    # Mirrors test_answer_is_always_freshly_generated_even_on_a_cache_hit
    # in test_db_query_service_branching.py, targeted at a causal
    # question: a second, different session following up with the same
    # question text must get an answer reflecting ITS OWN conversation
    # history, never a stale one cached from the first session's context.
    _stub_schema(monkeypatch)
    seen_contexts = []

    async def fake_generate(prompt):
        if _is_sql_gen_prompt(prompt):
            return "SELECT category, SUM(amount) AS total FROM sales GROUP BY category"
        seen_contexts.append("last month's promo" in prompt)
        return "answer mentioning the promo" if "last month's promo" in prompt else "plain answer"

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(
        db_query_service, "execute_readonly_query", lambda sql: [{"category": "A", "total": 10000}, {"category": "B", "total": 5000}]
    )

    _run(db_query_service.answer_database_question_full("Why did revenue drop?", conversation_context=""))
    _run(
        db_query_service.answer_database_question_full(
            "Why did revenue drop?",
            conversation_context="User: We ran last month's promo campaign then.\n\n",
        )
    )

    assert seen_contexts == [False, True], "the second follow-up's own context must reach the answer prompt"


def test_why_question_reuses_the_same_short_ttl_query_cache_as_plain_data_questions(monkeypatch):
    # Proves "reuse existing... caching" literally: asking the identical
    # causal question twice must skip SQL generation and the DB round
    # trip the second time, exactly like a repeated plain-value question
    # already does (see test_repeated_data_question_skips_sql_generation_
    # and_the_db_round_trip in test_db_query_service_branching.py) -
    # because it goes through the exact same _resolve_sql_and_rows/
    # _query_cache, not a second, separate cache.
    _stub_schema(monkeypatch)
    sql_calls = []
    db_calls = []

    async def fake_generate(prompt):
        if _is_sql_gen_prompt(prompt):
            sql_calls.append(prompt)
            return "SELECT category, SUM(amount) AS total FROM sales GROUP BY category"
        return "explanation"

    def fake_execute(sql):
        db_calls.append(sql)
        return [{"category": "A", "total": 10000}, {"category": "B", "total": 5000}]

    _stub_both(monkeypatch, fake_generate)
    monkeypatch.setattr(db_query_service, "execute_readonly_query", fake_execute)

    first = _run(db_query_service.answer_database_question_full("Why did revenue drop?"))
    second = _run(db_query_service.answer_database_question_full("Why did revenue drop?"))

    assert len(sql_calls) == 1, "SQL generation must not be repeated for a cache hit"
    assert len(db_calls) == 1, "the DB round trip must not be repeated for a cache hit"
    assert first["sql"] == second["sql"]
    assert first["rows"] == second["rows"]
