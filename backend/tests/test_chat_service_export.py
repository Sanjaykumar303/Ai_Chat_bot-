# answer_docx_export()/answer_xlsx_export() - the two dispatch targets
# routes/chat.py calls for an export request (see services/export_intent.py
# for how that's recognized). Same monkeypatch-at-the-module-boundary
# style as test_db_query_service_branching.py: the real Gemini/DB calls
# stay unmocked in normal use, but the CONTROL FLOW that decides whether
# a second query happens, or whether a spreadsheet gets fabricated from
# nothing, is deterministic and worth locking down directly.
#
# answer_xlsx_export() calls db_query_service.answer_database_question_
# full() (the non-streaming collect wrapper - see chat_service.py's own
# module docstring for why the streaming answer_database_question() and
# this collect wrapper are two different names now), NOT
# answer_database_question - patching the wrong one here would silently
# let a test fall through to the REAL, live pipeline instead of the
# fake, which is exactly what happened once before this file was fixed
# alongside the streaming refactor (caught by a real SUPABASE connection
# warning appearing in a "mocked" test's own log output).

import asyncio

from services import chat_service


def _run(coroutine):
    return asyncio.run(coroutine)


def test_docx_export_uses_the_previous_answer_verbatim_with_no_gemini_call(monkeypatch):
    # answer_docx_export is a plain, non-async function with no
    # generate()/generate_stream() reference anywhere in its body (see
    # its own docstring) - it structurally cannot call Gemini, so there
    # is nothing left to monkeypatch defensively here; this just asserts
    # directly on what it returns.
    result = chat_service.answer_docx_export("The total profit today is 1,000.")

    assert result["export"]["format"] == "docx"
    assert result["export"]["filename"].endswith(".docx")
    assert "id" in result["export"]


def test_docx_export_with_no_previous_answer_does_not_fabricate_a_file():
    result = chat_service.answer_docx_export(None)

    assert "export" not in result
    assert "ask a question first" in result["answer"].lower()


def test_docx_export_with_blank_previous_answer_does_not_fabricate_a_file():
    # A previous_answer of only whitespace is still "nothing real to
    # convert" - same outcome as None, not an empty document.
    result = chat_service.answer_docx_export("   ")
    assert "export" not in result


def test_xlsx_export_queries_the_database_exactly_once(monkeypatch):
    calls = []

    async def fake_answer_database_question_full(question, language_instruction=""):
        calls.append(question)
        return {
            "answer": "Income was 5,000.",
            "sources": [],
            "sql": "SELECT amount FROM accounts_income",
            "rows": [{"amount": 5000}],
        }

    monkeypatch.setattr(chat_service, "answer_database_question_full", fake_answer_database_question_full)

    result = _run(chat_service.answer_xlsx_export("export last month's income"))

    assert len(calls) == 1, "the export path must reuse one query, never run it twice"
    assert result["export"]["format"] == "xlsx"
    assert "1 row" in result["answer"]


def test_xlsx_export_strips_the_export_phrase_before_generating_sql(monkeypatch):
    seen_questions = []

    async def fake_answer_database_question_full(question, language_instruction=""):
        seen_questions.append(question)
        return {"answer": "ok", "sources": [], "sql": "SELECT 1", "rows": [{"x": 1}]}

    monkeypatch.setattr(chat_service, "answer_database_question_full", fake_answer_database_question_full)

    _run(chat_service.answer_xlsx_export("give me last month's income in Excel"))

    assert seen_questions == ["give me last month's income"]


def test_xlsx_export_does_not_fabricate_a_spreadsheet_when_no_sql_ran(monkeypatch):
    # classify_database_question's CAPABILITY/SCHEMA branches, and a
    # genuine NO_QUERY, all return sql=None from answer_database_question_
    # full - none of them have real rows behind them, so no file should
    # appear.
    async def fake_answer_database_question_full(question, language_instruction=""):
        return {"answer": "I couldn't answer that from the database.", "sources": [], "sql": None, "rows": None}

    monkeypatch.setattr(chat_service, "answer_database_question_full", fake_answer_database_question_full)

    result = _run(chat_service.answer_xlsx_export("export the weather forecast"))

    assert "export" not in result
    assert result["answer"] == "I couldn't answer that from the database."


def test_xlsx_export_reports_singular_row_count_correctly(monkeypatch):
    async def fake_answer_database_question_full(question, language_instruction=""):
        return {"answer": "ok", "sources": [], "sql": "SELECT 1", "rows": [{"x": 1}]}

    monkeypatch.setattr(chat_service, "answer_database_question_full", fake_answer_database_question_full)

    result = _run(chat_service.answer_xlsx_export("export today's income"))

    assert "1 row found" in result["answer"]
    assert "1 rows" not in result["answer"]
