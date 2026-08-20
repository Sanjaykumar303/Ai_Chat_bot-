# Tests for the persistent chat-memory feature (services/chat_memory.py)
# and the conversation_context wiring in the answer prompts.
#
# Same scope rule as the rest of this suite (see test_chat_service_guards.py
# / test_db_query_service_branching.py): the real Gemini/DB calls stay
# unmocked in normal use, but the DETERMINISTIC logic around them is worth
# locking down directly. Here that's two things:
#   1. the summarization THROTTLE - the one piece of logic that, if wrong,
#      silently either wastes a Gemini call on every message or stops
#      folding older messages into the summary at all. These monkeypatch
#      chat_memory's own DB helpers (so no real Postgres is needed) and
#      count generate() calls.
#   2. the context-block FORMATTING and the guarantee that an empty
#      context leaves every answer prompt byte-identical to before this
#      feature existed.

import asyncio

import pytest

from services import chat_memory


def _run(coroutine):
    return asyncio.run(coroutine)


# ---------------------------------------------------------------------------
# _format_context - pure, no DB
# ---------------------------------------------------------------------------


def test_format_context_empty_when_nothing_stored():
    # No summary and no messages -> empty string, which is what keeps the
    # answer prompts byte-identical for a brand-new session.
    assert chat_memory._format_context(None, []) == ""


def test_format_context_includes_summary_and_recent_and_ends_with_blank_line():
    block = chat_memory._format_context(
        "User asked about profit.",
        [
            {"role": "user", "content": "What was the profit?"},
            {"role": "assistant", "content": "It was 5000."},
        ],
    )

    assert "Summary of earlier conversation: User asked about profit." in block
    assert "User: What was the profit?" in block
    assert "Assistant: It was 5000." in block
    # Must end with a blank line so it can sit directly before "QUESTION:"
    # in a prompt without running into it.
    assert block.endswith("\n\n")


def test_format_context_recent_only_when_no_summary_yet():
    block = chat_memory._format_context(None, [{"role": "user", "content": "hi"}])

    assert "Summary of earlier conversation" not in block
    assert "User: hi" in block


# ---------------------------------------------------------------------------
# maybe_summarize - the throttle (the single most important behavior)
# ---------------------------------------------------------------------------


def _stub_generate(monkeypatch, calls, response="An updated summary."):
    async def fake_generate(prompt):
        calls.append(prompt)
        return response

    monkeypatch.setattr(chat_memory, "generate", fake_generate)


def test_no_summarization_below_trigger_count(monkeypatch):
    # A short session (<= SUMMARIZE_TRIGGER_COUNT total) is fully covered
    # by the recent window alone, so there is nothing to summarize and no
    # Gemini call must happen.
    monkeypatch.setattr(chat_memory, "count_messages", lambda sid: chat_memory.SUMMARIZE_TRIGGER_COUNT)
    calls = []
    _stub_generate(monkeypatch, calls)
    # If it were reached, this would blow up - proving it isn't reached.
    monkeypatch.setattr(chat_memory, "_aged_out_messages", lambda sid: 1 / 0)

    result = _run(chat_memory.maybe_summarize("s1"))

    assert result is None
    assert calls == []


def test_summarizes_once_when_threshold_first_crossed(monkeypatch):
    monkeypatch.setattr(chat_memory, "count_messages", lambda sid: chat_memory.SUMMARIZE_TRIGGER_COUNT + 2)
    monkeypatch.setattr(
        chat_memory,
        "_aged_out_messages",
        lambda sid: [
            {"role": "user", "content": "old q", "created_at": "t1"},
            {"role": "assistant", "content": "old a", "created_at": "t2"},
        ],
    )
    monkeypatch.setattr(chat_memory, "get_summary", lambda sid: None)
    upserts = []
    monkeypatch.setattr(
        chat_memory, "_upsert_summary", lambda sid, summary, watermark: upserts.append((sid, summary, watermark))
    )
    calls = []
    _stub_generate(monkeypatch, calls, response="Rolled-up summary.")

    result = _run(chat_memory.maybe_summarize("s1"))

    assert result == "Rolled-up summary."
    assert len(calls) == 1, "exactly one Gemini call the first time older messages age out"
    # Watermark must be the newest aged-out message's created_at ("t2"),
    # not wall-clock time - see _upsert_summary's docstring for why a
    # now()-based watermark would silently drop messages from context.
    assert upserts == [("s1", "Rolled-up summary.", "t2")]


def test_does_not_resummarize_when_nothing_new_aged_out(monkeypatch):
    # Session is long (past the trigger), but every older message has
    # already been folded into the existing summary (watermark logic in
    # _aged_out_messages returns nothing new) - so NO further Gemini call
    # must be made. This is the money-wasting / correctness trap the
    # throttle exists to prevent.
    monkeypatch.setattr(chat_memory, "count_messages", lambda sid: chat_memory.SUMMARIZE_TRIGGER_COUNT + 50)
    monkeypatch.setattr(chat_memory, "_aged_out_messages", lambda sid: [])
    calls = []
    _stub_generate(monkeypatch, calls)
    monkeypatch.setattr(chat_memory, "_upsert_summary", lambda sid, summary, watermark: 1 / 0)

    result = _run(chat_memory.maybe_summarize("s1"))

    assert result is None
    assert calls == [], "no Gemini call when nothing new has aged out since last summary"


def test_upsert_summary_watermark_is_last_message_created_at_not_now(monkeypatch):
    # Regression test for a real bug found via live-DB testing: if the
    # summary's watermark were now() instead of the newest folded-in
    # message's own created_at, a message still inside the recent window
    # at summarization time (correctly excluded that round) would later
    # fall out of the window without ever being folded into the summary -
    # permanently invisible to both the recent window AND the summary.
    # Anchoring the watermark to the last aged-out message's created_at
    # (not wall-clock time) is what prevents that gap.
    monkeypatch.setattr(chat_memory, "count_messages", lambda sid: chat_memory.SUMMARIZE_TRIGGER_COUNT + 1)
    monkeypatch.setattr(
        chat_memory,
        "_aged_out_messages",
        lambda sid: [
            {"role": "user", "content": "a", "created_at": "2026-01-01T00:00:00"},
            {"role": "user", "content": "b", "created_at": "2026-01-01T00:05:00"},
        ],
    )
    monkeypatch.setattr(chat_memory, "get_summary", lambda sid: None)
    upserts = []
    monkeypatch.setattr(
        chat_memory, "_upsert_summary", lambda sid, summary, watermark: upserts.append(watermark)
    )
    _stub_generate(monkeypatch, [], response="summary text")

    _run(chat_memory.maybe_summarize("s1"))

    assert upserts == ["2026-01-01T00:05:00"], "watermark must be the newest folded-in message's created_at"


def test_empty_generated_summary_is_not_stored(monkeypatch):
    # A blank model response must not overwrite a session's summary with
    # nothing - better to keep whatever we had than store an empty string.
    monkeypatch.setattr(chat_memory, "count_messages", lambda sid: chat_memory.SUMMARIZE_TRIGGER_COUNT + 2)
    monkeypatch.setattr(
        chat_memory, "_aged_out_messages", lambda sid: [{"role": "user", "content": "x", "created_at": "t1"}]
    )
    monkeypatch.setattr(chat_memory, "get_summary", lambda sid: None)
    monkeypatch.setattr(chat_memory, "_upsert_summary", lambda sid, summary, watermark: 1 / 0)
    calls = []
    _stub_generate(monkeypatch, calls, response="   ")

    result = _run(chat_memory.maybe_summarize("s1"))

    assert result is None


# ---------------------------------------------------------------------------
# Prompt regression: conversation_context="" must be byte-identical to the
# pre-feature prompts, and a populated context must land right before the
# QUESTION line.
# ---------------------------------------------------------------------------


def test_general_knowledge_prompt_byte_identical_with_empty_context():
    from services.chat_service import GENERAL_KNOWLEDGE_PROMPT

    rendered = GENERAL_KNOWLEDGE_PROMPT.format(question="What is ML?", conversation_context="")
    expected = (
        "You are a helpful, knowledgeable assistant.\n\n"
        "Answer the following question directly using your own general knowledge.\n\n"
        "QUESTION: What is ML?\n\n"
        "ANSWER:"
    )
    assert rendered == expected


def test_general_knowledge_prompt_inserts_context_before_question():
    from services.chat_service import GENERAL_KNOWLEDGE_PROMPT

    context = chat_memory._format_context("earlier stuff", [{"role": "user", "content": "hi"}])
    rendered = GENERAL_KNOWLEDGE_PROMPT.format(question="Follow up?", conversation_context=context)

    assert context in rendered
    # The whole context block sits immediately before the QUESTION line.
    assert rendered.index(context) < rendered.index("QUESTION: Follow up?")


@pytest.mark.parametrize(
    "prompt_name",
    ["ANSWER_PROMPT", "SCHEMA_SUMMARY_PROMPT"],
)
def test_db_answer_prompts_have_no_context_artifacts_when_empty(prompt_name):
    # The database answer prompts live in db_query_service; with an empty
    # context they must not sprout any stray "CONVERSATION SO FAR" header
    # or blank-line artifact that would change what the model sees.
    import services.db_query_service as dqs

    template = getattr(dqs, prompt_name)
    rendered = template.format(
        question="Q",
        results="[]" if prompt_name == "ANSWER_PROMPT" else None,
        schema="Table t(a int)" if prompt_name == "SCHEMA_SUMMARY_PROMPT" else None,
        conversation_context="",
        language_instruction="",
    )

    assert "CONVERSATION SO FAR" not in rendered
    assert "\n\nQUESTION: Q" in rendered


def test_pdf_and_hybrid_prompts_render_with_context():
    from services.chat_service import PDF_ANSWER_PROMPT, HYBRID_ANSWER_PROMPT

    context = chat_memory._format_context("prior", [{"role": "assistant", "content": "ok"}])

    pdf = PDF_ANSWER_PROMPT.format(
        question="Q", pdf_context="doc text", conversation_context=context, language_instruction=""
    )
    assert context in pdf
    assert pdf.index(context) < pdf.index("QUESTION: Q")

    hybrid = HYBRID_ANSWER_PROMPT.format(
        question="Q",
        pdf_context="doc",
        db_context="db",
        db_sql="SELECT 1",
        db_rows="[]",
        conversation_context=context,
        language_instruction="",
    )
    assert context in hybrid
    assert hybrid.index(context) < hybrid.index("QUESTION: Q")
