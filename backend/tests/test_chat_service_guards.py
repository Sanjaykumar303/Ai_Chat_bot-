# _language_instruction is the one piece of deterministic logic that
# stayed in chat_service.py - the anti-fabrication guards that used to
# live here (_all_values_zero_or_null/_is_single_day_question/
# _sql_has_exact_day_filter) moved to services/db_query_service.py so
# they can pre-empt generation instead of replacing already-generated
# text - see tests/test_db_query_service_guards.py for their tests now,
# and db_query_service.py's own module comment for why they moved.

from services.chat_service import _language_instruction


def test_language_instruction_for_supported_language():
    instruction = _language_instruction("ta")
    assert "Tamil" in instruction


def test_language_instruction_empty_for_english_or_unknown():
    assert _language_instruction("en") == ""
    assert _language_instruction(None) == ""
    assert _language_instruction("xx") == ""


def test_language_instruction_handles_locale_suffix():
    # /transcribe can hand back a code like "ta-IN" - the language
    # lookup should still match on just the primary subtag.
    assert "Tamil" in _language_instruction("ta-IN")
