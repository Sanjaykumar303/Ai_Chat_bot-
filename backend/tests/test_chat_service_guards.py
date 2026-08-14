# These guards exist to catch a specific, previously-observed failure
# mode: Gemini confidently stating a wrong/fabricated answer from a
# database result that only *looks* valid (a zero from an unmatched
# filter, or a "today" query that never actually filtered to today) -
# see chat_service.py's own docstrings for the real incidents that led
# to each one.

from services.chat_service import (
    _all_values_zero_or_null,
    _is_single_day_question,
    _sql_has_exact_day_filter,
    _language_instruction,
)


def test_all_zero_or_null_true_for_an_unmatched_aggregate():
    # SUM(...)/COALESCE(...,0) over zero matching rows produces exactly
    # this shape - the trap the guard exists to catch.
    assert _all_values_zero_or_null([{"total": 0, "count": None}]) is True


def test_all_zero_or_null_false_when_any_real_value_present():
    assert _all_values_zero_or_null([{"total": 0}, {"total": 15000}]) is False


def test_all_zero_or_null_false_for_empty_rows():
    # No rows at all is a different case (query found nothing to
    # aggregate) from a row full of zeros (query matched something and
    # it happened to sum to zero) - only the latter is what this guard
    # is for.
    assert _all_values_zero_or_null([]) is False


def test_all_zero_or_null_treats_zero_point_zero_and_decimal_as_zero():
    assert _all_values_zero_or_null([{"a": 0.0, "b": None}]) is True


def test_single_day_question_detects_today_and_yesterday():
    assert _is_single_day_question("What is our profit today?") is True
    assert _is_single_day_question("How much did we make yesterday?") is True


def test_single_day_question_false_for_ranges():
    # Deliberately not meant to fire on ranges - those legitimately use
    # a comparison instead of an exact-date match.
    assert _is_single_day_question("What was revenue this week?") is False
    assert _is_single_day_question("What was revenue last month?") is False


def test_single_day_question_does_not_match_substrings():
    # "todays" or a word merely containing "today" shouldn't trigger a
    # false positive via the word-boundary regex.
    assert _is_single_day_question("What happened yesteryear?") is False


def test_sql_has_exact_day_filter_true_for_equality():
    assert _sql_has_exact_day_filter("WHERE voucher_date = CURRENT_DATE") is True


def test_sql_has_exact_day_filter_false_for_a_range_comparison():
    # The exact bug this guard was added for: a "<=" filter computes a
    # running total, not a single day's figure, but looks superficially
    # similar.
    assert _sql_has_exact_day_filter("WHERE voucher_date <= CURRENT_DATE") is False


def test_sql_has_exact_day_filter_false_when_theres_no_date_filter_at_all():
    assert _sql_has_exact_day_filter("SELECT SUM(amount) FROM accounts_income") is False


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
