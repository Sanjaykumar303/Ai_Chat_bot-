# classify_database_question() is a second-stage classifier, applied
# only after classify_intent() has already decided a question is
# DATABASE_QUERY - see intent_router.py's module comment above
# _SCHEMA_STRUCTURE_RE for why a schema/overview question, a capability
# question, and an actual data question can't share the same answer
# machinery.

from services.intent_router import (
    classify_database_question,
    DATABASE_QUESTION_CAPABILITY,
    DATABASE_QUESTION_DATA,
    DATABASE_QUESTION_SCHEMA,
    DATABASE_QUESTION_WHY,
)


def test_summary_the_db_is_a_schema_question():
    assert classify_database_question("summary the db") == DATABASE_QUESTION_SCHEMA


def test_summary_of_my_database_is_a_schema_question():
    assert classify_database_question("give me the summary of my database") == DATABASE_QUESTION_SCHEMA


def test_what_are_tables_is_a_schema_question():
    assert classify_database_question("what are tables in my db?") == DATABASE_QUESTION_SCHEMA


def test_what_data_do_we_have_is_a_schema_question():
    # A vague "what's in there" question is answered better as an
    # overview of what's available than as one specific metric.
    assert classify_database_question("what data do we have?") == DATABASE_QUESTION_SCHEMA


def test_column_question_is_a_schema_question():
    assert classify_database_question("what columns are in the sales table?") == DATABASE_QUESTION_SCHEMA


def test_can_you_connect_is_a_capability_question():
    assert classify_database_question("Can you connect to our DB?") == DATABASE_QUESTION_CAPABILITY


def test_are_you_connected_is_a_capability_question():
    assert classify_database_question("Are you connected to the database?") == DATABASE_QUESTION_CAPABILITY


def test_do_you_have_access_is_a_capability_question():
    assert classify_database_question("Do you have access to our database?") == DATABASE_QUESTION_CAPABILITY


def test_how_many_customers_is_a_data_question():
    assert classify_database_question("how many customers are there?") == DATABASE_QUESTION_DATA


def test_total_revenue_is_a_data_question():
    assert classify_database_question("What is the total revenue this month?") == DATABASE_QUESTION_DATA


def test_capability_takes_priority_over_schema_vocabulary():
    # "see" + "tables" could plausibly match either check - it's asking
    # about ability, not content, so it has to resolve to CAPABILITY.
    assert classify_database_question("Can you see our tables?") == DATABASE_QUESTION_CAPABILITY


# --- DATABASE_QUESTION_WHY - the user's own literal example questions --


def test_why_is_profit_low_is_a_why_question():
    assert classify_database_question("Why is profit low?") == DATABASE_QUESTION_WHY


def test_why_did_revenue_drop_is_a_why_question():
    assert classify_database_question("Why did revenue drop?") == DATABASE_QUESTION_WHY


def test_what_caused_the_expense_increase_is_a_why_question():
    assert classify_database_question("What caused the expense increase?") == DATABASE_QUESTION_WHY


def test_what_is_the_reason_for_the_loss_is_a_why_question():
    assert classify_database_question("What is the reason for the loss?") == DATABASE_QUESTION_WHY


def test_what_led_to_the_increase_in_costs_is_a_why_question():
    assert classify_database_question("What led to the increase in costs?") == DATABASE_QUESTION_WHY


def test_bare_reason_clause_without_for_is_still_a_why_question():
    # "the reason revenue dropped" (no "for"/"behind") is just as much a
    # causal question as "the reason FOR the drop" - the clause itself
    # names what needs explaining.
    assert classify_database_question("What is the reason revenue dropped?") == DATABASE_QUESTION_WHY


def test_reason_as_a_literal_column_name_is_not_a_why_question():
    # "reason code"/"reason number" are realistic column names in other
    # schemas even though this project's own doesn't have one - asking
    # for that column's VALUE is a DATA question, not a causal one, so
    # the WHY regex must not swallow every sentence containing "reason".
    assert classify_database_question("What is the reason code for this transaction?") == DATABASE_QUESTION_DATA


def test_a_plain_value_question_is_still_a_data_question_not_why():
    # Regression guard: adding WHY must not steal any DATA question that
    # merely shares vocabulary ("increase", a metric name) without
    # actually asking why.
    assert classify_database_question("What is the total revenue this month?") == DATABASE_QUESTION_DATA


def test_capability_still_takes_priority_over_why_vocabulary():
    # "can you explain why the database exists" is asking about ability/
    # scope, not requesting an actual causal data analysis - capability
    # has to win the same way it already wins over schema vocabulary.
    assert classify_database_question("Can you tell me why you're connected to the database?") == DATABASE_QUESTION_CAPABILITY
