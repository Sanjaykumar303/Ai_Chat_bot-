# The routing decision table these cases cover is documented in
# services/intent_router.py's own module docstring, including the
# "representative examples" table this test mirrors directly - keeping
# these in sync catches a routing regression before it reaches a live
# question.

from services.intent_router import (
    classify_intent,
    is_ambiguous_followup,
    is_self_reference,
    resolve_intent,
    GENERAL_KNOWLEDGE,
    DATABASE_QUERY,
    PDF_QUERY,
    HYBRID_QUERY,
)

# Fragment-style, matching how db_client.get_schema_terms() actually
# builds this set in production: column names are split on "_" (e.g.
# "category_name" -> "category", "name"; "voucher_number" -> "voucher",
# "number"), not kept as whole compound strings - a fixture using whole
# compound names wouldn't exercise the same matching real questions do.
DATABASE_TERMS = {
    "tables": {"accounts_income", "tally_ledger", "voucher"},
    "columns": {"amount", "category", "name", "voucher", "number"},
}


def test_general_knowledge_with_no_document():
    assert classify_intent("What is machine learning?") == GENERAL_KNOWLEDGE


def test_database_phrase_trigger_with_no_document():
    assert classify_intent("How many income categories are there?") == DATABASE_QUERY


def test_general_knowledge_unaffected_by_an_attached_document():
    assert classify_intent("What is machine learning?", has_document=True) == GENERAL_KNOWLEDGE


def test_database_query_stays_database_query_even_with_a_document_attached():
    # is_doc is False here (no document-reference phrase, not
    # pdf_relevant) - falls straight through to the plain is_db check,
    # same as the no-document case.
    assert (
        classify_intent(
            "What are the available income categories?",
            database_terms=DATABASE_TERMS,
            has_document=True,
            pdf_relevant=False,
        )
        == DATABASE_QUERY
    )


def test_explicit_document_reference_routes_to_pdf_query():
    assert (
        classify_intent("What does the document say about MHS166?", has_document=True)
        == PDF_QUERY
    )


def test_weak_tier3_signal_alone_does_not_escalate_to_hybrid():
    # "amount" (column) + "what is the" (shape phrase) is exactly the
    # weak tier-3 signal strong_only is built to ignore once a document
    # is already relevant - see intent_router.py's own worked example.
    assert (
        classify_intent(
            "What is the expected payment amount?",
            database_terms=DATABASE_TERMS,
            has_document=True,
            pdf_relevant=True,
        )
        == PDF_QUERY
    )


def test_strong_database_signal_escalates_to_hybrid():
    # "records" is a tier-1 DATABASE_QUERY_PHRASES trigger - strong
    # enough to clear strong_only's higher bar.
    assert (
        classify_intent(
            "According to the document and payment records, how much does MHS166 still need to pay?",
            database_terms=DATABASE_TERMS,
            has_document=True,
            pdf_relevant=True,
        )
        == HYBRID_QUERY
    )


def test_table_name_singular_plural_matching():
    # _plural_variants() should match "vouchers" (plural, in the
    # question) against "voucher" (singular, in the schema terms) -
    # tier 2 fires on a bare table-name match, no shape phrase needed.
    assert (
        classify_intent(
            "show all vouchers",
            database_terms=DATABASE_TERMS,
        )
        == DATABASE_QUERY
    )


def test_falls_back_to_phrase_only_matching_without_database_terms():
    # No live schema available (DB unreachable) - only
    # DATABASE_QUERY_PHRASES should still work.
    assert classify_intent("What is the total revenue?") == DATABASE_QUERY
    assert classify_intent("What is the amount?") == GENERAL_KNOWLEDGE


# --- "refers to this app's connected database" regression cases -------
#
# These four were misrouted to GENERAL_KNOWLEDGE before _OUR_DATABASE_RE/
# _OWNED_DATA_RE were added: none of them contain a DATABASE_QUERY_PHRASES
# trigger or a schema table/column name, and only "database" (a plain
# substring match that would also wrongly fire on "What is a database?")
# used to cover any of this.


def test_db_abbreviation_with_possessive_routes_to_database():
    assert classify_intent("summary the db") == DATABASE_QUERY


def test_full_word_database_with_possessive_routes_to_database():
    assert classify_intent("give me the summary of my database") == DATABASE_QUERY


def test_tables_in_my_db_routes_to_database():
    assert classify_intent("what are tables in my db?") == DATABASE_QUERY


def test_what_data_do_we_have_routes_to_database():
    assert classify_intent("what data do we have?") == DATABASE_QUERY


def test_our_db_capability_question_routes_to_database():
    assert classify_intent("Can you connect to our DB?") == DATABASE_QUERY


# --- generic "database" concept questions must NOT route to the DB ----
#
# "database" used to be a bare DATABASE_QUERY_PHRASES substring, so any
# question containing the word - including one that's actually asking
# for a dictionary definition - was sent down the text-to-SQL pipeline.


def test_what_is_a_database_stays_general_knowledge():
    assert classify_intent("What is a database?") == GENERAL_KNOWLEDGE


def test_what_does_db_mean_stays_general_knowledge():
    assert classify_intent("What does DB mean?") == GENERAL_KNOWLEDGE


def test_define_database_stays_general_knowledge():
    assert classify_intent("Please define database for me") == GENERAL_KNOWLEDGE


def test_difference_between_databases_stays_general_knowledge():
    assert classify_intent("What's the difference between a database and a spreadsheet?") == GENERAL_KNOWLEDGE


def test_ordinary_general_knowledge_questions_are_unaffected():
    # Sanity check that none of the new regexes over-fire on plain
    # questions that share no vocabulary with "database" at all.
    assert classify_intent("What is machine learning?") == GENERAL_KNOWLEDGE
    assert classify_intent("How does photosynthesis work?") == GENERAL_KNOWLEDGE


# --- "earn"/"earnings" business-metric vocabulary ----------------------
#
# "How much did we earn?" has no other database signal at all (no
# table/column word, no other DATABASE_QUERY_PHRASES trigger) - it needs
# its own regex, not a bare substring, since "earn" as plain substring
# also matches inside "learn"/"learning"/"learned".


def test_earn_routes_to_database():
    assert classify_intent("How much did we earn?") == DATABASE_QUERY
    assert classify_intent("What did we earn last month?") == DATABASE_QUERY
    assert classify_intent("Show me our earnings") == DATABASE_QUERY


def test_earn_does_not_false_positive_on_learn():
    # The actual regression caught by this project's own test suite
    # while building the "earn" fix: a bare substring match on "earn"
    # also fires inside "learning".
    assert classify_intent("What is machine learning?") == GENERAL_KNOWLEDGE
    assert classify_intent("I want to learn about history") == GENERAL_KNOWLEDGE
    assert classify_intent("She learned a new skill") == GENERAL_KNOWLEDGE


# --- follow-up routing priority ----------------------------------------
#
# The bug: a question is classified using the PREVIOUS question's own
# topic when it should be classified on its own words first. Fixed by
# resolve_intent - classify_intent (unchanged, still exhaustively tested
# above) always runs first; only an ambiguous/topic-less follow-up ever
# falls back to the previous question's classification. See
# intent_router.py's own module comment above is_ambiguous_followup for
# the full design rationale.


def test_new_question_with_its_own_topic_is_classified_independently():
    # A previous GENERAL_KNOWLEDGE question must never drag a real,
    # standalone database question down with it - each of these has its
    # own recognizable topic and needs no context at all.
    previous = "What is My Health School?"
    assert resolve_intent("Get me the profit", previous, None) == DATABASE_QUERY
    assert resolve_intent("What is today's profit?", previous, None) == DATABASE_QUERY
    assert resolve_intent("How much did we earn?", previous, None) == DATABASE_QUERY
    assert resolve_intent("My Health School?", previous, None) == GENERAL_KNOWLEDGE


def test_is_ambiguous_followup_recognizes_bare_referent_questions():
    assert is_ambiguous_followup("Tell me more about it") is True
    assert is_ambiguous_followup("What about it?") is True
    assert is_ambiguous_followup("What about that?") is True


def test_is_ambiguous_followup_false_for_questions_with_their_own_topic():
    # A question that names its own subject is never ambiguous, however
    # short it is - only a question with NO topic of its own qualifies.
    assert is_ambiguous_followup("My Health School?") is False
    assert is_ambiguous_followup("Get me the profit") is False
    assert is_ambiguous_followup("What is today's profit?") is False
    assert is_ambiguous_followup("How much did we earn?") is False
    assert is_ambiguous_followup("What is machine learning?") is False


def test_ambiguous_followup_inherits_previous_database_intent():
    previous = "What is our total revenue?"
    assert resolve_intent("Tell me more about it", previous, None) == DATABASE_QUERY
    assert resolve_intent("What about it?", previous, None) == DATABASE_QUERY


def test_ambiguous_followup_inherits_previous_general_knowledge_intent():
    previous = "What is My Health School?"
    assert resolve_intent("Tell me more about it", previous, None) == GENERAL_KNOWLEDGE
    assert resolve_intent("What about it?", previous, None) == GENERAL_KNOWLEDGE


def test_ambiguous_followup_without_a_previous_question_stays_general_knowledge():
    # No context to inherit - same fallback classify_intent alone would
    # already give, not a crash or a guess.
    assert resolve_intent("Tell me more about it", None, None) == GENERAL_KNOWLEDGE
    assert resolve_intent("What about it?", None, None) == GENERAL_KNOWLEDGE


def test_resolve_intent_matches_classify_intent_when_not_ambiguous():
    # For every non-ambiguous question, resolve_intent must be a pure
    # passthrough to classify_intent - previous_question should never be
    # consulted at all.
    assert resolve_intent("What is machine learning?", "What is our total revenue?", None) == GENERAL_KNOWLEDGE
    assert resolve_intent("How many income categories are there?", "irrelevant", None) == DATABASE_QUERY


# --- first-person self-reference must not be misrouted to the database -
#
# The bug: a schema table name can also be an ordinary English noun
# ("department", "account", "product"), so a plain personal statement or
# question sharing that word used to fall into tier 2/3 and get routed to
# DATABASE_QUERY - even though it's clearly about the speaker themselves,
# not business records, and this app has no user-identity system to
# answer such a question from anyway. Found live via services/
# chat_memory.py's own verification: "My name is Alex and I manage the
# Marketing department." answered "I couldn't answer that from the
# database." instead of being treated as ordinary conversation.

_DEPARTMENT_TERMS = {
    "tables": {"departments", "department"},
    "columns": {"department", "name"},
}


def test_self_identification_statement_stays_general_knowledge():
    assert (
        classify_intent(
            "My name is Alex and I manage the Marketing department.",
            database_terms=_DEPARTMENT_TERMS,
        )
        == GENERAL_KNOWLEDGE
    )


def test_self_referential_question_stays_general_knowledge():
    assert (
        classify_intent(
            "What's my name, and which department do I manage?",
            database_terms=_DEPARTMENT_TERMS,
        )
        == GENERAL_KNOWLEDGE
    )


def test_who_am_i_and_call_me_stay_general_knowledge():
    assert classify_intent("Who am I?") == GENERAL_KNOWLEDGE
    assert classify_intent("You can call me Alex.") == GENERAL_KNOWLEDGE


def test_genuine_third_person_database_question_about_the_same_table_still_routes_correctly():
    # The fix must not over-fire: a real, third-person question sharing
    # the exact same table name as the self-reference examples above must
    # still route to DATABASE_QUERY - the veto is specifically about
    # first-person self-reference, not about the word "department" itself.
    assert (
        classify_intent("Which department has the highest headcount?", database_terms=_DEPARTMENT_TERMS)
        == DATABASE_QUERY
    )
    assert classify_intent("How many departments are there?", database_terms=_DEPARTMENT_TERMS) == DATABASE_QUERY


def test_explicit_our_database_reference_still_wins_over_self_reference():
    # Tier 0 (_OUR_DATABASE_RE) is checked before the self-reference veto,
    # same precedence _GENERIC_DB_CONCEPT_RE already has - an explicit
    # reference to the app's own database still routes to DATABASE_QUERY
    # even if the same sentence also happens to contain "am I".
    assert classify_intent("Am I connected to our database right now?") == DATABASE_QUERY


def test_is_self_reference_public_function():
    # Public specifically so services/entity_resolution.py can reuse this
    # exact signal (see its own extract_entity_candidates fix) - directly
    # exercised here so a future change to the wrapper itself (not just
    # the private regex it wraps) is caught.
    assert is_self_reference("My name is Alex and I manage the Marketing department.") is True
    assert is_self_reference("What's my name?") is True
    assert is_self_reference("Which department has the highest revenue?") is False
    assert is_self_reference("What is machine learning?") is False
