# The routing decision table these cases cover is documented in
# services/intent_router.py's own module docstring, including the
# "representative examples" table this test mirrors directly - keeping
# these in sync catches a routing regression before it reaches a live
# question.

from services.intent_router import (
    classify_intent,
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
