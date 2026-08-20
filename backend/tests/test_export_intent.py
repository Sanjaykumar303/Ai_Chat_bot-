# services/export_intent.py's regex-based classification - deliberately
# includes near-miss negative cases (see test_db_query_service_branching.py's
# own precedent for why: a regex/guard is only proven by testing what it
# should REJECT, not just what it should accept) so an export detector
# never silently swallows a normal chat/DB/PDF question.

from services import export_intent


def test_docx_examples_from_the_feature_request_are_all_detected():
    assert export_intent.detect_export_format("convert the summary into a document") == export_intent.DOCX
    assert export_intent.detect_export_format("make this into a Word document") == export_intent.DOCX
    assert export_intent.detect_export_format("download this answer as a document") == export_intent.DOCX


def test_xlsx_examples_from_the_feature_request_are_all_detected():
    assert export_intent.detect_export_format("give me last month's income in Excel") == export_intent.XLSX
    assert export_intent.detect_export_format("export last month's income") == export_intent.XLSX
    assert export_intent.detect_export_format("give me the records in Excel format") == export_intent.XLSX


def test_ordinary_questions_are_not_misdetected_as_exports():
    # The adversarial cases: real questions this app already answers
    # every day, none of which should ever be diverted into the export
    # pipeline.
    ordinary_questions = [
        "What is the total profit today?",
        "What is a database?",
        "summarize this document",
        "what does this document say about payment terms?",
        "show me students older than 20",
        "generate a profit report for this month",
        "export the pdf",  # export + a noun not on the data-noun list
        # Adversarial cases for the summarize/show/explain/tell-me verbs
        # added below - each contains the word "document" AND one of
        # those verbs, but never with "into/as/to" directly before
        # "document" the way a real export request has it, so none of
        # these should match either.
        "refer to the document for pricing details",
        "according to the document, what is the total?",
        "explain the document to me",
        "tell me about the document",
    ]
    for question in ordinary_questions:
        assert export_intent.detect_export_format(question) is None, question


def test_docx_first_message_combined_ask_and_format_requests_are_detected():
    # Real, reported bug: a user's very FIRST message in a fresh chat can
    # combine a new question with a requested output format - there's no
    # previous_answer yet to convert, but the request should still be
    # recognized as an export request (so chat_service.answer_docx_export
    # gives its own honest "ask a question first" message - see
    # test_chat_service_export.py's test_docx_export_with_no_previous_
    # answer_does_not_fabricate_a_file) rather than silently falling
    # through to the ordinary chat pipeline, where Gemini would try to
    # answer the question AND separately hallucinate that this app can't
    # produce documents at all (it can - see docx_export.py).
    assert export_intent.detect_export_format("Summarize the profit trend as a Word document") == export_intent.DOCX
    assert export_intent.detect_export_format("Show me the profit trend as a document") == export_intent.DOCX
    assert export_intent.detect_export_format("Explain the profit trend as a Word document") == export_intent.DOCX
    assert export_intent.detect_export_format("Tell me the profit trend as a document") == export_intent.DOCX


def test_docx_phrasing_does_not_also_match_as_xlsx():
    for phrase in [
        "convert the summary into a document",
        "make this into a Word document",
        "download this answer as a document",
    ]:
        assert export_intent.detect_export_format(phrase) == export_intent.DOCX


def test_strip_xlsx_phrase_leaves_the_underlying_question_intact():
    assert export_intent.strip_xlsx_phrase("give me last month's income in Excel") == "give me last month's income"
    assert export_intent.strip_xlsx_phrase("export last month's income") == "last month's income"
    assert export_intent.strip_xlsx_phrase("give me the records in Excel format") == "give me the records"


def test_strip_xlsx_phrase_falls_back_to_original_if_nothing_would_remain():
    assert export_intent.strip_xlsx_phrase("export") == "export"
