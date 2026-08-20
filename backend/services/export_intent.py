"""
Detects a request to export the CURRENT interaction as a downloadable
file - layered on top of, not inside, intent_router.py's own
DATABASE_QUERY/PDF_QUERY/HYBRID_QUERY/GENERAL_KNOWLEDGE classification.

routes/chat.py checks detect_export_format() FIRST, before any of the
existing follow-up rewriting/PDF relevance scoring/resolve_intent() logic
runs at all. If it matches, the request short-circuits to
services/docx_export.py or services/xlsx_export.py instead of the normal
answer_* pipeline; if it doesn't match (every existing question, typed or
spoken, in this app so far), every line of the existing routing logic
runs completely unchanged. Nothing in this module reads or writes
anything intent_router.py itself touches.
"""

import re

DOCX = "docx"
XLSX = "xlsx"

# "convert the summary into a document", "make this into a Word
# document", "download this answer as a document", "give me this as a
# doc" - references what's ALREADY been said in this chat, not a new
# question, so services/docx_export.py never calls Gemini; it just
# formats the previous answer text routes/chat.py is given (see
# ChatRequest.previous_answer).
#
# Also matches a combined ask+format request with NO prior answer yet -
# "summarize the profit trend as a Word document", "show me X as a
# document" - real, observed phrasing (a user's very first message in a
# fresh chat). This app's export pipeline only ever formats an EXISTING
# previous_answer (see answer_docx_export's own docstring - it never
# calls Gemini), so it still can't answer a brand-new question and
# export it in one step; what changed is that this phrasing now reaches
# that function at all, so a missing previous_answer gets chat_service's
# own honest "ask a question first, then ask to export it" message
# instead of silently falling through to the ordinary chat pipeline -
# where Gemini, seeing "...as a Word document" as part of the question
# itself with no idea this app has real export tooling, would answer the
# question badly (dumping raw schema/table names for a vague DB-shaped
# question like this) AND falsely claim it can't produce a document at
# all - a real, reported bug (see project_ux_review_2026-08-20 memory).
# summarize/show/explain/tell me were added carefully, not as a generic
# catch-all: each still requires "into/as/to" positioned directly before
# the document-phrase (see test_export_intent.py's adversarial cases -
# "refer to the document", "according to the document", "explain the
# document to me" - none of them have "into/as/to" immediately before
# the word "document", so none of them false-positive here).
_DOCX_RE = re.compile(
    r"\b(convert|turn|make|export|download|save|give\s+me|summarize|show|explain|tell\s+me)\b"
    r".{0,40}\b(into|as|to)\b.{0,15}\b"
    r"(word\s*doc(?:ument)?s?|\.docx|docx|documents?)\b",
    re.IGNORECASE,
)

# "give me last month's income in Excel", "export last month's income",
# "give me the records in Excel format" - a real spreadsheet request.
# Distinguished from the DOCX case above by naming excel/xlsx/spreadsheet
# directly, or by "export" paired with a data-shaped noun rather than
# "document"/"word doc".
_XLSX_RE = re.compile(
    r"\b(excel|xlsx|spreadsheet)\b"
    r"|\bexport\b.{0,30}\b(record|data|row|income|revenue|expense|sale|invoice|report)s?\b",
    re.IGNORECASE,
)

# Strips the export-format phrasing back out before the remaining text is
# handed to the EXISTING SQL-generation prompt (services/db_query_service.py,
# untouched by this feature) - "last month's income in Excel" becomes
# "last month's income", exactly the question that would have been asked
# without the export request, so SQL generation sees nothing new.
_XLSX_PHRASE_RE = re.compile(
    r"\b(in|as|to)\s+(an?\s+)?(excel|xlsx|spreadsheet)(\s+format)?\b"
    r"|\bexport\b"
    r"|\bdownload\b",
    re.IGNORECASE,
)


def detect_export_format(question):
    """Returns DOCX, XLSX, or None - never raises, never calls Gemini or
    the database. DOCX is checked first: a request naming both "document"
    and "excel" in the same sentence is unlikely, but if it ever happens
    the more specific "turn this answer into a document" phrasing should
    win over a bare "export" keyword."""

    if _DOCX_RE.search(question):
        return DOCX
    if _XLSX_RE.search(question):
        return XLSX
    return None


def strip_xlsx_phrase(question):
    """"last month's income in Excel" -> "last month's income" - see
    _XLSX_PHRASE_RE above. Falls back to the original text untouched if
    stripping would leave nothing usable (defensive; detect_export_format
    already required export-format wording to be present, not the whole
    question, so this should never actually happen)."""

    cleaned = _XLSX_PHRASE_RE.sub(" ", question)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.!")
    return cleaned or question
