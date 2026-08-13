"""
Classifies a chat message as a database question, a question about an
uploaded PDF, a question needing both, or a general knowledge question.

Classification is plain keyword/phrase matching, not a Gemini call - the
database triggers below are exhaustive enough that a classifier call per
message would only add cost and latency for no real gain, and the same is
true of the document-reference phrases used once a PDF is active. Anything
that isn't recognized as a database or document question falls back to
GENERAL_KNOWLEDGE.

DECISION TABLE - routing priority (see classify_intent for the actual code)
-----------------------------------------------------------------------

Two signals are computed per question, then combined in a fixed order:

  * is_doc  - is this question about the attached PDF? True if either
              a DOCUMENT_REFERENCE_PHRASES word appears ("document",
              "pdf", ...) OR the question scores above threshold against
              the PDF's own content (routes/chat.py's pdf_relevant, via
              pdf_retrieval.top_score() - catches questions that never
              say "document" at all, e.g. "What is the expected payment
              amount?" while a budget PDF is attached).
  * is_db   - is this a database question? See _is_database_query()'s
              three tiers: (1) a standalone phrase like "how many" or
              "records", (2) a live table name appearing in the text, or
              (3) a live column name (e.g. "amount") paired with a
              generic shape phrase ("what is the"). Tier 3 is the
              weakest signal - it's common enough (nearly every table
              has an "amount" column) that it's excluded when a document
              is already relevant (see "strong_only" below), so it can't
              single-handedly drag a PDF question into HYBRID_QUERY.

Priority, evaluated in this exact order:

  1. No document attached (has_document=False):
       is_db (all 3 tiers)  ->  DATABASE_QUERY
       otherwise             ->  GENERAL_KNOWLEDGE
     (PDF_QUERY/HYBRID_QUERY are structurally impossible here - this is
     also the exact pre-PDF-feature behavior, unchanged.)

  2. Document attached (has_document=True):
       a. is_doc first.
          - If True: only a STRONG is_db (tiers 1-2 only, via
            _is_database_query(..., strong_only=True)) can escalate:
                is_db (strong)  ->  HYBRID_QUERY
                otherwise       ->  PDF_QUERY
          - If False: falls through to a plain is_db check (all 3
            tiers, same as the no-document case):
                is_db  ->  DATABASE_QUERY
                otherwise -> GENERAL_KNOWLEDGE

  In short: once a document is relevant, database relevance has to be
  UNAMBIGUOUS (a real trigger phrase or an actual table name - not just
  a generic column + generic phrasing) to justify pulling in the DB too.
  This is deliberate: false positives on tier 3 are common ("amount",
  "name", "date" all exist as real columns), so requiring a strong
  signal keeps PDF-only questions from being needlessly routed to
  HYBRID_QUERY just because they mention a common business word.

Representative examples (verified against this project's live schema and
a real test PDF containing "Budgeted amount for MHS166 campaign: Rs.
400000" - see backend's own verification scripts, not committed here):

  +----------------------------------------------------------+-------+---------------------+
  | Question                                                  | doc?  | Result              |
  +----------------------------------------------------------+-------+---------------------+
  | "What is machine learning?"                                | no    | GENERAL_KNOWLEDGE   |
  | "How many income categories are there?"                    | no    | DATABASE_QUERY      |
  | "What is machine learning?"                                | yes   | GENERAL_KNOWLEDGE   |
  | "What are the available income categories?"                | yes   | DATABASE_QUERY      |
  | "What does the document say about MHS166?"                 | yes   | PDF_QUERY           |
  | "What is the expected payment amount?"                     | yes   | PDF_QUERY           |
  | "According to the document and payment records, how much   | yes   | HYBRID_QUERY        |
  |  does MHS166 still need to pay?"                            |       |                     |
  +----------------------------------------------------------+-------+---------------------+

  Why row 4 ("available income categories") stays DATABASE_QUERY, not
  PDF_QUERY: is_doc is False (no document-reference phrase, and the
  question shares no real vocabulary with a finance-budget PDF, so
  pdf_relevant scores ~0) - it falls straight to the plain is_db check.

  Why row 6 ("expected payment amount") is PDF_QUERY, not HYBRID_QUERY,
  even though "amount" is a real column and "what is the" is a real
  shape phrase: that's exactly the weak tier-3 signal strong_only is
  built to ignore once is_doc is already True.

  Why row 7 is HYBRID_QUERY: "document" makes is_doc True, and "records"
  is a tier-1 DATABASE_QUERY_PHRASES trigger - a strong signal, so it
  clears the higher bar strong_only requires.
"""

import re

GENERAL_KNOWLEDGE = "general_knowledge"
DATABASE_QUERY = "database_query"
PDF_QUERY = "pdf_query"
HYBRID_QUERY = "hybrid_query"

# Only checked when a document is actually attached (see classify_intent's
# has_document) - otherwise "document"/"pdf" in a general-knowledge
# question (e.g. "what is a pdf file?") would be misrouted with nothing
# to actually answer from. Deliberately broad substrings, same style as
# DATABASE_QUERY_PHRASES: these describe *referring to the upload*, not
# any particular document's content, so they don't need to be schema- or
# document-specific to be a reliable signal.
#
# "image"/"photo"/"picture"/"scan" cover direct image uploads and
# scanned/OCR'd PDFs (see services/image_service.py and
# pdf_service.extract_text_via_ocr) - an uploaded document isn't
# necessarily a PDF anymore, so "what does the image say about X" needs
# the same recognition "what does the document say about X" already
# gets.
DOCUMENT_REFERENCE_PHRASES = [
    "document", "pdf", "the file", "uploaded", "attachment", "attached",
    "the report", "image", "photo", "picture", "scan",
]

# Triggers for questions about the connected database, regardless of what
# tables/columns actually exist - aggregation/reporting words like these
# are never table- or column-specific, so they're always a strong signal
# on their own.
#
# The business-metric words below (revenue/profit/loss/cost/orders/
# discount) are a deliberate second category: they're near-universal
# business-analytics vocabulary, but they are usually *computed* values
# (e.g. revenue = quantity x selling_price x (1 - discount/100) - they
# don't exist as a literal "revenue" column anywhere), so no amount of
# schema introspection will ever find them - unlike table/column names,
# which are matched dynamically in _is_database_query() below, these
# have to be listed explicitly because nothing in the database can ever
# tell us "revenue" and "profit" are meaningful business terms. This is
# schema-agnostic (not specific to this project's actual table names,
# e.g. this schema has no "orders" table - it's called "sales" - "orders"
# is still listed because that's how many other schemas would name it).
DATABASE_QUERY_PHRASES = [
    "how many",
    "total",
    "average",
    "count of",
    "database",
    "records",
    "the table",
    "revenue",
    "profit",
    "loss",
    "cost",
    "costs",
    "orders",
    "discount",
]

# Question "shapes" common to record lookup/filter/sort requests -
# "show students older than 20" or "which department does student 3
# belong to". On their own these are too generic to mean "database"
# (e.g. "which" or "show" appear in all kinds of questions), so
# _is_database_query() only treats a shape-phrase match as a signal when
# it's ALSO paired with an actual column name from the live schema
# (see database_terms below) - the pairing is what keeps this from
# firing on ordinary general-knowledge questions that happen to say
# "show" or "which".
DATABASE_QUESTION_SHAPE_PHRASES = [
    "show", "list", "which", "what is the", "what are the", "who is the",
    "belong to", "belongs to", "with id", "older than", "younger than",
    "greater than", "less than", "more than", "at least", "at most",
    "sorted by", "order by", "highest", "lowest", "top ",
]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _plural_variants(word):
    """Return {word} plus its likely singular/plural counterpart(s).

    Schema terms (see db_client._load_schema()) are stored as singular
    column-name fragments ("category" from category_name/category_raw)
    and largely-unpluralized table names, but a real question just as
    often uses the plural ("categories", "payments", "customers") - so
    without this, a word that's plainly the same schema concept fails a
    literal set-intersection check purely over an -s. Deliberately just
    the two common English patterns (trailing "ies" <-> "y", trailing
    "s" <-> bare), not a full stemmer - good enough since the result is
    only ever used to intersect against the schema's own vocabulary
    (see _is_database_query), never treated as a match by itself.
    """

    variants = {word}

    if word.endswith("ies") and len(word) > 4:
        variants.add(word[:-3] + "y")  # categories -> category
    elif word.endswith("y") and len(word) > 2:
        variants.add(word[:-1] + "ies")  # category -> categories

    if word.endswith("s") and not word.endswith("ss") and len(word) > 1:
        variants.add(word[:-1])  # incomes/expenses/payments/... -> singular
    else:
        variants.add(word + "s")  # income/expense/payment/... -> plural

    return variants


def _is_database_query(lowered, database_terms=None, strong_only=False):
    """database_terms, when given, is {"tables": set[str], "columns":
    set[str]} introspected live from the connected database (see
    db_client.get_schema_terms()) - nothing here is hardcoded to any
    particular table/column name, so this keeps working unchanged if
    tables are added, removed, or renamed.

    Three tiers, from strongest to weakest signal:
    1. DATABASE_QUERY_PHRASES - standalone triggers, no schema needed.
    2. A table name appearing in the question ("students", or "student"
       via the naive-singular match already done when the terms were
       built) - a strong signal on its own, since table names tend to be
       distinctive nouns.
    3. A column name (e.g. "department", "age") appearing ALONGSIDE a
       DATABASE_QUESTION_SHAPE_PHRASES match - column names alone are
       often ordinary English words ("name", "city"), so this tier
       requires both, to avoid misrouting unrelated questions that
       happen to share a word with a column.

    strong_only=True skips tier 3. A generic column name like "amount"
    exists in almost every table this project has, so pairing it with an
    equally generic shape phrase ("what is the") is weak enough that it
    shouldn't, on its own, drag an otherwise PDF-only question (e.g.
    "What is the expected payment amount?" while a PDF is attached) into
    HYBRID_QUERY - see classify_intent, which only applies strong_only
    once a document is already known to be relevant.

    Falls back to tier 1 only when database_terms isn't available (DB
    not configured/reachable) - same graceful-degradation shape as every
    other DB-dependent path in this project.
    """

    if any(phrase in lowered for phrase in DATABASE_QUERY_PHRASES):
        return True

    if not database_terms:
        return False

    words = set()
    for word in _WORD_RE.findall(lowered):
        words |= _plural_variants(word)

    if words & database_terms.get("tables", set()):
        return True

    if strong_only:
        return False

    if words & database_terms.get("columns", set()):
        if any(phrase in lowered for phrase in DATABASE_QUESTION_SHAPE_PHRASES):
            return True

    return False


def _is_document_query(lowered):
    return any(phrase in lowered for phrase in DOCUMENT_REFERENCE_PHRASES)


def classify_intent(text, database_terms=None, has_document=False, pdf_relevant=False):
    """Return DATABASE_QUERY, PDF_QUERY, HYBRID_QUERY, or GENERAL_KNOWLEDGE.

    database_terms: optional {"tables": set[str], "columns": set[str]}
    introspected live from the connected database (routes/chat.py fetches
    this via db_query_service.get_routing_terms() before calling in) -
    lets _is_database_query() recognize record lookups/filters/sorts
    ("show students older than 20") without any schema-specific keyword
    living in this file. Omit it (or pass None) to fall back to the
    schema-independent DATABASE_QUERY_PHRASES only - e.g. in tests, or
    when the database isn't configured.

    has_document: whether a PDF is currently attached to this
    conversation (routes/chat.py passes this based on whether the
    request's document_id resolves to a live document_store entry).
    Defaults to False, which reproduces the exact pre-PDF-feature
    behavior (only DATABASE_QUERY/GENERAL_KNOWLEDGE are ever returned) -
    document-reference phrases are only checked at all once a document is
    actually attached, so a plain "what is a pdf file?" with nothing
    uploaded still falls through to GENERAL_KNOWLEDGE as before.

    pdf_relevant: whether the question actually scores as relevant to the
    attached PDF's content (routes/chat.py computes this via
    pdf_retrieval.top_score() against a small threshold). Needed because
    real questions about an uploaded document often don't say
    "document"/"pdf" at all ("What is the expected payment amount?") -
    DOCUMENT_REFERENCE_PHRASES alone would miss those and let them fall
    through to GENERAL_KNOWLEDGE (or a weak DB match) instead of
    PDF_QUERY. Once a document is relevant by either signal, only a
    *strong* DB signal (tier 1/2 - see _is_database_query's strong_only)
    is enough to escalate to HYBRID_QUERY; the weak tier-3 column+shape
    signal alone is deliberately ignored here, since a generic column
    like "amount" paired with "what is the" is common enough to fire on
    plain PDF questions too.
    """

    lowered = text.lower().strip()

    if not has_document:
        is_db = _is_database_query(lowered, database_terms)
        return DATABASE_QUERY if is_db else GENERAL_KNOWLEDGE

    is_doc = _is_document_query(lowered) or pdf_relevant

    if is_doc:
        is_db_strong = _is_database_query(lowered, database_terms, strong_only=True)
        return HYBRID_QUERY if is_db_strong else PDF_QUERY

    is_db = _is_database_query(lowered, database_terms)
    return DATABASE_QUERY if is_db else GENERAL_KNOWLEDGE
