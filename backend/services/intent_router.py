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

# ---------------------------------------------------------------------
# Referring to *this app's* database, vs. talking about databases
# ---------------------------------------------------------------------
#
# DATABASE_QUERY_PHRASES below matches plain substrings, which is fine
# for aggregation words ("how many", "average") that are only ever asked
# about real data. It is NOT fine for the word "database" itself, and
# that gap produced errors in both directions:
#
#   "summary the db"           -> GENERAL_KNOWLEDGE (missed: says "db", not "database")
#   "what are tables in my db?"-> GENERAL_KNOWLEDGE (missed: same reason)
#   "what data do we have?"    -> GENERAL_KNOWLEDGE (missed: never names the DB at all)
#   "What is a database?"      -> DATABASE_QUERY    (false positive: generic concept
#                                                    question, tried to generate SQL for it)
#
# What separates the two isn't the noun, it's whether the question points
# at a *specific, possessed* database ("my db", "our database", "the db")
# or at the concept in general ("a database", "what does DB mean"). The
# three regexes below encode exactly that distinction, and nothing about
# any particular schema.

# Abbreviations and spellings of the database noun itself.
_DB_NOUN = r"(?:db|dbs|databases?|data\s?bases?)"

# Nouns for the *contents* of a database, used by the ownership pattern
# below - "what data do we have?" never says "database" at all, but
# "data ... we have" is just as clearly about this app's records.
_DATA_NOUN = r"(?:data|records?|tables?|rows|entries|information|info)"

# A determiner that makes the database noun refer to one specific,
# already-known database: "my db", "our database", "the connected DB".
# Deliberately excludes the indefinite "a"/"an", which is what makes
# "what is a database?" a concept question - that single article is the
# entire difference between the two cases.
_OUR_DATABASE_RE = re.compile(
    r"\b(?:my|our|your|the|this|that|these|those|its|connected|attached|current|live)\s+"
    r"(?:\w+\s+){0,2}?"  # optional qualifiers: "the connected sql db"
    + _DB_NOUN + r"\b",
    re.IGNORECASE,
)

# First-person ownership of the *contents*, for questions that never name
# the database: "what data do we have?", "how much information do we
# store?", "our records".
_OWNED_DATA_RE = re.compile(
    r"\b(?:my|our)\s+(?:\w+\s+){0,2}?" + _DATA_NOUN + r"\b"
    r"|\b" + _DATA_NOUN + r"\b[^.?!]{0,40}?\b(?:we|i|you)\s+(?:have|store|keep|hold|got)\b",
    re.IGNORECASE,
)

# Definitional frames - the question is about what a database *is*, not
# about the one this app is connected to. Checked first and vetoes
# everything, so the bare "database" substring in DATABASE_QUERY_PHRASES
# can no longer drag a vocabulary question into the SQL pipeline.
_GENERIC_DB_CONCEPT_RE = re.compile(
    r"\bwhat\s+(?:is|are)\s+(?:an?\s+)?" + _DB_NOUN + r"\b"
    r"|\bwhat\s+does\s+" + _DB_NOUN + r"\s+(?:mean|stand\s+for)\b"
    r"|\b(?:define|explain)\s+(?:an?\s+)?" + _DB_NOUN + r"\b"
    r"|\bmeaning\s+of\s+(?:an?\s+)?" + _DB_NOUN + r"\b"
    r"|\bdifference\s+between\s+(?:an?\s+)?" + _DB_NOUN + r"\b",
    re.IGNORECASE,
)

# A question/statement about the SPEAKER'S OWN identity or role, not
# about this app's business records - "My name is Alex", "what's my
# name?", "which department do I manage?". A real, observed false
# positive once services/chat_memory.py let a conversation build up
# personal context across turns: several of this project's own table
# names are also perfectly ordinary English nouns ("department",
# "account", "product"), so a plain personal statement sharing one of
# those words used to fall straight into tier 2/3 below purely on that
# coincidence.
#
# This isn't just a style preference for THIS app specifically: there is
# no login/user-identity system at all (see [[project-known-issues-
# deferred]]) - no row anywhere maps "the current user" to anything, so a
# first-person question about the speaker's own attributes/relationship
# to the data ("do I...", "am I...", "which X do I...") can never be
# correctly answered by a real SQL query here, regardless of how it's
# phrased. Routing it to the database pipeline isn't just unlikely to
# help, it's structurally guaranteed to fail or fabricate.
#
# Checked in the same veto position as _GENERIC_DB_CONCEPT_RE just above
# (after the explicit "my database"/"our data" tier, before everything
# else) - an explicit reference to the app's own database still wins
# (e.g. "am I connected to our database?" matches _OUR_DATABASE_RE
# first), but a bare tier-2/3 table/column coincidence never overrides a
# clearly self-referential question.
_SELF_REFERENCE_RE = re.compile(
    r"\bmy\s+name\s+is\b"
    r"|\bwhat(?:'s|\s+is)\s+my\s+name\b"
    r"|\bwho\s+am\s+i\b"
    r"|\bcall\s+me\s+\w"
    r"|\bdo\s+i\b"
    r"|\bam\s+i\b"
    r"|\bi\s+(?:manage|run|lead|head|work\s+(?:in|for|at)|belong\s+to)\b",
    re.IGNORECASE,
)


def is_self_reference(text):
    """True if `text` is a first-person question/statement about the
    speaker's own identity or role - "My name is Alex", "what's my
    name?", "which department do I manage?" - rather than a question
    about this app's database or a request to identify some external
    named entity.

    Public (unlike the regex it wraps) specifically so
    services/entity_resolution.py can reuse this exact same signal:
    extract_entity_candidates() had its own related bug from the same
    root cause (a self-referential statement mentioning a schema-shaped
    word getting treated as if it named a real-world organization worth
    a web-search verification) - sharing one definition here keeps both
    fixes answering the same question ("is this about the speaker
    themselves?") the same way, rather than two regexes silently
    drifting apart over time.
    """

    return bool(_SELF_REFERENCE_RE.search(text.lower()))


# ---------------------------------------------------------------------
# Which *kind* of database question (see classify_database_question)
# ---------------------------------------------------------------------
#
# This is a SECOND-STAGE classification, applied only after classify_intent
# has already decided a question is DATABASE_QUERY - it doesn't add a new
# top-level intent, and routes/chat.py's dispatch (DATABASE_QUERY ->
# chat_service.answer_database_query, unchanged) doesn't need to know
# about it. db_query_service.answer_database_question calls it to decide
# HOW to answer, since a database question can be asking for four
# genuinely different things, and answering any of them with another's
# machinery produces nonsense:
#
#   CAPABILITY - "Can you connect to our DB?" Asking what the assistant
#                can do. Generating SQL for it answers a question nobody
#                asked.
#   SCHEMA     - "what are tables in my db?", "summary the db". Asking
#                what the database *contains structurally*. The answer is
#                already sitting in the introspected schema
#                (db_client.get_schema_description()) - generating SQL to
#                rediscover it is both slower and, since information_schema
#                isn't in sql_guard's table allowlist, doomed anyway.
#   WHY        - "Why is profit low?", "What caused the expense increase?"
#                Asking for an EXPLANATION of a metric, not its value. A
#                single number can't answer this - db_query_service.py
#                generates SQL shaped for causal analysis instead (a
#                comparison against a prior period, or a breakdown by
#                component), through the exact same SQL Guard/DB
#                connection/cache as every other SQL-backed kind here,
#                then asks Gemini to explain the cause using ONLY those
#                retrieved rows - never the whole table, never a guess
#                beyond what the data actually shows.
#   DATA       - "how many customers are there?" Asking for actual
#                records/metrics. This is the pre-existing text-to-SQL
#                path in db_query_service.py, completely unchanged - it's
#                also the default classify_database_question falls back
#                to for anything it doesn't specifically recognize as
#                CAPABILITY, SCHEMA, or WHY.

DATABASE_QUESTION_DATA = "data"
DATABASE_QUESTION_SCHEMA = "schema"
DATABASE_QUESTION_CAPABILITY = "capability"
DATABASE_QUESTION_WHY = "why"

# Structural vocabulary - asking about the shape of the data, not its values.
_SCHEMA_STRUCTURE_RE = re.compile(
    r"\b(?:schema|structure|columns?|fields?|table\s+names?|data\s+model|"
    r"what\s+tables?|which\s+tables?|how\s+many\s+tables?|list\s+(?:the\s+)?tables?|"
    r"tables?\s+(?:are|in|do|does|exist))\b",
    re.IGNORECASE,
)

# "Tell me what's in there" - an overview request. Only counts as a schema
# question when it's aimed at the database as a whole (see
# _database_question_kind); "give me a summary of sales in 2024" names a
# specific table and is a data question, not a structural one.
_DB_OVERVIEW_RE = re.compile(
    r"\b(?:summary|summari[sz]e|summari[sz]ing|overview|describe|"
    r"what\s+data|what\s+kind\s+of\s+data|what\s+sort\s+of\s+data|"
    r"what\s+information|what(?:'s|\s+is)\s+in)\b",
    re.IGNORECASE,
)

# Asking about the assistant's own access/ability rather than about content.
_DB_CAPABILITY_RE = re.compile(
    r"\b(?:can|could|are|is|do|does|will)\s+(?:you|we|the|it)\b[^.?!]{0,40}?"
    r"\b(?:connect|connected|access|reach|read|see|query|talk\s+to|linked)\b"
    r"|\b(?:do|does)\s+you\s+have\s+access\b"
    r"|\bare\s+you\s+(?:connected|linked|hooked)\b",
    re.IGNORECASE,
)

# Asking WHY a metric is what it is, not what its value IS - "Why is
# profit low?", "Why did revenue drop?", "What caused the expense
# increase?", "What's the reason for the loss?". A metric's own name
# ("profit"/"revenue"/...) is what gets a question like this recognized
# as DATABASE_QUERY at all in the first place (see DATABASE_QUERY_PHRASES
# above, checked by classify_intent before this second-stage classifier
# ever runs) - this regex only decides WHICH KIND of database question it
# is once that's already settled.
_WHY_ANALYSIS_RE = re.compile(
    r"\bwhy\s+(?:is|are|was|were|did|does|do|has|have|would|could|so)\b"
    r"|\bwhat\s+(?:is|was|'s)?\s*(?:the\s+)?(?:reason|cause)s?\s+(?:for|behind|why)\b"
    # A bare "what is/was the reason/cause" (optionally followed by a
    # clause like "revenue dropped") is still a causal question - EXCEPT
    # when it's immediately followed by a word that turns "reason" into a
    # compound noun naming an actual column ("reason code", "reason
    # number"), which asks for that column's plain VALUE, not a cause.
    r"|\bwhat(?:'s|\s+is|\s+was)\s+the\s+(?:reason|cause)\b(?!\s+(?:code|number|id|type)\b)"
    r"|\bwhat\s+caus(?:e|es|ed|ing)\b"
    r"|\breason\s+(?:for|behind|why)\b"
    r"|\bwhat\s+(?:led\s+to|drove|driving|is\s+driving)\b",
    re.IGNORECASE,
)

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
#
# Deliberately NOT "database"/"db": a bare substring match on that word
# can't tell "my database" (this app's) from "a database" (the general
# concept), which is exactly the false positive that used to send "What
# is a database?" into the SQL pipeline. _OUR_DATABASE_RE/_OWNED_DATA_RE/
# _GENERIC_DB_CONCEPT_RE above make that distinction properly - see
# _is_database_query, which checks them before this list.
DATABASE_QUERY_PHRASES = [
    "how many",
    "total",
    "average",
    "count of",
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

# "earn"/"earned"/"earning(s)" - the same business-metric category as
# revenue/profit/loss above ("How much did we earn?" has no other
# database signal at all: no table/column word, no other phrase here).
# NOT added to the plain-substring DATABASE_QUERY_PHRASES list above:
# "earn" as a bare substring also matches inside "l-earn-ing", "learned",
# "yearning" - a real false positive caught by this project's own test
# suite ("What is machine learning?" started coming back DATABASE_QUERY).
# A word-boundary regex is what "cost"/"costs" and the rest of that list
# get away without needing, since none of them collide with an unrelated
# English word the way "earn" collides with "learn".
_EARNINGS_RE = re.compile(r"\bearn(?:s|ed|ing|ings)?\b", re.IGNORECASE)

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

    Checked in this order, each a strong-enough signal on its own to
    return True immediately:
    0. An explicit reference to THIS app's database or its contents -
       _OUR_DATABASE_RE ("my db", "the connected database") or
       _OWNED_DATA_RE ("what data do we have?"). See the module-level
       comment above both for the false positives/negatives this tier
       exists to fix. Checked before everything else specifically so it
       can outrank _GENERIC_DB_CONCEPT_RE below for the (rare, and not
       specially handled) case where a question manages to reference both
       ("what is our database, technically?").
    0.5. _GENERIC_DB_CONCEPT_RE - "what is a database?", "what does DB
       mean?". A definitional question about the concept, not this app's
       data. Vetoes the rest of this function (returns False) rather than
       falling through to tier 1, since without this veto the word
       "database" itself would still need to live in DATABASE_QUERY_PHRASES
       and reintroduce the exact false positive tier 0 exists to prevent.
    0.75. _SELF_REFERENCE_RE - "my name is Alex", "which department do I
       manage?". A question/statement about the SPEAKER'S OWN identity or
       role, not about business records - vetoes the rest of this
       function the same way tier 0.5 does, and for a similar reason:
       this app has no login/user-identity system, so a first-person
       question about the speaker's own attributes can never be
       correctly answered from the database regardless of what schema
       vocabulary it happens to share a word with (e.g. "department").
    1. DATABASE_QUERY_PHRASES - standalone triggers, no schema needed.
    1.5. _EARNINGS_RE - "earn"/"earned"/"earnings", the same business-
       metric category as tier 1's revenue/profit/loss, kept as its own
       word-boundary regex rather than added to that plain-substring
       list because "earn" alone collides with "learn"/"learning".
    2. A table name appearing in the question ("students", or "student"
       via the naive-singular match already done when the terms were
       built) - a strong signal on its own, since table names tend to be
       distinctive nouns.
    3. A column name (e.g. "department", "age") appearing ALONGSIDE a
       DATABASE_QUESTION_SHAPE_PHRASES match - column names alone are
       often ordinary English words ("name", "city"), so this tier
       requires both, to avoid misrouting unrelated questions that
       happen to share a word with a column.

    strong_only=True skips tier 3 only - tier 0/0.5 always apply
    regardless, since an explicit "my database" reference is just as
    strong a signal as a tier-1 phrase or an actual table name, and the
    concept veto should suppress a false positive either way.  A generic
    column name like "amount" exists in almost every table this project
    has, so pairing it with an equally generic shape phrase ("what is
    the") is weak enough that it shouldn't, on its own, drag an otherwise
    PDF-only question (e.g. "What is the expected payment amount?" while
    a PDF is attached) into HYBRID_QUERY - see classify_intent, which
    only applies strong_only once a document is already known to be
    relevant.

    Falls back to tier 0/0.5/1 only when database_terms isn't available
    (DB not configured/reachable) - same graceful-degradation shape as
    every other DB-dependent path in this project.
    """

    if _OUR_DATABASE_RE.search(lowered) or _OWNED_DATA_RE.search(lowered):
        return True

    if _GENERIC_DB_CONCEPT_RE.search(lowered):
        return False

    if _SELF_REFERENCE_RE.search(lowered):
        return False

    if any(phrase in lowered for phrase in DATABASE_QUERY_PHRASES):
        return True

    if _EARNINGS_RE.search(lowered):
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


# ---------------------------------------------------------------------
# Ambiguous follow-ups - "Tell me more about it", "What about it?"
# ---------------------------------------------------------------------
#
# classify_intent above judges every question purely on its own words -
# correct for a standalone question, but a short follow-up that never
# names its own topic ("What about it?") has nothing of its own for
# classify_intent to recognize, so it always fell through to
# GENERAL_KNOWLEDGE regardless of what the conversation was actually
# about. That's wrong specifically when the previous question was a
# DATABASE_QUERY/PDF_QUERY/HYBRID_QUERY - "What is our total revenue?"
# followed by "Tell me more about it" should still be answered from the
# database, not answered as if "it" were some general-knowledge topic.
#
# The fix is NOT to rewrite the follow-up's text (that's
# followup_context.py's job, and only for bare time references like
# "yesterday" - a genuinely different, non-overlapping case: those
# splice onto the previous question's own words, these have no topic to
# splice onto at all). Instead, see resolve_intent below: classify the
# new question independently FIRST, and only fall back to the previous
# question's own classification when the new one is ambiguous - never
# the other way around, so a real (if short) standalone question like
# "Get me the profit" or "My Health School?" is never second-guessed
# just because a previous_question happens to be available.

# Words that carry no topic of their own - question/request scaffolding
# ("tell me", "what about", "please explain") rather than a subject.
# Deliberately the SAME kind of small, explicit, enumerable list this
# project already uses for TIME_PHRASES/DATABASE_QUERY_PHRASES, not a
# hardcoded reaction to any one example question.
_FOLLOWUP_FILLER_WORDS = {
    "tell", "me", "us", "more", "about", "what", "whats",
    "please", "explain", "elaborate", "further", "go", "on",
    "continue", "and", "so", "then", "also", "give", "show",
}

# A bare pronoun standing in for something named earlier in the
# conversation, not in this question - the actual signal that a question
# depends on prior context rather than stating its own subject.
_FOLLOWUP_REFERENT_WORDS = {"it", "that", "this", "them", "those", "these"}


def is_ambiguous_followup(text):
    """True if `text`, judged entirely on its own, names no topic of its
    own - just scaffolding words plus a bare pronoun referring to
    something said earlier ("tell me more about it", "what about it?").

    Deliberately conservative: EVERY word has to be filler or a referent
    pronoun, and at least one referent pronoun has to be present. A
    short question that still names its own subject ("My Health
    School?", "Get me the profit") is never flagged just for being
    short - only a question with no subject of its own at all is.
    """

    words = _WORD_RE.findall(text.lower())

    if not words:
        return False

    if not any(word in _FOLLOWUP_REFERENT_WORDS for word in words):
        return False

    return all(word in _FOLLOWUP_FILLER_WORDS or word in _FOLLOWUP_REFERENT_WORDS for word in words)


def resolve_intent(text, previous_question=None, database_terms=None, has_document=False, pdf_relevant=False):
    """The entry point routes/chat.py actually calls - classify_intent
    plus the ambiguous-follow-up fallback above.

    `text` is classified independently first, via the unmodified
    classify_intent (same signature, same rules, same result for every
    question that names its own topic). Only when that comes back
    GENERAL_KNOWLEDGE *and* `text` is itself ambiguous/incomplete (see
    is_ambiguous_followup) *and* a previous_question is actually
    available does this fall back to classifying previous_question
    instead - inheriting what the conversation was already about rather
    than defaulting to general knowledge just because the follow-up
    itself named nothing.

    has_document/pdf_relevant are passed through unchanged to both
    classify_intent calls - this never re-scores PDF relevance or
    touches document retrieval itself, it only decides which question's
    *text* gets handed to the exact same classify_intent already in use.
    """

    intent = classify_intent(text, database_terms, has_document=has_document, pdf_relevant=pdf_relevant)

    if intent != GENERAL_KNOWLEDGE:
        return intent

    if not previous_question or not is_ambiguous_followup(text):
        return intent

    return classify_intent(previous_question, database_terms, has_document=has_document, pdf_relevant=pdf_relevant)


def classify_database_question(text):
    """Return DATABASE_QUESTION_CAPABILITY, DATABASE_QUESTION_SCHEMA,
    DATABASE_QUESTION_WHY, or DATABASE_QUESTION_DATA for a question
    already known to be about this app's database (i.e. classify_intent
    already returned DATABASE_QUERY).

    Called by db_query_service.answer_database_question to decide HOW to
    answer - see the module comment above _SCHEMA_STRUCTURE_RE for why
    the four kinds can't share machinery. Order matters:

      1. CAPABILITY first - "can you see our tables?" would also match
         the SCHEMA check's "tables" vocabulary, but it's asking about
         ability, not content, so capability has to be checked before
         schema/overview gets a chance to claim it.
      2. SCHEMA next - structural vocabulary (_SCHEMA_STRUCTURE_RE) or an
         overview request (_DB_OVERVIEW_RE, which also covers "what data
         do we have?" - a vague content question is answered better as
         an overview of what's available than as a specific metric).
      3. WHY next - "why is profit low?" (_WHY_ANALYSIS_RE). Checked
         after CAPABILITY/SCHEMA since neither of those vocabularies
         overlaps with it in practice, but before the DATA default so a
         causal question doesn't fall through to a plain single-value
         answer.
      4. DATA is the default - same fallback shape as every other tier in
         this file, so a database question this function doesn't
         specifically recognize still gets a real answer attempt (the
         existing text-to-SQL path) instead of a refusal.
    """

    lowered = text.lower().strip()

    if _DB_CAPABILITY_RE.search(lowered):
        return DATABASE_QUESTION_CAPABILITY

    if _SCHEMA_STRUCTURE_RE.search(lowered) or _DB_OVERVIEW_RE.search(lowered):
        return DATABASE_QUESTION_SCHEMA

    if _WHY_ANALYSIS_RE.search(lowered):
        return DATABASE_QUESTION_WHY

    return DATABASE_QUESTION_DATA
