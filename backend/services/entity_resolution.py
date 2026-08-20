"""
Named-entity resolution for the GENERAL_KNOWLEDGE answer path.

THE BUG THIS EXISTS FOR
-----------------------
"What is My Health School in India?" came back as a confident description
of the *School Health and Wellness Programme* - a real, well-known Indian
government programme that shares most of the same words and none of the
same identity. Nothing in the pipeline was broken: GENERAL_KNOWLEDGE went
straight to a single Gemini call with the raw question, and a model asked
about a name it has never seen will happily answer about the nearest
famous thing instead of saying so. That is the single most dangerous
failure mode for a general-knowledge assistant, because the answer is
fluent, detailed, and about the wrong organisation.

The fix follows this project's established preference for a deterministic
guard over a prompt instruction alone (same reasoning as
transcription.py's silence pre-check and chat_service.py's zero-row and
exact-date guards): rather than *asking* Gemini not to substitute a
similarly-named entity, an entity that can't be verified never reaches a
free-form answer at all - it is routed to a fixed clarification response
instead.

THREE STAGES
------------
1. extract_entity_candidates() - pure text analysis, no API call. Finds
   the proper nouns / codes / product names in the question and returns
   them VERBATIM (requirement: never normalise, expand, or "correct" the
   entity the user actually typed - that rewriting is itself how a
   specific name drifts into a generic one).

2. verify_entities() - only runs when stage 1 found something. One
   Gemini call *with Google Search grounding* that looks the entity up on
   the live web and returns a structured FOUND / AMBIGUOUS / NOT_FOUND
   verdict plus what it actually found.

3. chat_service.answer_general_knowledge() branches on that verdict:
   FOUND answers with the search context attached, AMBIGUOUS and
   NOT_FOUND return a short clarification instead of an answer.

WHEN THE SEARCH DOES *NOT* HAPPEN
---------------------------------
Stage 1 is the cost gate, and it is deliberately the only one: a question
with no proper-noun-shaped span in it ("What is machine learning?", "How
does photosynthesis work?", "Explain recursion") produces no candidates,
takes the pre-existing single-call path unchanged, and never touches web
search. That covers the large majority of general-knowledge questions.

The trade-off taken here, stated plainly: a *single* capitalized word
that isn't a sentence-opening artifact does count as a candidate, so
"What is the capital of France?" will run a verification search it
doesn't strictly need. That's a deliberate choice over the alternative -
requiring two or more capitalized words - which would have been cheaper
but would have silently missed "Tell me about Integfarms", a one-word
obscure company name that is exactly the case this module exists for.
There is no way to tell "France" from "Integfarms" without either a
hardcoded list of famous entities (which the brief rules out, and which
would rot) or the very lookup being gated. Paying for an unnecessary
search on well-known nouns is the cheaper mistake than confidently
describing the wrong organisation.
"""

import logging
import re

from config import DEBUG_VOICE_PIPELINE
from services.gemini_client import generate_with_search, GeminiError
from services.intent_router import is_self_reference

logger = logging.getLogger("uvicorn")

FOUND = "FOUND"
AMBIGUOUS = "AMBIGUOUS"
NOT_FOUND = "NOT_FOUND"

# More than a handful of entities in one question is a sign the detector
# has latched onto ordinary title-cased prose rather than real names -
# and a longer list would bloat the research prompt for no gain. The
# first few candidates are the ones the question is actually about.
MAX_CANDIDATES = 4

# Words that begin a question purely as grammar, so their capital letter
# is an orthographic artifact rather than evidence of a proper noun.
# Only ever applied to the *leading* run of tokens (see _first_content_index):
# "Will" mid-sentence ("What is Will Smith's height?") is still a name.
#
# Deliberately NOT included: possessive/determiner words like "my",
# "our", "your". "My Health School" is a real organisation name whose
# first word is "My" - dropping it as a stopword would hand the next
# stage "Health School", a mangled name that searches straight back to
# the generic government programme this module exists to stop confusing
# it with.
QUESTION_OPENERS = {
    "what", "whats", "what's", "who", "whos", "who's", "whose", "where",
    "when", "why", "how", "which",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will", "has", "have", "had", "am", "there",
    "tell", "explain", "define", "describe", "give", "list", "show",
    "please", "me", "us", "about", "info", "information", "know",
}

# Lowercase words allowed to sit *inside* a multi-word name without
# ending it - "School Health and Wellness Programme", "Bank of Baroda",
# "Procter & Gamble".
#
# Locational prepositions ("in", "at", "on", "from", "near") are
# deliberately absent: in "My Health School in India" they separate the
# entity from where it is, and swallowing them would fuse two distinct
# entities into one name ("My Health School In India") that matches
# nothing on the web.
RUN_CONNECTORS = {"of", "and", "for", "the", "&", "de", "del", "la", "van", "von"}

# A token is any run of letters/digits plus the punctuation that occurs
# *inside* real names - the dot in "Recall.ai", the hyphen in "GPT-4",
# the apostrophe in "McDonald's", the ampersand in "AT&T".
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.\-&]*")

# Trailing sentence punctuation that the token pattern above sweeps up
# ("Programme." at the end of a sentence) but that is not part of the
# name. Stripped only from the end, so "Recall.ai" keeps its inner dot.
_TRAILING_PUNCTUATION = ".-'’&"

# A dotted product/domain name: "Recall.ai", "openai.com", "socket.io".
# Requires every segment to start with a letter, which keeps decimals and
# version numbers ("3.11") out.
_DOTTED_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-]*(?:\.[A-Za-z][A-Za-z0-9\-]*)+$")

_QUOTED_SPAN_RE = re.compile(r"[\"“]([^\"“”]{2,80})[\"”]")


def _tokenize(question):
    """Return [(start_offset, token_text)] for the question's word-like tokens."""

    tokens = []

    for match in _TOKEN_RE.finditer(question):
        text = match.group(0).rstrip(_TRAILING_PUNCTUATION)
        if text:
            tokens.append((match.start(), text))

    return tokens


def _first_content_index(tokens):
    """Index of the first token that isn't a grammatical question opener.

    Every capitalized token *before* this index is capitalized only
    because it starts the sentence, so it carries no evidence of being a
    name. Everything from this index on is judged on its own merits -
    which is what lets a bare "Integfarms" (index 0, no opener in front
    of it) be recognised as an entity while "What" in "What is machine
    learning?" is not.
    """

    for index, (_, token) in enumerate(tokens):
        if token.lower() not in QUESTION_OPENERS:
            return index

    return len(tokens)


def _is_capitalized(token):
    # The pronoun "I" is always capitalized regardless of grammatical
    # position ("Alex and I manage...") - that capitalization is a fixed
    # orthographic rule, not evidence of being part of a proper name, so
    # it must never be treated as continuing or starting a capitalized
    # run. A real, observed bug: "Alex and I" got extracted as if it
    # were one entity name, mixing a real name with a grammatical
    # pronoun purely because the run-detector saw two capitalized tokens
    # in a row.
    if token == "I":
        return False
    return token[:1].isupper()


def _has_letters_and_digits(token):
    return any(c.isalpha() for c in token) and any(c.isdigit() for c in token)


def _is_dotted_name(token):
    return bool(_DOTTED_NAME_RE.match(token))


# A run consisting of exactly ONE of these words, and nothing else, is
# never a valid entity candidate by itself - "My"/"Our"/"Your" alone at
# the start of an ordinary sentence ("My name is Alex...") is just the
# possessive pronoun, capitalized purely because it opens the sentence,
# not evidence of naming anything. This does NOT affect a longer run
# that starts with one of these words ("My Health School" still comes
# through as a 3-token run below, unaffected) - only a bare, standalone
# occurrence with nothing capitalized following it.
_STANDALONE_DETERMINER_WORDS = {"my", "our", "your"}


def _capitalized_runs(tokens, first_content):
    """Yield (offset, text) for each run of capitalized tokens, allowing
    RUN_CONNECTORS between them, skipping sentence-opening artifacts."""

    runs = []
    index = 0

    while index < len(tokens):
        offset, token = tokens[index]

        if index < first_content or not _is_capitalized(token):
            index += 1
            continue

        # Walk forward over capitalized tokens and the lowercase
        # connectors that join them. last_capitalized is tracked
        # separately so a run never ends on a dangling connector
        # ("Ministry of" -> "Ministry").
        end = index + 1
        last_capitalized = index

        while end < len(tokens):
            next_token = tokens[end][1]

            if _is_capitalized(next_token):
                last_capitalized = end
                end += 1
            elif (
                next_token.lower() in RUN_CONNECTORS
                and end + 1 < len(tokens)
                and _is_capitalized(tokens[end + 1][1])
            ):
                end += 1
            else:
                break

        run_text = " ".join(text for _, text in tokens[index:last_capitalized + 1])
        if run_text.lower() not in _STANDALONE_DETERMINER_WORDS:
            runs.append((offset, run_text))
        index = last_capitalized + 1

    return runs


# Tail nouns that mark a "my/our ..." phrase as naming an INSTITUTION
# rather than describing something the user personally owns. Used by
# _possessive_institution_runs below to catch a real proper name that
# begins with the word "my" (e.g. "My Health School") when it's typed in
# plain lowercase - very common casual/mobile/voice-transcribed input -
# where _capitalized_runs above has nothing capitalized to latch onto at
# all, and "my health school" would otherwise read as indistinguishable
# from the possessive pronoun "my" plus an ordinary noun.
#
# This is a general shape (any org/institution name of this form), not a
# list of specific entities - nothing here is "My Health School" itself.
_INSTITUTION_NOUNS = {
    "school", "college", "university", "institute", "academy", "hospital",
    "clinic", "foundation", "company", "corporation", "firm", "agency",
    "bank", "app", "platform", "program", "programme", "service",
    "network", "store", "studio", "brand", "group", "society",
    "association", "club", "league", "team", "mission", "project",
    "initiative",
}


def _possessive_institution_runs(tokens):
    """Yield (offset, text) for "my/our <word> [<word>] <institution
    noun>" runs - the lowercase counterpart to _capitalized_runs.

    A single generic noun after "my"/"our" ("my school", "my account")
    is ordinary possession and is deliberately left alone. Two or more
    words ending in an _INSTITUTION_NOUNS word ("my health school") is
    what marks this as naming a specific institution instead - that tail
    noun is the one thing that tells "my health school" apart from "my
    account balance" or "my email address", which don't end in one of
    these words and correctly produce no candidate here.

    The backward search for the institution noun (rather than requiring
    it to be the LAST word scanned) mirrors why _capitalized_runs stops
    before "in": "my health school in India" should resolve to "my
    health school", not fail to match just because "in" trails after it
    within the scanned word window.
    """

    runs = []
    index = 0

    while index < len(tokens):
        offset, token = tokens[index]

        if token.lower() not in ("my", "our"):
            index += 1
            continue

        window_end = index + 1
        while window_end < len(tokens) and window_end - index <= 3 and tokens[window_end][1].isalpha():
            window_end += 1

        match_end = None
        for candidate_end in range(window_end - 1, index + 1, -1):
            if tokens[candidate_end][1].lower() in _INSTITUTION_NOUNS:
                match_end = candidate_end
                break

        if match_end is not None:
            runs.append((offset, " ".join(text for _, text in tokens[index:match_end + 1])))
            index = match_end + 1
        else:
            index += 1

    return runs


def _deduplicate(spans):
    """Order by position in the question, dropping any candidate wholly
    contained in another (so "Recall.ai" survives and the bare "Recall"
    the capitalized-run pass also produced does not)."""

    kept = []

    for _, text in sorted(spans, key=lambda span: span[0]):
        lowered = text.lower()

        if any(lowered in existing.lower() for existing in kept):
            continue

        kept = [existing for existing in kept if existing.lower() not in lowered]
        kept.append(text)

    return kept


def extract_entity_candidates(question):
    """Return the phrases in `question` that look like named entities.

    Verbatim and in order of appearance - the exact strings the user
    typed, never a normalised or expanded form.

    Five shapes are recognised, all of them structural rather than
    knowledge-based (nothing here knows any particular entity's name):

      * a double-quoted span - an explicit "treat this as a name" signal
      * a dotted product/domain name: Recall.ai, openai.com
      * a token mixing letters and digits: MHS166, GPT-4, A320
      * a run of capitalized words, optionally joined by RUN_CONNECTORS,
        that isn't just the sentence's opening grammar: My Health School,
        School Health and Wellness Programme, Integfarms
      * a lowercase "my/our <word(s)> <institution noun>" phrase - see
        _possessive_institution_runs: catches a name like "my health
        school" even with no capital letter at all, without treating an
        ordinary possessive like "my school" or "my account balance" as
        an entity.

    The dotted-name, letters-plus-digits, and possessive-institution
    shapes are checked regardless of capitalization, so a lowercase
    "what is mhs166?"/"what is my health school?" is still recognised;
    the capitalized-run shape is by definition case-sensitive, so a
    lowercase "tell me about integfarms" is not. That asymmetry is
    accepted rather than worked around: without a capital letter or a
    distinctive shape there is genuinely nothing to separate a
    typed-lowercase company name from an ordinary noun, and guessing
    would put every question back through web search.
    """

    if not question or not question.strip():
        return []

    # A first-person question/statement about the SPEAKER'S OWN identity
    # or role ("My name is Alex...", "which department do I manage?")
    # was never asking to identify a real-world organization in the
    # first place - there is nothing here for a web-search verification
    # to usefully check. A real, observed bug: even after the "I"-pronoun
    # and lone-determiner fixes above, a plain personal statement could
    # still surface an ordinary capitalized word ("Marketing" in "I
    # manage the Marketing department") as a lone candidate, sending a
    # completely mundane sentence through entity verification and coming
    # back NOT_FOUND/AMBIGUOUS - refusing to just have the conversation.
    # Shares the exact same signal services/intent_router.py uses to keep
    # the same class of sentence out of the database pipeline, so both
    # fixes agree on what counts as "about the speaker themselves."
    if is_self_reference(question):
        return []

    tokens = _tokenize(question)

    if not tokens:
        return []

    first_content = _first_content_index(tokens)

    spans = [(match.start(1), match.group(1).strip()) for match in _QUOTED_SPAN_RE.finditer(question)]

    for offset, token in tokens:
        if _is_dotted_name(token) or _has_letters_and_digits(token):
            spans.append((offset, token))

    spans.extend(_capitalized_runs(tokens, first_content))
    spans.extend(_possessive_institution_runs(tokens))

    return _deduplicate(spans)[:MAX_CANDIDATES]


# Search for the name AS WRITTEN is the whole point: the failure being
# guarded against is a near-miss match being reported as the real thing,
# and a model left to its own devices treats "close enough" as found.
# The worked example is deliberately concrete and deliberately about the
# actual observed failure, matching how HYBRID_ANSWER_PROMPT in
# chat_service.py spells out its own zero-row trap - a rule stated in the
# abstract was not enough there either.
ENTITY_RESEARCH_PROMPT = """You are verifying what specific named entities actually refer to, before another assistant answers a question about them.

Use web search to look up each entity below EXACTLY as written. Do not correct its spelling, expand an abbreviation, or search for a better-known organisation with a similar name.

QUESTION: {question}
ENTITIES TO VERIFY: {entities}

Reply in EXACTLY this format, and nothing else:

VERDICT: <FOUND or AMBIGUOUS or NOT_FOUND>
FINDINGS: <what the search actually shows, in 2-5 sentences. For AMBIGUOUS, list the distinct candidates by name. For NOT_FOUND, say what you searched for and what the closest non-matching results were.>

VERDICT rules:
- FOUND: the search results clearly identify an entity carrying this name, and they agree on what it is.
- AMBIGUOUS: two or more genuinely different real entities carry this name and the question does not say which is meant.
- NOT_FOUND: no search result identifies an entity carrying this name.

Critically: an organisation whose name merely RESEMBLES the entity is not a match. If the closest thing you can find is a differently-named programme, company, product, or person, the verdict is NOT_FOUND - not FOUND - no matter how well-known that near-miss is.

Worked example of the failure to avoid: asked to verify "My Health School", finding the Indian government's "School Health and Wellness Programme" is NOT a match. It is a different name for a different thing, and reporting it as FOUND would make the next assistant describe the wrong organisation in confident detail. The correct verdict there is NOT_FOUND, with the near-miss named in FINDINGS as a near-miss."""

_RESEARCH_RESPONSE_RE = re.compile(
    r"VERDICT:\s*(?P<verdict>[A-Z_]+)\s*(?:FINDINGS:\s*(?P<findings>.*))?",
    re.DOTALL,
)

_VERDICTS = {FOUND, AMBIGUOUS, NOT_FOUND}

# Grounded responses interleave citation markers like "[2.1, 2.3]" into
# their prose, keyed to grounding chunks the user never sees. They're
# harmless in the FOUND path (that text is only ever context for a second
# Gemini call, which ignores them), but the AMBIGUOUS/NOT_FOUND paths
# show findings to the user verbatim - observed live, that shipped a
# clarification reading "...do not identify a distinct entity [2.1, 2.2,
# 2.3]." Deliberately narrow: only bracketed groups made entirely of
# numbers, dots, and commas, so a real bracketed aside survives intact.
_CITATION_MARKER_RE = re.compile(r"\s*\[\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*\]")


def _strip_citation_markers(text):
    return _CITATION_MARKER_RE.sub("", text)


def _parse_research_response(text):
    """Return (verdict, findings) from the research call's reply.

    An unparseable reply degrades to (FOUND, whole_text) rather than to a
    refusal: the text still came back from a real grounded web search, so
    passing it on as context keeps the answer anchored to something real,
    which is strictly better than the pre-existing behavior of answering
    from the model's own memory with no context at all. Refusing outright
    on a formatting slip would be the worse trade - it would turn a
    cosmetic parse failure into a user-visible non-answer.
    """

    text = (text or "").strip()

    if not text:
        return None, ""

    match = _RESEARCH_RESPONSE_RE.search(text)

    if not match:
        return FOUND, _strip_citation_markers(text)

    verdict = match.group("verdict").strip().upper()
    findings = _strip_citation_markers((match.group("findings") or "").strip())

    if verdict not in _VERDICTS:
        return FOUND, _strip_citation_markers(text)

    return verdict, findings or _strip_citation_markers(text)


async def verify_entities(question, candidates):
    """Look the candidates up on the live web via one grounded Gemini call.

    Returns (verdict, findings, sources), or (None, "", []) when
    verification could not be completed - a missing key, a Gemini/search
    failure, or an empty reply. The None verdict is the caller's signal to
    fall back to the plain, pre-existing single-call answer path: a search
    outage should degrade this feature, never fail the whole request, the
    same way db_client failures degrade the database guards in
    chat_service.answer_database_query rather than erroring the endpoint.
    """

    prompt = ENTITY_RESEARCH_PROMPT.format(
        question=question,
        entities=", ".join(f'"{candidate}"' for candidate in candidates),
    )

    try:
        text, sources = await generate_with_search(prompt)
    except GeminiError as error:
        logger.warning(f"[entity-resolution] verification unavailable, answering without it: {error}")
        return None, "", []

    verdict, findings = _parse_research_response(text)

    if DEBUG_VOICE_PIPELINE:
        logger.info(f"[entity-resolution] CANDIDATES: {candidates}")
        logger.info(f"[entity-resolution] VERDICT: {verdict}")
        logger.info(f"[entity-resolution] FINDINGS: {findings[:300]!r}")
        logger.info(f"[entity-resolution] SOURCES: {[source['url'] for source in sources]}")

    return verdict, findings, sources


def _quoted(candidates):
    return ", ".join(f'"{candidate}"' for candidate in candidates)


def ambiguous_clarification(candidates, findings):
    """The short clarification returned instead of an answer when a name
    maps to several real entities. Deterministic text, not a Gemini call:
    the whole point is that no answer gets generated about a guessed
    entity, and generating the question to ask would reopen exactly that
    door."""

    return (
        f"Before I answer, I need to know which {_quoted(candidates)} you mean - "
        f"the name matches more than one real thing, and answering about the wrong one "
        f"would be worse than asking.\n\n"
        f"{findings}\n\n"
        f"Which of these did you have in mind?"
    )


def not_found_clarification(candidates, findings):
    """The response when the web search can't confirm the entity exists at
    all. Deliberately does NOT fall through to a normal answer: answering
    anyway is precisely how "My Health School" became a description of the
    School Health and Wellness Programme."""

    return (
        f"I couldn't verify anything specifically called {_quoted(candidates)}, so I'd "
        f"rather ask than describe some other organisation with a similar-sounding name.\n\n"
        f"{findings}\n\n"
        f"Could you add a little context - roughly where it's based, what sector it's in, "
        f"or where you came across the name? I'll look again with that."
    )
