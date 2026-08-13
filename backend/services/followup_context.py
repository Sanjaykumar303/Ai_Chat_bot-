"""
Lightweight conversation context for short follow-up questions - "what is
the total profit today?" followed by just "yesterday" should be answered
as "what is the total profit yesterday?", not misrouted to
GENERAL_KNOWLEDGE for having no recognizable database vocabulary of its
own.

Deliberately plain text/regex matching, not a Gemini call - same
rationale intent_router.py already uses for its own phrase-list routing:
a classifier/rewriter call per message would only add cost and latency,
and the set of ways someone phrases "same question, different day" is
small and enumerable.

This only ever rewrites the text that reaches classify_intent() and the
downstream answer prompts - it never touches conversation state itself.
The caller (routes/chat.py) is responsible for remembering what the
previous *resolved* question was (see ChatRequest.previous_question) and
handing it in each time; this module is pure text-in, text-out.
"""

import re

# The exact phrases this module recognizes as "the same question, just a
# different time period" - deliberately a small, explicit set rather
# than a general date parser, matching this project's existing
# lightweight-keyword-matching style (see intent_router.py).
TIME_PHRASES = [
    "today",
    "yesterday",
    "this week",
    "last week",
    "this month",
    "last month",
    "this year",
    "last year",
    "this quarter",
    "last quarter",
]

# Common ways a follow-up is phrased around the bare time reference -
# "and last week?", "what about yesterday", "how about last month?".
_LEADING_FILLERS = ["and", "what about", "how about", "what's", "whats"]

_TIME_PHRASE_ALTERNATION = "|".join(re.escape(phrase) for phrase in sorted(TIME_PHRASES, key=len, reverse=True))

# Matches one of TIME_PHRASES sitting at the end of a sentence (optionally
# followed by punctuation) - used to strip the previous question's own
# time reference before splicing the new one on.
_TRAILING_TIME_PHRASE_RE = re.compile(
    r"\b(?:" + _TIME_PHRASE_ALTERNATION + r")\b[?.!]*\s*$",
    re.IGNORECASE,
)


def _clean(text):
    return text.strip().lower().rstrip("?.!").strip()


def _strip_leading_filler(text):
    for filler in _LEADING_FILLERS:
        if text == filler:
            return ""
        if text.startswith(filler + " "):
            return text[len(filler):].strip()
    return text


def _match_time_phrase(text):
    """Return the canonical TIME_PHRASES entry `text` reduces to, or None
    if it isn't just a bare time reference. Tolerates a trailing "s"
    typo ("yesterdays") the same way intent_router._plural_variants
    tolerates plurals elsewhere in this project."""

    cleaned = _strip_leading_filler(_clean(text))

    if cleaned in TIME_PHRASES:
        return cleaned

    if cleaned.endswith("s") and cleaned[:-1] in TIME_PHRASES:
        return cleaned[:-1]

    return None


def is_followup_time_query(text):
    """True if text is nothing more than a time-period reference
    ("yesterday", "and last week?") with no question content of its
    own - the signal that it's a follow-up needing the previous
    question's context, not a standalone question."""

    return _match_time_phrase(text) is not None


def rewrite_with_context(question, previous_question):
    """Return the question to actually route/answer.

    If `question` is just a bare time reference (is_followup_time_query)
    and a `previous_question` is available, splice the new time period
    onto the previous question in place of its own trailing time
    reference (if it had one) - "What is the total profit today?" +
    "yesterday" -> "What is the total profit yesterday?". Otherwise
    `question` is returned unchanged, exactly as if this module didn't
    exist - a standalone question is never rewritten.
    """

    if not previous_question:
        return question

    time_phrase = _match_time_phrase(question)
    if time_phrase is None:
        return question

    base = _TRAILING_TIME_PHRASE_RE.sub("", previous_question).strip()
    base = base.rstrip("?.!").strip()

    if not base:
        # previous_question was itself nothing but a time phrase (chained
        # follow-ups with no real metric ever stated) - nothing sensible
        # to splice onto, so fall back to it unchanged rather than
        # producing a blank/garbled question.
        return previous_question

    return f"{base} {time_phrase}?"
