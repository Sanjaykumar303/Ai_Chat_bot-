# Regression tests for the entity-resolution stage in front of the
# GENERAL_KNOWLEDGE answer path.
#
# The bug being locked down: "What is My Health School in India?" was
# answered as a description of the School Health and Wellness Programme -
# a real, well-known, and completely different Indian government
# programme. See services/entity_resolution.py's module docstring.
#
# Same scope rule as the rest of this suite (see
# test_chat_service_guards.py): these cover the deterministic logic -
# which questions trigger a lookup, what spans get extracted, how a
# verdict is parsed, and which branch each verdict takes. The live web
# search itself is stubbed, never called, matching this project's
# practice of verifying anything touching a real Gemini/DB call by
# running it for real instead of mocking it into a test.

import asyncio

import pytest

from services import chat_service, entity_resolution
from services.entity_resolution import (
    AMBIGUOUS,
    FOUND,
    NOT_FOUND,
    _parse_research_response,
    extract_entity_candidates,
)


# --- Stage 1: which questions have a named entity in them at all -------
#
# This is the cost gate. Everything below the line "no candidates" takes
# the pre-existing single-Gemini-call path and never touches web search.


def test_multiword_name_is_extracted_verbatim():
    # The headline case. "My Health School" has to survive as typed -
    # dropping the possessive "My" as a stopword would leave "Health
    # School", which searches straight back to the generic programme this
    # whole module exists to stop confusing it with.
    assert "My Health School" in extract_entity_candidates("What is My Health School in India?")


def test_locational_preposition_does_not_get_absorbed_into_the_name():
    # "in India" says where the entity is, it isn't part of its name -
    # fusing them would produce a string that matches nothing.
    candidates = extract_entity_candidates("What is My Health School in India?")
    assert "My Health School" in candidates
    assert not any("India" in candidate and "School" in candidate for candidate in candidates)


def test_bare_single_word_company_name_is_extracted():
    # A one-word obscure name submitted on its own. Nothing precedes it
    # to explain its capital letter, so the capital is real evidence.
    assert extract_entity_candidates("Integfarms") == ["Integfarms"]


def test_single_word_company_name_mid_sentence_is_extracted():
    assert "Integfarms" in extract_entity_candidates("Tell me about Integfarms")


def test_real_programme_name_is_also_extracted():
    # The genuine entity that the buggy answer was describing. It has to
    # be verified too - the fix is "resolve the entity", not "distrust
    # this one particular name".
    assert extract_entity_candidates("What is the School Health and Wellness Programme?") == [
        "School Health and Wellness Programme"
    ]


def test_connector_words_stay_inside_a_multiword_name():
    assert "Bank of Baroda" in extract_entity_candidates("Who runs Bank of Baroda?")


def test_alphanumeric_code_is_extracted():
    assert extract_entity_candidates("What is MHS166?") == ["MHS166"]


def test_alphanumeric_code_is_extracted_even_lowercase():
    # A code's letters-and-digits shape is evidence on its own, so it
    # doesn't need the capital letter a plain word name does.
    assert extract_entity_candidates("what is mhs166?") == ["mhs166"]


def test_dotted_product_name_is_extracted_whole():
    # "Recall.ai", not the bare "Recall" that the capitalized-run pass
    # also produces - the dotted form is the real product name.
    assert extract_entity_candidates("What is Recall.ai?") == ["Recall.ai"]


def test_no_entity_in_a_plain_concept_question():
    # The cost gate doing its job: no proper noun, so no web search, and
    # this question takes exactly the path it always did.
    assert extract_entity_candidates("What is machine learning?") == []


@pytest.mark.parametrize(
    "question",
    [
        "How does photosynthesis work?",
        "Explain recursion to me",
        "why is the sky blue",
        "Tell me a joke",
        "What are the main causes of inflation?",
    ],
)
def test_ordinary_general_knowledge_questions_trigger_no_lookup(question):
    assert extract_entity_candidates(question) == []


def test_sentence_opening_capital_is_not_treated_as_a_name():
    # "What"/"Tell"/"Is" are capitalized by orthography, not because
    # they're names.
    assert extract_entity_candidates("What is inflation?") == []
    assert extract_entity_candidates("Is inflation rising?") == []


# --- "My Health School" typed in plain lowercase -----------------------
#
# The bug this section locks down: "my health school" (no capital letter
# at all - common casual/mobile/voice-transcribed input) produced NO
# entity candidate, so the question fell straight through to the plain
# GENERAL_KNOWLEDGE_PROMPT path with nothing telling Gemini this wasn't
# an ordinary possessive - it answered as if asked about the user's own,
# personal, unnamed school. See
# entity_resolution._possessive_institution_runs's docstring for the
# general (not entity-specific) rule that fixes this: two or more words
# after "my"/"our" ending in a recognised institution-type noun
# ("school", "hospital", "foundation", ...) is what marks the phrase as
# naming an institution rather than describing personal property.


def test_lowercase_my_health_school_is_recognised_as_an_entity():
    assert "my health school" in extract_entity_candidates("What is my health school?")


def test_lowercase_my_health_school_recognised_mid_sentence_too():
    assert "my health school" in extract_entity_candidates("tell me about my health school")


def test_my_school_stays_personal_not_an_entity():
    # A single generic noun after "my" is ordinary possession - "my
    # school" must NOT be treated as a named entity, so this question
    # keeps answering as an unresolved personal reference exactly as it
    # did before this fix, not routed into web-search verification.
    assert extract_entity_candidates("What is my school?") == []


def test_a_health_school_stays_general_knowledge_not_an_entity():
    # No "my"/"our" possessive at all, and nothing capitalized - a plain
    # concept question, unaffected by either entity rule.
    assert extract_entity_candidates("What is a health school?") == []


def test_my_health_school_still_recognised_alongside_a_trailing_location():
    # "in India" shouldn't prevent "my health school" from being found -
    # same reasoning _capitalized_runs uses to stop a name before a
    # trailing preposition, applied to the lowercase possessive shape.
    candidates = extract_entity_candidates("what is my health school in India?")
    assert "my health school" in candidates


def test_ordinary_possessive_phrases_are_not_mistaken_for_entities():
    # The false-positive check for the new rule: a multi-word possessive
    # that does NOT end in an institution-type noun must stay untouched,
    # so this fix doesn't turn every "my ___ ___" question into a web
    # search.
    assert extract_entity_candidates("What is my account balance?") == []
    assert extract_entity_candidates("what is my email address?") == []
    assert extract_entity_candidates("what is my monthly report?") == []


def test_empty_and_whitespace_questions_are_safe():
    assert extract_entity_candidates("") == []
    assert extract_entity_candidates("   ") == []
    assert extract_entity_candidates("???") == []


# --- self-reference: a personal statement/question is not an entity ---
#
# Found live via services/chat_memory.py's own verification: "My name is
# Alex and I manage the Marketing department." extracted ["My", "Alex
# and I", "Marketing"] as if they were named entities to verify on the
# live web, and the assistant refused to just have the conversation.
# Three separate, compounding bugs, each covered below: the pronoun "I"
# joining a real name into a garbled span, a lone possessive determiner
# counting as its own candidate, and (the actual fix) recognizing the
# whole sentence as self-referential in the first place.


def test_self_referential_statement_produces_no_candidates():
    assert extract_entity_candidates("My name is Alex and I manage the Marketing department.") == []


def test_self_referential_question_produces_no_candidates():
    assert extract_entity_candidates("What's my name, and which department do I manage?") == []


def test_who_am_i_and_call_me_produce_no_candidates():
    assert extract_entity_candidates("Who am I?") == []
    assert extract_entity_candidates("You can call me Alex.") == []


def test_pronoun_i_never_joins_a_capitalized_run():
    # Defense in depth, independent of the self-reference gate above:
    # even a sentence the gate doesn't catch must never fuse the pronoun
    # "I" into an adjacent name - "Alex and I" mixing a real name with
    # grammar was the most visibly broken part of the original bug.
    candidates = entity_resolution._capitalized_runs(
        entity_resolution._tokenize("Ask Alex and I about it"), first_content=0
    )
    texts = [text for _, text in candidates]
    assert "Alex and I" not in texts
    assert not any(" I" in text or text.endswith(" I") for text in texts)


def test_lone_possessive_determiner_is_not_a_candidate():
    # "My"/"Our"/"Your" alone (nothing capitalized following) is never a
    # valid entity by itself - only a longer run starting with one of
    # these words is (see test_lowercase_my_health_school_is_recognised_
    # as_an_entity and test_multiword_name_is_extracted_verbatim above,
    # both still passing, proving this fix doesn't disturb that case).
    runs = entity_resolution._capitalized_runs(
        entity_resolution._tokenize("My favorite color is blue"), first_content=0
    )
    assert ("my" not in [text.lower() for _, text in runs])


def test_candidate_list_is_capped():
    question = "Compare Alpha Corp, Beta Industries, Gamma Holdings, Delta Systems and Epsilon Labs"
    assert len(extract_entity_candidates(question)) <= entity_resolution.MAX_CANDIDATES


# --- Stage 2: parsing the verification verdict ------------------------


def test_parse_found_verdict():
    verdict, findings = _parse_research_response(
        "VERDICT: FOUND\nFINDINGS: Recall.ai is an API for meeting recording bots."
    )
    assert verdict == FOUND
    assert "meeting recording" in findings


def test_parse_not_found_verdict_keeps_the_near_miss_text():
    # The near-miss has to survive into the findings: it's what the
    # clarification shows the user so they can confirm or correct.
    verdict, findings = _parse_research_response(
        "VERDICT: NOT_FOUND\nFINDINGS: No organisation called My Health School was found. "
        "The closest result was the School Health and Wellness Programme, a different name."
    )
    assert verdict == NOT_FOUND
    assert "School Health and Wellness Programme" in findings


def test_parse_ambiguous_verdict():
    verdict, _ = _parse_research_response("VERDICT: AMBIGUOUS\nFINDINGS: Two companies use this name.")
    assert verdict == AMBIGUOUS


def test_unparseable_reply_degrades_to_found_with_the_raw_text():
    # A formatting slip shouldn't cost the user their answer - the text
    # still came from a real grounded search, so it's still usable
    # context. See _parse_research_response's docstring.
    verdict, findings = _parse_research_response("Recall.ai is a meeting-bot API company.")
    assert verdict == FOUND
    assert findings == "Recall.ai is a meeting-bot API company."


def test_unrecognized_verdict_word_degrades_the_same_way():
    verdict, _ = _parse_research_response("VERDICT: MAYBE\nFINDINGS: unclear")
    assert verdict == FOUND


def test_grounding_citation_markers_are_stripped_from_findings():
    # Observed live: the NOT_FOUND clarification shown to the user read
    # "...do not identify a distinct entity [2.1, 2.2, 2.3]." Those
    # markers key to grounding chunks the user never sees.
    _, findings = _parse_research_response(
        "VERDICT: NOT_FOUND\nFINDINGS: No entity carries this name [2.1, 2.2]. "
        "The closest result was a truck part [3.1]."
    )
    assert "[" not in findings
    assert findings == "No entity carries this name. The closest result was a truck part."


def test_a_real_bracketed_aside_is_not_stripped():
    # The strip is deliberately narrow - only all-numeric bracket groups.
    _, findings = _parse_research_response(
        "VERDICT: FOUND\nFINDINGS: Recall.ai [formerly Recall API] is a meeting-bot platform."
    )
    assert "[formerly Recall API]" in findings


def test_empty_reply_has_no_verdict():
    # None is the caller's signal to fall back to the plain path.
    assert _parse_research_response("") == (None, "")


# --- Stage 3: which branch each verdict takes -------------------------


def _run(coroutine):
    return asyncio.run(coroutine)


async def _collect(stream):
    """Collect chat_service.answer_general_knowledge()'s {"type": "chunk"/
    "done", ...} event stream (see its own docstring) into the plain
    {"answer": str, "sources": [...]} shape this test file's assertions
    were written against, before that function became a streaming async
    generator. No production code needs this (routes/chat.py always
    relays the stream onward, never collects it) - it's a test-only
    convenience."""

    chunks = []
    sources = []
    async for event in stream:
        if event["type"] == "chunk":
            chunks.append(event["text"])
        else:
            sources = event.get("sources", [])
    return {"answer": "".join(chunks), "sources": sources}


def _stub_verification(monkeypatch, verdict, findings="findings text", sources=None):
    async def fake_verify(question, candidates):
        return verdict, findings, sources if sources is not None else []

    monkeypatch.setattr(entity_resolution, "verify_entities", fake_verify)


def _stub_generate(monkeypatch, record):
    # answer_general_knowledge streams its answer via generate_stream()
    # now, not the single-shot generate() this used to patch - yields
    # the fixed response as one chunk, which is all these control-flow
    # tests need (real incremental chunking is covered in gemini_client's
    # own live verification, not duplicated here).
    async def fake_generate_stream(prompt):
        record.append(prompt)
        yield "generated answer"

    monkeypatch.setattr(chat_service, "generate_stream", fake_generate_stream)


def test_question_without_entities_never_calls_verification(monkeypatch):
    prompts = []
    _stub_generate(monkeypatch, prompts)

    async def explode(question, candidates):  # pragma: no cover - must not run
        raise AssertionError("verify_entities was called for a question with no named entity")

    monkeypatch.setattr(entity_resolution, "verify_entities", explode)

    result = _run(_collect(chat_service.answer_general_knowledge("What is machine learning?")))

    assert result["answer"] == "generated answer"
    assert result["sources"] == []
    assert "ENTITY CONTEXT" not in prompts[0]


def test_found_entity_answers_with_the_search_context_attached(monkeypatch):
    prompts = []
    _stub_generate(monkeypatch, prompts)
    _stub_verification(
        monkeypatch,
        FOUND,
        findings="Recall.ai sells an API for meeting recording bots.",
        sources=[{"type": "web", "title": "Recall.ai", "url": "https://recall.ai"}],
    )

    result = _run(_collect(chat_service.answer_general_knowledge("What is Recall.ai?")))

    assert result["answer"] == "generated answer"
    assert result["sources"][0]["url"] == "https://recall.ai"
    # The verified context and the verbatim entity both have to reach the
    # answer prompt - the context so the answer is grounded, the entity
    # so the no-substitution rule has something to name.
    assert "meeting recording bots" in prompts[0]
    assert '"Recall.ai"' in prompts[0]


def test_unverifiable_entity_asks_instead_of_answering(monkeypatch):
    # The core regression. An entity the web can't confirm must NOT reach
    # a free-form Gemini answer, because that is exactly how "My Health
    # School" came back as the School Health and Wellness Programme.
    prompts = []
    _stub_generate(monkeypatch, prompts)
    _stub_verification(
        monkeypatch,
        NOT_FOUND,
        findings="Closest match was the School Health and Wellness Programme, a different name.",
    )

    result = _run(_collect(chat_service.answer_general_knowledge("What is My Health School in India?")))

    assert prompts == [], "no answer should have been generated for an unverified entity"
    assert "My Health School" in result["answer"]
    assert "?" in result["answer"], "the response should be asking something"


def test_ambiguous_entity_asks_which_one(monkeypatch):
    prompts = []
    _stub_generate(monkeypatch, prompts)
    _stub_verification(monkeypatch, AMBIGUOUS, findings="Two different companies use this name.")

    result = _run(_collect(chat_service.answer_general_knowledge("What is Integfarms?")))

    assert prompts == []
    assert "Integfarms" in result["answer"]
    assert "Two different companies" in result["answer"]


def test_verification_failure_falls_back_to_the_plain_answer(monkeypatch):
    # A search outage should degrade this feature, not fail the request -
    # same graceful-degradation rule the database guards follow.
    prompts = []
    _stub_generate(monkeypatch, prompts)
    _stub_verification(monkeypatch, None, findings="")

    result = _run(_collect(chat_service.answer_general_knowledge("What is Recall.ai?")))

    assert result["answer"] == "generated answer"
    assert result["sources"] == []
    assert "ENTITY CONTEXT" not in prompts[0]


def test_language_instruction_still_applies_on_the_grounded_path(monkeypatch):
    # Voice input in Tamil must keep answering in Tamil after entity
    # resolution, exactly as it did before.
    prompts = []
    _stub_generate(monkeypatch, prompts)
    _stub_verification(monkeypatch, FOUND, findings="context")

    _run(_collect(chat_service.answer_general_knowledge("What is Recall.ai?", language="ta")))

    assert "Tamil" in prompts[0]


def test_lowercase_my_health_school_reaches_verification_not_the_plain_path(monkeypatch):
    # End-to-end version of the bug fix: previously this question had no
    # entity candidate at all, so verify_entities was never called and
    # Gemini saw only the raw question with no signal that "my health
    # school" wasn't personal possession - it answered as if asked about
    # the user's own unnamed school.
    prompts = []
    _stub_generate(monkeypatch, prompts)
    _stub_verification(
        monkeypatch,
        FOUND,
        findings="My Health School is a health education platform in Chennai, India.",
        sources=[{"type": "web", "title": "My Health School", "url": "https://myhealthschool.in"}],
    )

    result = _run(_collect(chat_service.answer_general_knowledge("What is my health school?")))

    assert result["sources"][0]["url"] == "https://myhealthschool.in"
    assert "Chennai" in prompts[0]
    assert '"my health school"' in prompts[0]


def test_my_school_still_takes_the_plain_personal_path(monkeypatch):
    # No entity candidate for "my school" - must never call verification,
    # and must keep answering as an unresolved personal reference exactly
    # as before this fix.
    prompts = []
    _stub_generate(monkeypatch, prompts)

    async def explode(question, candidates):  # pragma: no cover - must not run
        raise AssertionError("verify_entities was called for a plain possessive, not a name")

    monkeypatch.setattr(entity_resolution, "verify_entities", explode)

    result = _run(_collect(chat_service.answer_general_knowledge("What is my school?")))

    assert result["answer"] == "generated answer"
    assert result["sources"] == []
    assert "ENTITY CONTEXT" not in prompts[0]
