# Regression tests for the manual Web Search toggle
# (chat_service.answer_general_knowledge's web_search parameter,
# routes/chat.py's ChatRequest.web_search).
#
# Scope, matching test_entity_resolution.py's own rule: cover the
# deterministic branching (does web_search=True call generate_with_search
# and skip entity resolution entirely; does web_search=False/omitted stay
# byte-identical to before this feature). The live Gemini search call
# itself is stubbed, never called - verified for real separately (see
# [[project_web_search_toggle_handoff_2026-08-19]]).

import asyncio

from services import chat_service, entity_resolution


def _run(coroutine):
    return asyncio.run(coroutine)


async def _collect(stream):
    chunks = []
    sources = []
    async for event in stream:
        if event["type"] == "chunk":
            chunks.append(event["text"])
        else:
            sources = event.get("sources", [])
    return {"answer": "".join(chunks), "sources": sources}


def _stub_generate_stream(monkeypatch, record):
    async def fake_generate_stream(prompt):
        record.append(prompt)
        yield "plain answer"

    monkeypatch.setattr(chat_service, "generate_stream", fake_generate_stream)


def _stub_generate_with_search(monkeypatch, record, text="web answer", sources=None):
    async def fake_generate_with_search(prompt):
        record.append(prompt)
        return text, sources if sources is not None else [{"type": "web", "title": "Example", "url": "https://example.com"}]

    monkeypatch.setattr(chat_service, "generate_with_search", fake_generate_with_search)


def test_web_search_true_calls_generate_with_search_and_never_entity_resolution(monkeypatch):
    search_prompts = []
    _stub_generate_with_search(monkeypatch, search_prompts)

    def explode(question):  # pragma: no cover - must not run
        raise AssertionError("entity_resolution.extract_entity_candidates was called with web_search=True")

    monkeypatch.setattr(entity_resolution, "extract_entity_candidates", explode)

    async def explode_verify(question, candidates):  # pragma: no cover - must not run
        raise AssertionError("entity_resolution.verify_entities was called with web_search=True")

    monkeypatch.setattr(entity_resolution, "verify_entities", explode_verify)

    result = _run(_collect(chat_service.answer_general_knowledge("What is the latest news on Recall.ai?", web_search=True)))

    assert result["answer"] == "web answer"
    assert result["sources"] == [{"type": "web", "title": "Example", "url": "https://example.com"}]
    assert len(search_prompts) == 1
    assert "What is the latest news on Recall.ai?" in search_prompts[0]


def test_web_search_false_never_calls_generate_with_search(monkeypatch):
    prompts = []
    _stub_generate_stream(monkeypatch, prompts)

    async def explode(prompt):  # pragma: no cover - must not run
        raise AssertionError("generate_with_search was called with web_search=False")

    monkeypatch.setattr(chat_service, "generate_with_search", explode)

    # A plain question with no named entity, so this also never reaches
    # verify_entities - isolates this assertion to just the web_search flag.
    result = _run(_collect(chat_service.answer_general_knowledge("What is machine learning?", web_search=False)))

    assert result["answer"] == "plain answer"
    assert result["sources"] == []


def test_web_search_omitted_defaults_to_false_and_is_byte_identical(monkeypatch):
    prompts = []
    _stub_generate_stream(monkeypatch, prompts)

    async def explode(prompt):  # pragma: no cover - must not run
        raise AssertionError("generate_with_search was called when web_search was omitted")

    monkeypatch.setattr(chat_service, "generate_with_search", explode)

    with_flag = _run(_collect(chat_service.answer_general_knowledge("What is machine learning?", web_search=False)))
    prompts.clear()
    omitted = _run(_collect(chat_service.answer_general_knowledge("What is machine learning?")))

    assert with_flag == omitted
