# services/export_store.py - the same "TTL-cached, lazily evicted on
# lookup" shape as test_document_store.py's own tests exercise, just for
# a much shorter-lived, much smaller kind of stored object (a generated
# export file, not an uploaded document with a retrieval index).

import pytest

from services import export_store


@pytest.fixture(autouse=True)
def _reset_export_store():
    export_store._store.clear()
    yield
    export_store._store.clear()


def test_create_then_get_returns_the_exact_bytes_and_metadata():
    export_id = export_store.create_export(b"docx-bytes", filename="answer.docx", content_type="application/x-docx")

    entry = export_store.get_export(export_id)

    assert entry["bytes"] == b"docx-bytes"
    assert entry["filename"] == "answer.docx"
    assert entry["content_type"] == "application/x-docx"


def test_unknown_export_id_returns_none():
    assert export_store.get_export("does-not-exist") is None


def test_expired_export_is_evicted_on_lookup(monkeypatch):
    monkeypatch.setattr(export_store, "EXPORT_TTL_SECONDS", -1)  # already "expired" the instant it's created
    export_id = export_store.create_export(b"bytes", filename="f.xlsx", content_type="application/x-xlsx")

    assert export_store.get_export(export_id) is None
    assert export_id not in export_store._store


def test_fresh_export_is_not_evicted():
    monkeypatched_id = export_store.create_export(b"bytes", filename="f.xlsx", content_type="application/x-xlsx")
    assert export_store.get_export(monkeypatched_id) is not None
