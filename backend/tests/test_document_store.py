# get_document() alone only evicts an entry the next time that exact
# document_id is looked up - a document uploaded and never revisited
# (tab closed, conversation abandoned) would otherwise stay in memory
# forever. sweep_expired() is the periodic backstop for that (see
# main.py's background task) - these tests cover it directly.

import pytest

from services import document_store


@pytest.fixture(autouse=True)
def _reset_document_store():
    document_store._documents.clear()
    yield
    document_store._documents.clear()


def test_sweep_removes_an_expired_document_never_looked_up(monkeypatch):
    monkeypatch.setattr(document_store, "DOCUMENT_TTL_SECONDS", -1)  # already "expired" the instant it's created
    doc_id = document_store.create_document("test.pdf", ["chunk"], None, None)

    removed = document_store.sweep_expired()

    assert removed == 1
    assert doc_id not in document_store._documents


def test_sweep_leaves_a_fresh_document_alone(monkeypatch):
    monkeypatch.setattr(document_store, "DOCUMENT_TTL_SECONDS", 3600)
    doc_id = document_store.create_document("test.pdf", ["chunk"], None, None)

    removed = document_store.sweep_expired()

    assert removed == 0
    assert doc_id in document_store._documents


def test_sweep_only_removes_expired_entries_not_everything(monkeypatch):
    # DOCUMENT_TTL_SECONDS is read fresh at sweep time (not frozen at
    # creation time), so simulating "one old, one new" means backdating
    # created_at directly rather than juggling the TTL setting itself
    # between the two create_document() calls.
    monkeypatch.setattr(document_store, "DOCUMENT_TTL_SECONDS", 3600)
    fresh_id = document_store.create_document("fresh.pdf", ["chunk"], None, None)
    expired_id = document_store.create_document("expired.pdf", ["chunk"], None, None)
    document_store._documents[expired_id]["created_at"] -= 7200

    removed = document_store.sweep_expired()

    assert removed == 1
    assert fresh_id in document_store._documents
    assert expired_id not in document_store._documents
