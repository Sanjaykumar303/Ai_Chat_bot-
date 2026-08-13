"""
Temporary, in-memory holding area for uploaded-PDF chunks + their
retrieval index, keyed by document_id - never written to disk, and never
anywhere near Supabase/Postgres (that's the whole point of this feature:
the PDF itself must never be stored in the database).

Keyed by document_id rather than a single global variable so more than
one upload can exist at once (e.g. two browser tabs, or a future
multi-user deployment) without one replacing another's context.
document_id doubles as the "session" handle the frontend holds onto -
there's no separate session/auth layer in this project to hang it off of.

Same module-level dict + lock shape db_client.py's schema cache and
rag_service.py's (since-removed) FAISS index used - built up at runtime,
guarded against concurrent access, nothing persisted.
"""

import os
import threading
import time
import uuid

DOCUMENT_TTL_SECONDS = int(os.getenv("DOCUMENT_TTL_SECONDS", str(60 * 60)))

_lock = threading.RLock()
_documents = {}  # document_id -> {filename, chunks, vectorizer, matrix, created_at}


def create_document(filename, chunks, vectorizer, matrix):
    """Store a newly processed document's chunks + retrieval index.
    Returns the new document_id. Each call gets its own id - uploading a
    replacement never touches an earlier document's entry, so an old
    document_id the frontend has already discarded simply expires on its
    own (or can be deleted explicitly via delete_document)."""

    document_id = uuid.uuid4().hex

    with _lock:
        _documents[document_id] = {
            "filename": filename,
            "chunks": chunks,
            "vectorizer": vectorizer,
            "matrix": matrix,
            "created_at": time.time(),
        }

    return document_id


def get_document(document_id):
    """Return the stored entry for document_id, or None if it doesn't
    exist or has expired (past DOCUMENT_TTL_SECONDS since upload - lazily
    evicted here rather than via a background sweep, which is enough for
    a single-process dev deployment like this one)."""

    if not document_id:
        return None

    with _lock:
        doc = _documents.get(document_id)

        if doc is None:
            return None

        if time.time() - doc["created_at"] > DOCUMENT_TTL_SECONDS:
            del _documents[document_id]
            return None

        return doc


def delete_document(document_id):
    """Remove a document's temporary context immediately (user clicked
    "Remove PDF", or is replacing it). Returns True if something was
    actually removed."""

    if not document_id:
        return False

    with _lock:
        return _documents.pop(document_id, None) is not None
