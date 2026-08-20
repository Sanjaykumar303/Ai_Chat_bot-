"""
Tiny in-memory holding pen for a just-generated export file (.docx/.xlsx)
between the moment routes/chat.py creates it and the moment the frontend
fetches it via GET /export/download/{export_id} (routes/export.py).

Deliberately much simpler than services/document_store.py (no
vectorizer/chunks, no periodic background sweep task registered in
main.py): an export file is small (a handful of KB) and downloaded
within moments of being created, so lazy eviction on lookup - the same
mechanism document_store.get_document() used before its own periodic
sweep was added for a much longer-lived, much larger kind of stored
object - is proportionate here.
"""

import threading
import time
import uuid

# Generous for "click the download link that just appeared" without
# holding files in memory indefinitely if a response is never opened.
EXPORT_TTL_SECONDS = 600

_lock = threading.Lock()
_store = {}  # {export_id: {"bytes", "filename", "content_type", "created_at"}}


def create_export(file_bytes, filename, content_type):
    export_id = uuid.uuid4().hex
    with _lock:
        _store[export_id] = {
            "bytes": file_bytes,
            "filename": filename,
            "content_type": content_type,
            "created_at": time.time(),
        }
    return export_id


def get_export(export_id):
    """Returns {"bytes", "filename", "content_type"} or None if missing
    or expired. An expired entry is dropped right here rather than left
    for a background task - see the module docstring for why that's
    enough for something this short-lived."""

    with _lock:
        entry = _store.get(export_id)
        if entry is None:
            return None
        if time.time() - entry["created_at"] > EXPORT_TTL_SECONDS:
            del _store[export_id]
            return None
        return entry
