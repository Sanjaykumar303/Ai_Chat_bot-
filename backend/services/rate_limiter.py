"""
Minimal in-memory rate limiter, keyed by client IP - caps how many
requests one client can make, per window, to the endpoints that trigger
a paid Gemini call (/chat, /transcribe, /documents/upload). Without
this, one client (scripted or just a runaway retry loop) can run up the
Gemini bill or exhaust the shared API quota for everyone else, since
this project has no authentication to fall back on (see the project's
own deferred-issues list) - this doesn't identify who's making
requests, it just bounds how fast any one source can spend money.

Same module-level dict + lock shape document_store.py's document cache
uses - built up at runtime, guarded against concurrent access, nothing
persisted, sized for a single-process deployment like this one rather
than a distributed store (Redis, etc.) a multi-worker production
deployment would eventually want.
"""

import os
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

_lock = threading.Lock()
_requests_by_client = {}  # client_key -> deque of request timestamps (time.monotonic())


def _client_key(request: Request):
    """The client's IP, preferring the first hop of X-Forwarded-For (set
    by Render's edge/any reverse proxy in front of this app) over the
    direct peer address, which would otherwise just be the proxy itself.
    Trusting a client-supplied header is normally a spoofing risk, but
    with no auth layer to begin with, a client that wants to dodge this
    limit could already just connect from a different real IP - this
    only has to raise the bar above "one accidental retry loop", not
    stand up to a determined attacker."""

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request):
    """FastAPI dependency - raises HTTPException(429) once this client
    has made RATE_LIMIT_MAX_REQUESTS requests (to whichever endpoint(s)
    also depend on this) within the trailing RATE_LIMIT_WINDOW_SECONDS.

    A client's entry is only cleaned up the next time that same client
    is checked again (same lazy-eviction tradeoff document_store.py's
    get_document() makes, "enough for a single-process dev deployment
    like this one") - a one-off visitor who never returns leaves a tiny
    (one string key + a short deque of floats) entry behind rather than
    the large per-document content document_store.py holds, so the same
    tradeoff is far cheaper to leave unswept here.
    """

    key = _client_key(request)
    now = time.monotonic()

    with _lock:
        timestamps = _requests_by_client.get(key)

        if timestamps is not None:
            while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW_SECONDS:
                timestamps.popleft()
            if not timestamps:
                del _requests_by_client[key]
                timestamps = None

        if timestamps is not None and len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests. Please wait {retry_after}s and try again.",
                headers={"Retry-After": str(retry_after)},
            )

        _requests_by_client.setdefault(key, deque()).append(now)
