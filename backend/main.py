import asyncio
import logging
import os

# Loaded first, before any other import reads an environment variable.
import config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# Import Routers
from routes.chat import router as chat_router
from routes.transcribe import router as transcribe_router
from routes.documents import router as documents_router
from services import db_client, document_store

# How often the background sweep below removes expired uploaded
# documents that were never looked up again (see
# document_store.sweep_expired()'s own docstring for why this needs to
# be periodic, not just the lazy per-lookup eviction get_document()
# already does).
DOCUMENT_SWEEP_INTERVAL_SECONDS = int(os.getenv("DOCUMENT_SWEEP_INTERVAL_SECONDS", str(15 * 60)))

logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="AI Document Assistant",
    description="Backend API for AI Document Assistant",
    version="1.0.0"
)

# CORS Configuration
#
# The browser blocks requests to this API unless the address of the
# React app is allowed here.
#
# Vite normally runs on port 5173, but it quietly moves to 5174, 5175
# and so on when that port is already busy. The regex below accepts
# localhost on ANY port, so local dev keeps working after such a move.
#
# The deployed frontend (e.g. a Vercel domain) isn't localhost, so it's
# allowed separately via config.FRONTEND_ORIGINS - a comma-separated list
# of the real site address(es), set in the host's environment variables,
# not committed to source. Unset in local dev, where it's not needed.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_origins=config.FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routes
app.include_router(chat_router)
app.include_router(transcribe_router)
app.include_router(documents_router)


@app.on_event("startup")
def warm_up_db_schema():
    """Introspect the database schema once so the first database question
    isn't slow. Best-effort, same as the two hooks above - a missing/bad
    SUPABASE_DB_URL logs a warning and the app still boots; /chat's
    database path will surface a clear failure at request time instead.
    """

    try:
        tables = db_client.get_table_allowlist()
        logger.info(f"Database schema loaded: {len(tables)} table(s) available for questions.")
    except Exception as error:
        logger.warning(f"Could not load the database schema at startup: {error}")


async def _sweep_expired_documents_forever():
    """Runs for the life of the process, sweeping document_store.py's
    in-memory dict every DOCUMENT_SWEEP_INTERVAL_SECONDS - see
    sweep_expired()'s own docstring for why lazy eviction alone isn't
    enough. Sleeps first so this doesn't do pointless work checking an
    empty/freshly-started store immediately at boot."""

    while True:
        await asyncio.sleep(DOCUMENT_SWEEP_INTERVAL_SECONDS)
        try:
            removed = await run_in_threadpool(document_store.sweep_expired)
            if removed:
                logger.info(f"Swept {removed} expired document(s) from memory.")
        except Exception as error:
            logger.warning(f"Document sweep failed (will retry next interval): {error}")


@app.on_event("startup")
async def start_document_sweep():
    asyncio.create_task(_sweep_expired_documents_forever())


# Home Route
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Backend Running"
    }


@app.get("/health")
async def health():
    """Real connectivity check, unlike "/" (which just proves the process
    is up) - a monitoring tool hitting "/" alone can't tell "up" apart
    from "up but the database or Gemini key is broken".

    Database: a live, uncached ping (db_client.ping() - see its own
    docstring for why this isn't the TTL-cached schema instead).

    Gemini: presence of GEMINI_API_KEY only, not a live generate() call -
    an actual request per health check would cost real money and add
    latency on every hit from a monitoring tool that might poll every
    few seconds; a missing/absent key is by far the most common way this
    dependency is actually broken (see gemini_client._require_api_key()),
    and is checked the same lightweight way /chat's own pre-check does.

    Returns 200 if every checked component is healthy, 503 if any isn't -
    the response body always lists each component's individual status
    either way, so a human or a dashboard doesn't have to guess which one
    failed from the status code alone.
    """

    components = {}

    try:
        await run_in_threadpool(db_client.ping)
        components["database"] = "ok"
    except Exception as error:
        components["database"] = f"unreachable: {error}"

    components["gemini"] = "configured" if config.GEMINI_API_KEY else "GEMINI_API_KEY is not set"

    healthy = components["database"] == "ok" and components["gemini"] == "configured"

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "healthy" if healthy else "unhealthy", "components": components},
    )