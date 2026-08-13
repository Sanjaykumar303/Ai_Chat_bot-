import logging

# Loaded first, before any other import reads an environment variable.
import config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import Routers
from routes.chat import router as chat_router
from routes.transcribe import router as transcribe_router
from routes.documents import router as documents_router
from services import transcription
from services import db_client

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
def warm_up_whisper():
    """Load the Whisper model once so the first voice question isn't slow.

    Downloads the model from Hugging Face on first run only, then reuses
    the local cache. Failure here (e.g. no internet on first run) doesn't
    crash startup - /transcribe will surface a clear error instead.
    """

    try:
        transcription.load_model()
    except Exception as error:
        logger.warning(f"Could not load the Whisper model at startup: {error}")


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


# Home Route
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Backend Running"
    }