"""
Central place for environment-derived settings that more than one module
needs. Imported first (before any other local import) so .env is loaded
exactly once, regardless of import order elsewhere.

Settings used by only a single file stay defined in that file instead
(e.g. DB_QUERY_ROW_LIMIT in services/db_client.py, CHUNK_SIZE in
services/pdf_service.py) - moving those here would just relocate them
away from the code that uses them, not remove any duplication.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEBUG_VOICE_PIPELINE = os.getenv("DEBUG_VOICE_PIPELINE", "false").lower() == "true"

# Comma-separated list of deployed frontend origin(s) (e.g. a Vercel
# domain), set in the host's environment variables, not committed to
# source. Unset in local dev, where routes/../main.py's regex-based
# localhost allowance already covers it.
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGIN", "").split(",")
    if origin.strip()
]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
