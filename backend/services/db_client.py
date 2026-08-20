"""
Talks to the Supabase Postgres database - connection, schema introspection,
and query execution only. No Gemini calls and no SQL safety validation live
here; sql_guard.py owns that, kept separate so this module stays a plain,
independently-testable "talk to Postgres" layer, and db_query_service.py
owns the orchestration between the two.

Connecting here should be a dedicated READ-ONLY Postgres role, not the
default Supabase "postgres" superuser - see the read-only role setup in
the project plan. This module can't enforce that (a connection string is
just a connection string), so it logs a startup warning if the configured
user looks like the superuser, but the real enforcement is sql_guard.py's
structural validation plus whatever the DB role actually grants.
"""

import logging
import os
import threading
import time
from urllib.parse import quote_plus

from sqlalchemy import create_engine, inspect, text

logger = logging.getLogger("uvicorn")

DB_SCHEMA_NAME = os.getenv("DB_SCHEMA_NAME", "public")
DB_QUERY_ROW_LIMIT = int(os.getenv("DB_QUERY_ROW_LIMIT", "100"))
DB_QUERY_TIMEOUT_MS = int(os.getenv("DB_QUERY_TIMEOUT_MS", "5000"))
DB_SCHEMA_CACHE_TTL = int(os.getenv("DB_SCHEMA_CACHE_TTL", "600"))

# Restricts which tables the Gemini SQL-generation path can even see.
# Empty (unset) means no restriction - every table in DB_SCHEMA_NAME is
# exposed, which is almost certainly not what you want on a database with
# tables you haven't reviewed yet. Filtering happens in _load_schema()
# itself, before the schema description/allowlist/routing vocabulary are
# built, so an excluded table isn't just rejected later by sql_guard -
# Gemini never sees its name or columns in the first place.
DB_QUERY_ALLOWED_TABLES = {
    name.strip().lower()
    for name in os.getenv("DB_QUERY_ALLOWED_TABLES", "").split(",")
    if name.strip()
}


class DatabaseError(Exception):
    """Base class for problems raised by this module."""


class DatabaseConnectionError(DatabaseError):
    """Raised when the database connection isn't configured or can't be reached."""


class DatabaseQueryError(DatabaseError):
    """Raised when an already-validated query fails to execute."""


# Same module-level cache + lock shape as rag_service.py's FAISS index -
# built once per process and reused, protected against concurrent access.
# RLock (not Lock): _ensure_schema_loaded() holds this while calling
# _load_schema(), which calls get_engine(), which also acquires this same
# lock - a plain Lock would deadlock on that nested acquisition.
_lock = threading.RLock()
_engine = None
_schema_cache = None  # {"description", "tables", "routing_terms", "cached_at"}


def _build_connection_url():
    """Prefer a single SUPABASE_DB_URL; fall back to discrete PG* vars
    (the standard libpq environment variable names) if that isn't set."""

    url = os.getenv("SUPABASE_DB_URL")
    if url:
        return url

    host = os.getenv("PGHOST")
    user = os.getenv("PGUSER")

    if not host or not user:
        return None

    port = os.getenv("PGPORT", "5432")
    password = os.getenv("PGPASSWORD", "")
    database = os.getenv("PGDATABASE", "postgres")
    sslmode = os.getenv("PGSSLMODE", "require")

    return (
        f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{database}?sslmode={sslmode}"
    )


def _warn_if_privileged_role(user):
    """Best-effort nudge, not a security control: Supabase's pooler prefixes
    the username with the role name (e.g. "postgres.<project-ref>" for the
    default superuser, "readonly_chat_role.<project-ref>" for a dedicated
    role) - if it still looks like the superuser, say so loudly."""

    role_name = user.split(".")[0] if user else ""
    if role_name == "postgres":
        logger.warning(
            "SUPABASE/PG connection is using the 'postgres' superuser, not a dedicated "
            "read-only role. sql_guard.py still blocks writes at the app layer, but the "
            "database itself is not enforcing read-only - see the plan's read-only role "
            "setup SQL to close this gap."
        )


def get_engine():
    """Build (once) and return the cached SQLAlchemy engine.

    Raises DatabaseConnectionError with a user-facing message if no
    connection is configured - mirrors gemini_client._require_api_key()'s
    "clear message, don't crash startup" style.
    """

    global _engine

    if _engine is not None:
        return _engine

    with _lock:
        if _engine is None:
            url = _build_connection_url()
            if not url:
                raise DatabaseConnectionError(
                    "No database connection configured. Set SUPABASE_DB_URL, or "
                    "PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE, in backend/.env"
                )

            user = os.getenv("PGUSER") or ""
            if user:
                _warn_if_privileged_role(user)

            _engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=2,
                connect_args={"connect_timeout": 10},
            )

    return _engine


def ping():
    """Run a trivial query to confirm the database is actually reachable
    right now. Deliberately not get_table_allowlist()/get_schema_terms()
    - those are TTL-cached (DB_SCHEMA_CACHE_TTL), so a health check built
    on them could report "healthy" from stale cached data even if the
    database just went down. Raises DatabaseError (via get_engine() or
    the query itself) on any failure - callers decide what to do with
    that, this only proves or disproves connectivity."""

    engine = get_engine()

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as error:
        raise DatabaseQueryError(f"Database ping failed: {error}") from error


def _schema_cache_expired():
    if _schema_cache is None:
        return True
    return (time.time() - _schema_cache["cached_at"]) > DB_SCHEMA_CACHE_TTL


def _find_date_column(columns):
    """Return the first date/timestamp-typed column name, or None.

    Best-effort heuristic (first match by column type, not by name) - a
    table with several date columns (e.g. bill_date and due_date) only
    gets one checked. Good enough for surfacing "this table doesn't cover
    the period you're asking about" without hardcoding any table's actual
    column name.
    """

    for col in columns:
        type_name = str(col["type"]).lower()
        if "date" in type_name or "timestamp" in type_name:
            return col["name"]
    return None


# Small, bounded set of the actual distinct values a low-cardinality
# character column holds (e.g. a status/category column) - shown inline
# in the schema description below so the SQL-generation prompt can filter
# on the REAL stored spelling/casing (e.g. "overdue") instead of guessing
# one (e.g. "Overdue") and silently matching nothing. A genuinely
# free-text column (name, email, description) will have far more than
# this many distinct values and is silently skipped - no per-column-name
# guessing needed, same "show a fact about the live data instead of
# leaving the model to invent one" idea the date-range annotation below
# already uses for date columns.
MAX_DISTINCT_VALUES = 12


def _is_character_type(sqlalchemy_type):
    type_name = str(sqlalchemy_type).lower()
    return any(marker in type_name for marker in ("char", "text"))


def _table_extra_facts(conn, table_name, columns):
    """One combined query per table computing BOTH the date-range
    annotation and every character column's low-cardinality distinct-
    value set, in a single round trip.

    This exists because round-trip COUNT, not per-query complexity, is
    what actually dominates schema-load latency against a remote/hosted
    DB - confirmed live against this project's own Supabase instance,
    where a trivial `SELECT 1` and a real `SELECT DISTINCT ...` both cost
    ~130ms. An earlier version of this annotation issued one query per
    date column PLUS one query per character column, which pushed a full
    schema reload (DB_SCHEMA_CACHE_TTL) past 25 seconds on this project's
    real 22-table schema (43 character columns alone) - a real regression
    introduced while adding the distinct-value annotation, caught and
    fixed the same session via live timing, not just by review. This
    brings a full reload back down to roughly one round trip per table.

    Built as a single SELECT with no FROM clause at the top level - every
    value is instead its own self-contained scalar subquery (each with
    its own FROM/WHERE/LIMIT) - so it always returns exactly one row
    regardless of whether the underlying table is empty, with no
    aggregate-vs-non-aggregate row-count ambiguity to handle.

    Returns (date_range_note: str, column_values: {col_name: [values]}).
    Both come back empty on any failure or if the table has neither a
    date column nor a character column - non-fatal, matching this
    function's own two predecessors' behavior.
    """

    date_column = _find_date_column(columns)
    char_columns = [col for col in columns if _is_character_type(col["type"])]

    select_parts = []
    if date_column is not None:
        select_parts.append(f'(SELECT MIN("{date_column}") FROM "{table_name}") AS date_min')
        select_parts.append(f'(SELECT MAX("{date_column}") FROM "{table_name}") AS date_max')
    for col in char_columns:
        select_parts.append(
            f'(SELECT array_agg(v) FROM (SELECT DISTINCT "{col["name"]}" AS v FROM "{table_name}" '
            f'WHERE "{col["name"]}" IS NOT NULL LIMIT {MAX_DISTINCT_VALUES + 1}) sub) AS "vals__{col["name"]}"'
        )

    if not select_parts:
        return "", {}

    try:
        row = conn.execute(text("SELECT " + ", ".join(select_parts))).mappings().first()
    except Exception:
        return "", {}  # non-fatal - schema description just won't have these facts for this table

    if row is None:
        return "", {}

    date_range_note = ""
    if date_column is not None:
        min_date, max_date = row["date_min"], row["date_max"]
        if min_date is None or max_date is None:
            date_range_note = f" -- {date_column}: no rows"
        else:
            date_range_note = f" -- {date_column} data available from {min_date} to {max_date}"

    column_values = {}
    for col in char_columns:
        values = row.get(f"vals__{col['name']}")
        if values and len(values) <= MAX_DISTINCT_VALUES:
            column_values[col["name"]] = list(values)

    return date_range_note, column_values


def _load_schema():
    """Introspect live tables/columns. Not cached forever like the FAISS
    index - unlike documents (which only change via this app's own
    /upload), the Supabase schema can change externally (Studio,
    migrations), so it's re-read after DB_SCHEMA_CACHE_TTL seconds."""

    engine = get_engine()

    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names(schema=DB_SCHEMA_NAME)

        if DB_QUERY_ALLOWED_TABLES:
            table_names = [name for name in table_names if name.lower() in DB_QUERY_ALLOWED_TABLES]

        # get_multi_columns() reflects every table's columns in ONE round
        # trip instead of one per table via get_columns() in a loop -
        # confirmed live against this project's real 22-table schema:
        # get_columns() x22 cost ~13s (each call itself issuing several
        # catalog queries), get_multi_columns() for the same 22 tables
        # cost ~1.5s. This was the single largest contributor to a slow
        # cold schema load, bigger than the per-table _table_extra_facts
        # calls below - fixed the same session those were added in, once
        # live timing showed it was the actual bottleneck.
        all_columns = inspector.get_multi_columns(schema=DB_SCHEMA_NAME)

        lines = []
        table_terms = set()
        column_terms = set()

        with engine.connect() as conn:
            for table_name in table_names:
                columns = all_columns[(DB_SCHEMA_NAME, table_name)]
                date_range_note, column_values = _table_extra_facts(conn, table_name, columns)

                column_parts = []
                for col in columns:
                    part = f"{col['name']} {col['type']}"
                    distinct_values = column_values.get(col["name"])
                    if distinct_values:
                        # Doubled single quotes, same escaping SQL itself
                        # uses for a literal - keeps a value like "Owner's
                        # Equity" from breaking out of its own quoting in
                        # the displayed schema, and shows the value in
                        # exactly the form a WHERE clause would need to
                        # write it.
                        escaped_values = (value.replace("'", "''") for value in distinct_values)
                        values_text = ", ".join(f"'{value}'" for value in escaped_values)
                        part += f" [values: {values_text}]"
                    column_parts.append(part)
                column_descriptions = ", ".join(column_parts)
                lines.append(f"Table {table_name}({column_descriptions}){date_range_note}")

                lower_name = table_name.lower()
                table_terms.add(lower_name)
                # Naive singular (strip a trailing "s") so a question about
                # "student 3" or "the student" still matches a "students"
                # table - doesn't handle irregular plurals (e.g. "children"),
                # but covers ordinary table-naming conventions.
                if lower_name.endswith("s") and len(lower_name) > 1:
                    table_terms.add(lower_name[:-1])

                for col in columns:
                    col_name = col["name"].lower()
                    column_terms.add(col_name)
                    # Also add each snake_case part on its own ("cost_price"
                    # -> "cost", "price") - a real question says "cost" or
                    # "price", never the literal column identifier, so
                    # without this a column can exist in the schema and still
                    # never match a natural-language question about it. Pure
                    # string splitting, not tied to any specific column name.
                    if "_" in col_name:
                        column_terms.update(part for part in col_name.split("_") if part)

    except Exception as error:
        raise DatabaseConnectionError(f"Could not read the database schema: {error}") from error

    return {
        "description": "\n".join(lines) if lines else "(no tables found)",
        "tables": {name.lower() for name in table_names},
        # Vocabulary intent_router.py uses to recognize a database
        "routing_terms": {"tables": table_terms, "columns": column_terms},
        "cached_at": time.time(),
    }


def _ensure_schema_loaded():
    global _schema_cache

    with _lock:
        if _schema_cache_expired():
            _schema_cache = _load_schema()

    return _schema_cache


def get_schema_description():
    """Return a text block describing every table/column - what goes into
    the Gemini SQL-generation prompt."""

    return _ensure_schema_loaded()["description"]


def get_table_allowlist():
    """Return the cached set of lowercase table names sql_guard.py checks
    generated SQL against."""

    return _ensure_schema_loaded()["tables"]


def get_schema_terms():
    """Return {"tables": set[str], "columns": set[str]} derived from the
    live schema - the vocabulary intent_router.py uses to recognize a
    database-shaped question without any table/column name being
    hardcoded in Python. Updates automatically as tables/columns change,
    same TTL cache as get_schema_description()."""

    return _ensure_schema_loaded()["routing_terms"]


def execute_readonly_query(sql):
    """Run already-validated SQL (see sql_guard.validate_and_limit) and
    return the rows as a list of dicts. Blocking/sync - call this via
    run_in_threadpool, the same pattern rag_service.retrieve() uses for
    FAISS search."""

    engine = get_engine()

    try:
        with engine.connect() as conn:
            # Belt-and-suspenders alongside the DB role and sql_guard's
            # validation: this session can neither write nor run long.
            conn.execute(text(f"SET statement_timeout = {DB_QUERY_TIMEOUT_MS}"))
            conn.execute(text("SET TRANSACTION READ ONLY"))
            result = conn.execute(text(sql))
            rows = [dict(row) for row in result.mappings().all()]
    except Exception as error:
        raise DatabaseQueryError(f"Database query failed: {error}") from error

    return rows
