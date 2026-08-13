"""
Validates SQL that Gemini generated before it's ever executed.

A read-only database role (see db_client.py's docstring) is the primary
defense - it can't write no matter what runs. This module is the second,
independent layer: even a manipulated prompt or a compromised role
grant shouldn't be enough on its own, so generated SQL is parsed and
checked structurally rather than trusted as text. The executed string is
always the re-rendered, validated AST - never Gemini's raw output.
"""

import sqlglot
from sqlglot import exp

# Node types that mean this isn't a plain read: writes, DDL, permission
# changes, or anything that isn't a single SELECT.
_FORBIDDEN_NODE_TYPES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Create,
    exp.Grant, exp.Command, exp.Into, exp.Copy, exp.TruncateTable,
)

# Server-side functions with no business being called from a chat answer,
# even inside an otherwise-plain SELECT.
_FORBIDDEN_FUNCTIONS = {
    "pg_terminate_backend", "pg_cancel_backend", "pg_read_file",
    "pg_read_binary_file", "pg_ls_dir", "lo_import", "lo_export",
    "pg_sleep", "dblink", "dblink_exec",
}


class SqlValidationError(Exception):
    """Raised with a reason suitable for both logging and a corrective retry prompt."""


def _check_single_select(parsed):
    if len(parsed) != 1:
        raise SqlValidationError("Only a single SQL statement is allowed.")

    statement = parsed[0]
    if statement is None or not isinstance(statement, exp.Select):
        raise SqlValidationError("Only a single read-only SELECT statement is allowed.")

    return statement


def _check_no_forbidden_nodes(statement):
    for node in statement.walk():
        node = node[0] if isinstance(node, tuple) else node
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            raise SqlValidationError(f"'{type(node).__name__}' is not allowed in a read-only query.")

        if isinstance(node, exp.Anonymous) and node.name and node.name.lower() in _FORBIDDEN_FUNCTIONS:
            raise SqlValidationError(f"Calling '{node.name}' is not allowed.")

        if isinstance(node, exp.Func):
            func_name = node.sql_name().lower() if hasattr(node, "sql_name") else ""
            if func_name in _FORBIDDEN_FUNCTIONS:
                raise SqlValidationError(f"Calling '{func_name}' is not allowed.")


def _check_tables_allowed(statement, allowed_tables):
    if allowed_tables is None:
        return

    # A WITH clause's CTE names (e.g. "WITH revenue AS (...) SELECT ...")
    # parse as exp.Table nodes too where they're referenced in the outer
    # query's FROM/JOIN - sqlglot doesn't distinguish "real table" from
    # "name defined earlier in this same query" at the syntax level, so
    # without this exclusion every derived-metric query using a CTE
    # (needed to compute something like profit, which spans several
    # tables) would be incorrectly rejected as touching a disallowed
    # "table" that's actually just its own subquery's alias.
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}

    for table in statement.find_all(exp.Table):
        # Compare bare table names - schema-qualification handled by the
        # allowlist itself already only containing intended tables.
        name = table.name.lower()
        if name in cte_names:
            continue
        if name not in allowed_tables:
            raise SqlValidationError(
                f"Table '{name}' is not part of the tables this assistant is allowed to read."
            )


def _apply_row_limit(statement, row_limit):
    existing_limit = statement.args.get("limit")

    if existing_limit is not None:
        try:
            current_value = int(existing_limit.expression.this)
        except (AttributeError, TypeError, ValueError):
            current_value = row_limit + 1  # malformed/non-literal limit - clamp defensively

        if current_value > row_limit:
            statement.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
    else:
        statement.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))

    return statement


def validate_and_limit(sql, allowed_tables, row_limit):
    """Validate Gemini-generated SQL and return a safe, re-rendered query string.

    allowed_tables: lowercase set of table names this query is permitted to
    touch (pass None to skip the table check, e.g. if relying solely on DB
    grants - not recommended, kept only as an escape hatch).

    Raises SqlValidationError with a human-readable reason on any violation,
    including from Gemini producing unparseable SQL.
    """

    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
    except Exception as error:
        raise SqlValidationError(f"Generated SQL could not be parsed: {error}") from error

    statement = _check_single_select(parsed)
    _check_no_forbidden_nodes(statement)
    _check_tables_allowed(statement, allowed_tables)
    statement = _apply_row_limit(statement, row_limit)

    return statement.sql(dialect="postgres")
