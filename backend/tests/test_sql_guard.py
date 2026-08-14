# Covers the second, independent defense layer between Gemini-generated
# SQL and the database (see services/sql_guard.py's own docstring) -
# these are the exact cases that layer exists to catch, so regressions
# here are the highest-cost kind this project could have.

import pytest

from services.sql_guard import validate_and_limit, SqlValidationError

ALLOWED_TABLES = {"accounts_income", "accounts_marketing_expense", "tally_ledger"}


def test_allows_a_plain_select():
    result = validate_and_limit("SELECT * FROM accounts_income", ALLOWED_TABLES, 100)
    assert "accounts_income" in result
    assert "LIMIT 100" in result


def test_allows_a_cte_using_only_allowed_tables():
    sql = (
        "WITH total AS (SELECT SUM(amount) AS s FROM accounts_income) "
        "SELECT s FROM total"
    )
    result = validate_and_limit(sql, ALLOWED_TABLES, 100)
    assert "total" in result.lower()


@pytest.mark.parametrize("sql", [
    "DELETE FROM accounts_income",
    "UPDATE accounts_income SET amount = 0",
    "INSERT INTO accounts_income (amount) VALUES (1)",
    "DROP TABLE accounts_income",
    "ALTER TABLE accounts_income ADD COLUMN x int",
    "GRANT ALL ON accounts_income TO public",
    "TRUNCATE accounts_income",
])
def test_rejects_writes_and_ddl(sql):
    with pytest.raises(SqlValidationError):
        validate_and_limit(sql, ALLOWED_TABLES, 100)


def test_rejects_a_second_statement_stacked_after_the_select():
    with pytest.raises(SqlValidationError):
        validate_and_limit(
            "SELECT * FROM accounts_income; DROP TABLE accounts_income",
            ALLOWED_TABLES,
            100,
        )


def test_rejects_a_table_outside_the_allowlist():
    with pytest.raises(SqlValidationError):
        validate_and_limit("SELECT * FROM pg_shadow", ALLOWED_TABLES, 100)


@pytest.mark.parametrize("function_call", [
    "SELECT pg_sleep(10)",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT pg_terminate_backend(1)",
])
def test_rejects_forbidden_functions(function_call):
    with pytest.raises(SqlValidationError):
        validate_and_limit(function_call, ALLOWED_TABLES, 100)


def test_rejects_unparseable_sql():
    with pytest.raises(SqlValidationError):
        validate_and_limit("this is not sql at all", ALLOWED_TABLES, 100)


def test_caps_an_oversized_explicit_limit():
    result = validate_and_limit("SELECT * FROM accounts_income LIMIT 999999", ALLOWED_TABLES, 100)
    assert "LIMIT 100" in result
    assert "999999" not in result


def test_leaves_a_smaller_explicit_limit_alone():
    result = validate_and_limit("SELECT * FROM accounts_income LIMIT 5", ALLOWED_TABLES, 100)
    assert "LIMIT 5" in result


def test_none_allowed_tables_skips_the_table_check():
    # The documented escape hatch (sql_guard.py's own docstring calls
    # this "not recommended, kept only as an escape hatch") - any table
    # name should pass when allowed_tables is None.
    result = validate_and_limit("SELECT * FROM literally_anything", None, 100)
    assert "literally_anything" in result
