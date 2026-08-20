# Regression tests for _table_extra_facts (services/db_client.py) - the
# schema-annotation logic that adds a date-range note and low-cardinality
# column value lists to the schema description Gemini's SQL generation
# sees.
#
# Two real, observed bugs led here, both found and fixed live in the same
# session:
#   1. Correctness: Gemini's generated SQL guessed a status column's
#      stored values ("Overdue", "Paid") instead of the real ones
#      ("overdue", "paid"), silently matching nothing and falling back to
#      a much looser date-based heuristic that overcounted overdue
#      invoices (236 instead of the true 30). Fixed by showing the
#      column's REAL distinct values in the schema description.
#   2. Efficiency: an earlier version issued one query per date column
#      PLUS one query per character column - against this project's real
#      22-table/43-character-column schema, that pushed a full schema
#      reload past 25 seconds (each round trip to the hosted DB measured
#      at ~130ms regardless of query complexity). _table_extra_facts
#      collapses both facts into ONE query per table.
#
# Same scope rule as the rest of this suite: a real DB connection isn't
# needed to lock down the SQL-BUILDING / ROW-PARSING logic this function
# owns - that's pure logic once given a connection whose .execute()
# returns a fixed row, so it's tested with a fake connection here rather
# than a live Postgres one (the live round trip itself - real column
# types, real data, real latency - is covered by this project's own
# "verify live" practice, not duplicated as a unit test).

from services import db_client


class _FakeColumn(dict):
    """Mimics one row of inspector.get_columns() - a dict with 'name'/
    'type' keys, where 'type' stringifies like a real SQLAlchemy type."""


class _FakeType:
    def __init__(self, rendered):
        self._rendered = rendered

    def __str__(self):
        return self._rendered


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def mappings(self):
        return self

    def first(self):
        return self._row


class _FakeConnection:
    """Stands in for a SQLAlchemy Connection - hands back one fixed,
    pre-built row (a dict) for whatever single combined query
    _table_extra_facts issues, or raises to simulate a transient
    failure. Also records whether .execute() was called at all, so a
    test can assert NO query was issued for a table with nothing to
    annotate."""

    def __init__(self, row=None, raises=None):
        self._row = row
        self._raises = raises
        self.executed = False

    def execute(self, statement, params=None):
        self.executed = True
        if self._raises is not None:
            raise self._raises
        return _FakeResult(self._row)


def _column(name, type_str):
    return _FakeColumn(name=name, type=_FakeType(type_str))


def test_date_column_and_character_columns_combined():
    row = {
        "date_min": "2026-01-01",
        "date_max": "2026-08-01",
        "vals__status": ["paid", "overdue", "pending"],
    }
    conn = _FakeConnection(row=row)
    columns = [_column("id", "INTEGER"), _column("expense_date", "DATE"), _column("status", "VARCHAR(20)")]

    date_note, column_values = db_client._table_extra_facts(conn, "invoices", columns)

    assert date_note == " -- expense_date data available from 2026-01-01 to 2026-08-01"
    assert column_values == {"status": ["paid", "overdue", "pending"]}


def test_date_column_with_no_rows():
    row = {"date_min": None, "date_max": None}
    conn = _FakeConnection(row=row)
    columns = [_column("created_at", "DATE")]

    date_note, column_values = db_client._table_extra_facts(conn, "products", columns)

    assert date_note == " -- created_at: no rows"
    assert column_values == {}


def test_high_cardinality_column_is_excluded():
    # A genuinely free-text column (name, email, description) legitimately
    # has more distinct values than the cap - excluded from the schema
    # annotation rather than dumping a huge, useless value list.
    row = {"vals__description": [f"v{i}" for i in range(db_client.MAX_DISTINCT_VALUES + 1)]}
    conn = _FakeConnection(row=row)
    columns = [_column("description", "TEXT")]

    _, column_values = db_client._table_extra_facts(conn, "expenses", columns)

    assert column_values == {}


def test_exactly_at_the_cap_is_kept():
    values = [f"v{i}" for i in range(db_client.MAX_DISTINCT_VALUES)]
    row = {"vals__status": values}
    conn = _FakeConnection(row=row)
    columns = [_column("status", "VARCHAR(20)")]

    _, column_values = db_client._table_extra_facts(conn, "t", columns)

    assert column_values == {"status": values}


def test_null_or_empty_distinct_values_are_excluded():
    row = {"vals__status": None}
    conn = _FakeConnection(row=row)
    columns = [_column("status", "VARCHAR(20)")]

    _, column_values = db_client._table_extra_facts(conn, "t", columns)

    assert column_values == {}


def test_no_date_or_character_columns_issues_no_query():
    # A table with only numeric/boolean/id columns has nothing to
    # annotate - _table_extra_facts must not issue a query at all, the
    # single most direct way this stays cheap for tables that don't need it.
    conn = _FakeConnection(raises=AssertionError("should never query when there is nothing to annotate"))
    columns = [_column("id", "INTEGER"), _column("is_active", "BOOLEAN"), _column("amount", "NUMERIC(12, 2)")]

    date_note, column_values = db_client._table_extra_facts(conn, "t", columns)

    assert conn.executed is False
    assert date_note == ""
    assert column_values == {}


def test_query_failure_is_non_fatal():
    conn = _FakeConnection(raises=RuntimeError("simulated DB error"))
    columns = [_column("status", "VARCHAR(20)"), _column("created_at", "DATE")]

    date_note, column_values = db_client._table_extra_facts(conn, "t", columns)

    assert date_note == ""
    assert column_values == {}


def test_only_character_columns_no_date_column():
    row = {"vals__vendor_type": ["supplier", "contractor"]}
    conn = _FakeConnection(row=row)
    columns = [_column("id", "INTEGER"), _column("vendor_type", "VARCHAR(80)")]

    date_note, column_values = db_client._table_extra_facts(conn, "vendors", columns)

    assert date_note == ""
    assert column_values == {"vendor_type": ["supplier", "contractor"]}


def test_is_character_type_matches_char_and_text_variants():
    assert db_client._is_character_type(_FakeType("VARCHAR(20)"))
    assert db_client._is_character_type(_FakeType("TEXT"))
    assert db_client._is_character_type(_FakeType("CHAR(10)"))
    assert not db_client._is_character_type(_FakeType("INTEGER"))
    assert not db_client._is_character_type(_FakeType("NUMERIC(12, 2)"))
    assert not db_client._is_character_type(_FakeType("BOOLEAN"))
    assert not db_client._is_character_type(_FakeType("DATE"))
