r"""Contract tests for the gated T3 SQL helper (no DB connection).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_sql -v
"""

from __future__ import annotations

import unittest
from typing import Any

from lens.tools.sql import sql_is_allowed, validate_sql


class SqlGateTestCase(unittest.TestCase):
    def test_recorded_runs_never_allow_sql(self) -> None:
        self.assertFalse(sql_is_allowed(recorded=True, allow_sql=True))

    def test_dev_opt_in(self) -> None:
        self.assertFalse(sql_is_allowed(recorded=False, allow_sql=False))
        self.assertTrue(sql_is_allowed(recorded=False, allow_sql=True))

    def test_select_and_with_are_allowed(self) -> None:
        self.assertEqual(validate_sql("SELECT 1"), "SELECT 1")
        self.assertEqual(
            validate_sql("WITH x AS (SELECT 1) SELECT * FROM x"),
            "WITH x AS (SELECT 1) SELECT * FROM x",
        )

    def test_write_statements_are_rejected(self) -> None:
        for sql in (
            "DELETE FROM papers",
            "DROP TABLE chunks",
            "UPDATE papers SET title = 'x'",
            "INSERT INTO papers VALUES (1)",
        ):
            with self.assertRaises(ValueError, msg=sql):
                validate_sql(sql)

    def test_multiple_statements_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_sql("SELECT 1; DROP TABLE papers")

    def test_one_trailing_semicolon_is_allowed(self) -> None:
        # psql habits: a single trailing ';' is not a second statement.
        self.assertEqual(validate_sql("SELECT 1;"), "SELECT 1")

    def test_keyword_check_respects_word_boundaries(self) -> None:
        # A column called ``updated_at`` must not trip the ``update`` keyword.
        self.assertEqual(validate_sql("SELECT updated_at FROM runs"), "SELECT updated_at FROM runs")


class FakeCursor:
    """Minimal cursor double: records statements, returns fixed rows."""

    def __init__(self, rows: list[tuple], columns: list[str], enforce_cap: bool = True) -> None:
        from types import SimpleNamespace

        self.rows = rows
        self.description = [SimpleNamespace(name=name) for name in columns]
        self.statements: list[tuple[str, Any]] = []
        self.limit: int | None = None
        self.enforce_cap = enforce_cap

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        self.statements.append((sql, params))

    def fetchmany(self, limit: int) -> list[tuple]:
        self.limit = limit
        return self.rows[:limit] if self.enforce_cap else self.rows


class FakeConn:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.cursor_calls = 0

    def cursor(self) -> FakeCursor:
        self.cursor_calls += 1
        return self._cursor


class QuerySqlTestCase(unittest.TestCase):
    """``query_sql`` without a database: the transaction shape and the row cap."""

    def test_read_only_first_then_timeout_then_the_query(self) -> None:
        from lens.tools.sql import query_sql

        cursor = FakeCursor(rows=[(1, "a"), (2, "b"), (3, "c")], columns=["n", "s"])
        conn = FakeConn(cursor)

        rows = query_sql(conn, "SELECT 1", row_cap=2, timeout_ms=5000)

        self.assertEqual(rows, [{"n": 1, "s": "a"}, {"n": 2, "s": "b"}])
        self.assertEqual(cursor.statements[0][0], "SET TRANSACTION READ ONLY")
        self.assertIn("set_config", cursor.statements[1][0])
        self.assertEqual(cursor.statements[1][1], ("5000",))
        self.assertEqual(cursor.statements[2], ("SELECT 1", None))
        self.assertEqual(cursor.limit, 2)  # the cap, not the query's LIMIT

    def test_writes_are_rejected_before_the_connection_is_touched(self) -> None:
        from lens.tools.sql import query_sql

        conn = FakeConn(FakeCursor(rows=[], columns=[]))

        with self.assertRaises(ValueError):
            query_sql(conn, "DELETE FROM papers")

        self.assertEqual(conn.cursor_calls, 0)


if __name__ == "__main__":
    unittest.main()
