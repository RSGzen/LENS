r"""Contract tests for the gated T3 SQL helper (no DB connection).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_sql -v
"""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
