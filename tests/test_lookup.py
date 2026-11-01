"""Contract tests for the corpus lookups (fake connection; no database)."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from lens import lookup


class _Cursor:
    """Minimal DB-API cursor: records the query, returns canned rows."""

    def __init__(self, rows: list[tuple], columns: list[str]) -> None:
        self._rows = rows
        self.description = [(name,) for name in columns]
        self.sql = ""
        self.params: tuple | None = None

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.sql = sql
        self.params = params

    def fetchall(self) -> list[tuple]:
        return self._rows


class _Conn:
    def __init__(self, rows: list[tuple], columns: list[str]) -> None:
        self.cursor_obj = _Cursor(rows, columns)

    def __enter__(self) -> "_Conn":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def cursor(self) -> _Cursor:
        return self.cursor_obj


PAPER_ROW = (
    "2407.21049",
    "Evaluating Long Range Dependency Handling",
    "A. Author",
    ["cs.CL"],
    None,
    None,
    "v2",
    "2025-07-01",
    "abstract text",
    "1/768",
)


class PaperLookupTestCase(unittest.TestCase):
    def test_a_row_is_returned_as_a_mapping(self) -> None:
        conn = _Conn([PAPER_ROW], list(lookup.PAPER_COLUMNS))

        row = lookup.paper_row(conn, "2407.21049")

        assert row is not None
        self.assertEqual(row["arxiv_id"], "2407.21049")
        self.assertEqual(row["categories"], ["cs.CL"])

    def test_the_query_is_parameterised(self) -> None:
        conn = _Conn([], list(lookup.PAPER_COLUMNS))

        self.assertIsNone(lookup.paper_row(conn, "2407.21049"))
        self.assertIn("WHERE arxiv_id = %s", conn.cursor_obj.sql)
        self.assertEqual(conn.cursor_obj.params, ("2407.21049",))

    def test_no_embedding_column_is_selected(self) -> None:
        # A 512-float vector is the largest thing in the table and no reader can use it.
        self.assertNotIn("abstract_embedding", lookup.PAPER_COLUMNS)
        self.assertNotIn("chunk_embedding", lookup.CHUNK_COLUMNS)

    def test_a_missing_paper_is_reported_not_faked(self) -> None:
        conn = _Conn([], list(lookup.PAPER_COLUMNS))

        payload = lookup.lookup_paper(conn, "0000.00000")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["found"], [])
        self.assertEqual(payload["missing"], ["0000.00000"])

    def test_a_bad_id_is_refused_before_any_query(self) -> None:
        conn = _Conn([PAPER_ROW], list(lookup.PAPER_COLUMNS))

        payload = lookup.lookup_paper(conn, "not-an-id")

        self.assertFalse(payload["ok"])
        self.assertEqual(conn.cursor_obj.sql, "")

    def test_long_text_is_capped_with_a_marker(self) -> None:
        long_abstract = "x" * (lookup.TEXT_MAX_CHARS + 500)
        row = (*PAPER_ROW[:8], long_abstract, PAPER_ROW[9])
        conn = _Conn([row], list(lookup.PAPER_COLUMNS))

        paper = lookup.paper_row(conn, "2407.21049")

        assert paper is not None
        self.assertIn("more characters", paper["abstract"])


class ChunkLookupTestCase(unittest.TestCase):
    CHUNK_A = "11111111-1111-5111-8111-111111111111"
    CHUNK_B = "22222222-2222-5222-8222-222222222222"

    def _row(self, chunk_id: str, text: str = "body") -> tuple:
        return (chunk_id, "2407.21049", "Results", text, 512, 3, "1/768")

    def test_rows_come_back_in_the_order_asked(self) -> None:
        conn = _Conn(
            [self._row(self.CHUNK_B), self._row(self.CHUNK_A)],
            list(lookup.CHUNK_COLUMNS),
        )

        rows = lookup.chunk_rows(conn, [self.CHUNK_A, self.CHUNK_B])

        self.assertEqual([row["chunk_uuid"] for row in rows], [self.CHUNK_A, self.CHUNK_B])

    def test_missing_ids_are_listed(self) -> None:
        conn = _Conn([self._row(self.CHUNK_A)], list(lookup.CHUNK_COLUMNS))

        payload = lookup.lookup_chunks(conn, [self.CHUNK_A, self.CHUNK_B])

        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["found"]), 1)
        self.assertEqual(payload["missing"], [self.CHUNK_B])

    def test_an_empty_list_is_an_error_not_an_empty_result(self) -> None:
        self.assertFalse(lookup.lookup_chunks(_Conn([], []), [])["ok"])

    def test_a_non_chunk_id_is_refused(self) -> None:
        payload = lookup.lookup_chunks(_Conn([], []), ["not-a-chunk"])

        self.assertFalse(payload["ok"])
        self.assertIn("not a chunk id", payload["error"])

    def test_the_list_is_capped(self) -> None:
        ids = [self.CHUNK_A] * (lookup.MAX_CHUNK_IDS + 1)

        payload = lookup.lookup_chunks(_Conn([], []), ids)

        self.assertFalse(payload["ok"])
        self.assertIn("at most", payload["error"])


class MainTestCase(unittest.TestCase):
    def test_it_prints_one_json_object(self) -> None:
        conn = _Conn([PAPER_ROW], list(lookup.PAPER_COLUMNS))
        buffer = io.StringIO()

        with patch("lens.db.connect.connect", return_value=conn):
            with redirect_stdout(buffer):
                code = lookup.main(["paper", "2407.21049"])

        payload = json.loads(buffer.getvalue())

        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["found"][0]["arxiv_id"], "2407.21049")

    def test_a_missing_database_is_a_message_not_a_traceback(self) -> None:
        buffer = io.StringIO()

        with patch("lens.db.connect.connect", side_effect=OSError("no server")):
            with redirect_stdout(buffer):
                code = lookup.main(["paper", "2407.21049"])

        payload = json.loads(buffer.getvalue())

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertIn("database unavailable", payload["error"])


if __name__ == "__main__":
    unittest.main()
