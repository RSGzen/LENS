r"""!!! DESTRUCTIVE DB TEST - WIPES ITS DATABASE. DO NOT POINT AT ``lens``. !!!

Contract tests for the M2 pgvector schema (no API key, zero cost).

WHY THIS FILE LIVES IN ``tests/DESTRUCTIVE_db_tests/``
------------------------------------------------------
Every test calls ``drop_schema()`` + ``apply_schema()``, which **drop the
``papers`` and ``chunks`` tables**. Against the real ``lens`` database that
destroys the full-corpus data (166,704 papers / ~4 M chunks). These tests run
against a **throwaway** database ``<LENS_DB_NAME>_test`` (default ``lens_test``)
- never ``lens``.

The normal suite ignores this folder: ``unittest discover -s tests`` does not
descend into a subdirectory without ``__init__.py``. Run it **only on purpose**:

    # one-time: create the throwaway DB (lens-pg container must be up)
    docker exec lens-pg psql -U lens -d lens -c "CREATE DATABASE lens_test"

    .\.venv\Scripts\python.exe -m unittest discover -s tests\DESTRUCTIVE_db_tests -p "test_*.py" -v

Do NOT run these against ``lens``. See ``tests/DESTRUCTIVE_db_tests/README.md``.
"""

from __future__ import annotations

import unittest

import psycopg

from lens import config
from lens.db.connect import connect
from lens.db.schema import apply_schema, drop_schema, tables_exist

# Throwaway DB this destructive suite is confined to (NEVER the live ``lens``).
_TEST_DB = f"{config.LENS_DB_NAME}_test"


class SchemaLiveTest(unittest.TestCase):
    """M2 verify clause: papers/chunks exist; lens_ro reads but cannot write."""

    admin: psycopg.Connection

    @classmethod
    def setUpClass(cls) -> None:
        try:
            # connect_timeout so a down container skips fast instead of hanging.
            cls.admin = connect(autocommit=True, connect_timeout=3, dbname=_TEST_DB)
        except psycopg.Error as exc:  # DB down / test DB missing -> skip, not fail
            raise unittest.SkipTest(
                f"no throwaway {_TEST_DB} database reachable: {exc}"
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.admin.close()

    # Fresh schema per test; drop_schema must be idempotent on a first run.
    def setUp(self) -> None:
        drop_schema(self.admin)
        apply_schema(self.admin)

    def tearDown(self) -> None:
        drop_schema(self.admin)

    # ------------------------------------------------------------- existence
    def test_tables_exist(self) -> None:
        state = tables_exist(self.admin)
        self.assertTrue(state["papers"])
        self.assertTrue(state["chunks"])

    def test_apply_schema_is_idempotent(self) -> None:
        apply_schema(self.admin)  # a second and third run must change nothing
        apply_schema(self.admin)
        self.assertTrue(all(tables_exist(self.admin).values()))

    def test_embedding_columns_are_halfvec_512(self) -> None:
        expected = f"{config.EMBEDDING_PGVECTOR_TYPE}({config.EMBEDDING_DIM})"
        checks = [("papers", "abstract_embedding"), ("chunks", "chunk_embedding")]
        with self.admin.cursor() as cur:
            for table, column in checks:
                cur.execute(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = %s::regclass AND attname = %s",
                    (table, column),
                )
                self.assertEqual(cur.fetchone()[0], expected)

    def test_embedding_index_uses_matching_op_class(self) -> None:
        with self.admin.cursor() as cur:
            for index in ("papers_abstract_embedding_hnsw", "chunks_chunk_embedding_hnsw"):
                cur.execute(
                    "SELECT indexdef FROM pg_indexes WHERE indexname = %s", (index,)
                )
                self.assertIn(config.EMBEDDING_INDEX_OPS, cur.fetchone()[0])

    def test_chunks_fk_to_papers(self) -> None:
        # A chunk for an unknown paper must be rejected by the FK.
        with self.admin.cursor() as cur:
            with self.assertRaises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    "INSERT INTO chunks (arxiv_id) VALUES ('does-not-exist')"
                )

    # -------------------------------------------------------------- readonly
    def test_readonly_role_can_select(self) -> None:
        with connect(readonly=True) as ro:
            ro.execute("SELECT count(*) FROM papers")
            ro.execute("SELECT count(*) FROM chunks")

    def test_readonly_role_write_denied(self) -> None:
        with connect(readonly=True) as ro:
            with self.assertRaises(psycopg.Error):
                ro.execute("INSERT INTO papers (arxiv_id, title) VALUES ('x', 'y')")
            ro.rollback()


if __name__ == "__main__":
    unittest.main()
