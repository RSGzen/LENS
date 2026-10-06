r"""Contract tests for the M2 pgvector schema (no API key, zero cost).

These tests need a **running** ``lens-pg`` container but no OpenRouter key. If
the database is unreachable the class is skipped, so ``unittest discover`` still
passes on a cold machine.

Start the DB (Docker Desktop running):

    docker run -d --name lens-pg --restart unless-stopped `
        -e POSTGRES_USER=lens -e POSTGRES_PASSWORD=lens -e POSTGRES_DB=lens `
        -p 5433:5432 -v D:\lens-data\postgres:/var/lib/postgresql/data `
        pgvector/pgvector:pg16

Host port 5433 keeps clear of the native ``postgresql-x64-17`` service on 5432;
``LENS_DB_PORT=5433`` in ``.env`` points the harness at it. One-time creation;
afterwards ``docker start lens-pg`` (or ``--restart`` + Docker Desktop). Compose
for DB + sandbox is deferred to M8.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import psycopg

from lens import config
from lens.db.connect import connect
from lens.db.schema import apply_schema, drop_schema, tables_exist


class SchemaLiveTest(unittest.TestCase):
    """M2 verify clause: papers/chunks exist; lens_ro reads but cannot write."""

    admin: psycopg.Connection

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.admin = connect(autocommit=True)
        except psycopg.Error as exc:  # DB down -> skip, not fail
            raise unittest.SkipTest(f"no LENS database reachable: {exc}")

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
