r"""!!! DESTRUCTIVE DB TEST - WIPES ITS DATABASE. DO NOT POINT AT ``lens``. !!!

Contract tests for M4 chunk loading (no key, zero cost).

WHY THIS FILE LIVES IN ``tests/DESTRUCTIVE_db_tests/``
------------------------------------------------------
``setUp`` calls ``drop_schema()`` + ``apply_schema()``, which **drop and recreate**
``papers``/``chunks``. Against the real ``lens`` database that destroys the
full-corpus data. These tests run against a **throwaway** database
``<LENS_DB_NAME>_test`` (default ``lens_test``) - never ``lens``.

Run it **only on purpose** (the normal suite does not descend into this folder):

    .\.venv\Scripts\python.exe -m unittest discover -s tests\DESTRUCTIVE_db_tests -p "test_*.py" -v

See ``tests/DESTRUCTIVE_db_tests/README.md``.
"""

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import psycopg

from lens import config
from lens.dataset.chunks import chunk_uuid, load_chunks
from lens.dataset.papers import load_papers
from lens.db.connect import connect
from lens.db.schema import apply_schema, drop_schema

# Throwaway DB this destructive suite is confined to (NEVER the live ``lens``).
_TEST_DB = f"{config.LENS_DB_NAME}_test"

WORDS = lambda s: len(s.split())  # noqa: E731 — test token counter

TEI_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body>
    <div><head n="1">Introduction</head><p>intro words here</p></div>
    <div><head n="2">Method</head><p>method words here</p></div>
  </body></text>
</TEI>
"""


def paper_entry(arxiv_id: str) -> dict:
    """A minimal manifest entry carrying the fields the M3/M4 loaders read."""
    return {
        "arxiv_id": arxiv_id,
        "version": "v1",
        "tei_xml": f"{arxiv_id}v1.tei.xml",
        "title": f"title {arxiv_id}",
        "authors": "A. Author",
        "categories": ["cs.AI"],
        "arxiv_doi": None,
        "journal_ref": None,
        "update_date": "2020-01-01",
        "abstract": f"abstract for {arxiv_id}",
    }


class FakeEmbedder:
    """Deterministic stand-in: one distinct vector of ``config.EMBEDDING_DIM`` per text."""

    def __init__(self, dim: int = config.EMBEDDING_DIM) -> None:
        self.dim = dim
        self.seen: list[list[str]] = []

    def encode_documents(self, texts, task_prefix: str, batch_size: int) -> np.ndarray:
        texts = list(texts)
        self.seen.append(texts)
        return np.arange(len(texts) * self.dim, dtype=np.float32).reshape(len(texts), self.dim)


class ChunkUuidTest(unittest.TestCase):
    def test_deterministic_and_distinct(self) -> None:
        a = chunk_uuid("1501.00601", "Methodology", 3)
        self.assertEqual(a, chunk_uuid("1501.00601", "Methodology", 3))
        self.assertNotEqual(a, chunk_uuid("1501.00601", "Methodology", 4))
        self.assertNotEqual(a, chunk_uuid("1501.00602", "Methodology", 3))
        self.assertIsInstance(a, uuid.UUID)


class LoadChunksLiveTest(unittest.TestCase):
    """M4 verify clause: sane chunk counts; no NULL embeddings; idempotent upsert."""

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

    def setUp(self) -> None:
        drop_schema(self.admin)
        apply_schema(self.admin)
        self._dir = tempfile.TemporaryDirectory()
        self.xml_dir = Path(self._dir.name)
        self.entries = [paper_entry("0001.00001"), paper_entry("0002.00002")]
        for entry in self.entries:
            (self.xml_dir / entry["tei_xml"]).write_text(TEI_TEMPLATE, encoding="utf-8")
        load_papers(self.admin, self.entries, process_chunk_size=2,
                    encode_batch_size=2, embedder=FakeEmbedder())

    def tearDown(self) -> None:
        self._dir.cleanup()
        drop_schema(self.admin)

    def _load(self) -> tuple[int, int]:
        return load_chunks(
            self.admin,
            self.entries,
            xml_dir=self.xml_dir,
            count_tokens=WORDS,
            embedder=FakeEmbedder(),
            papers_per_batch=1,
            encode_batch_size=2,
            threshold=50,
        )

    def _chunk_counts(self) -> tuple[int, int]:
        with self.admin.cursor() as cur:
            cur.execute("SELECT count(*), count(chunk_embedding) FROM chunks")
            return cur.fetchone()

    def test_load_chunks_inserts_rows_with_embeddings(self) -> None:
        papers, chunks = self._load()

        self.assertEqual(papers, len(self.entries))
        self.assertEqual(chunks, 2 * len(self.entries))  # 2 sections -> 2 chunks/paper
        total, embedded = self._chunk_counts()
        self.assertEqual(total, chunks)
        self.assertEqual(embedded, chunks)  # no NULL embeddings

    def test_load_chunks_maps_columns(self) -> None:
        self._load()

        with self.admin.cursor() as cur:
            cur.execute(
                "SELECT section_type, section_order, token_count, embedding_version "
                "FROM chunks WHERE arxiv_id = %s ORDER BY section_order",
                (self.entries[0]["arxiv_id"],),
            )
            rows = cur.fetchall()

        self.assertEqual([r[0] for r in rows], ["Background Study", "Methodology"])
        self.assertEqual([r[1] for r in rows], [0, 1])
        self.assertEqual([r[2] for r in rows], [3, 3])
        self.assertEqual(rows[0][3], config.EMBEDDING_VERSION)

    def test_load_chunks_stores_configured_dims(self) -> None:
        self._load()

        with self.admin.cursor() as cur:
            cur.execute("SELECT vector_dims(chunk_embedding) FROM chunks LIMIT 1")
            self.assertEqual(cur.fetchone()[0], config.EMBEDDING_DIM)

    def test_load_chunks_is_idempotent(self) -> None:
        self._load()
        self._load()

        total, _ = self._chunk_counts()
        self.assertEqual(total, 2 * len(self.entries))


if __name__ == "__main__":
    unittest.main()
