r"""Contract tests for M3 paper loading (no key, zero cost).

The load tests need a **running** ``lens-pg`` container; if the DB is
unreachable the class is skipped. They fail with ``NotImplementedError`` until
the fill point is done.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np
import psycopg

from lens import config
from lens.dataset.papers import load_papers
from lens.db.connect import connect
from lens.db.schema import apply_schema, drop_schema


def entry(arxiv_id: str, title: str = "t") -> dict:
    """A minimal manifest entry carrying the fields :func:`load_papers` reads."""
    return {
        "arxiv_id": arxiv_id,
        "version": "v1",
        "tei_xml": f"{arxiv_id}v1.tei.xml",
        "title": title,
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

    def encode_documents(self, texts) -> np.ndarray:
        texts = list(texts)
        self.seen.append(texts)
        return np.arange(len(texts) * self.dim, dtype=np.float32).reshape(len(texts), self.dim)


class LoadPapersLiveTest(unittest.TestCase):
    """M3 verify clause: rows == entry count; no NULL embeddings; idempotent."""

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

    def setUp(self) -> None:
        drop_schema(self.admin)
        apply_schema(self.admin)
        self.entries = [entry(f"000{n}.00001", title=f"paper {n}") for n in range(3)]

    def tearDown(self) -> None:
        drop_schema(self.admin)

    def _paper_counts(self) -> tuple[int, int]:
        with self.admin.cursor() as cur:
            cur.execute("SELECT count(*), count(abstract_embedding) FROM papers")
            return cur.fetchone()

    def test_load_papers_inserts_rows_with_embeddings(self) -> None:
        count = load_papers(self.admin, self.entries, FakeEmbedder())

        self.assertEqual(count, len(self.entries))
        total, embedded = self._paper_counts()
        self.assertEqual(total, len(self.entries))
        self.assertEqual(embedded, len(self.entries))  # no NULL embeddings

    def test_load_papers_embeds_abstracts(self) -> None:
        embedder = FakeEmbedder()

        load_papers(self.admin, self.entries, embedder)

        self.assertEqual(embedder.seen, [[e["abstract"] for e in self.entries]])

    def test_load_papers_maps_manifest_fields(self) -> None:        load_papers(self.admin, self.entries, FakeEmbedder())

        with self.admin.cursor() as cur:
            cur.execute(
                "SELECT current_ver, categories, abstract FROM papers WHERE arxiv_id = %s",
                (self.entries[0]["arxiv_id"],),
            )
            current_ver, categories, abstract = cur.fetchone()

        self.assertEqual(current_ver, "v1")
        self.assertEqual(categories, ["cs.AI"])
        self.assertEqual(abstract, self.entries[0]["abstract"])

    def test_load_papers_stores_configured_dims(self) -> None:
        load_papers(self.admin, self.entries, FakeEmbedder())

        with self.admin.cursor() as cur:
            cur.execute("SELECT vector_dims(abstract_embedding) FROM papers LIMIT 1")
            self.assertEqual(cur.fetchone()[0], config.EMBEDDING_DIM)

    def test_load_papers_is_idempotent(self) -> None:
        load_papers(self.admin, self.entries, FakeEmbedder())
        load_papers(self.admin, self.entries, FakeEmbedder())

        total, _ = self._paper_counts()
        self.assertEqual(total, len(self.entries))


if __name__ == "__main__":
    unittest.main()
