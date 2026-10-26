r"""Contract tests for M5 retrieval tools (no DB, no key, zero cost).

Pure helpers are tested directly; the funnel functions are exercised with a
``FakeConn`` + ``FakeEmbedder`` + ``FakeReranker`` (and ``fetch_papers`` /
``fetch_chunks`` patched where the DB row shape is not the point). No DB, no
torch. The retrieval tools are implemented; these are contract tests.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from lens import config
from lens.tools.retrieval import (
    PAPER_COLUMNS,
    deep_match,
    format_query,
    hnsw_config_sql,
    rerank_chunks,
    rerank_papers,
    row_to_dict,
    shallow_match,
    top_by_score,
    vector_literal,
    year_from_arxiv_id,
)


class FakeCursor:
    """Records every executed statement; ``fetchall`` returns canned rows."""

    def __init__(self, rows) -> None:
        self._rows = list(rows)
        self.statements: list[tuple[str, object]] = []

    def execute(self, sql, params=None) -> None:
        self.statements.append((sql, params))

    def fetchall(self):
        return list(self._rows)

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class FakeConn:
    """One shared cursor so every ``with conn.cursor()`` sees the same statements."""

    def __init__(self, rows) -> None:
        self._cursor = FakeCursor(rows)

    def cursor(self) -> FakeCursor:
        return self._cursor


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str, int]] = []

    def encode_documents(self, texts, task_prefix, batch_size):
        self.calls.append((list(texts), task_prefix, batch_size))
        return np.zeros((len(texts), config.EMBEDDING_DIM), dtype=np.float32)


class FakeReranker:
    def __init__(self, scores) -> None:
        self._scores = list(scores)
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, texts, batch_size=config.RERANK_BATCH_SIZE):
        self.calls.append((query, list(texts)))
        return list(self._scores)


class FormatQueryTest(unittest.TestCase):
    def test_str_passes_through(self) -> None:
        self.assertEqual(format_query("hello world"), "hello world")

    def test_mapping_joins_lists_in_insertion_order(self) -> None:
        buckets = {"objective": ["improve Y"], "problem": ["why X"]}
        self.assertEqual(format_query(buckets), "improve Y why X")

    def test_multiple_items_in_one_bucket_flatten(self) -> None:
        buckets = {"objectives": ["Improve Y", "Test Y"], "problems": ["Y is costly"]}
        self.assertEqual(format_query(buckets), "Improve Y Test Y Y is costly")

    def test_skips_empty_and_appends_unknown_keys(self) -> None:
        buckets = {"problem": ["p"], "objective": [], "note": ["n"]}
        self.assertEqual(format_query(buckets), "p n")

    def test_string_value_is_one_clause_not_characters(self) -> None:
        # The M9 smoke bug: the RLM handed over {"query": "..."} and iterating the
        # string embedded "s o m e   q u e r y", so the funnel returned unrelated
        # papers. A string value is one clause.
        self.assertEqual(
            format_query({"query": "long context aggregation"}), "long context aggregation"
        )

    def test_nested_and_non_text_values_are_tolerated(self) -> None:
        buckets = {"a": [["deep", "nested"]], "b": "flat", "c": None, "d": 3}
        self.assertEqual(format_query(buckets), "deep nested flat")


class YearFromArxivIdTest(unittest.TestCase):
    def test_two_digit_prefix(self) -> None:
        self.assertEqual(year_from_arxiv_id("2103.12345"), 2021)
        self.assertEqual(year_from_arxiv_id("1501.00601"), 2015)
        self.assertEqual(year_from_arxiv_id("2601.0001"), 2026)

    def test_non_numeric_prefix_is_none(self) -> None:
        self.assertIsNone(year_from_arxiv_id("acc-phys/9411001"))
        self.assertIsNone(year_from_arxiv_id("2a01.0001"))
        self.assertIsNone(year_from_arxiv_id(""))


class RowToDictTest(unittest.TestCase):
    def test_zips_columns_and_row(self) -> None:
        self.assertEqual(row_to_dict(("a", "b"), (1, 2)), {"a": 1, "b": 2})


class VectorLiteralTest(unittest.TestCase):
    def test_round_trips_as_json(self) -> None:
        vec = np.array([0.5, -1.0, 0.25], dtype=np.float32)
        self.assertEqual(json.loads(vector_literal(vec)), [0.5, -1.0, 0.25])


class HnswConfigSqlTest(unittest.TestCase):
    def test_always_sets_ef_search(self) -> None:
        stmts = hnsw_config_sql(64)
        self.assertEqual(len(stmts), 1)
        self.assertIn("64", stmts[0])

    def test_iterative_scan_adds_relaxed_order(self) -> None:
        stmts = " ".join(hnsw_config_sql(40, iterative_scan=True))
        self.assertIn("iterative_scan", stmts)
        self.assertIn("relaxed_order", stmts)


class TopByScoreTest(unittest.TestCase):
    def test_sorts_desc_and_truncates(self) -> None:
        out = top_by_score(["a", "b", "c"], [0.1, 0.9, 0.5], top=2)
        self.assertEqual(out, [("b", 0.9), ("c", 0.5)])

    def test_ties_are_stable(self) -> None:
        out = top_by_score(["a", "b", "c"], [0.5, 0.5, 0.9], top=3)
        self.assertEqual(out, [("c", 0.9), ("a", 0.5), ("b", 0.5)])

    def test_non_positive_top_is_empty(self) -> None:
        self.assertEqual(top_by_score(["a"], [1.0], top=0), [])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            top_by_score(["a", "b"], [1.0], top=1)


class ShallowMatchTest(unittest.TestCase):
    def test_returns_ids_in_ann_order_with_query_prefix(self) -> None:
        conn = FakeConn([("2101.00001",), ("2102.00002",), ("2103.00003",)])
        emb = FakeEmbedder()

        ids = shallow_match(conn, "hello world", emb, k=3)

        self.assertEqual(ids, ["2101.00001", "2102.00002", "2103.00003"])
        self.assertEqual(emb.calls[0][1], config.EMBEDDING_QUERY_PREFIX)
        self.assertEqual(emb.calls[0][0], ["hello world"])

    def test_ef_search_is_raised_to_k(self) -> None:
        # pgvector caps an unfiltered HNSW scan at ef_search rows, so k must not
        # exceed it (default 40). k=100 -> ef_search = max(config, 100).
        conn = FakeConn([("x",)])
        emb = FakeEmbedder()

        shallow_match(conn, "q", emb, k=100)

        sql_blob = " ".join(s for s, _ in conn._cursor.statements).lower()
        self.assertIn(f"ef_search = {max(config.HNSW_EF_SEARCH, 100)}", sql_blob)


class DeepMatchTest(unittest.TestCase):
    def test_filtered_search_uses_iterative_scan(self) -> None:
        conn = FakeConn([("uuid-1",), ("uuid-2",)])
        emb = FakeEmbedder()

        ids = deep_match(conn, ["2101.00001"], "hello", emb, candidate_chunks=2)

        self.assertEqual(ids, ["uuid-1", "uuid-2"])
        sql_blob = " ".join(s for s, _ in conn._cursor.statements).lower()
        self.assertIn("iterative_scan", sql_blob)

    def test_ef_search_is_raised_to_candidate_chunks(self) -> None:
        conn = FakeConn([("uuid-1",)])
        emb = FakeEmbedder()

        deep_match(conn, ["2101.00001"], "q", emb, candidate_chunks=100)

        sql_blob = " ".join(s for s, _ in conn._cursor.statements).lower()
        self.assertIn(f"ef_search = {max(config.HNSW_EF_SEARCH, 100)}", sql_blob)

    def test_empty_papers_returns_empty_without_embedding(self) -> None:
        conn = FakeConn([])
        emb = FakeEmbedder()

        self.assertEqual(deep_match(conn, [], "hello", emb, candidate_chunks=2), [])
        self.assertEqual(emb.calls, [])


class RerankPapersTest(unittest.TestCase):
    def test_reorders_to_input_then_keeps_top_n(self) -> None:
        rows = {
            "a": {"arxiv_id": "a", "title": "A", "abstract": "abs-a"},
            "b": {"arxiv_id": "b", "title": "B", "abstract": "abs-b"},
            "c": {"arxiv_id": "c", "title": "C", "abstract": "abs-c"},
        }
        # DB returns the papers unordered (c, a, b); scores map to input order (a,b,c).
        with mock.patch(
            "lens.tools.retrieval.fetch_papers",
            return_value=[rows["c"], rows["a"], rows["b"]],
        ):
            rr = FakeReranker([0.1, 0.9, 0.5])
            out = rerank_papers(None, ["a", "b", "c"], "q", rr, n=2)

        self.assertEqual([o["arxiv_id"] for o in out], ["b", "c"])
        self.assertEqual(rr.calls[0][1], ["abs-a", "abs-b", "abs-c"])


class RerankChunksTest(unittest.TestCase):
    def test_returns_hydrated_payload_top_c(self) -> None:
        def chunk(uuid, section_order):
            return {
                "chunk_uuid": uuid,
                "arxiv_id": "2101.00001",
                "section_type": "Methodology",
                "section_order": section_order,
                "chunk_text": f"text-{uuid}",
                "token_count": 5,
            }

        papers = [{"arxiv_id": "2101.00001", "title": "T", "authors": "A", "abstract": "abs"}]
        with (
            mock.patch(
                "lens.tools.retrieval.fetch_chunks",
                return_value=[chunk("u2", 1), chunk("u1", 0), chunk("u3", 2)],
            ),
            mock.patch("lens.tools.retrieval.fetch_papers", return_value=papers),
        ):
            rr = FakeReranker([0.2, 0.8, 0.4])
            out = rerank_chunks(None, ["u1", "u2", "u3"], "q", rr, top_chunks=2)

        self.assertEqual([c["chunk_uuid"] for c in out["chunks"]], ["u2", "u3"])
        self.assertEqual(out["chunks"][0]["asset_refs"], [])
        self.assertEqual(out["papers"]["2101.00001"]["year"], 2021)
        self.assertEqual(out["assets"], {})

    def test_paper_columns_align_with_schema(self) -> None:
        self.assertIn("arxiv_id", PAPER_COLUMNS)
        self.assertEqual(len(PAPER_COLUMNS), 9)


if __name__ == "__main__":
    unittest.main()
