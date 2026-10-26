r"""Contract tests for the / gold-sample helpers (no DB, no key)."""

from __future__ import annotations

import unittest

from lens.kg_gold import (
    clean_surface,
    format_triple_query,
    is_benchmark_span,
    parse_gold_entities,
    score_entities,
    score_triples,
    snippet,
    stratified_sample,
    wilson_ci,
)


def triple(relation, i=0):
    return {"relation": relation, "head": f"h{i}", "tail": f"t{i}"}


class StratifiedSampleTest(unittest.TestCase):
    def test_takes_requested_count_and_covers_relations(self) -> None:
        triples = [triple("outperforms", i) for i in range(90)] + [
            triple("compares_to", 1),
            triple("addresses", 2),
        ]
        sampled = stratified_sample(triples, 10, seed=0)
        self.assertEqual(len(sampled), 10)
        self.assertEqual(
            {t["relation"] for t in sampled},
            {"outperforms", "compares_to", "addresses"},
        )

    def test_empty_or_zero(self) -> None:
        self.assertEqual(stratified_sample([], 10), [])
        self.assertEqual(stratified_sample([triple("uses")], 0), [])


class WilsonTest(unittest.TestCase):
    def test_bounds(self) -> None:
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))
        lo, hi = wilson_ci(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


class TripleScoreTest(unittest.TestCase):
    def test_precision(self) -> None:
        rows = [
            {
                "relation": "uses",
                "gold_relation_ok": "y",
                "gold_head_type_ok": "y",
                "gold_tail_type_ok": "y",
            },
            {
                "relation": "uses",
                "gold_relation_ok": "y",
                "gold_head_type_ok": "n",
                "gold_tail_type_ok": "y",
            },
            {
                "relation": "uses",
                "gold_relation_ok": "",  # unlabeled -> ignored
                "gold_head_type_ok": "",
                "gold_tail_type_ok": "",
            },
        ]
        report = score_triples(rows)
        self.assertEqual(report["n"], 2)
        self.assertEqual(report["relation_precision"], 1.0)
        self.assertEqual(report["triple_precision"], 0.5)
        self.assertEqual(report["by_relation"]["uses"]["n"], 2)


class EntityScoreTest(unittest.TestCase):
    def test_parse(self) -> None:
        self.assertEqual(
            parse_gold_entities("QASPER:Dataset; QuALITY:Dataset; BM25"),
            [("QASPER", "Dataset"), ("QuALITY", "Dataset"), ("BM25", "Concept")],
        )

    def test_typed_vs_untyped_recall(self) -> None:
        rows = [{"chunk_uuid": "u1", "gold_entities": "QASPER:Dataset; QuALITY:Dataset"}]
        predictions = {
            "u1": [
                {"entity": "QASPER", "type": "Method"},  # name matched, type wrong
                {"entity": "narrativeqa", "type": "Dataset"},  # not in gold
            ]
        }
        report = score_entities(rows, predictions)
        self.assertEqual(report["gold_entities"], 2)
        self.assertEqual(report["recall_untyped"], 0.5)
        self.assertEqual(report["recall_typed"], 0.0)
        self.assertAlmostEqual(report["precision_untyped"], 0.5)


class BenchmarkLenientTest(unittest.TestCase):
    def test_is_benchmark_span(self) -> None:
        for name in ("QASPER", "2WikiMultihopQA", "WCEP", "MultiFieldQA"):
            self.assertTrue(is_benchmark_span(name), name)
        self.assertFalse(is_benchmark_span("BERT"))

    def test_lenient_counts_dataset_or_method(self) -> None:
        rows = [
            {
                "relation": "evaluated_on",
                "head": "QASPER",
                "head_type": "Method",
                "tail": "SQuAD",
                "tail_type": "Dataset",
                "gold_relation_ok": "y",
                "gold_head_type_ok": "n",
                "gold_tail_type_ok": "y",
            }
        ]
        strict = score_triples(rows)
        lenient = score_triples(rows, benchmark_lenient=True)
        self.assertEqual(strict["triple_precision"], 0.0)
        self.assertEqual(lenient["triple_precision_lenient"], 1.0)


class SurfaceTextTest(unittest.TestCase):
    def test_clean_dotted(self) -> None:
        self.assertEqual(clean_surface("GPT 3 . 5"), "GPT 3.5")
        self.assertEqual(clean_surface("OntoNotes-5 . 0"), "OntoNotes-5.0")
        self.assertEqual(clean_surface("( BERT )"), "(BERT)")

    def test_format_triple_query(self) -> None:
        row = {
            "head": "QASPER",
            "head_type": "Dataset",
            "relation": "uses",
            "tail": "GPT-4",
            "tail_type": "Method",
            "arxiv_id": "2401.1",
        }
        self.assertEqual(
            format_triple_query(row),
            "Is it true that QASPER (Dataset) uses GPT-4 (Method) at arxiv paper 2401.1",
        )

    def test_snippet_centers_on_term(self) -> None:
        text = "Intro. We evaluate GPT-3.5 on SQuAD and HotpotQA. More text."
        out = snippet(text, ["SQuAD"], width=40)
        self.assertIn("SQuAD", out)
        self.assertLessEqual(len(out), 48)  # width + ellipses

    def test_snippet_without_term_returns_prefix(self) -> None:
        self.assertEqual(snippet("abcdef", ["zzz"], width=3), "abc")


if __name__ == "__main__":
    unittest.main()
