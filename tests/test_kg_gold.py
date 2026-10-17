r"""Contract tests for the K-04/K-05 gold-sample helpers (no DB, no key)."""

from __future__ import annotations

import unittest

from lens.kg_gold import (
    parse_gold_entities,
    score_entities,
    score_triples,
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
            {"relation": "uses", "gold_relation_ok": "y",
             "gold_head_type_ok": "y", "gold_tail_type_ok": "y"},
            {"relation": "uses", "gold_relation_ok": "y",
             "gold_head_type_ok": "n", "gold_tail_type_ok": "y"},
            {"relation": "uses", "gold_relation_ok": "",  # unlabeled -> ignored
             "gold_head_type_ok": "", "gold_tail_type_ok": ""},
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
                {"entity": "QASPER", "type": "Method"},     # name matched, type wrong
                {"entity": "narrativeqa", "type": "Dataset"},  # not in gold
            ]
        }
        report = score_entities(rows, predictions)
        self.assertEqual(report["gold_entities"], 2)
        self.assertEqual(report["recall_untyped"], 0.5)
        self.assertEqual(report["recall_typed"], 0.0)
        self.assertAlmostEqual(report["precision_untyped"], 0.5)


if __name__ == "__main__":
    unittest.main()
