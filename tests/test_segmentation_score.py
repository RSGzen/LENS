"""Tests for the segmentation scorer (no key, no DB)."""

from __future__ import annotations

import unittest

from lens.experiments.segmentation_score import (
    BUCKETS,
    bucket_scores,
    bucket_union_scores,
    clause_matches,
    normalise_clause,
    score_buckets,
)


class ClauseMatchTests(unittest.TestCase):
    def test_normalise_collapses_and_strips(self) -> None:
        self.assertEqual(normalise_clause("  Compare  RLM. "), "compare rlm")

    def test_containment_passes(self) -> None:
        self.assertTrue(clause_matches("published 2025-2026", "papers published 2025-2026"))

    def test_a_content_word_replaced_fails(self) -> None:
        self.assertFalse(clause_matches("published 2025-2026", "published 2024-2025"))

    def test_a_short_clause_with_no_content_words_fails(self) -> None:
        self.assertFalse(clause_matches("of the", "anything"))


class BucketScoreTests(unittest.TestCase):
    def test_recall_and_precision_counts(self) -> None:
        score = bucket_scores(
            ["compare where aggregation happens", "published 2025-2026"],
            ["compare where aggregation happens", "an invented constraint"],
        )

        self.assertEqual(score["gold"], 2)
        self.assertEqual(score["produced"], 2)
        self.assertEqual(score["matched_gold"], 1)
        self.assertEqual(score["matched_produced"], 1)
        self.assertEqual(score["recall"], 0.5)
        self.assertEqual(score["precision"], 0.5)

    def test_empty_buckets_are_none_not_zero(self) -> None:
        score = bucket_scores([], [])

        self.assertIsNone(score["recall"])
        self.assertIsNone(score["precision"])

    def test_all_buckets_are_returned(self) -> None:
        scored = score_buckets({"objectives": ["a long objective"]}, {})

        self.assertEqual(tuple(scored), BUCKETS)
        self.assertEqual(scored["objectives"]["recall"], 0.0)
        self.assertIsNone(scored["problems"]["recall"])


class UnionRecallTests(unittest.TestCase):
    """A clause filed under another bucket is present, not lost (D-26)."""

    def test_a_clause_filed_elsewhere_counts_under_the_union(self) -> None:
        gold = {"problems": ["the threshold removes every detector in the set"]}
        produced = {
            "constraints": ["if that threshold removes every detector in the set"],
            "problems": ["the exclusion is the problem I need explained"],
        }

        strict = score_buckets(gold, produced)
        union = bucket_union_scores(gold, produced)

        self.assertEqual(strict["problems"]["recall"], 0.0)
        self.assertEqual(union["problems"]["recall_union"], 1.0)
        self.assertEqual(union["problems"]["matched_union"], 1)

    def test_a_clause_nowhere_scores_zero(self) -> None:
        union = bucket_union_scores(
            {"constraints": ["published between 2020 and 2023"]},
            {"objectives": ["rank the five detectors"]},
        )

        self.assertEqual(union["constraints"]["matched_union"], 0)
        self.assertEqual(union["constraints"]["recall_union"], 0.0)

    def test_an_empty_bucket_is_none_not_zero(self) -> None:
        union = bucket_union_scores({"problems": []}, {"objectives": ["anything"]})

        self.assertIsNone(union["problems"]["recall_union"])

    def test_all_buckets_are_returned(self) -> None:
        union = bucket_union_scores({}, {})

        self.assertEqual(tuple(union), BUCKETS)


if __name__ == "__main__":
    unittest.main()
