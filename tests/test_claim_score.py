"""Tests for the gold-claim scorer (no key, no DB, no Docker)."""

from __future__ import annotations

import unittest

from lens.experiments.claim_score import (
    citation_precision,
    citation_recall,
    claim_recall,
    gold_chunk_recall,
    ledger_is_stale,
    normalise_value,
    value_matches,
)

CHUNK_A = "11111111-1111-1111-1111-111111111111"
CHUNK_B = "22222222-2222-2222-2222-222222222222"


class ValueMatchTests(unittest.TestCase):
    def test_a_number_matches_within_the_tolerance(self) -> None:
        self.assertTrue(value_matches("78.9", "the score was 78.9%"))
        self.assertTrue(value_matches("3", "there are 3 papers"))
        self.assertTrue(value_matches("3", "there are 3.0 papers"))

    def test_a_number_does_not_match_a_different_number(self) -> None:
        self.assertFalse(value_matches("3", "there are 13 papers"))
        self.assertFalse(value_matches("78.9", "the score was 79.4%"))

    def test_a_phrase_matches_by_content_words(self) -> None:
        self.assertTrue(value_matches("human evaluation", "All five report a human evaluation."))
        self.assertFalse(value_matches("human evaluation", "All five report an automatic metric."))

    def test_normalise_drops_commas_and_edge_punctuation(self) -> None:
        self.assertEqual(normalise_value(" 1,234.5% "), "1234.5%")


class ClaimRecallTests(unittest.TestCase):
    def test_a_value_in_any_sentence_counts(self) -> None:
        gold = [
            {"value": "7", "chunk_uuid": "c1", "arxiv_id": "2512.24601"},
            {"value": "human evaluation", "chunk_uuid": "c2", "arxiv_id": "2505.00023"},
        ]
        answer = "There are 7 relevant papers. Three report a human evaluation."

        matched, indexes = claim_recall(gold, answer)

        self.assertEqual(matched, 2)
        self.assertEqual(indexes, [0, 1])

    def test_a_missing_value_is_not_counted(self) -> None:
        matched, _ = claim_recall([{"value": "7"}], "There are nine papers.")

        self.assertEqual(matched, 0)


class CitationPrecisionTests(unittest.TestCase):
    def test_a_cited_chunk_that_contains_the_value_counts(self) -> None:
        gold = [{"value": "7"}]
        answer = f"There are 7 relevant papers [{CHUNK_A}]."
        texts = {CHUNK_A: "We count 7 relevant papers."}

        supported, judged = citation_precision(gold, answer, texts)

        self.assertEqual((supported, judged), (1, 1))

    def test_a_cited_chunk_that_does_not_contain_the_value_does_not(self) -> None:
        gold = [{"value": "7"}]
        answer = f"There are 7 relevant papers [{CHUNK_A}]."
        texts = {CHUNK_A: "We count nine relevant papers."}

        supported, judged = citation_precision(gold, answer, texts)

        self.assertEqual((supported, judged), (0, 1))

    def test_a_sentence_without_a_gold_value_is_not_judged(self) -> None:
        gold = [{"value": "7"}]
        answer = f"The papers differ in where they aggregate [{CHUNK_A}]."
        texts = {CHUNK_A: "They differ in locus."}

        supported, judged = citation_precision(gold, answer, texts)

        self.assertEqual((supported, judged), (0, 0))


class ChunkRecallTests(unittest.TestCase):
    def test_gold_chunk_recall_and_citation_recall(self) -> None:
        gold = [{"chunk_uuid": CHUNK_A}, {"chunk_uuid": CHUNK_B}]

        self.assertEqual(gold_chunk_recall(gold, {CHUNK_A, "other"}), (1, 2))
        self.assertEqual(citation_recall(gold, f"A claim [{CHUNK_B}]."), (1, 2))

    def test_a_claim_without_a_chunk_is_not_a_denominator(self) -> None:
        self.assertEqual(gold_chunk_recall([{"value": "7"}], {"c1"}), (0, 0))


class StaleLedgerTests(unittest.TestCase):
    def test_a_ledger_for_an_earlier_draft_is_stale(self) -> None:
        payload = {
            "units": 2,
            "frames": [{"text": "the earlier draft's bullet"}, {"text": "another bullet"}],
        }

        self.assertTrue(ledger_is_stale("ok", payload))

    def test_a_ledger_whose_frames_are_in_the_answer_is_not_stale(self) -> None:
        payload = {"units": 1, "frames": [{"text": "the final answer's bullet"}]}

        self.assertFalse(ledger_is_stale("The final answer's bullet says X.", payload))

    def test_an_empty_ledger_is_not_stale(self) -> None:
        self.assertFalse(ledger_is_stale("ok", {"units": 0, "frames": []}))


if __name__ == "__main__":
    unittest.main()
