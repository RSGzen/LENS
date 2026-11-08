"""Tests for the blind claim-truth sampler and scorer."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


truth = _load("truth_sample")


def _frame(index: int, status: str, chunk_id: str = "c1") -> dict:
    return {
        "unit_index": index,
        "text": f"claim {index}",
        "chunk_id": chunk_id,
        "verdict": {"status": status},
    }


class BuildRowsTests(unittest.TestCase):
    def test_sampling_is_stratified_and_blind(self) -> None:
        frames = [
            _frame(1, "grounded"),
            _frame(2, "grounded"),
            _frame(3, "unverifiable"),
        ]

        rows, sidecar = truth.build_rows(
            frames, {"c1": "the cited excerpt"}, run_id="run-1", per_status=1, seed=3
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            set(rows[0]),
            {"claim_id", "unit_index", "claim", "quote", "cited_excerpt", "label"},
        )
        self.assertEqual(
            sorted(entry["status"] for entry in sidecar.values()),
            ["grounded", "unverifiable"],
        )
        self.assertTrue(all(row["label"] == "" for row in rows))

    def test_a_bullet_split_into_quotes_is_one_row_per_quote(self) -> None:
        # One bullet quoting two sentences is two checks, not one claim printed
        # twice. The quote is the field that distinguishes them, and the claim
        # text is the assertion, not the bullet that contained it.
        frames = [
            {
                "unit_index": 12,
                "text": "the whole bullet",
                "subject": "SRLM",
                "predicate": "reports",
                "object": "context is external",
                "quote": "quote one",
                "chunk_id": "c1",
                "verdict": {"status": "grounded"},
            },
            {
                "unit_index": 12,
                "text": "the whole bullet",
                "subject": "SRLM",
                "predicate": "reports",
                "object": "no explicit self-query",
                "quote": "quote two",
                "chunk_id": "c1",
                "verdict": {"status": "unverifiable"},
            },
            {
                "unit_index": 12,
                "text": "the whole bullet",
                "subject": "SRLM",
                "predicate": "reports",
                "object": "context is external",
                "quote": "quote one",
                "chunk_id": "c1",
                "verdict": {"status": "grounded"},
            },
        ]

        rows, _ = truth.build_rows(
            frames, {"c1": "the cited excerpt"}, run_id="run-1", per_status=5, seed=3
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["quote"] for row in rows}, {"quote one", "quote two"})
        self.assertIn("SRLM reports", rows[0]["claim"])
        self.assertNotIn("whole bullet", rows[0]["claim"])

    def test_a_status_with_no_frames_is_skipped(self) -> None:
        rows, sidecar = truth.build_rows(
            [_frame(1, "grounded")],
            {"c1": "the cited excerpt"},
            run_id="run-1",
            per_status=5,
            seed=3,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(sidecar), 1)

    def test_a_frame_with_no_cited_excerpt_is_not_sampled(self) -> None:
        frames = [
            _frame(1, "grounded"),
            _frame(2, "unverifiable", chunk_id=""),
        ]

        rows, sidecar = truth.build_rows(
            frames,
            {"c1": "the cited excerpt"},
            run_id="run-1",
            per_status=5,
            seed=3,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual([entry["status"] for entry in sidecar.values()], ["grounded"])

    def test_quote_evidence_uses_the_verifier_span(self) -> None:
        frames = [
            {
                "unit_index": 1,
                "text": "the bullet",
                "subject": "SRLM",
                "predicate": "reports",
                "object": "context is external",
                "quote": "the quoted span",
                "chunk_id": "c1",
                "verdict": {"status": "grounded"},
            }
        ]

        rows, sidecar = truth.build_rows(
            frames,
            {"c1": "the chunk text"},
            run_id="run-1",
            per_status=1,
            seed=3,
            evidence="quote",
        )

        self.assertEqual(rows[0]["cited_excerpt"], "the quoted span")
        self.assertEqual(sidecar[rows[0]["claim_id"]]["evidence"], "quote")


class ScoreRowsTests(unittest.TestCase):
    def test_rates_are_conditional_on_decided_labels(self) -> None:
        sidecar = {
            "a": {"status": "grounded"},
            "b": {"status": "grounded"},
            "c": {"status": "unverifiable"},
        }
        labels = {"a": "supported", "b": "not supported", "c": "cannot judge"}

        result = truth.score_rows(labels, sidecar)

        self.assertEqual(result["grounded"]["decided"], 2)
        self.assertEqual(result["grounded"]["rate"], 0.5)
        self.assertEqual(result["unverifiable"]["decided"], 0)
        self.assertIsNone(result["unverifiable"]["rate"])

    def test_wilson_interval_contains_the_rate(self) -> None:
        low, high = truth.wilson(8, 10)

        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)


if __name__ == "__main__":
    unittest.main()
