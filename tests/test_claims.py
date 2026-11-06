"""Contract tests for the claim-frame verifier.

No key, no database, no Docker: every test builds a frame and the chunk text a
run would have recorded, then asserts the status the verifier must return.
"""

from __future__ import annotations

import unittest

from lens.observability.claims import (
    CONTRADICTED,
    GROUNDED,
    UNVERIFIABLE,
    ClaimFrame,
    parse_frames,
    verify_frame,
    verify_units,
)

CHUNK_ID = "3b4365f9-0000-0000-0000-000000000001"


def _frame(**overrides: object) -> ClaimFrame:
    """Return a measurement frame with the fields a test does not override."""
    base = {
        "unit_index": 0,
        "text": "RLM reaches 56.0 accuracy on OOLONG trec_coarse.",
        "predicate": "measures",
        "subject": "RLM",
        "benchmark": "OOLONG trec_coarse",
        "metric": "accuracy",
        "value": "56.0",
        "direction": None,
        "quote": "reaches 56.0 accuracy on the OOLONG trec_coarse benchmark",
        "chunk_id": CHUNK_ID,
    }
    base.update(overrides)

    return ClaimFrame(**base)  # type: ignore[arg-type]


class MeasurementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.texts = {
            CHUNK_ID: (
                "The RLM method reaches 56.0 accuracy on the OOLONG "
                "trec_coarse benchmark, against 44.0 for the flat baseline."
            )
        }

    def test_value_in_the_span_is_grounded(self) -> None:
        verdict = verify_frame(_frame(), self.texts)

        self.assertEqual(verdict.status, GROUNDED, verdict.reason)

    def test_a_different_value_is_contradicted(self) -> None:
        verdict = verify_frame(_frame(value="61.0"), self.texts)

        self.assertEqual(verdict.status, CONTRADICTED)
        self.assertIn("56.0", verdict.reason)

    def test_a_missing_benchmark_is_unverifiable(self) -> None:
        verdict = verify_frame(_frame(benchmark="S-NIAH"), self.texts)

        self.assertEqual(verdict.status, UNVERIFIABLE)

    def test_a_decimal_survives_normalisation(self) -> None:
        texts = {CHUNK_ID: "The method improves nDCG by 5.6 points."}

        verdict = verify_frame(
            _frame(
                predicate="measures",
                benchmark=None,
                metric="nDCG",
                value="5.6",
                quote="improves nDCG by 5.6 points",
            ),
            texts,
        )

        self.assertEqual(verdict.status, GROUNDED, verdict.reason)

    def test_a_retyped_quote_is_unverifiable(self) -> None:
        verdict = verify_frame(_frame(quote="RLM reaches 56.0 accuracy on OOLONG."), self.texts)

        self.assertEqual(verdict.status, UNVERIFIABLE)

    def test_a_missing_chunk_is_unverifiable(self) -> None:
        verdict = verify_frame(_frame(), {})

        self.assertEqual(verdict.status, UNVERIFIABLE)


class AttributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.texts = {
            CHUNK_ID: (
                "We introduce SliSum, a method that groups retrieved chunks "
                "into longer contexts before summarisation."
            )
        }

    def test_named_subject_and_object_are_grounded(self) -> None:
        frame = _frame(
            predicate="reports",
            subject="SliSum",
            object="groups retrieved chunks into longer contexts",
            benchmark=None,
            metric=None,
            value=None,
            quote="SliSum, a method that groups retrieved chunks into longer contexts",
        )

        verdict = verify_frame(frame, self.texts)

        self.assertEqual(verdict.status, GROUNDED, verdict.reason)

    def test_an_absent_object_is_unverifiable(self) -> None:
        frame = _frame(
            predicate="reports",
            subject="SliSum",
            object="uses a graph database",
            benchmark=None,
            metric=None,
            value=None,
            quote="We introduce SliSum, a method that groups retrieved chunks",
        )

        verdict = verify_frame(frame, self.texts)

        self.assertEqual(verdict.status, UNVERIFIABLE)


class ComparisonTests(unittest.TestCase):
    def test_the_stated_direction_is_grounded(self) -> None:
        texts = {CHUNK_ID: "RLM outperforms naive RAG on the aggregation task."}
        frame = _frame(
            predicate="compares",
            subject="RLM",
            object="naive RAG",
            benchmark=None,
            metric=None,
            value=None,
            direction=">",
            quote="RLM outperforms naive RAG on the aggregation task.",
        )

        verdict = verify_frame(frame, texts)

        self.assertEqual(verdict.status, GROUNDED, verdict.reason)

    def test_the_opposite_direction_is_contradicted(self) -> None:
        texts = {CHUNK_ID: "Naive RAG outperforms RLM on the aggregation task."}

        verdict = verify_frame(
            _frame(
                predicate="compares",
                subject="RLM",
                object="naive RAG",
                benchmark=None,
                metric=None,
                value=None,
                direction=">",
                quote="Naive RAG outperforms RLM on the aggregation task.",
            ),
            texts,
        )

        self.assertEqual(verdict.status, CONTRADICTED)

    def test_no_cue_is_unverifiable(self) -> None:
        texts = {CHUNK_ID: "RLM and naive RAG were run on the aggregation task."}

        verdict = verify_frame(
            _frame(
                predicate="compares",
                subject="RLM",
                object="naive RAG",
                benchmark=None,
                metric=None,
                value=None,
                direction=">",
                quote="RLM and naive RAG were run on the aggregation task.",
            ),
            texts,
        )

        self.assertEqual(verdict.status, UNVERIFIABLE)


class ParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.units = [("First claim.", [CHUNK_ID]), ("Second claim.", ["other-id"])]

    def test_a_frame_addressing_an_unknown_unit_is_dropped(self) -> None:
        frames, errors = parse_frames(
            {"frames": [{"unit_index": 7, "predicate": "measures", "subject": "X"}]},
            self.units,
        )

        self.assertEqual(frames, [])
        self.assertTrue(errors)

    def test_a_frame_citing_an_unrequested_chunk_is_dropped(self) -> None:
        frames, errors = parse_frames(
            {
                "frames": [
                    {
                        "unit_index": 1,
                        "predicate": "measures",
                        "subject": "X",
                        "chunk_id": CHUNK_ID,
                    }
                ]
            },
            self.units,
        )

        self.assertEqual(frames, [])
        self.assertTrue(any("did not cite" in error for error in errors))

    def test_a_single_cited_chunk_is_stamped_when_the_model_omits_it(self) -> None:
        frames, errors = parse_frames(
            {"frames": [{"unit_index": 0, "predicate": "measures", "subject": "X"}]},
            self.units,
        )

        self.assertEqual(errors, [])
        self.assertEqual(frames[0].chunk_id, CHUNK_ID)

    def test_a_non_list_reply_is_an_error(self) -> None:
        frames, errors = parse_frames({"frames": None}, self.units)

        self.assertEqual(frames, [])
        self.assertEqual(len(errors), 1)

    def test_a_frame_for_a_unit_the_call_did_not_show_is_dropped(self) -> None:
        frames, errors = parse_frames(
            {"frames": [{"unit_index": 1, "predicate": "measures", "subject": "X"}]},
            self.units,
            allowed_units={0},
        )

        self.assertEqual(frames, [])
        self.assertTrue(any("not shown" in error for error in errors))


class ReportTests(unittest.TestCase):
    def test_coverage_counts_units_not_frames(self) -> None:
        units = [("First claim.", [CHUNK_ID]), ("Second claim.", [CHUNK_ID])]
        frames = [
            _frame(unit_index=0),
            _frame(unit_index=0, value="61.0"),
            _frame(unit_index=1, quote="not in the chunk at all"),
        ]
        texts = {
            CHUNK_ID: ("The RLM method reaches 56.0 accuracy on the OOLONG trec_coarse benchmark.")
        }

        report = verify_units(units, frames, texts, run_id="run-1")

        self.assertEqual(report.units, 2)
        self.assertEqual(report.counts[GROUNDED], 1)
        self.assertEqual(report.counts[CONTRADICTED], 1)
        self.assertEqual(report.counts[UNVERIFIABLE], 1)
        self.assertEqual(report.covered_units, 1)
        self.assertEqual(report.coverage, 0.5)
        self.assertEqual(report.to_dict()["run_id"], "run-1")


if __name__ == "__main__":
    unittest.main()
