r"""Tests for the calibration kit - no key, no database, no Docker.

The kit decides a threshold from labels, so its arithmetic and its decision rule are
checked against hand-built cases before it is ever pointed at real labels.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_calibration -v
"""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_set = _load("build_calibration_set")
variants_mod = _load("build_perturbations")
fitter = _load("fit_thresholds")


class BandAndWilsonTests(unittest.TestCase):
    def test_bands_are_half_open(self) -> None:
        self.assertEqual(build_set.band_of(0.0), 0)
        self.assertEqual(build_set.band_of(0.199), 0)
        self.assertEqual(build_set.band_of(0.2), 1)
        self.assertEqual(build_set.band_of(0.5), 2)
        self.assertEqual(build_set.band_of(0.8), 3)
        self.assertEqual(build_set.band_of(1.0), 3)

    def test_wilson_matches_a_known_interval(self) -> None:
        low, high = fitter.wilson(50, 100)

        self.assertAlmostEqual(low, 0.404, places=3)
        self.assertAlmostEqual(high, 0.596, places=3)

    def test_wilson_bounds_hold_at_the_extremes(self) -> None:
        self.assertEqual(fitter.wilson(0, 0), (0.0, 0.0))
        low, high = fitter.wilson(10, 10)
        self.assertLess(low, 1.0)
        self.assertLessEqual(high, 1.0)


class TruthTests(unittest.TestCase):
    def test_partial_counts_as_not_supported(self) -> None:
        # The gate's question is binary and a half-supported claim misleads a reader.
        self.assertEqual(fitter.truth_of("partial"), 0)
        self.assertEqual(fitter.truth_of("unsupported"), 0)
        self.assertEqual(fitter.truth_of("supported"), 1)
        self.assertIsNone(fitter.truth_of(""))


class CorruptionTests(unittest.TestCase):
    def test_a_number_is_swapped_for_one_that_is_absent(self) -> None:
        claim = "The method uses 32 layers of recursion."
        corrupted, ok = variants_mod.corrupt(claim, ["the method uses 32 layers"])

        self.assertTrue(ok)
        self.assertNotEqual(corrupted, claim)
        self.assertNotIn("32 layers", corrupted)

    def test_a_claim_with_no_number_cannot_be_corrupted(self) -> None:
        corrupted, ok = variants_mod.corrupt("The method is described as recursive.", ["recursive"])

        self.assertFalse(ok)
        self.assertEqual(corrupted, "The method is described as recursive.")

    def test_a_claim_whose_only_replacements_are_present_is_skipped(self) -> None:
        # 39, 64, 33 and 31 all appear: no clean negative is available.
        claim = "The window holds 32 rounds."
        excerpts = ["32 rounds", "39 rounds", "64 rounds", "33 rounds", "31 rounds"]

        _corrupted, ok = variants_mod.corrupt(claim, excerpts)

        self.assertFalse(ok)


class VariantBuildTests(unittest.TestCase):
    def rows(self) -> list[dict[str, str]]:
        return [
            {
                "pair_id": "pair-001",
                "claim_text": "ReCAP keeps 64 dialogue rounds in the window.",
                "cited_ids": "c1",
                "excerpt_1": "a sliding window of K rounds",
                "excerpt_2": "",
                "excerpt_3": "",
                "your_label": "supported",
            },
            {
                "pair_id": "pair-002",
                "claim_text": "The dataset is SQuAD.",
                "cited_ids": "c2",
                "excerpt_1": "",
                "excerpt_2": "",
                "excerpt_3": "",
                "your_label": "unsupported",
            },
        ]

    def test_only_supported_claims_get_a_twin(self) -> None:
        variants, skipped = variants_mod.build_variants(self.rows())

        kinds = [(variant["pair_id"], variant["kind"]) for variant in variants]

        self.assertIn(("pair-001", "corrupted"), kinds)
        self.assertNotIn(("pair-002", "corrupted"), kinds)
        self.assertEqual(skipped, 0)

    def test_a_corrupted_twin_is_labelled_unsupported(self) -> None:
        variants, _skipped = variants_mod.build_variants(self.rows())
        twin = [variant for variant in variants if variant["kind"] == "corrupted"][0]

        self.assertEqual(twin["label"], "unsupported")

    def test_unlabelled_rows_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"

            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["pair_id", "claim_text", "cited_ids", "excerpt_1", "your_label"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "pair-001",
                        "claim_text": "a claim",
                        "cited_ids": "",
                        "excerpt_1": "",
                        "your_label": "",
                    }
                )

            self.assertEqual(variants_mod.load_worksheet(path), [])


class DecisionTests(unittest.TestCase):
    def rows(
        self,
        negatives_aggregate: float = 0.9,
        positives: tuple[float, ...] = (0.10, 0.20, 0.30),
        negatives: tuple[float, ...] = (0.80, 0.90, 0.95),
    ) -> list[dict[str, object]]:
        return [{"truth": 1, "claim": p, "aggregate": p, "kind": "original"} for p in positives] + [
            {"truth": 0, "claim": p, "aggregate": negatives_aggregate, "kind": "original"}
            for p in negatives
        ]

    def test_a_separable_set_chooses_the_rule(self) -> None:
        results = fitter.sweep(self.rows())
        decision = fitter.choose(results)

        self.assertIsNotNone(decision["chosen"])
        chosen = decision["chosen"]
        self.assertEqual(chosen["precision"], 1.0)
        self.assertEqual(chosen["tnr"], 1.0)

    def test_an_inseparable_set_reports_uncalibrated(self) -> None:
        # The two classes interleave (0.10, 0.50, 0.60 against 0.20, 0.55, 0.90), so
        # no threshold separates them without rejecting good claims: the honest
        # answer is "uncalibrated", not a threshold that only looks decisive.
        results = fitter.sweep(
            self.rows(
                negatives_aggregate=0.5,
                positives=(0.10, 0.50, 0.60),
                negatives=(0.20, 0.55, 0.90),
            )
        )
        decision = fitter.choose(results)

        self.assertIsNone(decision["chosen"])
        self.assertIn("uncalibrated", decision["reason"])

    def test_metrics_are_computable(self) -> None:
        result = fitter.evaluate(self.rows(), "aggregate", 0.5, 1.1)

        self.assertEqual(result["positives"], 3)
        self.assertEqual(result["negatives"], 3)
        self.assertEqual(result["tpr"], 1.0)
        self.assertEqual(result["tnr"], 1.0)
        self.assertEqual(result["false_reject"], 0)
        self.assertEqual(result["pass_rate"], 0.5)

    def test_a_bad_claim_inside_the_band_is_a_false_accept(self) -> None:
        result = fitter.evaluate(self.rows(negatives_aggregate=0.25), "aggregate", 0.5, 1.1)

        self.assertEqual(result["tpr"], 0.0)
        self.assertEqual(result["precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
