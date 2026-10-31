r"""Contract tests for the polarity probe (no network, no key).

The probe exists because the gate maps ``noul`` at 0.5, and that mapping is a
property of a third-party endpoint: a sign flip or a renamed field turns every
verdict into noise while the pipeline keeps reporting confidence. Four probes cost
about $0.00002 and catch it before a wave spends money.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_polarity -v
"""

from __future__ import annotations

import unittest

from lens.experiments.polarity import MIDPOINT, cases, probe_polarity
from lens.models import JevResult


def jev(*values: float | None, shape: str = "noul") -> callable:
    """Return a fake ``jev_call`` that yields ``values`` in order."""
    queue = list(values)

    def call(state, questions):
        value = queue.pop(0)
        answer = {} if value is None else {shape: value}
        return JevResult(
            answers={"polarity": answer},
            model="fake",
            provider="fake",
            tokens_in=10,
            cost_usd=0.000005,
            latency_ms=1,
        )

    return call


class PolarityProbeTest(unittest.TestCase):
    def test_the_four_cases_cover_both_signs(self) -> None:
        names = [name for name, _high, _state, _question in cases()]

        self.assertEqual(
            names,
            ["true-claim", "false-claim", "supported-answer", "contradicted-answer"],
        )
        self.assertEqual([high for _n, high, _s, _q in cases()], [True, False] * 2)

    def test_a_correctly_polarised_layer_passes(self) -> None:
        result = probe_polarity(jev(0.92, 0.03, 0.89, 0.01))

        self.assertTrue(result.ok)
        self.assertEqual(result.skewed, [])
        self.assertAlmostEqual(result.cost_usd, 4 * 0.000005)

    def test_a_flipped_sign_is_reported_by_name(self) -> None:
        # The failure mode that matters: every verdict inverted, pipeline healthy.
        result = probe_polarity(jev(0.03, 0.92, 0.01, 0.89))

        self.assertFalse(result.ok)
        self.assertEqual(
            result.skewed,
            ["true-claim", "false-claim", "supported-answer", "contradicted-answer"],
        )

    def test_one_wrong_probe_is_enough_to_fail(self) -> None:
        result = probe_polarity(jev(0.92, 0.03, 0.51, 0.49))

        self.assertTrue(result.ok)  # 0.51 and 0.49 straddle the midpoint correctly

        result = probe_polarity(jev(0.92, 0.03, 0.40, 0.01))
        self.assertFalse(result.ok)
        self.assertEqual(result.skewed, ["supported-answer"])

    def test_the_midpoint_is_the_gates_own(self) -> None:
        self.assertEqual(MIDPOINT, 0.5)
        result = probe_polarity(jev(0.5, 0.499, 0.5, 0.0))

        self.assertTrue(result.ok)  # exactly 0.5 counts as high

    def test_an_unreadable_answer_shape_fails_with_a_reason(self) -> None:
        # A renamed field is the realistic breakage; it must not read as "skewed"
        # without saying which probe could not be read.
        result = probe_polarity(jev(*([None] * 4)))

        self.assertFalse(result.ok)
        self.assertIn("no numeric `noul`", result.probes[0].error)
        self.assertTrue(all(probe.noul is None for probe in result.probes))

    def test_to_dict_is_json_safe(self) -> None:
        payload = probe_polarity(jev(0.92, 0.03, 0.89, 0.01)).to_dict()

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["skewed"], [])
        self.assertEqual(len(payload["probes"]), 4)
        self.assertIsInstance(payload["probes"][0]["noul"], float)


if __name__ == "__main__":
    unittest.main()
