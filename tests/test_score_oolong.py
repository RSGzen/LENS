r"""The W3 scorer's rules: an empty answer is excluded, and types are reported apart."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import score_oolong  # noqa: E402


class UsableTest(unittest.TestCase):
    def test_a_normal_answer_is_usable(self) -> None:
        self.assertTrue(score_oolong.usable("completed", "", "Label: spam"))

    def test_an_empty_answer_is_not_usable(self) -> None:
        # The observed failure: finish_reason=length, 131k reasoning tokens, 2-byte answer.
        self.assertFalse(score_oolong.usable("completed", "flat", "\n"))

    def test_a_failed_run_is_not_usable(self) -> None:
        self.assertFalse(score_oolong.usable("failed", "error", "Run failed before an answer"))

    def test_an_error_stop_is_not_usable(self) -> None:
        self.assertFalse(score_oolong.usable("partial", "error", "half"))


class SummariseTest(unittest.TestCase):
    def rows(self) -> list[dict]:
        return [
            {"arm": "a0-doc", "instance_id": "i1", "score": 1.0, "usable": True},
            {"arm": "a0-doc", "instance_id": "i2", "score": 0.0, "usable": True},
            {"arm": "a0-doc", "instance_id": "i3", "score": 0.0, "usable": False},
            {"arm": "a2-doc", "instance_id": "i1", "score": 1.0, "usable": True},
            {"arm": "a2-doc", "instance_id": "i2", "score": 1.0, "usable": True},
        ]

    def test_excluded_rows_never_enter_a_mean(self) -> None:
        summary = score_oolong.summarise(
            self.rows(),
            {"i1": "ANSWER_TYPE.LABEL", "i2": "ANSWER_TYPE.NUMERIC", "i3": "ANSWER_TYPE.NUMERIC"},
        )

        self.assertEqual(summary["excluded"], 1)
        self.assertEqual(summary["arms"]["a0-doc"], (2, 0.5))
        self.assertEqual(summary["arms"]["a2-doc"], (2, 1.0))

    def test_types_are_reported_separately(self) -> None:
        summary = score_oolong.summarise(
            self.rows(),
            {"i1": "ANSWER_TYPE.LABEL", "i2": "ANSWER_TYPE.NUMERIC", "i3": "ANSWER_TYPE.NUMERIC"},
        )

        self.assertEqual(summary["types"]["ANSWER_TYPE.LABEL"], (2, 1.0))
        self.assertEqual(summary["types"]["ANSWER_TYPE.NUMERIC"], (2, 0.5))


if __name__ == "__main__":
    unittest.main()
