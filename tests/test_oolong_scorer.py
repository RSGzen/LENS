r"""The vendored OOLONG scorer must behave exactly as the release does."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import oolong_scorer  # noqa: E402


def datapoint(answer: str, answer_type: str = "ANSWER_TYPE.LABEL") -> dict:
    return {"id": "x", "answer": answer, "answer_type": answer_type, "dataset": "trec_coarse"}


class SynthParseTest(unittest.TestCase):
    def test_the_answer_after_the_last_colon_wins(self) -> None:
        trimmed, confidence = oolong_scorer.synth_attempt_answer_parse("Label: numeric value")

        self.assertEqual(trimmed, "numeric value")
        self.assertEqual(confidence, "vhigh")

    def test_markdown_and_brackets_are_stripped_but_a_space_survives(self) -> None:
        # The release strips the text *before* removing the decorations, so a bolded
        # bracketed answer keeps a leading space and then fails the equality test.
        # That is the release's behaviour, vendored unchanged.
        trimmed, _ = oolong_scorer.synth_attempt_answer_parse("**Answer:** [location]")

        self.assertEqual(trimmed, " location")

    def test_no_colon_and_short_returns_the_whole_text(self) -> None:
        trimmed, confidence = oolong_scorer.synth_attempt_answer_parse("abbreviation")

        self.assertEqual(trimmed, "abbreviation")
        self.assertEqual(confidence, "low")


class SynthScoreTest(unittest.TestCase):
    def test_an_exact_label_scores_one(self) -> None:
        scored = oolong_scorer.synth_process_response(datapoint("['spam']"), "Label: spam")

        self.assertEqual(scored["score"], 1)

    def test_a_comparison_phrase_scores_through_the_mapping(self) -> None:
        scored = oolong_scorer.synth_process_response(
            datapoint("['more common than']", "ANSWER_TYPE.COMPARISON"), "Answer: more common"
        )

        self.assertEqual(scored["score"], 1)

    def test_a_numeric_near_miss_gets_partial_credit(self) -> None:
        scored = oolong_scorer.synth_process_response(
            datapoint("['5']", "ANSWER_TYPE.NUMERIC"), "Answer: 3"
        )

        self.assertAlmostEqual(scored["score"], 0.75**2, places=6)

    def test_a_wrong_label_scores_zero(self) -> None:
        scored = oolong_scorer.synth_process_response(datapoint("['spam']"), "Label: ham")

        self.assertEqual(scored["score"], 0)

    def test_prose_after_the_answer_loses_the_parse(self) -> None:
        # The release parses after the last colon: an explanation that contains one
        # is silently mis-parsed, which is why the prompt must end with a bare line.
        scored = oolong_scorer.synth_process_response(
            datapoint("['numeric value']"),
            "Label: numeric value. I checked all 3,182 records: 412 matched",
        )

        self.assertEqual(scored["score"], 0)
        self.assertNotEqual(scored["attempted_parse"], "numeric value")


class ScoreAdapterTest(unittest.TestCase):
    def test_the_adapter_reads_the_instrument_shape(self) -> None:
        instance = {
            "instance_id": "oolong-tc-1",
            "gold": {"answer": "['spam']"},
            "answer_type": "ANSWER_TYPE.LABEL",
            "meta": {"context_window_id": 6, "dataset": "trec_coarse"},
        }

        self.assertEqual(oolong_scorer.score("Answer: spam", instance), 1.0)
        self.assertEqual(oolong_scorer.score("Answer: ham", instance), 0.0)


if __name__ == "__main__":
    unittest.main()
