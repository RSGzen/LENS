r"""Contract tests for shared types (no API key, no network, zero cost).

These pin how :class:`lens.models.Question` validates ``criteria`` per ``type``.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from lens.models import CallResult, JevResult, Question


class CallResultTests(unittest.TestCase):
    def test_fields_and_default_finish_reason(self) -> None:
        result = CallResult(
            text="hi",
            message={"role": "assistant", "content": "hi"},
            model="openai/gpt-5",
            provider="OpenAI",
            model_snapshot=None,
            tokens_in=11,
            tokens_out=2,
            latency_ms=340,
            cost_usd=0.00042,
        )
        self.assertEqual(result.text, "hi")
        self.assertEqual(result.tokens_in, 11)
        self.assertIsNone(result.finish_reason)


class JevResultTests(unittest.TestCase):
    def test_fields(self) -> None:
        result = JevResult(
            answers={"bucket": {"type": "choice", "choice": "problem"}},
            model="typesafe/jev-1.13-20260917",
            provider="TypeSafe",
            tokens_in=476,
            cost_usd=0.000019992,
            latency_ms=210,
        )
        self.assertEqual(result.answers["bucket"]["choice"], "problem")
        self.assertEqual(result.model, "typesafe/jev-1.13-20260917")


class QuestionTests(unittest.TestCase):
    def test_score_accepts_ordered_list(self) -> None:
        question = Question(
            type="score",
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"],
        )
        self.assertEqual(question.type, "score")

    def test_choice_accepts_labeled_dict(self) -> None:
        question = Question(
            type="choice",
            instructions="Which bucket?",
            criteria={"problem": "Problem statement", "objective": "Objective"},
        )
        self.assertEqual(question.type, "choice")

    def test_noul_accepts_true_false_dict(self) -> None:
        question = Question(
            type="noul",
            instructions="Is this a problem statement?",
            criteria={"true": "Yes", "false": "No"},
        )
        self.assertEqual(question.type, "noul")

    def test_score_rejects_dict(self) -> None:
        with self.assertRaises(ValidationError):
            Question(type="score", instructions="x", criteria={"low": "Low"})

    def test_choice_rejects_list(self) -> None:
        with self.assertRaises(ValidationError):
            Question(type="choice", instructions="x", criteria=["a", "b"])

    def test_noul_rejects_list(self) -> None:
        with self.assertRaises(ValidationError):
            Question(type="noul", instructions="x", criteria=["true", "false"])

    def test_noul_requires_true_false_keys(self) -> None:
        with self.assertRaises(ValidationError):
            Question(
                type="noul",
                instructions="x",
                criteria={"yes": "Yes", "no": "No"},
            )

    def test_choice_requires_at_least_two_options(self) -> None:
        with self.assertRaises(ValidationError):
            Question(type="choice", instructions="x", criteria={"only": "One"})

    def test_score_requires_at_least_two_levels(self) -> None:
        with self.assertRaises(ValidationError):
            Question(type="score", instructions="x", criteria=["Only"])


if __name__ == "__main__":
    unittest.main()
