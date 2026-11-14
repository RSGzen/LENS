r"""Contract tests for the blind LLM judge (no API key, no network, zero cost).

The judge is a measurement instrument: these tests pin what it is shown, because a
judge that grades an answer against the wrong question measures nothing.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import judge_answers  # noqa: E402

RUBRIC = {
    "version": "test",
    "facets": {
        "coverage": "states the gold items",
        "grounding": "every claim carries a citation",
        "scope_honesty": "states what it inspected",
        "usability": "a synthesis",
    },
}


class FakeClient:
    """Returns the same all-pass verdict and records every request."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, role, messages, *, trace_ctx, **kw):
        self.calls.append({"role": role, "messages": messages, **kw})

        class Result:
            text = json.dumps(
                {
                    "coverage": "pass",
                    "grounding": "pass",
                    "scope_honesty": "pass",
                    "usability": "pass",
                    "reason": "ok",
                }
            )

        return Result()


def instrument(tmp: str) -> Path:
    path = Path(tmp) / "gold.jsonl"
    path.write_text(
        json.dumps(
            {
                "instance_id": "i5-f4-01",
                "query": "What SHD does arXiv 2310.13576 report on Sachs?",
                "gold": {"answer": "10.6", "claims": [{"arxiv_id": "2310.13576", "value": "10.6"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class LoadInstrumentTest(unittest.TestCase):
    def test_question_and_gold_are_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            instances = judge_answers.load_instrument(instrument(tmp))

        self.assertEqual(instances["i5-f4-01"]["gold"], ["10.6"])
        self.assertIn("2310.13576", instances["i5-f4-01"]["query"])


class UserPromptTest(unittest.TestCase):
    def test_gold_answer_heads_the_block(self) -> None:
        prompt = judge_answers.user_prompt("Q?", "A.", RUBRIC, ["10.6"], "10.6")

        self.assertIn("Gold answer (what coverage must see)\n10.6", prompt)
        self.assertIn("Supporting items", prompt)

    def test_supporting_items_are_not_the_required_answer(self) -> None:
        # Coverage is judged against the answer; an instance may also list comparison
        # values that a correct answer does not have to repeat.
        prompt = judge_answers.user_prompt("Q?", "A.", RUBRIC, ["10.6", "8.2"], "10.6")

        self.assertIn("- 8.2", prompt)
        self.assertIn("not required to repeat all of them", prompt)

    def test_no_gold_leaves_no_heading(self) -> None:
        prompt = judge_answers.user_prompt("Q?", "A.", RUBRIC)

        self.assertNotIn("Gold answer", prompt)
        self.assertNotIn("Supporting items", prompt)


class JudgeRowsTest(unittest.TestCase):
    def test_the_question_comes_from_the_instrument_not_the_id(self) -> None:
        # The campaign CSV carries no query text; grading against ``query_id``
        # would show the judge the string "i5-f4-01" instead of the question.
        with tempfile.TemporaryDirectory() as tmp:
            instances = judge_answers.load_instrument(instrument(tmp))

        client = FakeClient()
        rows = [{"run_id": "run-1", "query_id": "i5-f4-01", "campaign_arm": "a2"}]

        judge_answers.judge_rows(
            rows,
            RUBRIC,
            client=client,
            answers=lambda run_id: "The SHD is 10.6.",
            instances=instances,
        )

        shown = client.calls[0]["messages"][1]["content"]
        self.assertIn("What SHD does arXiv 2310.13576 report", shown)
        self.assertNotIn("Question\ni5-f4-01", shown)

    def test_the_arm_name_never_reaches_the_prompt(self) -> None:
        client = FakeClient()
        rows = [{"run_id": "run-1", "query_id": "i5-f4-01", "campaign_arm": "a2"}]

        judge_answers.judge_rows(
            rows,
            RUBRIC,
            client=client,
            answers=lambda run_id: "answer",
            instances={"i5-f4-01": {"query": "Q?", "gold": ["10.6"]}},
        )

        self.assertNotIn("a2", json.dumps(client.calls[0]["messages"]))


if __name__ == "__main__":
    unittest.main()
