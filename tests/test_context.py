r"""Contract tests for root-context assembly (no key, no network).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_context -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lens.agent import context


class ContextTestCase(unittest.TestCase):
    def test_extract_python_block(self) -> None:
        text = "Here is the step:\n```python\nprint('hi')\n```\nDone."
        self.assertEqual(context.extract_python_block(text), "print('hi')")

    def test_extract_python_block_absent(self) -> None:
        self.assertIsNone(context.extract_python_block("FINAL_ANSWER: ok"))

    def test_extract_final_answer(self) -> None:
        text = "some reasoning\nFINAL_ANSWER: The answer is [1]."
        self.assertEqual(context.extract_final_answer(text), "The answer is [1].")

    def test_extract_final_answer_absent(self) -> None:
        self.assertIsNone(context.extract_final_answer("```python\nprint(1)\n```"))

    def test_count_tokens_positive(self) -> None:
        self.assertGreater(context.count_tokens("hello world"), 0)

    def test_truncate_observation(self) -> None:
        out = context.truncate_observation("x" * 100, max_chars=10)
        self.assertLessEqual(len(out), 60)
        self.assertIn("truncated", out.lower())

    def test_build_system_prompt_fills_placeholders(self) -> None:
        prompt = context.build_system_prompt(
            tool_manifest="- shallow_match(...)",
            query_buckets={"problems": ["aggregate long contexts"]},
            budget_snapshot={"left": {"iterations": 10}},
        )
        self.assertIn("shallow_match", prompt)
        self.assertIn("aggregate long contexts", prompt)
        self.assertNotIn("<<TOOL_MANIFEST>>", prompt)
        self.assertIn("FINAL_ANSWER:", prompt)

    def test_build_messages_starts_with_system(self) -> None:
        messages = context.build_messages(
            system_prompt="sys",
            memory=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(messages[0], {"role": "system", "content": "sys"})
        self.assertEqual(messages[-1]["content"], "hi")

    def test_compact_messages_keeps_system_and_recent_turn(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            *[{"role": "assistant", "content": "x" * 100} for _ in range(20)],
            {"role": "user", "content": "latest"},
        ]
        compacted = context.compact_messages(messages, token_limit=10, ratio=0.5)
        self.assertEqual(compacted[0]["role"], "system")
        self.assertEqual(compacted[-1]["content"], "latest")
        self.assertLessEqual(len(compacted), len(messages))

    def test_write_context_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = context.write_context_files(
                tmp,
                query_buckets={"problems": ["p1"]},
                tool_manifest="- tool()",
                budget_snapshot={"left": {"iterations": 3}},
            )
            self.assertTrue(paths["query.json"].is_file())
            self.assertTrue(paths["tools.md"].is_file())
            self.assertTrue(paths["budget.md"].is_file())
            self.assertTrue(paths["AGENTS.md"].is_file())
            data = json.loads(paths["query.json"].read_text(encoding="utf-8"))
            self.assertEqual(data["problems"], ["p1"])


if __name__ == "__main__":
    unittest.main()
