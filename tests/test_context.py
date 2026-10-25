r"""Contract tests for root-context assembly (no key, no network).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_context -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lens.agent import context


class ToolBindingTestCase(unittest.TestCase):
    """The tools must be callable as the manifest prints them (LOG-115).

    The M9 smoke failed with ``NameError: shallow_match is not defined`` on every
    turn: the prompt shows ``shallow_match(buckets: ...)`` but only
    ``rlm_api.tool()`` existed, so the model reported "retrieval unavailable" and
    never retrieved anything.
    """

    def _exec_bindings(self, parameters: dict, calls: list | None = None) -> dict:
        """Exec the generated block with a stub ``rlm_api``; return its namespace."""
        import sys
        import types

        module = types.ModuleType("rlm_api")
        module.tool = lambda name, **kwargs: (
            calls.append((name, kwargs)) if calls is not None else None
        )
        original = sys.modules.get("rlm_api")
        sys.modules["rlm_api"] = module

        try:
            namespace: dict = {}
            exec(context.tool_bindings(parameters), namespace)
            return namespace
        finally:
            if original is None:
                sys.modules.pop("rlm_api", None)
            else:
                sys.modules["rlm_api"] = original

    def test_bindings_define_a_callable_per_tool(self) -> None:
        calls: list[tuple[str, dict]] = []
        namespace = self._exec_bindings({"shallow_match": ["buckets", "k"]}, calls)

        # Both calling styles must reach one dispatch path.
        self.assertIsNone(namespace["shallow_match"](buckets={"q": 1}, k=50))
        namespace["shallow_match"]({"q": 2})

        self.assertEqual(
            calls,
            [("shallow_match", {"buckets": {"q": 1}, "k": 50}),
             ("shallow_match", {"buckets": {"q": 2}})],
        )

    def test_too_many_positionals_is_a_clear_type_error(self) -> None:
        namespace = self._exec_bindings({"fetch_papers": ["arxiv_ids"]})

        with self.assertRaises(TypeError) as caught:
            namespace["fetch_papers"]([["a"]], "extra")

        self.assertIn("at most 1 positional", str(caught.exception))

    def test_bindings_are_valid_python_for_no_tools(self) -> None:
        # An empty registry must still produce an executable block.
        self.assertEqual(self._exec_bindings({})["TOOL_PARAMETERS"], {})

    def test_prompt_names_both_calling_forms(self) -> None:
        prompt = context.build_system_prompt("1. shallow_match(buckets)", {}, {})

        self.assertIn("bound as a plain function", prompt)
        self.assertIn("rlm_api.tool(", prompt)

    def test_prompt_states_payload_shapes_and_variable_search(self) -> None:
        # The S6 smoke: gpt-5 lost turns to `TypeError: string indices must be
        # integers` (guessing return shapes) and then concluded "no chunk contains
        # OOLONG" while 88 retrieved chunks did — it had searched its printed digest
        # instead of the namespace variable (LOG-117).
        prompt = context.build_system_prompt("1. shallow_match(buckets)", {}, {})

        self.assertIn("Payload shapes", prompt)
        self.assertIn("list[str]", prompt)
        self.assertIn("Search the variable, not the digest", prompt)
        self.assertIn("negative finding is a claim", prompt)


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

    def test_sentinel_inside_a_fence_is_not_an_answer(self) -> None:
        # print("FINAL_ANSWER: ...") belongs to the block; it must not end the run.
        text = "```python\nprint('FINAL_ANSWER: not yet')\n```\nstill working"
        self.assertIsNone(context.extract_final_answer(text))

    def test_answer_after_the_fence_counts(self) -> None:
        # The opposite failure: code + a real answer used to be discarded, so the
        # run looped to the iteration cap with an empty answer.md (LOG-115).
        text = "```python\nprint('ran')\n```\nFINAL_ANSWER: The answer is [1]."
        self.assertEqual(context.extract_final_answer(text), "The answer is [1].")

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
        self.assertIn("FINAL_ANSWER:", prompt)
        # Every placeholder must be gone: a prompt shipped to gpt-5 with a raw
        # `<<BUDGET>>` is a silent quality bug, not a crash.
        self.assertNotIn("<<", prompt)
        self.assertNotIn(">>", prompt)

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
                tool_manifest="- tool()",
                budget_snapshot={"left": {"iterations": 3}},
            )
            self.assertTrue(paths["tools.md"].is_file())
            self.assertTrue(paths["budget.md"].is_file())
            self.assertTrue(paths["AGENTS.md"].is_file())
            # query.json is M6's file (write_query_json), not this module's:
            # one writer per file or the artifact drifts from the context.
            self.assertNotIn("query.json", paths)

    def test_write_context_files_creates_the_workspace(self) -> None:
        # The runtime seeds <run_dir>/work before the first turn; on a fresh run
        # that directory may not exist yet.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "work"
            paths = context.write_context_files(
                target, tool_manifest="- tool()", budget_snapshot={}
            )
            self.assertTrue(target.is_dir())
            agents = paths["AGENTS.md"].read_text(encoding="utf-8")
            self.assertIn("FINAL_ANSWER:", agents)   # not an empty placeholder file

    def test_truncate_observation_leaves_short_text_alone(self) -> None:
        # Prefixing "Truncated:" to text that fits is a lie the model would read.
        self.assertEqual(context.truncate_observation("ok"), "ok")

    def test_extract_python_block_accepts_common_fences(self) -> None:
        # The prompt asks for ```python, but models also emit py / python3 / bare.
        self.assertEqual(context.extract_python_block("```py\nprint(1)\n```"), "print(1)")
        self.assertEqual(context.extract_python_block("```\nprint(2)\n```"), "print(2)")
        self.assertEqual(context.extract_python_block("```Python3\nprint(3)\n```"), "print(3)")
        self.assertEqual(context.extract_python_block("```python\n  x = 1\n```"), "  x = 1")
        # ... and a non-Python block must never be executed.
        self.assertIsNone(context.extract_python_block('```json\n{"a": 1}\n```'))

    def test_budget_warning_is_silent_below_threshold(self) -> None:
        snapshot = {"limits": {"tokens": 100}, "used": {"tokens_in": 10}}
        self.assertIsNone(context.budget_warning(snapshot))

    def test_budget_warning_fires_near_a_cap(self) -> None:
        snapshot = {"limits": {"tokens": 100}, "used": {"tokens_in": 85}}
        warning = context.budget_warning(snapshot)
        self.assertIsNotNone(warning)
        self.assertIn("FINAL_ANSWER", warning)

    def test_budget_warning_reports_the_binding_cap_with_numbers(self) -> None:
        # cost is 95 %, tokens 85 %: the *most consumed* cap must win, and
        # {which} carries the absolute burn (a bare percentage is not actionable).
        snapshot = {
            "limits": {"iterations": 8, "tokens": 100, "cost_usd": 1.0, "backtracks": 3},
            "used": {"iterations": 1, "tokens_in": 50, "tokens_out": 35, "cost_usd": 0.95},
        }
        warning = context.budget_warning(snapshot) or ""
        self.assertIn("95%", warning)
        self.assertIn("cost", warning)

    def test_budget_warning_counts_input_plus_output_tokens(self) -> None:
        # The cap is total work (LOG-96): 50 + 35 = 85 % of 100.
        snapshot = {"limits": {"tokens": 100}, "used": {"tokens_in": 50, "tokens_out": 35}}
        self.assertIn("85%", context.budget_warning(snapshot) or "")


if __name__ == "__main__":
    unittest.main()
