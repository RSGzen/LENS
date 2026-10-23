r"""Contract tests for the host tool registry (no key, no DB).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_tool_registry -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import lens.observability.trajectory as trajectory_module
from lens.agent.tool_registry import ToolRegistry, args_hash
from lens.observability.trajectory import TraceCtx, Trajectory


def echo(text: str) -> dict:
    """Echo one string back."""
    return {"echo": text}


class ToolRegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_root = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = self._tmp.name
        self.trajectory = Trajectory("run-test")
        self.registry = ToolRegistry()
        self.registry.register("echo", echo, description="Echo a string.")
        self.ctx = TraceCtx(trace_id="t1", step_idx=0, actor="tool")

    def tearDown(self) -> None:
        trajectory_module.LENS_RUNS_ROOT_PATH = self._orig_root
        self._tmp.cleanup()

    def test_manifest_has_name_signature_and_doc(self) -> None:
        manifest = self.registry.manifest()
        self.assertIn("echo", manifest)
        self.assertIn("text", manifest)
        self.assertIn("Echo a string", manifest)
        self.assertNotIn("def echo", manifest)

    def test_dispatch_returns_and_logs(self) -> None:
        result = self.registry.dispatch(
            "echo", {"text": "hi"}, trace_ctx=self.ctx, trajectory=self.trajectory
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.result, {"echo": "hi"})
        self.assertIsNotNone(result.result_ref)
        events = [
            json.loads(line)
            for line in self.trajectory.trajectory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(events[-1]["tool"], "echo")
        self.assertEqual(events[-1]["args_hash"], args_hash({"text": "hi"}))
        self.assertIn("result_ref", events[-1])

    def test_dispatch_unknown_tool_is_structured_error(self) -> None:
        result = self.registry.dispatch(
            "missing", {}, trace_ctx=self.ctx, trajectory=self.trajectory
        )
        self.assertFalse(result.ok)
        self.assertIn("error", result.to_dict())

    def test_args_hash_is_deterministic_and_order_insensitive(self) -> None:
        self.assertEqual(args_hash({"a": 1, "b": 2}), args_hash({"b": 2, "a": 1}))
        self.assertNotEqual(args_hash({"a": 1}), args_hash({"a": 2}))

    def test_summary_is_bounded(self) -> None:
        summary = self.registry.summarize_result({"x": "y" * 10000})
        self.assertLessEqual(len(summary), 2500)

    # ------------------------------------------------- regressions (D1-D10)
    def test_duplicate_registration_is_rejected(self) -> None:
        # The inverted guard (`if name not in ...: raise`) made *every* first
        # registration fail and let a duplicate silently replace a tool.
        with self.assertRaises(ValueError):
            self.registry.register("echo", echo)

    def test_names_and_manifest_agree_on_registration_order(self) -> None:
        self.registry.register("alpha", echo, description="a")

        def zulu(x: int = 1) -> int:
            """Last registered."""
            return x

        self.registry.register("zulu", zulu)

        # Registration order (pipeline order), not alphabetical.
        self.assertEqual(self.registry.names(), ["echo", "alpha", "zulu"])
        self.assertLess(self.registry.manifest().index("1. echo"),
                        self.registry.manifest().index("3. zulu"))

    def test_manifest_never_leaks_a_function_repr(self) -> None:
        # A repr carries a memory address -> the system prompt changes every run
        # and prefix caching never hits.
        manifest = self.registry.manifest()
        self.assertNotIn("0x", manifest)
        self.assertNotIn("function echo", manifest)
        self.assertEqual(manifest, self.registry.manifest())   # deterministic

    def test_manifest_falls_back_to_the_docstring_first_line(self) -> None:
        self.registry.register("documented", echo)  # no explicit description
        self.assertIn("Echo one string back.", self.registry.manifest())

    def test_failed_call_returns_and_logs_a_structured_error(self) -> None:
        def boom(text: str) -> dict:
            """Always fails."""
            raise RuntimeError("kaboom")

        self.registry.register("boom", boom)
        result = self.registry.dispatch(
            "boom", {"text": "x"}, trace_ctx=self.ctx, trajectory=self.trajectory
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error["type"], "RuntimeError")

        event = json.loads(
            self.trajectory.trajectory_path.read_text(encoding="utf-8").splitlines()[-1]
        )
        self.assertEqual(event["action"], "tool.dispatch")   # not the tool name
        self.assertEqual(event["tool"], "boom")
        self.assertEqual(event["error"]["type"], "RuntimeError")
        self.assertIsInstance(event["latency_ms"], int)      # ms, not seconds

    def test_envelope_always_carries_ok_and_result(self) -> None:
        failed = self.registry.dispatch(
            "missing", {}, trace_ctx=self.ctx, trajectory=self.trajectory
        )
        envelope = failed.to_dict()
        self.assertIn("ok", envelope)
        self.assertIn("result", envelope)
        self.assertNotIn("result_summary", envelope)         # None fields omitted


if __name__ == "__main__":
    unittest.main()
