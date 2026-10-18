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


if __name__ == "__main__":
    unittest.main()
