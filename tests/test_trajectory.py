r"""
Contract tests for the observability sink (no API key, no network, zero cost).

These pin the contract of ``trajectory.py`` (content-addressed blobs + one JSONL
schema) through an injected sink; no torch, no DB, no network.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import lens.observability.trajectory as trajectory_module
from lens.observability.trajectory import TraceCtx, Trajectory


class TrajectoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # The sink reads its root from config, not a constructor arg. Point it at a
        # temp dir so tests never touch the real <runs_root>.
        self._tmp = tempfile.TemporaryDirectory()
        self.runs_root = Path(self._tmp.name)
        self._orig_runs_root = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = str(self.runs_root)
        self.traj = Trajectory("run-test")

    def tearDown(self) -> None:
        trajectory_module.LENS_RUNS_ROOT_PATH = self._orig_runs_root
        self._tmp.cleanup()

    def _events(self) -> list[dict]:
        raw = self.traj.trajectory_path.read_text(encoding="utf-8")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    # ---------------------------------------------------------------- layout
    def test_layout_is_created_eagerly(self) -> None:
        self.assertTrue(self.traj.run_dir.is_dir())
        self.assertTrue(self.traj.blobs_dir.is_dir())
        self.assertEqual(self.traj.trajectory_path.name, "trajectory.jsonl")

    # ----------------------------------------------------------------- blobs
    def test_store_blob_is_content_addressed(self) -> None:
        ref = self.traj.store_blob("hello world")
        self.assertTrue(ref.startswith("sha256:"), ref)
        hexdigest = ref.split(":", 1)[1]
        blob = self.traj.blobs_dir / hexdigest
        self.assertTrue(blob.is_file())
        self.assertEqual(blob.read_text(encoding="utf-8"), "hello world")

    def test_store_blob_dedupes_identical_text(self) -> None:
        ref_a = self.traj.store_blob("same payload")
        ref_b = self.traj.store_blob("same payload")
        self.assertEqual(ref_a, ref_b)
        self.assertEqual(len(list(self.traj.blobs_dir.iterdir())), 1)

    def test_store_blob_distinct_text_distinct_ref(self) -> None:
        self.assertNotEqual(self.traj.store_blob("a"), self.traj.store_blob("b"))

    # ---------------------------------------------------------------- events
    def test_log_event_appends_one_line_per_call(self) -> None:
        ctx = TraceCtx(trace_id="t1", step_idx=0, actor="root")
        self.traj.log_event(
            trace_ctx=ctx, call_type="chat", action="chat.completion",
            model="openai/gpt-5",
        )
        self.traj.log_event(trace_ctx=ctx, call_type="jev", action="jev.decisions")
        self.assertEqual(len(self._events()), 2)

    def test_log_event_includes_identity_and_timestamp(self) -> None:
        ctx = TraceCtx(trace_id="t1", step_idx=3, actor="sub")
        self.traj.log_event(trace_ctx=ctx, call_type="chat", action="chat.completion")
        event = self._events()[0]
        self.assertEqual(event["trace_id"], "t1")
        self.assertEqual(event["step_idx"], 3)
        self.assertEqual(event["actor"], "sub")
        self.assertEqual(event["call_type"], "chat")
        self.assertEqual(event["action"], "chat.completion")
        datetime.fromisoformat(event["timestamp"])  # must parse

    def test_log_event_omits_none_fields(self) -> None:
        ctx = TraceCtx(trace_id="t1", step_idx=0, actor="root")
        self.traj.log_event(
            trace_ctx=ctx, call_type="chat", action="chat.completion",
            cost_usd=0.0004,
        )
        event = self._events()[0]
        self.assertEqual(event["cost_usd"], 0.0004)
        self.assertNotIn("provider", event)
        self.assertNotIn("error", event)

    def test_log_event_keeps_structured_error(self) -> None:
        ctx = TraceCtx(trace_id="t1", step_idx=1, actor="root")
        err = {"type": "RateLimit", "status": 429, "retry": 2}
        self.traj.log_event(
            trace_ctx=ctx, call_type="chat", action="chat.retry", error=err,
        )
        self.assertEqual(self._events()[0]["error"], err)

    # --------------------------------------------------------------- tracectx
    def test_trace_ctx_is_frozen(self) -> None:
        ctx = TraceCtx(trace_id="t1", step_idx=0, actor="root")
        with self.assertRaises(FrozenInstanceError):
            ctx.step_idx = 1  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
