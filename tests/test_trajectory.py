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

    # ---------------------------------------------------------------- totals
    def test_totals_sums_cost_and_tokens(self) -> None:
        ctx = TraceCtx(trace_id="t1", step_idx=0, actor="root")
        self.traj.log_event(
            trace_ctx=ctx, call_type="chat", action="chat.completion",
            tokens_in=100, tokens_out=10, cost_usd=0.001,
        )
        self.traj.log_event(
            trace_ctx=ctx, call_type="jev", action="jev.decisions",
            tokens_in=50, cost_usd=0.0002,
        )
        self.traj.log_event(
            trace_ctx=ctx, call_type="tool", action="tool.dispatch", tool="build_kg",
        )
        totals = self.traj.totals()
        self.assertEqual(totals["calls"], 3)
        self.assertEqual(totals["tokens_in"], 150)
        self.assertEqual(totals["tokens_out"], 10)
        self.assertAlmostEqual(totals["cost_usd"], 0.0012)
        # Tool/admin events carry no model spend and must not be counted as LLM calls.
        self.assertEqual(totals["by_action"]["tool.dispatch"]["calls"], 1)
        self.assertAlmostEqual(totals["by_action"]["tool.dispatch"]["cost_usd"], 0.0)

    def test_totals_on_empty_log_is_zeroed(self) -> None:
        totals = self.traj.totals()
        self.assertEqual(totals["cost_usd"], 0.0)
        self.assertEqual(totals["calls"], 0)

    def test_totals_does_not_rewrite_the_log(self) -> None:
        ctx = TraceCtx(trace_id="t1", step_idx=0, actor="root")
        self.traj.log_event(trace_ctx=ctx, call_type="chat", action="chat.completion")
        before = self.traj.trajectory_path.read_bytes()
        self.traj.totals()
        self.assertEqual(self.traj.trajectory_path.read_bytes(), before)

    # --------------------------------------------------------------- rebuild
    def _log_turn(self, step: int, messages: list[dict], code: str, stdout: str) -> None:
        """Append one realistic turn: root chat + exec, the way the loop does it."""
        root = TraceCtx(trace_id="t1", step_idx=step, actor="root")
        prompt_ref = self.traj.store_blob(json.dumps({"model": "openai/gpt-5", "messages": messages}))
        completion = f"```python\n{code}\n```"
        completion_ref = self.traj.store_blob(completion)
        self.traj.log_event(
            trace_ctx=root, call_type="chat", action="chat.completion",
            prompt_ref=prompt_ref, completion_ref=completion_ref,
            tokens_in=100, tokens_out=10, cost_usd=0.001,
        )
        self.traj.log_event(
            trace_ctx=TraceCtx(trace_id="t1", step_idx=step, actor="tool"),
            call_type="exec", action="exec.block",
            code_ref=self.traj.store_blob(code),
            stdout_ref=self.traj.store_blob(stdout),
            latency_ms=7,
        )

    def test_rebuild_on_empty_log_is_zeroed(self) -> None:
        state = self.traj.rebuild()
        self.assertEqual(state.messages, [])
        self.assertEqual(state.budget, {})
        self.assertEqual(state.emits, {})
        self.assertEqual(state.last_seq, 0)

    def test_rebuild_folds_messages_and_next_seq(self) -> None:
        base = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "the query"},
        ]
        self._log_turn(0, base, "print(1)", "1\n")
        second = base + [
            {"role": "assistant", "content": "```python\nprint(1)\n```"},
            {"role": "user", "content": "1\n"},
        ]
        self._log_turn(1, second, "print(2)", "2\n")

        state = self.traj.rebuild()
        self.assertEqual(state.last_seq, 2)
        # The last chat carried the context up to that point; the trailing exec
        # contributes the assistant block + its observation.
        self.assertEqual(state.messages[:2], base)
        self.assertEqual(state.messages[2]["role"], "assistant")
        self.assertIn("print(2)", state.messages[2]["content"])
        self.assertEqual(state.messages[3], {"role": "user", "content": "2\n"})

    def test_rebuild_collects_budget_emits_and_tool_results(self) -> None:
        ctx = TraceCtx(trace_id="t1", step_idx=0, actor="root")
        snapshot = {"used": {"tokens_in": 100, "cost_usd": 0.001}, "left": {"iterations": 7}}
        self.traj.log_event(
            trace_ctx=ctx, call_type="note", action="budget", extra={"budget": snapshot}
        )
        self.traj.log_event(
            trace_ctx=ctx, call_type="emit", action="emit.store",
            result_ref="sha256:abc", extra={"var": "chunks"},
        )
        self.traj.log_event(
            trace_ctx=ctx, call_type="tool", action="tool.dispatch",
            tool="shallow_match", args_hash="hash1", result_ref="sha256:def",
        )
        state = self.traj.rebuild()
        self.assertEqual(state.budget, snapshot)
        self.assertEqual(state.emits, {"chunks": "sha256:abc"})
        self.assertEqual(state.tool_results, {"shallow_match:hash1": "sha256:def"})

    def test_rebuild_marks_an_interrupted_exec(self) -> None:
        # Crash mid-block: the chat completed, the exec did not record stdout.
        ctx = TraceCtx(trace_id="t1", step_idx=0, actor="root")
        prompt_ref = self.traj.store_blob(json.dumps({"messages": [{"role": "system", "content": "s"}]}))
        self.traj.log_event(
            trace_ctx=ctx, call_type="chat", action="chat.completion",
            prompt_ref=prompt_ref, completion_ref=self.traj.store_blob("```python\nx=1\n```"),
        )
        self.traj.log_event(trace_ctx=ctx, call_type="exec", action="exec.block")
        state = self.traj.rebuild()
        self.assertEqual(state.messages[-1]["role"], "user")
        self.assertIn("interrupt", state.messages[-1]["content"].lower())

    def test_rebuild_is_deterministic(self) -> None:
        self._log_turn(0, [{"role": "user", "content": "q"}], "print(1)", "1\n")
        self.assertEqual(self.traj.rebuild(), self.traj.rebuild())

    def test_load_blob_round_trips(self) -> None:
        ref = self.traj.store_blob("payload")
        self.assertEqual(self.traj.load_blob(ref), "payload")
        with self.assertRaises(FileNotFoundError):
            self.traj.load_blob("sha256:" + "0" * 64)


if __name__ == "__main__":
    unittest.main()
