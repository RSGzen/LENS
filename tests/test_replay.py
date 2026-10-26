r"""Contract tests for turn-level replay (no key, no DB, no Docker).

``replay_turns`` is the only data source the M10b viewer needs. It must be a
faithful, read-only fold of the log: one turn per ``step_idx``, blobs resolved,
and **tolerant** of a damaged bundle (a demo that crashes on one missing blob is
worse than one that says "missing").

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_replay -v
"""

from __future__ import annotations

import json
import tempfile
import unittest

import lens.observability.trajectory as trajectory_module
from lens.observability.replay import replay_summary, replay_turns
from lens.observability.trajectory import TraceCtx, Trajectory


class ReplayTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = self._tmp.name
        self.traj = Trajectory("run-replay")

    def tearDown(self) -> None:
        trajectory_module.LENS_RUNS_ROOT_PATH = self._orig
        self._tmp.cleanup()

    # ---------------------------------------------------------------- helpers
    def _chat(
        self, step: int, completion: str, *, tokens_in: int = 100, cost: float = 0.01
    ) -> None:
        self.traj.log_event(
            trace_ctx=TraceCtx("run-replay", step, "root"),
            call_type="chat",
            action="chat.completion",
            model="openai/gpt-5",
            provider="openai",
            prompt_ref=self.traj.store_blob(json.dumps({"messages": []})),
            completion_ref=self.traj.store_blob(completion),
            tokens_in=tokens_in,
            tokens_out=20,
            cost_usd=cost,
            latency_ms=900,
        )

    def _exec(
        self,
        step: int,
        code: str,
        stdout: str | None,
        *,
        summary: str | None = None,
        error: dict | None = None,
    ) -> None:
        self.traj.log_event(
            trace_ctx=TraceCtx("run-replay", step, "tool"),
            call_type="exec",
            action="exec.block",
            code_ref=self.traj.store_blob(code),
            stdout_ref=self.traj.store_blob(stdout) if stdout is not None else None,
            result_summary=summary,
            latency_ms=42,
            error=error,
        )

    def _tool(
        self, step: int, name: str, args: dict, result: dict, *, gate: dict | None = None
    ) -> None:
        self.traj.log_event(
            trace_ctx=TraceCtx("run-replay", step, "tool"),
            call_type="tool",
            action="tool.dispatch",
            tool=name,
            args_hash="hash-" + name,
            args_ref=self.traj.store_blob(json.dumps(args)),
            result_ref=self.traj.store_blob(json.dumps(result)),
            result_summary=f"{name}: 3 rows",
            latency_ms=120,
            extra={"gate": gate} if gate else None,
        )

    # ------------------------------------------------------------------ shape
    def test_empty_log_has_no_turns(self) -> None:
        self.assertEqual(replay_turns(self.traj), [])

    def test_turns_are_grouped_by_step_idx(self) -> None:
        self._chat(0, "```python\nprint(1)\n```")
        self._exec(0, "print(1)", "1\n")
        self._chat(1, "FINAL_ANSWER: done", tokens_in=150, cost=0.02)
        self._exec(1, "print(2)", "2\n")

        turns = replay_turns(self.traj)
        self.assertEqual([t.turn for t in turns], [0, 1])
        self.assertEqual(turns[0].model, "openai/gpt-5")
        self.assertEqual(turns[0].tokens_in, 100)
        self.assertAlmostEqual(turns[1].cost_usd, 0.02)
        self.assertIn("print(1)", turns[0].code or "")
        self.assertEqual(turns[0].stdout_full, "1\n")

    def test_tool_steps_carry_args_latency_and_gate(self) -> None:
        gate = {"verdict": "pass", "confidence": 0.82, "tier": "jev"}
        self._chat(0, "```python\nx = tool('shallow_match', buckets=b)\n```")
        self._tool(
            0,
            "shallow_match",
            {"buckets": {"problems": ["p"]}, "k": 100},
            {"ids": ["a", "b"]},
            gate=gate,
        )
        self._exec(0, "x", "2 ids\n")

        turn = replay_turns(self.traj)[0]
        self.assertEqual(len(turn.tools), 1)
        step = turn.tools[0]
        self.assertEqual(step.name, "shallow_match")
        self.assertEqual(step.args["k"], 100)  # args come from args_ref
        self.assertEqual(step.args_hash, "hash-shallow_match")
        self.assertEqual(step.latency_ms, 120)
        self.assertEqual(step.gate, gate)
        self.assertIsNotNone(step.result_ref)

    def test_observation_is_what_the_model_saw(self) -> None:
        self._chat(0, "```python\nprint(x)\n```")
        self._exec(0, "print(x)", "x" * 10_000, summary="trimmed summary")
        self._exec(0, "print(y)", "y" * 10_000)
        t = replay_turns(self.traj)[0]
        # Explicit summary wins; otherwise the stdout is truncated, never the full capture.
        self.assertEqual(t.observation, "trimmed summary")
        self.assertIsNotNone(t.stdout_full)
        self.assertLessEqual(len(t.observation or ""), 3000)

    def test_emits_and_budget_attach_to_the_turn(self) -> None:
        self._chat(0, "```python\nemit(chunks, label='chunks')\n```")
        self.traj.log_event(
            trace_ctx=TraceCtx("run-replay", 0, "tool"),
            call_type="emit",
            action="emit.store",
            result_ref=self.traj.store_blob('["c1"]'),
            extra={"var": "chunks"},
        )
        snapshot = {"used": {"tokens_in": 100}, "left": {"iterations": 7}}
        self.traj.log_event(
            trace_ctx=TraceCtx("run-replay", 0, "root"),
            call_type="note",
            action="budget",
            extra={"budget": snapshot},
        )
        turn = replay_turns(self.traj)[0]
        self.assertEqual(turn.emits, ["chunks"])
        self.assertEqual(turn.budget, snapshot)

    # ------------------------------------------------------------- tolerance
    def test_missing_blob_is_reported_not_raised(self) -> None:
        self._chat(0, "```python\nprint(1)\n```")
        self.traj.log_event(
            trace_ctx=TraceCtx("run-replay", 0, "tool"),
            call_type="exec",
            action="exec.block",
            code_ref="sha256:" + "0" * 64,  # never stored
            stdout_ref=self.traj.store_blob("1\n"),
        )
        turn = replay_turns(self.traj)[0]
        self.assertIsNone(turn.code)
        self.assertTrue(turn.missing_refs)
        self.assertEqual(turn.stdout_full, "1\n")

    def test_interrupted_last_turn_is_marked(self) -> None:
        self._chat(0, "```python\nprint(1)\n```")
        self._exec(0, "print(1)", None)  # crash mid-block: no stdout blob
        turn = replay_turns(self.traj)[0]
        self.assertIsNone(turn.ok)
        self.assertIsNone(turn.stdout_full)
        self.assertIsNotNone(turn.code)

    # --------------------------------------------------------------- summary
    def test_summary_cost_includes_gate_calls_but_tokens_stay_chat_only(self) -> None:
        # The M10b cost meter compares total spend against the cap, while the
        # context-growth chart must show chat tokens only. One number for both was
        # short by the whole decision layer.
        self._chat(0, "```python\nprint(1)\n```", tokens_in=1000, cost=0.01)
        self.traj.log_event(
            trace_ctx=TraceCtx("run-replay", 0, "tool"),
            call_type="jev",
            action="jev.gate",
            tokens_in=500,
            cost_usd=0.001,
        )

        summary = replay_summary(replay_turns(self.traj))

        self.assertAlmostEqual(summary["cost_usd"], 0.011)
        self.assertAlmostEqual(summary["chat_cost_usd"], 0.01)
        self.assertEqual(summary["tokens_in"], 1000)  # chat only: the chart
        self.assertEqual(summary["jev_tokens_in"], 500)  # reported separately

    def test_summary_matches_trajectory_totals(self) -> None:
        self._chat(0, "```python\nprint(1)\n```")
        self._tool(0, "echo", {"text": "hi"}, {"echo": "hi"})
        self._exec(0, "print(1)", "1\n")
        self._chat(1, "FINAL_ANSWER: done", tokens_in=150, cost=0.02)

        turns = replay_turns(self.traj)
        summary = replay_summary(turns)
        totals = self.traj.totals()
        self.assertEqual(summary["turns"], 2)
        self.assertEqual(summary["tokens_in"], totals["tokens_in"])
        self.assertAlmostEqual(summary["cost_usd"], totals["cost_usd"])
        self.assertEqual(summary["tool_calls"], 1)


class ReplayRegressionTestCase(unittest.TestCase):
    """Edge cases the nine contracts cannot see."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = self._tmp.name
        self.traj = Trajectory("run-edge")
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(setattr, trajectory_module, "LENS_RUNS_ROOT_PATH", self._orig)

    def _chat(self, step: int, text: str, *, actor: str = "root", tokens: int = 100) -> None:
        self.traj.log_event(
            trace_ctx=TraceCtx("run-edge", step, actor),  # type: ignore[arg-type]
            call_type="chat",
            action="chat.completion",
            model="openai/gpt-5" if actor == "root" else "openai/gpt-5-mini",
            completion_ref=self.traj.store_blob(text),
            tokens_in=tokens,
            tokens_out=1,
            cost_usd=0.001,
        )

    def test_corrupt_line_is_skipped_not_fatal(self) -> None:
        self._chat(0, "print(1)")
        with open(self.traj.trajectory_path, "a", encoding="utf-8") as handle:
            handle.write("{not json\n")

        turns = replay_turns(self.traj)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].completion, "print(1)")

    def test_step_index_order_is_ascending_not_file_order(self) -> None:
        self._chat(2, "third")
        self._chat(0, "first")

        self.assertEqual([t.turn for t in replay_turns(self.traj)], [0, 2])

    def test_sub_call_does_not_clobber_the_root_completion(self) -> None:
        self._chat(0, "root completion", tokens=100)
        self._chat(0, "sub completion", actor="sub", tokens=7)

        turn = replay_turns(self.traj)[0]
        self.assertEqual(turn.completion, "root completion")
        self.assertEqual(turn.tokens_in, 107)  # accumulated, not overwritten
        self.assertEqual(turn.model, "openai/gpt-5")  # the root role wins

    def test_invalid_json_blob_is_reported_as_missing(self) -> None:
        bad_ref = self.traj.store_blob("{not json")
        self.traj.log_event(
            trace_ctx=TraceCtx("run-edge", 0, "tool"),
            call_type="tool",
            action="tool.dispatch",
            tool="echo",
            args_hash="h",
            args_ref=bad_ref,
        )

        turn = replay_turns(self.traj)[0]
        self.assertIsNone(turn.tools[0].args)
        self.assertIn(bad_ref, turn.missing_refs)

    def test_summary_counts_gate_verdicts(self) -> None:
        self._chat(0, "code")
        for step, verdict in ((0, "pass"), (1, "fail")):
            self.traj.log_event(
                trace_ctx=TraceCtx("run-edge", step, "tool"),
                call_type="tool",
                action="tool.dispatch",
                tool="gated",
                result_ref=self.traj.store_blob("{}"),
                extra={"gate": {"verdict": verdict}},
            )

        summary = replay_summary(replay_turns(self.traj))
        self.assertEqual(summary["gates_passed"], 1)
        self.assertEqual(summary["gates_failed"], 1)
        self.assertEqual(summary["tool_calls"], 2)


if __name__ == "__main__":
    unittest.main()
