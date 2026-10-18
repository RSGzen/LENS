r"""Contract tests for the orchestrator (fake client/sandbox; no key, no Docker).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_runtime -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lens.observability.trajectory as trajectory_module
from lens.agent.budget import Budget
from lens.agent.runtime import Runtime
from lens.agent.sandbox_session import ExecResult
from lens.agent.tool_registry import ToolRegistry, args_hash
from lens.models import CallResult, JevResult
from lens.observability.trajectory import TraceCtx, Trajectory


def echo(text: str) -> dict:
    """Echo one string back."""
    return {"echo": text}


class FakeClient:
    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.chat_calls: list[dict[str, Any]] = []

    def chat(self, role: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> CallResult:
        self.chat_calls.append({"role": role, "messages": list(messages), **kwargs})
        if role == "sub" and kwargs.get("response_format") is not None:
            text = json.dumps(
                {
                    "objectives": ["answer the query"],
                    "problems": ["long-context aggregation"],
                    "constraints": [],
                    "requirements": [],
                }
            )
        else:
            text = self.replies.pop(0) if self.replies else "FINAL_ANSWER: fallback"
        return CallResult(
            text=text,
            message={"role": "assistant", "content": text},
            model="openai/gpt-5",
            provider="openai",
            model_snapshot=None,
            tokens_in=10,
            tokens_out=5,
            latency_ms=1,
            cost_usd=0.001,
        )

    def jev(self, state: str, questions: Any, **kwargs: Any) -> JevResult:
        return JevResult(
            answers={},
            model="typesafe/jev-1.13",
            provider="TypeSafe",
            tokens_in=1,
            cost_usd=0.0,
            latency_ms=1,
        )


class FakeSandbox:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.executed: list[str] = []

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def exec(self, code: str) -> ExecResult:
        self.executed.append(code)
        return ExecResult(ok=True, stdout="observation\n", duration_ms=1)


class RuntimeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_root = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = self._tmp.name
        self.trajectory = Trajectory("run-test")
        # The sandbox mounts <run_dir>/work; the artifacts stay in run_dir.
        self.sandbox = FakeSandbox(self.trajectory.run_dir / "work")
        self.registry = ToolRegistry()
        self.registry.register("echo", echo, description="Echo a string.")
        self.client = FakeClient(
            [
                "```python\nprint('step')\n```",
                "FINAL_ANSWER: The grounded answer.",
            ]
        )
        self.runtime = Runtime(
            "test query",
            "run-test",
            client=self.client,  # type: ignore[arg-type]
            trajectory=self.trajectory,
            conn=None,
            embedder=None,
            reranker=None,
            extractor=None,
            sandbox=self.sandbox,  # type: ignore[arg-type]
            registry=self.registry,
            budget=Budget(),
        )

    def tearDown(self) -> None:
        trajectory_module.LENS_RUNS_ROOT_PATH = self._orig_root
        self._tmp.cleanup()

    def test_run_returns_final_answer(self) -> None:
        result = self.runtime.run()
        self.assertFalse(result.partial)
        self.assertIn("The grounded answer", result.answer)
        self.assertTrue(result.answer_path.is_file())

    def test_workspace_never_exposes_run_artifacts(self) -> None:
        # The RLM can write anywhere in its workspace, so the workspace must be
        # <run_dir>/work — trajectory.jsonl / tasks.json / answer.md stay outside
        # it and stay trustworthy.
        result = self.runtime.run()
        self.assertEqual(self.runtime.workspace, self.trajectory.run_dir / "work")
        self.assertEqual(result.answer_path.parent, self.trajectory.run_dir)
        self.assertFalse((self.runtime.workspace / "trajectory.jsonl").exists())
        self.assertFalse((self.runtime.workspace / "answer.md").exists())

    def test_log_is_rebuildable_after_a_run(self) -> None:
        # M8.5: the root context must be a fold of the log, not a second copy.
        self.runtime.run()
        state = self.trajectory.rebuild()
        self.assertGreaterEqual(state.last_seq, 1)
        self.assertTrue(state.messages)
        self.assertEqual(state.messages[0]["role"], "system")
        contents = " ".join(str(m.get("content", "")) for m in state.messages)
        self.assertIn("step", contents)  # the observation of the executed block

    def test_manifest_records_terminal_status(self) -> None:
        result = self.runtime.run()
        from lens.agent import runstore

        manifest = runstore.read_manifest(self.trajectory.run_dir)
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertIn(manifest.status, {"completed", "partial"})
        self.assertEqual(manifest.run_id, "run-test")
        self.assertEqual(manifest.query, "test query")
        self.assertTrue(manifest.artifacts)
        self.assertAlmostEqual(manifest.totals["cost_usd"], result.cost_usd)

    def test_resume_folds_the_prior_turn(self) -> None:
        # A previous process logged one turn and died before the next chat.
        ctx = TraceCtx(trace_id="run-test", step_idx=0, actor="root")
        prompt_ref = self.trajectory.store_blob(
            json.dumps({"messages": [{"role": "system", "content": "sys"},
                                     {"role": "user", "content": "earlier question"}]})
        )
        self.trajectory.log_event(
            trace_ctx=ctx, call_type="chat", action="chat.completion",
            prompt_ref=prompt_ref,
            completion_ref=self.trajectory.store_blob("```python\nprint('earlier')\n```"),
        )
        self.trajectory.log_event(
            trace_ctx=ctx, call_type="exec", action="exec.block",
            code_ref=self.trajectory.store_blob("print('earlier')"),
            stdout_ref=self.trajectory.store_blob("earlier\n"),
        )

        runtime = Runtime(
            "test query",
            "run-test",
            client=self.client,
            trajectory=self.trajectory,
            conn=None,
            embedder=None,
            reranker=None,
            extractor=None,
            sandbox=self.sandbox,
            registry=self.registry,
            budget=Budget(),
            resume=True,
        )
        result = runtime.run()
        self.assertFalse(result.partial)
        # The earlier turn is in the prompt the model was sent, so nothing was lost.
        first_prompt = self.client.chat_calls[0]["messages"]
        joined = " ".join(str(m.get("content", "")) for m in first_prompt)
        self.assertIn("earlier question", joined)
        self.assertIn("earlier", joined)
        # History is appended to, never rewritten.
        actions = [json.loads(line).get("action") for line in
                   self.trajectory.trajectory_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertIn("run.resume", actions)
        self.assertEqual(actions[-1], "chat.completion")

    def test_replay_serves_the_recorded_tool_result(self) -> None:
        ref = self.trajectory.store_blob(json.dumps({"echo": "hi"}))
        self.trajectory.log_event(
            trace_ctx=TraceCtx(trace_id="run-test", step_idx=0, actor="tool"),
            call_type="tool", action="tool.dispatch", tool="echo",
            args_hash=args_hash({"text": "hi"}), result_ref=ref,
        )
        response = self.runtime.handle_rpc("/tool", {"name": "echo", "args": {"text": "hi"}})
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"], {"echo": "hi"})
        self.assertTrue(response.get("replayed"))

    def test_handle_rpc_budget(self) -> None:
        response = self.runtime.handle_rpc("/budget", {})
        self.assertTrue(response["ok"])
        self.assertIn("left", response["result"])

    def test_handle_rpc_tool(self) -> None:
        response = self.runtime.handle_rpc("/tool", {"name": "echo", "args": {"text": "hi"}})
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"], {"echo": "hi"})

    def test_handle_rpc_unknown_endpoint(self) -> None:
        response = self.runtime.handle_rpc("/nope", {})
        self.assertFalse(response["ok"])


if __name__ == "__main__":
    unittest.main()
