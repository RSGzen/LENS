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
from lens.agent.tool_registry import ToolRegistry
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
