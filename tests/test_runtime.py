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
from lens import config
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
    """A transport double that also **logs**, like :class:`lens.clients.LensClient`.

    The real client owns the ``chat.completion`` event and the ``prompt_ref`` blob
    (that is what :meth:`Trajectory.rebuild` folds), so a fake that only returned a
    ``CallResult`` would make the fold untestable and would let the orchestrator
    double-log in production without anything noticing.
    """

    def __init__(self, replies: Sequence[str], trajectory: Trajectory | None = None) -> None:
        self.replies = list(replies)
        self.chat_calls: list[dict[str, Any]] = []
        self.trajectory = trajectory

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

        result = CallResult(
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

        if self.trajectory is not None:
            trace_ctx = kwargs.get("trace_ctx") or TraceCtx(
                trace_id="run-test", step_idx=0, actor="root" if role == "root" else "sub"
            )
            self.trajectory.log_event(
                trace_ctx=trace_ctx,
                call_type="chat",
                action="chat.completion",
                model=result.model,
                provider=result.provider,
                prompt_ref=self.trajectory.store_blob(json.dumps({"messages": list(messages)})),
                completion_ref=self.trajectory.store_blob(text),
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
            )

        return result

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
            ],
            self.trajectory,
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

    def test_a_turn_can_run_code_and_answer(self) -> None:
        # The smoke failure mode: gpt-5-mini ran a block and appended a real
        # FINAL_ANSWER after the fence on every turn; "code wins" discarded all of
        # them and the run ended at the iteration cap with an empty answer.md.
        self.runtime.client = FakeClient(
            [
                "```python\nprint('ran')\n```\nFINAL_ANSWER: The grounded answer.",
            ],
            self.trajectory,
        )

        result = self.runtime.run()

        self.assertFalse(result.partial)
        self.assertIn("The grounded answer", result.answer)
        # The block still executed: evidence first, then the answer.
        self.assertIn("print('ran')", self.sandbox.executed)

    def test_a_sentinel_printed_from_code_does_not_end_the_run(self) -> None:
        self.runtime.client = FakeClient(
            [
                "```python\nprint('FINAL_ANSWER: inside the block')\n```",
                "FINAL_ANSWER: The grounded answer.",
            ],
            self.trajectory,
        )

        result = self.runtime.run()

        self.assertFalse(result.partial)
        # The answer came from the *second* turn: the sentinel inside the block was
        # ignored, so the loop asked again instead of stopping on it.
        self.assertEqual(result.answer.strip(), "The grounded answer.")
        self.assertTrue(any("inside the block" in code for code in self.sandbox.executed))

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
        # : the root context must be a fold of the log, not a second copy.
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

    def test_manifest_records_the_arm(self) -> None:
        # /: a run whose k/n/KG/gate flags are unknown is unattributable
        # months later, so the arm travels with the manifest.
        self.runtime.run()
        from lens.agent import runstore

        manifest = runstore.read_manifest(self.trajectory.run_dir)
        assert manifest is not None
        arm = manifest.extra["arm"]
        self.assertEqual(arm["retrieval"]["k"], config.RETRIEVAL_K)
        self.assertEqual(arm["retrieval"]["c"], config.RETRIEVAL_C_TOP)
        self.assertEqual(arm["mode"]["recorded"], self.runtime.recorded)
        self.assertEqual(arm["models"]["root"], config.MODEL_REGISTRY["root"].id)
        # The tag is not reproducible; the digest slot exists even when unknown.
        self.assertIn("image_digest", manifest.created_by)

    def test_resume_keeps_the_original_arm(self) -> None:
        # A resumed run must not be re-fingerprinted with today's defaults: the
        # arm is written once at creation because ``extra`` is not updatable.
        self.runtime.run()
        from lens.agent import runstore

        manifest = runstore.read_manifest(self.trajectory.run_dir)
        assert manifest is not None
        manifest.extra["arm"]["sentinel"] = "original"
        runstore.write_manifest(self.trajectory.run_dir, manifest)

        resumed = Runtime(
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
        resumed.run()

        after = runstore.read_manifest(self.trajectory.run_dir)
        assert after is not None
        self.assertEqual(after.extra["arm"]["sentinel"], "original")

    def test_resume_folds_the_prior_turn(self) -> None:
        # A previous process logged one turn and died before the next chat.
        ctx = TraceCtx(trace_id="run-test", step_idx=0, actor="root")
        prompt_ref = self.trajectory.store_blob(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "earlier question"},
                    ]
                }
            )
        )
        self.trajectory.log_event(
            trace_ctx=ctx,
            call_type="chat",
            action="chat.completion",
            prompt_ref=prompt_ref,
            completion_ref=self.trajectory.store_blob("```python\nprint('earlier')\n```"),
        )
        self.trajectory.log_event(
            trace_ctx=ctx,
            call_type="exec",
            action="exec.block",
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
        actions = [
            json.loads(line).get("action")
            for line in self.trajectory.trajectory_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertIn("run.resume", actions)
        # Budget snapshots are bookkeeping and may follow the last chat; what must
        # not change is that the resume note comes first and new turns follow it.
        self.assertTrue(
            all(a not in {"run.resume"} for a in actions[actions.index("run.resume") + 1 :])
        )
        self.assertIn("chat.completion", actions[actions.index("run.resume") :])

    def test_replay_serves_the_recorded_tool_result(self) -> None:
        ref = self.trajectory.store_blob(json.dumps({"echo": "hi"}))
        self.trajectory.log_event(
            trace_ctx=TraceCtx(trace_id="run-test", step_idx=0, actor="tool"),
            call_type="tool",
            action="tool.dispatch",
            tool="echo",
            args_hash=args_hash({"text": "hi"}),
            result_ref=ref,
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


class RuntimeRegressionTestCase(unittest.TestCase):
    """Behaviour the nine contracts do not pin."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_root = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = self._tmp.name
        self.trajectory = Trajectory("run-reg")
        self.sandbox = FakeSandbox(self.trajectory.run_dir / "work")
        self.registry = ToolRegistry()
        self.registry.register("echo", echo, description="Echo a string.")
        self.client = FakeClient(["FINAL_ANSWER: done"], self.trajectory)
        self.runtime = Runtime(
            "q",
            "run-reg",
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

    def test_manifest_records_model_ids_and_no_fake_commit(self) -> None:
        self.runtime.run()
        from lens.agent import runstore

        manifest = runstore.read_manifest(self.trajectory.run_dir)
        assert manifest is not None
        # ModelSpec is a dataclass and would break json.dump; ids only.
        self.assertEqual(manifest.models["root"], "openai/gpt-5")
        self.assertEqual(manifest.models["jev"], "typesafe/jev-1.13")
        self.assertIsNone(manifest.created_by["commit"])  # unknown, never 0
        self.assertTrue(manifest.caps)

    def test_bad_rpc_payload_is_a_structured_error(self) -> None:
        response = self.runtime.handle_rpc("/tool", {})
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["type"], "BadRequest")

    def test_replay_off_calls_the_tool(self) -> None:
        runtime = Runtime(
            "q",
            "run-reg",
            client=self.client,
            trajectory=self.trajectory,
            conn=None,
            embedder=None,
            reranker=None,
            extractor=None,
            sandbox=self.sandbox,
            registry=self.registry,
            budget=Budget(),
            replay=False,
        )
        response = runtime.handle_rpc("/tool", {"name": "echo", "args": {"text": "hi"}})
        self.assertTrue(response["ok"])
        self.assertNotIn("replayed", response)

    def test_emit_writes_the_file_and_names_the_variable(self) -> None:
        response = self.runtime.handle_rpc("/emit", {"value": {"a": 1}, "label": "chunks"})
        self.assertTrue(response["ok"])
        emitted = response["result"]
        self.assertTrue(emitted["path"].startswith("/workspace/emits/"))
        on_disk = self.runtime.workspace / "emits" / "chunks.json"
        self.assertEqual(json.loads(on_disk.read_text(encoding="utf-8")), {"a": 1})
        # The label is the variable name: resume depends on this map.
        self.assertEqual(self.runtime._emits["chunks"], emitted["ref"])

    def test_checkpoint_writes_the_resume_cache(self) -> None:
        from lens.agent import runstore

        self.runtime.run()
        cache = runstore.read_resume_cache(self.trajectory.run_dir)
        self.assertIsNotNone(cache)
        assert cache is not None
        self.assertIn("namespace", cache)
        self.assertIn("query.json", cache["workspace"])

    def test_finalize_marks_a_crashed_run_failed(self) -> None:
        from lens.agent import runstore

        self.runtime._open_manifest()
        self.runtime._finalize("half an answer", partial=True, stop_reason="error")
        manifest = runstore.read_manifest(self.trajectory.run_dir)
        assert manifest is not None
        self.assertEqual(manifest.status, "failed")
        self.assertTrue((self.trajectory.run_dir / "answer.md").is_file())

    def test_budget_warning_is_appended_last_not_in_the_system_prompt(self) -> None:
        from lens.agent.context import budget_warning

        self.runtime.budget.cost_usd = self.runtime.budget.limits.max_cost_usd * 0.99
        warning = budget_warning(self.runtime.budget.snapshot())
        self.assertTrue(warning)

        messages = self.runtime._build_context()
        self.assertEqual(messages[0]["role"], "system")
        # The static prefix must not change, or prefix caching stops hitting.
        self.assertNotIn(warning, messages[0]["content"])
        self.assertEqual(messages[-1]["content"], warning)


if __name__ == "__main__":
    unittest.main()
