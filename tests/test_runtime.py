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
from unittest.mock import patch

import lens.observability.trajectory as trajectory_module
from lens import config
from lens.agent.budget import Budget, BudgetLimits
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


class SubTaskClient:
    """Return a canned sub-task reply and keep the request for assertions."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def chat(self, role: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> CallResult:
        self.calls.append({"role": role, "messages": list(messages), **kwargs})
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)

        return CallResult(
            text=text,
            message={"role": "assistant", "content": text},
            model="openai/gpt-5-mini",
            provider="openai",
            model_snapshot=None,
            tokens_in=10,
            tokens_out=5,
            latency_ms=1,
            cost_usd=0.001,
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
        # Two failure modes meet here. First: a real FINAL_ANSWER after a fence
        # used to be discarded on every turn, so the run ended with an empty
        # answer. Second: accepting it immediately let the model answer a blind
        # question — it had not seen the block's output, and reported the names of
        # its own variables. So the block runs, the answer is deferred one turn,
        # and the second turn's answer is the one kept.
        self.runtime.client = FakeClient(
            [
                "```python\nprint('ran')\n```\nFINAL_ANSWER: The grounded answer.",
                "FINAL_ANSWER: The grounded answer.",
            ],
            self.trajectory,
        )

        result = self.runtime.run()

        self.assertFalse(result.partial)
        self.assertIn("The grounded answer", result.answer)
        self.assertEqual(result.answer, "The grounded answer.")
        # The block still executed: evidence first, then the answer.
        self.assertIn("print('ran')", self.sandbox.executed)

    def test_the_final_gate_sees_abbreviated_ids_as_full_uuids(self) -> None:
        # Measured: an answer citing 8-character prefixes reached the gate with no
        # chunk ids and no evidence, and was rejected four times.
        full = "bc4e8dcb-1111-5111-8111-111111111111"

        def retrieval_funnel(buckets: object) -> dict:
            """The funnel."""
            return {"chunks": [{"chunk_uuid": full, "chunk_text": "t"}], "papers": {}}

        self.registry.register("retrieval_funnel", retrieval_funnel)
        self.runtime.handle_rpc("/tool", {"name": "retrieval_funnel", "args": {"buckets": {}}})

        states: list[str] = []
        client = FakeClient(
            ["FINAL_ANSWER: ReCAP re-injects the parent plan [chunk_uuid=bc4e8dcb]."],
            self.trajectory,
        )
        client.jev = lambda state, questions, **kw: (  # type: ignore[method-assign]
            states.append(state) or FakeClient.jev(client, state, questions)
        )
        self.runtime.client = client

        with patch.object(config, "GATE_ENABLED", True):
            result = self.runtime.run()

        summary = json.loads(states[-1])["summary"]
        self.assertEqual(summary["citations"], [full])
        self.assertIn(full, summary["answer"])
        self.assertIn("[chunk_uuid=bc4e8dcb]", result.answer)  # answer.md keeps the model's text

    def test_a_stage_gate_verdict_reaches_the_observation(self) -> None:
        # The façade returns only ``result``, so the envelope's gate never reached
        # the model: 24 of 24 recorded retrieval-gate verdicts were invisible to it.
        with (
            patch.object(config, "GATE_ENABLED", True),
            patch("lens.agent.runtime.GATED_STAGES", {"echo": "retrieval"}),
        ):
            self.runtime.handle_rpc("/tool", {"name": "echo", "args": {"text": "hi"}})
            observation = self.runtime._execute_block("print('x')")
            later = self.runtime._execute_block("print('y')")

        self.assertTrue(observation.startswith("observation"))
        self.assertIn("[gate] retrieval stage: uncertain", observation.splitlines()[-1])
        self.assertNotIn("[gate]", later)  # shown once, on the turn that produced it

    def test_a_failed_block_stores_its_traceback(self) -> None:
        # stderr_ref was declared in the schema and never written, so nothing but
        # the next turn's text could say why a block failed: a replay showed the
        # observation and a bundle had no traceback at all.
        def failing(code: str) -> ExecResult:
            return ExecResult(
                ok=False,
                stdout="partial\n",
                stderr="Traceback (most recent call last):\nValueError: bad\n",
                error={"type": "ValueError", "message": "bad"},
                duration_ms=2,
            )

        self.sandbox.exec = failing  # type: ignore[method-assign]

        observation = self.runtime._execute_block("raise ValueError('bad')")

        # The model still gets the diagnosis in the observation ...
        self.assertIn("ValueError: bad", observation)

        with open(self.trajectory.run_dir / "trajectory.jsonl", encoding="utf-8") as handle:
            event = json.loads(handle.read().splitlines()[-1])

        self.assertEqual(event["error"]["type"], "ValueError")
        self.assertIn("ValueError: bad", self.trajectory.load_blob(event["stderr_ref"]))

    def test_an_answer_written_before_its_own_block_is_not_final(self) -> None:
        self.runtime.client = FakeClient(
            [
                "```python\nprint(4)\n```\nFINAL_ANSWER: method = count_method records",
                "FINAL_ANSWER: method = 4 records, dataset = 2 records",
            ],
            self.trajectory,
        )

        result = self.runtime.run()

        self.assertEqual(result.answer, "method = 4 records, dataset = 2 records")

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

    def test_a_rejected_answer_is_not_rewritten_forever(self) -> None:
        # Measured on a recorded run: five turns of rewriting an unchanged answer,
        # 48 % of the run's cost, and the rejection was the aggregate's alone every
        # time. Past the cap the answer stands and the verdicts stay in the log.
        from lens.agent.runtime import MAX_FINAL_GATE_REJECTIONS

        claim = "11111111-1111-5111-8111-111111111111"
        client = FakeClient(
            [
                f"FINAL_ANSWER: The window is K rounds and the cost stays bounded [{claim}] "
                f"(draft {index})."
                for index in range(1, 7)
            ],
            self.trajectory,
        )
        client.jev = lambda state, questions, **kw: JevResult(  # type: ignore[method-assign]
            answers={"gate": {"noul": 0.9}, "claim-01": {"noul": 0.9}},
            model="typesafe/jev-1.13",
            provider="TypeSafe",
            tokens_in=1,
            cost_usd=0.0,
            latency_ms=1,
        )
        self.runtime.client = client

        with patch.object(config, "GATE_ENABLED", True):
            result = self.runtime.run()

        root_calls = [call for call in client.chat_calls if call["role"] == "root"]
        self.assertEqual(len(root_calls), MAX_FINAL_GATE_REJECTIONS + 1)
        self.assertIn(f"(draft {MAX_FINAL_GATE_REJECTIONS + 1})", result.answer)
        # The gate's disagreement does not become the run's stop reason: it is not
        # an acceptance criterion, and the status vocabulary stays
        # iterations/tokens/cost/backtracks.
        self.assertFalse(result.partial)

        events = [
            json.loads(line)
            for line in self.trajectory.trajectory_path.read_text(encoding="utf-8").splitlines()
        ]
        notes = [
            (event.get("extra") or {}).get("text", "")
            for event in events
            if event.get("call_type") == "note"
        ]

        self.assertTrue(any("accepted with the verdicts recorded" in text for text in notes))

    def test_an_aggregate_only_rejection_does_not_buy_a_turn(self) -> None:
        # Measured: the one rejection in m11-check-v5 scored P=0.80 with every claim
        # below 0.41, and the rewrite it bought deleted a cited method family. With
        # nothing named there is nothing to fix, so the answer stands.
        client = FakeClient(["FINAL_ANSWER: draft one", "FINAL_ANSWER: draft two"])
        client.jev = lambda state, questions, **kw: JevResult(  # type: ignore[method-assign]
            answers={"gate": {"noul": 0.8}},
            model="typesafe/jev-1.13",
            provider="TypeSafe",
            tokens_in=1,
            cost_usd=0.0,
            latency_ms=1,
        )
        self.runtime.client = client

        with patch.object(config, "GATE_ENABLED", True):
            result = self.runtime.run()

        root_calls = [call for call in client.chat_calls if call["role"] == "root"]
        self.assertEqual(len(root_calls), 1)
        self.assertEqual(result.answer, "draft one")
        self.assertFalse(result.partial)

        events = [
            json.loads(line)
            for line in self.trajectory.trajectory_path.read_text(encoding="utf-8").splitlines()
        ]
        notes = [
            (event.get("extra") or {}).get("text", "")
            for event in events
            if event.get("call_type") == "note"
        ]

        self.assertTrue(any("on the aggregate alone" in text for text in notes))

    def test_the_rejection_names_the_threshold_that_fired(self) -> None:
        # The root used to read "claim-10 (P=0.50)" with tau_claim nowhere in the
        # message, so a claim the rule never rejected read as a failure.
        claim = "11111111-1111-5111-8111-111111111111"
        client = FakeClient(
            [
                f"FINAL_ANSWER: The window is K rounds and the cost stays bounded [{claim}].",
                f"FINAL_ANSWER: The window is K rounds and the cost is bounded [{claim}].",
            ]
        )
        client.jev = lambda state, questions, **kw: JevResult(  # type: ignore[method-assign]
            answers={"gate": {"noul": 0.1}, "claim-01": {"noul": 0.93}},
            model="typesafe/jev-1.13",
            provider="TypeSafe",
            tokens_in=1,
            cost_usd=0.0,
            latency_ms=1,
        )
        self.runtime.client = client

        with patch.object(config, "GATE_ENABLED", True):
            self.runtime.run()

        rejection = " ".join(
            str(message.get("content") or "")
            for call in client.chat_calls[1:]
            for message in call["messages"]
            if "final gate rejected" in str(message.get("content") or "")
        )

        self.assertIn("failed on one claim", rejection)
        self.assertIn("tau_claim=0.70", rejection)
        self.assertIn("tau_agg=0.50", rejection)

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

    def test_a_corrupt_query_json_is_noted_on_resume(self) -> None:
        # The input file is the run's retrieval contract. Swallowing a parse error
        # as {} would resume the run with no buckets and nothing saying so.
        (self.runtime.workspace).mkdir(parents=True, exist_ok=True)
        (self.runtime.workspace / "query.json").write_text("{not json", encoding="utf-8")

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
        runtime.run()

        events = [
            json.loads(line)
            for line in self.trajectory.trajectory_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        notes = [
            (event.get("extra") or {}).get("text", "")
            for event in events
            if event.get("call_type") == "note"
        ]
        self.assertTrue(
            any("query.json unreadable on resume" in text for text in notes),
            notes,
        )

    def test_the_seed_block_binds_the_kg_helpers(self) -> None:
        # The helpers exist in the image but are not RPC tools, so nothing else
        # puts them in the namespace: without this, tools.md lists build_kg and the
        # model has no way to walk the graph.
        self.runtime.run()

        seed = next(code for code in self.sandbox.executed if "query = " in code)

        for name in ("kg_neighbors", "kg_triples", "kg_subgraph"):
            self.assertIn(name, seed)

    def test_the_seed_block_skips_the_kg_helpers_when_the_kg_is_off(self) -> None:
        with patch.object(config, "KG_ENABLED", False):
            self.runtime.run()

        seed = next(code for code in self.sandbox.executed if "query = " in code)

        self.assertNotIn("kg_neighbors", seed)

    def _tool_events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.trajectory.trajectory_path.read_text(encoding="utf-8").splitlines()
            if line and json.loads(line).get("action") == "tool.dispatch"
        ]

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
        with patch("lens.agent.runtime.REPLAYABLE_TOOLS", frozenset({"echo"})):
            response = self.runtime.handle_rpc("/tool", {"name": "echo", "args": {"text": "hi"}})
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"], {"echo": "hi"})
        self.assertTrue(response.get("replayed"))

    def test_a_repeated_read_only_call_is_served_from_the_first_result(self) -> None:
        # The live run recomputed the same funnel five times: the in-run cache was
        # built from the log once, before any tool result existed, and never grew.
        with patch("lens.agent.runtime.REPLAYABLE_TOOLS", frozenset({"echo"})):
            first = self.runtime.handle_rpc("/tool", {"name": "echo", "args": {"text": "hi"}})
            second = self.runtime.handle_rpc("/tool", {"name": "echo", "args": {"text": "hi"}})

        self.assertFalse(first.get("replayed"))
        self.assertTrue(second.get("replayed"))
        self.assertEqual(second["result"], first["result"])

        events = self._tool_events()
        self.assertEqual(len(events), 2)  # the repeat stays visible in the log
        self.assertEqual(events[1]["result_ref"], events[0]["result_ref"])
        self.assertTrue(events[1]["extra"]["replayed"])

    def test_a_state_changing_tool_is_never_replayed(self) -> None:
        self.runtime.handle_rpc("/tool", {"name": "echo", "args": {"text": "hi"}})
        second = self.runtime.handle_rpc("/tool", {"name": "echo", "args": {"text": "hi"}})

        self.assertFalse(second.get("replayed"))

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

    # ------------------------------------------------------------ sub-tasks
    def test_sub_task_text_mode_parses_the_reply(self) -> None:
        self.runtime.client = SubTaskClient(
            {"summaries": [{"id": "input", "summary": "the paper bounds cost"}]}
        )

        response = self.runtime.handle_rpc(
            "/sub_task",
            {"variant": "summarise", "params": {"text": "long text", "focus": "cost"}},
        )

        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["summaries"][0]["summary"], "the paper bounds cost")
        self.assertEqual(response["result"]["missing_ids"], [])

    def test_a_sub_task_asks_for_a_schema_and_an_output_cap(self) -> None:
        client = SubTaskClient({"summaries": []})
        self.runtime.client = client

        self.runtime.handle_rpc(
            "/sub_task", {"variant": "summarise", "params": {"text": "long text"}}
        )

        call = client.calls[0]
        self.assertEqual(call["role"], "sub")
        self.assertEqual(call["response_format"]["type"], "json_schema")
        self.assertIsInstance(call["max_tokens"], int)
        self.assertEqual(call["extra"]["sub_task"], "summarise")

    def test_sub_task_unknown_variant_is_a_structured_error(self) -> None:
        response = self.runtime.handle_rpc("/sub_task", {"variant": "invent", "params": {}})

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["type"], "UnknownSubTask")

    def test_sub_task_rejects_a_parameter_it_does_not_take(self) -> None:
        response = self.runtime.handle_rpc(
            "/sub_task", {"variant": "summarise", "params": {"text": "x", "depth": 2}}
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["type"], "BadRequest")

    def test_sub_task_is_refused_once_the_call_limit_is_reached(self) -> None:
        self.runtime.budget = Budget(limits=BudgetLimits(max_sub_tasks=0))

        response = self.runtime.handle_rpc(
            "/sub_task", {"variant": "summarise", "params": {"text": "x"}}
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["type"], "SubTaskLimit")

    def test_sub_task_counts_the_call_and_its_tokens(self) -> None:
        self.runtime.client = SubTaskClient({"summaries": []})

        self.runtime.handle_rpc("/sub_task", {"variant": "summarise", "params": {"text": "x"}})

        self.assertEqual(self.runtime.budget.sub_tasks, 1)
        self.assertEqual(self.runtime.budget.tokens_in, 10)

    def test_sub_task_ids_are_hydrated_and_stamped(self) -> None:
        self.runtime.conn = object()
        self.runtime.client = SubTaskClient(
            {
                "records": [
                    {"id": "c-1", "claims": ["a claim"]},
                    {"id": "c-999", "claims": ["invented"]},
                ]
            }
        )

        rows = [{"chunk_uuid": "c-1", "chunk_text": "the chunk text"}]

        with patch("lens.tools.retrieval.fetch_chunks", return_value=rows):
            response = self.runtime.handle_rpc(
                "/sub_task", {"variant": "extract", "params": {"chunk_ids": ["c-1", "c-2"]}}
            )

        self.assertTrue(response["ok"])
        self.assertEqual([record["id"] for record in response["result"]["records"]], ["c-1"])
        self.assertEqual(response["result"]["missing_ids"], ["c-2"])
        self.assertTrue(any("c-999" in error for error in response["result"]["errors"]))

    def test_sub_task_without_a_connection_reports_it(self) -> None:
        response = self.runtime.handle_rpc(
            "/sub_task", {"variant": "extract", "params": {"chunk_ids": ["c-1"]}}
        )

        self.assertFalse(response["ok"])
        self.assertIn("no chunk text", response["error"]["message"])


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


class FlatBaselineTestCase(unittest.TestCase):
    """The orchestration-off arm: one retrieval, one model turn, no loop."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_root = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = self._tmp.name
        self.trajectory = Trajectory("flat-test")
        self.sandbox = FakeSandbox(self.trajectory.run_dir / "work")
        self.registry = ToolRegistry()

        def retrieval_funnel(buckets: object) -> dict:
            """The whole funnel."""
            return {
                "chunks": [
                    {"chunk_uuid": "c1", "chunk_text": "alpha"},
                    {"chunk_uuid": "c2", "chunk_text": "beta"},
                ],
                "papers": {},
            }

        self.registry.register("retrieval_funnel", retrieval_funnel)

        self.client = FakeClient(["The answer [c1]."], self.trajectory)
        self.runtime = Runtime(
            "how many datasets",
            "flat-test",
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

    def test_one_turn_and_no_loop(self) -> None:
        result = self.runtime.run_flat()

        self.assertEqual(result.stop_reason, "flat")
        self.assertFalse(result.partial)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.answer, "The answer [c1].")
        self.assertEqual(len(self.client.chat_calls), 2)  # segmentation + one answer

    def test_the_funnel_is_dispatched_so_citations_resolve(self) -> None:
        self.runtime.run_flat()

        events = [
            json.loads(line)
            for line in self.trajectory.trajectory_path.read_text(encoding="utf-8").splitlines()
        ]
        tools = [event.get("tool") for event in events if event.get("action") == "tool.dispatch"]

        self.assertEqual(tools, ["retrieval_funnel"])

    def test_the_prompt_is_the_flat_variant_with_the_chunks_inline(self) -> None:
        self.runtime.run_flat()

        messages = self.client.chat_calls[-1]["messages"]
        self.assertNotIn("retrieval_funnel", messages[0]["content"])
        self.assertIn("alpha", messages[1]["content"])
        self.assertIn("beta", messages[1]["content"])

    def test_an_over_size_chunk_is_dropped_and_counted(self) -> None:
        with patch.object(config, "FLAT_EVIDENCE_TOKEN_LIMIT", 1):
            self.runtime.run_flat()

        events = [
            json.loads(line)
            for line in self.trajectory.trajectory_path.read_text(encoding="utf-8").splitlines()
        ]
        notes = [event["extra"]["text"] for event in events if event.get("call_type") == "note"]

        self.assertTrue(any("dropped for size" in text for text in notes))


class GateInputTestCase(unittest.TestCase):
    """What the gate is shown, which is what decides whether it can answer."""

    def test_retrieval_summary_carries_content_not_just_counts(self) -> None:
        from lens.agent.runtime import GATE_SAMPLE_CHARS, _retrieval_gate_summary

        result = {
            "chunks": [
                {
                    "chunk_uuid": f"id-{index}",
                    "arxiv_id": "2511.02817",
                    "section_type": "methods",
                    "chunk_text": "x" * 900,
                }
                for index in range(20)
            ],
            # The funnel's real shape: a map keyed by arxiv id. A list here is how
            # the bug that left ``paper_ids`` empty in every live run stayed hidden.
            "papers": {"2511.02817": {"title": "a"}, "2601.00001": {"title": "b"}},
        }

        summary = _retrieval_gate_summary(result)

        # Counts alone cannot answer "does this cover every objective the query
        # names": the gate needs something to read.
        self.assertEqual(summary["chunks"], 20)
        self.assertEqual(summary["paper_ids"], ["2511.02817", "2601.00001"])
        # The old state was six 300-character samples — 1.2 % of the funnel's text —
        # asked whether that output covered the query. The sample now runs to the
        # character budget, capped per chunk.
        self.assertGreater(len(summary["chunk_samples"]), 6)
        self.assertEqual(len(summary["chunk_samples"][0]["text"]), 900)
        self.assertLessEqual(
            sum(len(sample["text"]) for sample in summary["chunk_samples"]), GATE_SAMPLE_CHARS
        )

    def test_the_retrieval_sample_spans_papers_not_just_the_top_ranked(self) -> None:
        from lens.agent.runtime import _retrieval_gate_summary

        # The question is coverage, so the sample must not be the best five chunks of
        # one paper: that answers a narrower question and biases the verdict.
        result = {
            "chunks": [
                {
                    "chunk_uuid": f"id-{paper}-{index}",
                    "arxiv_id": paper,
                    "section_type": "methods",
                    "chunk_text": "x" * 2000,
                }
                for paper in ("2511.02817", "2601.00001", "2602.00002")
                for index in range(30)
            ],
            "papers": {},
        }

        summary = _retrieval_gate_summary(result)
        papers = [sample["paper"] for sample in summary["chunk_samples"]]

        self.assertGreaterEqual(len(set(papers)), 3)
        self.assertEqual(papers[:3], ["2511.02817", "2601.00001", "2602.00002"])

    def test_a_tool_summary_is_passed_through_not_counted(self) -> None:
        from lens.agent.runtime import _bounded_artifact

        artifact = _bounded_artifact(
            {
                "summary": {"n_nodes": 5, "relations": {"uses": 3}},
                "graph": {"nodes": [], "links": []},
            }
        )

        # {"summary": {...}} used to become {"summary": 2} — the number of keys.
        self.assertEqual(artifact["summary"]["n_nodes"], 5)
        self.assertEqual(artifact["summary"]["relations"], {"uses": 3})
        self.assertIsNone(artifact["score"])

    def test_the_kg_gate_sees_real_triples_not_counts(self) -> None:
        from lens.agent.runtime import _kg_gate_summary

        result = {
            "summary": {"n_nodes": 12, "n_edges": 9, "top_entities": ["ReCAP"]},
            "graph": {
                "nodes": [
                    {"id": "recap", "label": "ReCAP"},
                    {"id": "squad", "label": "SQuAD"},
                ],
                "links": [{"source": "recap", "target": "squad", "relation": "evaluated_on"}],
            },
        }

        summary = _kg_gate_summary(result)

        self.assertEqual(summary["n_nodes"], 12)
        self.assertEqual(
            summary["sample_edges"],
            [{"head": "ReCAP", "relation": "evaluated_on", "tail": "SQuAD"}],
        )

    def test_every_cited_chunk_is_fetched_not_the_first_five(self) -> None:
        runtime = Runtime(
            "q",
            "gate-input-test",
            client=FakeClient([]),  # type: ignore[arg-type]
            trajectory=Trajectory("gate-input-test"),
            conn=object(),
            embedder=None,
            reranker=None,
            extractor=None,
            sandbox=FakeSandbox(Path(tempfile.gettempdir()) / "gate-input-work"),  # type: ignore[arg-type]
            registry=ToolRegistry(),
            budget=Budget(),
        )
        rows = [
            {"chunk_uuid": f"c{index}-0000-0000-0000-000000000000", "chunk_text": "y" * 100}
            for index in range(8)
        ]

        with patch("lens.tools.retrieval.fetch_chunks", return_value=rows):
            evidence, truncated = runtime._citation_evidence([row["chunk_uuid"] for row in rows])

        self.assertEqual(len(evidence), 8)
        self.assertFalse(truncated)

    def test_the_evidence_budget_flags_truncation(self) -> None:
        runtime = Runtime(
            "q",
            "gate-trunc-test",
            client=FakeClient([]),  # type: ignore[arg-type]
            trajectory=Trajectory("gate-trunc-test"),
            conn=object(),
            embedder=None,
            reranker=None,
            extractor=None,
            sandbox=FakeSandbox(Path(tempfile.gettempdir()) / "gate-trunc-work"),  # type: ignore[arg-type]
            registry=ToolRegistry(),
            budget=Budget(),
        )
        rows = [
            {"chunk_uuid": f"c{index}-0000-0000-0000-000000000000", "chunk_text": "y" * 2000}
            for index in range(70)
        ]

        with patch("lens.tools.retrieval.fetch_chunks", return_value=rows):
            evidence, truncated = runtime._citation_evidence([row["chunk_uuid"] for row in rows])

        self.assertLess(len(evidence), 70)
        self.assertGreater(len(evidence), 20)
        self.assertTrue(truncated)

    def test_truncation_keeps_the_most_cited_chunks(self) -> None:
        runtime = Runtime(
            "q",
            "gate-order-test",
            client=FakeClient([]),  # type: ignore[arg-type]
            trajectory=Trajectory("gate-order-test"),
            conn=object(),
            embedder=None,
            reranker=None,
            extractor=None,
            sandbox=FakeSandbox(Path(tempfile.gettempdir()) / "gate-order-work"),  # type: ignore[arg-type]
            registry=ToolRegistry(),
            budget=Budget(),
        )
        rare = "aaaa1111-0000-0000-0000-000000000000"
        common = "bbbb2222-0000-0000-0000-000000000000"
        rows = [
            {"chunk_uuid": rare, "chunk_text": "r" * 2000},
            {"chunk_uuid": common, "chunk_text": "c" * 2000},
        ]
        answer = f"a claim [{rare}]. " + " ".join([f"again [{common}]."] * 5)

        # Room for one chunk only: the one the answer leans on must survive.
        with (
            patch("lens.tools.retrieval.fetch_chunks", return_value=rows),
            patch("lens.agent.runtime.EVIDENCE_FETCH_CHARS", 2000),
        ):
            evidence, truncated = runtime._citation_evidence([rare, common], answer=answer)

        self.assertEqual(list(evidence), [common])
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
