r"""Contract tests for the per-stage gate and the final groundedness check.

No key, no network: the tier-1 path and the ablation arm are pure.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_evaluate -v
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from lens import config
from lens.models import Question
from lens.observability.trajectory import TraceCtx
from lens.tools.evaluate import (
    EVIDENCE_MAX_CHARS,
    GATED_STAGES,
    StageVerdict,
    build_groundedness_question,
    build_sufficiency_question,
    evaluate_stage,
)


class EvaluateTestCase(unittest.TestCase):
    def test_verdict_is_json_serializable(self) -> None:
        verdict = StageVerdict(verdict="pass", confidence=0.9, tier="threshold")
        json.dumps(verdict.to_dict())

    def test_sufficiency_question_is_noul(self) -> None:
        question = build_sufficiency_question("retrieval", {"score": 0.5})
        self.assertIsInstance(question, Question)
        self.assertEqual(question.type, "noul")
        self.assertEqual(set(question.criteria), {"true", "false"})
        self.assertIn("retrieval", question.instructions.lower())

    def test_sufficiency_question_carries_the_query(self) -> None:
        # "covering every objective and problem the query names" is unjudgeable
        # when the question text does not say what the query is.
        question = build_sufficiency_question(
            "retrieval", {"score": None}, query="which methods aggregate long context?"
        )

        self.assertIn("which methods aggregate long context?", question.instructions)

    def test_state_carries_the_query(self) -> None:
        client = _FakeJevClient({"gate": {"noul": 0.9}})

        evaluate_stage(
            "retrieval",
            {"score": None, "summary": {"chunks": 100}},
            client=client,
            trace_ctx=TraceCtx(trace_id="t", step_idx=0, actor="root"),
            query="which methods aggregate long context?",
        )

        state = json.loads(client.calls[0][0])
        self.assertEqual(state["query"], "which methods aggregate long context?")
        self.assertEqual(state["stage"], "retrieval")

    def test_groundedness_question_is_noul(self) -> None:
        question = build_groundedness_question("A claim [c1].", ["c1"])
        self.assertEqual(question.type, "noul")
        self.assertEqual(set(question.criteria), {"true", "false"})

    def test_groundedness_question_carries_bounded_evidence(self) -> None:
        # An id alone cannot be verified: without the chunk text the gate rejected
        # a correct, cited answer on every call until the iteration cap.
        question = build_groundedness_question(
            "The sky is blue [chunk-a].", ["chunk-a"], {"chunk-a": "x" * 5000}
        )

        self.assertIn("The sky is blue", question.instructions)
        self.assertIn("[chunk-a]", question.instructions)
        # Bounded: the excerpt is cut at EVIDENCE_MAX_CHARS, not carried whole.
        self.assertLess(question.instructions.count("x"), EVIDENCE_MAX_CHARS + 1)
        self.assertLessEqual(len(question.instructions), EVIDENCE_MAX_CHARS + 2000 + 900)

    def test_gated_stages_map_only_stage_completing_tools(self) -> None:
        # shallow_match / rerank_papers return ids only: gating them asked JEV
        # whether a list of arxiv_ids was "sufficient".
        self.assertEqual(set(GATED_STAGES), {"rerank_chunks", "build_kg"})
        self.assertEqual(set(GATED_STAGES.values()), {"retrieval", "kg"})

    def test_gate_disabled_returns_none_without_a_client(self) -> None:
        # The ablation arm: no verdict, no JEV call, no `gate` key.
        with patch.object(config, "GATE_ENABLED", False):
            verdict = evaluate_stage(
                "retrieval",
                {"score": 0.9, "summary": {}},
                client=None,  # type: ignore[arg-type]
                trace_ctx=TraceCtx(trace_id="t1", step_idx=0, actor="root"),
            )
        self.assertIsNone(verdict)


class _FakeJevClient:
    """Transport double: one canned ``answers`` object, no network."""

    trajectory = None

    def __init__(self, answers: dict | None = None, exc: Exception | None = None) -> None:
        self.answers = answers if answers is not None else {}
        self.exc = exc
        self.calls: list[tuple[str, dict]] = []

    def jev(self, state, questions, *, trace_ctx):
        from lens.models import JevResult

        self.calls.append((state, questions))

        if self.exc is not None:
            raise self.exc

        return JevResult(
            answers=self.answers,
            model="typesafe/jev-1.13",
            provider="TypeSafe",
            tokens_in=1,
            cost_usd=0.0,
            latency_ms=1,
        )


class _FakeBudget:
    def __init__(self) -> None:
        self.jev_calls = 0

    def record_jev(self, result) -> None:
        self.jev_calls += 1


class EvaluateTierTestCase(unittest.TestCase):
    """Tier 1 / tier 2 / degradation — the paths the five contracts skip."""

    ctx = TraceCtx(trace_id="t", step_idx=0, actor="root")

    def test_tier1_accepts_on_a_calibrated_score_without_a_client(self) -> None:
        verdict = evaluate_stage(
            "retrieval",
            {"score": 0.83, "summary": {}},
            client=None,
            trace_ctx=self.ctx,
            threshold=0.7,  # type: ignore[arg-type]
        )
        assert verdict is not None
        self.assertEqual(verdict.verdict, "pass")
        self.assertEqual(verdict.tier, "threshold")
        self.assertAlmostEqual(verdict.confidence or 0, 0.83)

    def test_low_score_goes_to_tier2_not_to_fail(self) -> None:
        # One threshold can only accept: below it we ask JEV, never reject.
        client = _FakeJevClient({"gate": {"answer": True, "confidence": 0.66}})

        verdict = evaluate_stage(
            "retrieval",
            {"score": 0.10, "summary": {"chunks": 20}},
            client=client,
            trace_ctx=self.ctx,
            threshold=0.7,  # type: ignore[arg-type]
        )

        assert verdict is not None
        self.assertEqual(verdict.tier, "jev")
        self.assertEqual(verdict.verdict, "pass")
        self.assertEqual(len(client.calls), 1)
        self.assertIn('"chunks"', client.calls[0][0])  # the bounded summary is the state

    def test_noul_answer_is_read_not_ignored(self) -> None:
        # The real JEV shape for a noul question is {"type": "noul", "noul": 0.03}
        # — no `answer` key. Reading only `answer`/`value`/`verdict` made every
        # verdict `uncertain`, and only `fail` rejects, so the final gate accepted
        # a template answer that cited nothing.
        low = evaluate_stage(
            "final",
            {"score": None, "summary": {"answer": "see above", "citations": []}},
            client=_FakeJevClient({"gate": {"type": "noul", "noul": 0.03}}),  # type: ignore[arg-type]
            trace_ctx=self.ctx,
        )
        high = evaluate_stage(
            "final",
            {"score": None, "summary": {"answer": "cited", "citations": ["c1"]}},
            client=_FakeJevClient({"gate": {"type": "noul", "noul": 0.96}}),  # type: ignore[arg-type]
            trace_ctx=self.ctx,
        )

        assert low is not None and high is not None
        self.assertEqual(low.verdict, "fail")
        self.assertEqual(high.verdict, "pass")
        self.assertAlmostEqual(high.confidence or 0, 0.96)

    def test_final_gate_rejects_and_carries_the_criterion(self) -> None:
        client = _FakeJevClient(
            {"gate": {"answer": False, "confidence": 0.91, "failing_criterion": "claim 2 uncited"}}
        )

        verdict = evaluate_stage(
            "final",
            {"score": None, "summary": {"answer": "A claim [c1].", "citations": ["c1"]}},
            client=client,
            trace_ctx=self.ctx,  # type: ignore[arg-type]
        )

        assert verdict is not None
        self.assertEqual(verdict.verdict, "fail")
        self.assertEqual(verdict.failing_criterion, "claim 2 uncited")

    def test_unrecognised_answer_shape_is_uncertain_not_a_crash(self) -> None:
        client = _FakeJevClient({"gate": {"probability": 0.4}})

        verdict = evaluate_stage(
            "kg",
            {"score": None, "summary": {}},
            client=client,
            trace_ctx=self.ctx,  # type: ignore[arg-type]
        )

        assert verdict is not None
        self.assertEqual(verdict.verdict, "uncertain")

    def test_dead_decision_layer_degrades_to_uncertain(self) -> None:
        client = _FakeJevClient(exc=RuntimeError("timeout"))

        verdict = evaluate_stage(
            "kg",
            {"score": None, "summary": {}},
            client=client,
            trace_ctx=self.ctx,  # type: ignore[arg-type]
        )

        assert verdict is not None
        self.assertEqual(verdict.verdict, "uncertain")
        self.assertIn("unavailable", verdict.failing_criterion or "")

    def test_jev_cost_is_recorded_against_the_budget(self) -> None:
        budget = _FakeBudget()
        client = _FakeJevClient({"gate": {"answer": True, "confidence": 0.9}})

        evaluate_stage(
            "retrieval",
            {"score": None, "summary": {}},
            client=client,
            trace_ctx=self.ctx,
            budget=budget,  # type: ignore[arg-type]
        )

        self.assertEqual(budget.jev_calls, 1)

    def test_verdict_dict_keeps_a_stable_shape(self) -> None:
        keys = set(StageVerdict(verdict="uncertain").to_dict())
        self.assertEqual(keys, {"verdict", "confidence", "failing_criterion", "tier"})


if __name__ == "__main__":
    unittest.main()
