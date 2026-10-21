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

    def test_groundedness_question_is_noul(self) -> None:
        question = build_groundedness_question("A claim [c1].", ["c1"])
        self.assertEqual(question.type, "noul")
        self.assertEqual(set(question.criteria), {"true", "false"})

    def test_gated_stages_map_only_stage_completing_tools(self) -> None:
        # shallow_match / rerank_papers return ids only: gating them asked JEV
        # whether a list of arxiv_ids was "sufficient" (G1).
        self.assertEqual(set(GATED_STAGES), {"rerank_chunks", "build_kg"})
        self.assertEqual(set(GATED_STAGES.values()), {"retrieval", "kg"})

    def test_gate_disabled_returns_none_without_a_client(self) -> None:
        # The M11 ablation arm: no verdict, no JEV call, no `gate` key.
        with patch.object(config, "GATE_ENABLED", False):
            verdict = evaluate_stage(
                "retrieval",
                {"score": 0.9, "summary": {}},
                client=None,  # type: ignore[arg-type]
                trace_ctx=TraceCtx(trace_id="t1", step_idx=0, actor="root"),
            )
        self.assertIsNone(verdict)


if __name__ == "__main__":
    unittest.main()
