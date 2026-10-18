r"""Contract tests for the per-stage gate (no key, no network).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_evaluate -v
"""

from __future__ import annotations

import json
import unittest

from lens.models import Question
from lens.tools.evaluate import StageVerdict, build_sufficiency_question


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


if __name__ == "__main__":
    unittest.main()
