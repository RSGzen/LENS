r"""Tests for the campaign scoring scripts (no key, no database, no Docker).

The three scripts under test are the measuring instruments, so their arithmetic is
checked against hand-computed values rather than against themselves.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_eval_scripts -v
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score_retrieval = _load("score_retrieval")
score_instruments = _load("score_instruments")
judge_answers = _load("judge_answers")
freeze_manifest = _load("freeze_manifest")


class RetrievalMetricTests(unittest.TestCase):
    def test_ndcg_matches_a_hand_computation(self) -> None:
        # qrels 3/2/1; ranked a, c, b.
        # DCG  = 7 + 1/log2(3) + 3/log2(4)
        # IDCG = 7 + 3/log2(3) + 1/log2(4)
        judged = {"a": 3, "b": 2, "c": 1}
        value = score_retrieval.ndcg_at_k(["a", "c", "b"], judged, 10)

        self.assertAlmostEqual(value, 9.130929 / 9.392789, places=5)

    def test_a_perfect_ranking_scores_one(self) -> None:
        self.assertEqual(score_retrieval.ndcg_at_k(["a", "b"], {"a": 3, "b": 2}, 10), 1.0)

    def test_a_query_with_no_relevant_documents_scores_zero(self) -> None:
        self.assertEqual(score_retrieval.ndcg_at_k(["a"], {"a": 0}, 10), 0.0)

    def test_recall_mrr_and_precision(self) -> None:
        judged = {"a": 3, "b": 0, "c": 1}
        ranked = ["b", "c", "a"]

        self.assertEqual(score_retrieval.recall_at_k(ranked, judged, 10), 1.0)
        self.assertEqual(score_retrieval.recall_at_k(ranked, judged, 1), 0.0)
        self.assertAlmostEqual(score_retrieval.reciprocal_rank(ranked, judged), 0.5)
        self.assertAlmostEqual(score_retrieval.precision_at_k(ranked, judged, 3), 2 / 3)

    def test_a_config_that_missed_a_query_scores_zero_for_it(self) -> None:
        qrels = {"q1": {"a": 3}, "q2": {"b": 2}}
        rankings = {("cfg", "q1"): ["a"]}

        rows = score_retrieval.score_config(rankings, qrels)

        self.assertEqual(rows[0]["queries"], 2)
        self.assertAlmostEqual(float(rows[0]["ndcg@10"]), 0.5)

    def test_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "qrels.csv").write_text(
                "query_id,doc_id,rel\nq1,a,3\nq1,b,0\n", encoding="utf-8"
            )
            (root / "rankings.csv").write_text(
                "config,query_id,rank,doc_id\ncfg,q1,1,a\ncfg,q1,2,b\n", encoding="utf-8"
            )

            qrels = score_retrieval.load_qrels(root / "qrels.csv")
            rankings = score_retrieval.load_rankings(root / "rankings.csv")
            rows = score_retrieval.score_config(rankings, qrels)

            self.assertEqual(qrels, {"q1": {"a": 3, "b": 0}})
            self.assertEqual(rankings[("cfg", "q1")], ["a", "b"])
            self.assertEqual(rows[0]["ndcg@10"], 1.0)


class InstrumentScoringTests(unittest.TestCase):
    def test_integer_ignores_prose(self) -> None:
        correct, _predicted = score_instruments.score_answer("The count is 7.", 7, "integer")

        self.assertTrue(correct)

    def test_integer_rejects_a_different_number(self) -> None:
        correct, _predicted = score_instruments.score_answer("The count is 8.", 7, "integer")

        self.assertFalse(correct)

    def test_id_list_is_order_insensitive_but_not_content_blind(self) -> None:
        self.assertTrue(
            score_instruments.score_answer(
                "2511.02817, 2601.00001", "2601.00001,2511.02817", "sorted-id-list"
            )[0]
        )
        self.assertFalse(
            score_instruments.score_answer("2511.02817", "2601.00001,2511.02817", "sorted-id-list")[
                0
            ]
        )

    def test_uuid_is_found_inside_a_sentence(self) -> None:
        gold = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

        self.assertTrue(
            score_instruments.score_answer(f"The access code is {gold}.", gold, "uuid")[0]
        )
        self.assertFalse(score_instruments.score_answer("I could not find it.", gold, "uuid")[0])

    def test_free_format_requires_the_gold_text(self) -> None:
        self.assertTrue(
            score_instruments.score_answer("It evaluates on TREC coarse.", "trec coarse", "free")[0]
        )
        self.assertFalse(score_instruments.score_answer("No idea.", "trec coarse", "free")[0])

    def test_a_missing_answer_is_recorded_not_scored(self) -> None:
        instances = [{"instance_id": "i1", "gold": {"answer": 7}, "answer_format": "integer"}]
        campaign = {"i1": [{"run_id": "r1", "campaign_arm": "a"}]}

        rows = score_instruments.score_instances(instances, campaign, answers=lambda _rid: None)

        self.assertEqual(rows[0]["correct"], 0)
        self.assertEqual(rows[0]["note"], "missing answer.md")

    def test_summary_reports_accuracy_per_arm(self) -> None:
        results = [
            {"instance_id": "i1", "arm": "a", "correct": 1},
            {"instance_id": "i2", "arm": "a", "correct": 0},
            {"instance_id": "i1", "arm": "b", "correct": 1},
        ]

        summary = {row["arm"]: row["correct"] for row in score_instruments.summarise(results)}

        self.assertEqual(summary, {"a": 0.5, "b": 1.0})


class JudgeTests(unittest.TestCase):
    RUBRIC = {
        "facets": {
            "coverage": "c",
            "grounding": "g",
            "scope_honesty": "s",
            "usability": "u",
        }
    }

    def test_verdicts_parse_from_a_json_block(self) -> None:
        text = (
            '{"coverage": "pass", "grounding": "FAIL", "scope_honesty": "unknown", '
            '"usability": "pass"}'
        )

        verdicts = judge_answers.parse_verdicts(text, list(self.RUBRIC["facets"]))

        self.assertEqual(verdicts["coverage"], "pass")
        self.assertEqual(verdicts["grounding"], "fail")
        self.assertEqual(verdicts["scope_honesty"], "unknown")

    def test_an_unparseable_reply_is_unknown_not_fail(self) -> None:
        verdicts = judge_answers.parse_verdicts("I think it is good", list(self.RUBRIC["facets"]))

        self.assertEqual(set(verdicts.values()), {"unknown"})

    def test_the_arm_name_never_reaches_the_judge(self) -> None:
        calls: list[list[dict[str, str]]] = []

        class FakeClient:
            def chat(self, role: str, messages, **kwargs):  # noqa: ANN001, ANN003
                calls.append(list(messages))
                return SimpleNamespace(
                    text='{"coverage": "pass", "grounding": "pass", '
                    '"scope_honesty": "pass", "usability": "pass"}'
                )

        rows = [{"run_id": "r1", "query_id": "q1", "campaign_arm": "rlm-full"}]
        results = judge_answers.judge_rows(
            rows, self.RUBRIC, client=FakeClient(), answers=lambda _rid: "the answer"
        )

        self.assertEqual(len(results), 4)
        self.assertNotIn("rlm-full", calls[0][1]["content"])
        self.assertIn("the answer", calls[0][1]["content"])


class FreezeTests(unittest.TestCase):
    def record(self, **overrides: object) -> dict:
        base = {
            "commit": "a" * 40,
            "branch": "main",
            "git_dirty": False,
            "models": {"root": "openai/gpt-5"},
            "jev": "typesafe/jev-1.13",
            "caps": {"iterations": 8},
            "embedding": "nomic-embed-text-v1.5@512-halfvec",
            "python": "3.12.14",
            "platform": "Windows",
            "prompt": "full",
            "run_mode": "rlm",
        }
        base.update(overrides)
        return base

    def test_a_complete_record_passes(self) -> None:
        self.assertEqual(freeze_manifest.check_freeze(self.record()), [])

    def test_a_missing_field_fails(self) -> None:
        problems = freeze_manifest.check_freeze(self.record(commit=None))

        self.assertTrue(any("commit" in problem for problem in problems))

    def test_a_dirty_tree_fails_unless_allowed(self) -> None:
        record = self.record(git_dirty=True)

        self.assertTrue(freeze_manifest.check_freeze(record))
        self.assertEqual(freeze_manifest.check_freeze(record, allow_dirty=True), [])

    def test_a_short_hash_fails(self) -> None:
        problems = freeze_manifest.check_freeze(self.record(commit="abc123"))

        self.assertTrue(any("40-character" in problem for problem in problems))

    def test_instrument_hashes_cover_jsonl_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "cag.jsonl").write_text("{}\n", encoding="utf-8")
            (Path(tmp) / "notes.md").write_text("x", encoding="utf-8")

            hashes = freeze_manifest.instrument_hashes(Path(tmp))

            self.assertEqual(set(hashes), {"cag.jsonl"})


if __name__ == "__main__":
    unittest.main()
