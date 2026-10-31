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
from unittest.mock import patch

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
run_campaign = _load("run_campaign")


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


class CampaignRowTests(unittest.TestCase):
    """The CSV is the campaign's only durable output, so its columns are tested."""

    def test_cached_and_reasoning_tokens_are_reported_as_shares(self) -> None:
        manifest = SimpleNamespace(
            status="partial",
            iterations=3,
            stop_reason="iterations",
            totals={
                "cost_usd": 0.25,
                "tokens_in": 1000,
                "tokens_out": 400,
                "cached_tokens": 750,
                "reasoning_tokens": 200,
            },
        )
        citations = SimpleNamespace(counts={"cited": 3, "resolved": 3, "unresolved": 0})
        job = {"arm": "a", "query_id": "q", "expect": "answerable", "run_id": "r1"}

        with (
            patch.object(run_campaign.runstore, "read_manifest", return_value=manifest),
            patch.object(run_campaign, "resolve_citations", return_value=citations),
        ):
            row = run_campaign._row(job, 0, 1.5)

        self.assertEqual(row["cached_share"], 0.75)
        self.assertEqual(row["reasoning_share"], 0.5)
        self.assertEqual(row["stop_reason"], "iterations")
        self.assertAlmostEqual(row["cost_usd_with_fee"], 0.25 * 1.055, places=6)

    def test_a_missing_run_still_produces_a_row(self) -> None:
        job = {"arm": "a", "query_id": "q", "expect": "", "run_id": "r1"}

        with patch.object(run_campaign.runstore, "read_manifest", return_value=None):
            row = run_campaign._row(job, 1, 0.2)

        self.assertEqual(row["status"], "missing")
        self.assertEqual(row["cached_share"], 0.0)


class CheckRunShapeTests(unittest.TestCase):
    """The gate must assert the shape a mode can produce, not the loop's shape."""

    def setUp(self) -> None:
        self.check_run = _load("check_run")

    def test_rlm_requires_the_full_loop(self) -> None:
        required, allowed, ledger = self.check_run._event_shape("rlm")

        self.assertEqual(required, 4)
        self.assertIsNone(allowed)
        self.assertTrue(ledger)

    def test_flat_needs_no_exec_and_no_ledger(self) -> None:
        required, allowed, ledger = self.check_run._event_shape("naive")

        self.assertEqual(required, 2)
        self.assertNotIn("exec", allowed or frozenset())
        self.assertFalse(ledger)

    def test_document_needs_no_tool_calls(self) -> None:
        required, allowed, ledger = self.check_run._event_shape("document")

        self.assertEqual(required, 2)
        self.assertNotIn("tool", allowed or frozenset())
        self.assertFalse(ledger)

    def test_an_unknown_mode_falls_back_to_the_full_loop(self) -> None:
        self.assertEqual(
            self.check_run._event_shape("something-new"), self.check_run._event_shape("rlm")
        )


class RegradeTests(unittest.TestCase):
    """The regrade probe: old verdict from the log, new verdict from the current gate."""

    def test_a_claim_failure_names_itself_in_the_row(self) -> None:
        regrade = _load("regrade_gate")
        one = "11111111-1111-5111-8111-111111111111"

        class FakeClient:
            def jev(self, state, questions, *, trace_ctx):
                from lens.models import JevResult

                return JevResult(
                    answers={"gate": {"noul": 0.1}, "claim-01": {"noul": 0.92}},
                    model="typesafe/jev-1.13",
                    provider="TypeSafe",
                    tokens_in=1,
                    cost_usd=0.0,
                    latency_ms=1,
                )

        answer = f"The method uses a sliding window [{one}]."
        row = regrade.regrade_run(
            "recorded-run",
            client=FakeClient(),
            conn=None,
            arm="rlm-full",
            query_id="q1",
        )

        # No run directory here, so the answer is empty and the row says so.
        self.assertEqual(row["new"], "skipped")

        # With an answer supplied, the claim is evaluated and named.
        with (
            patch.object(regrade, "read_answer", return_value=answer),
            patch.object(regrade, "recorded_final_verdict", return_value=("fail", 0.4)),
            patch.object(regrade, "evidence_for", return_value={one: "text"}),
        ):
            row = regrade.regrade_run(
                "recorded-run", client=FakeClient(), conn=None, arm="rlm-full", query_id="q1"
            )

        self.assertEqual(row["old"], "fail")
        self.assertEqual(row["new"], "fail")
        self.assertIn("claim-01", row["criterion"])


if __name__ == "__main__":
    unittest.main()
