r"""Contract tests for the UI data layer (no Streamlit, no key, no DB, no Docker).

``lens.ui.data`` is the only thing between the Streamlit page and the artifacts, so
the two failures that matter are tested here rather than discovered in the demo:

- **opening a bad id must create nothing on disk** (``Trajectory`` ``mkdir``s), and
- **the cost cap comes from the manifest**, not from today's ``config`` — runs
  recorded ``0.5`` while the default is now ``1.00``.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_ui_data -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import lens.config as config_module
import lens.observability.trajectory as trajectory_module
from lens.agent import runstore
from lens.observability.trajectory import TraceCtx, Trajectory
from lens.ui import data

CHUNK_A = "11111111-1111-5111-8111-111111111111"
CHUNK_B = "22222222-2222-5222-8222-222222222222"


class UiDataTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_config = config_module.LENS_RUNS_ROOT_PATH
        self._orig_trajectory = trajectory_module.LENS_RUNS_ROOT_PATH
        config_module.LENS_RUNS_ROOT_PATH = self.root
        trajectory_module.LENS_RUNS_ROOT_PATH = self.root

    def tearDown(self) -> None:
        config_module.LENS_RUNS_ROOT_PATH = self._orig_config
        trajectory_module.LENS_RUNS_ROOT_PATH = self._orig_trajectory
        self._tmp.cleanup()

    # ---------------------------------------------------------------- helpers
    def _chat(self, run_id: str, step: int, *, tokens_in: int = 100) -> None:
        traj = Trajectory(run_id)
        traj.log_event(
            trace_ctx=TraceCtx(run_id, step, "root"),
            call_type="chat",
            action="chat.completion",
            model="openai/gpt-5",
            provider="openai",
            prompt_ref=traj.store_blob(json.dumps({"messages": []})),
            completion_ref=traj.store_blob(f"turn {step}"),
            tokens_in=tokens_in,
            tokens_out=20,
            cost_usd=0.01,
            latency_ms=900,
        )

    def _chunks(self, run_id: str, step: int, rows: list[dict]) -> None:
        traj = Trajectory(run_id)
        traj.log_event(
            trace_ctx=TraceCtx(run_id, step, "tool"),
            call_type="tool",
            action="tool.dispatch",
            tool="rerank_chunks",
            args_hash=f"hash-{step}",
            args_ref=traj.store_blob(json.dumps({"chunk_ids": []})),
            result_ref=traj.store_blob(json.dumps({"chunks": rows, "papers": []})),
            result_summary=f"rerank_chunks: {len(rows)} rows",
            latency_ms=120,
        )

    def _gate(
        self,
        run_id: str,
        step: int,
        *,
        stage: str = "final",
        verdict: str = "fail",
        confidence: float = 0.96,
        scores: dict[str, float] | None = None,
        failed: list[str] | None = None,
        criterion: str | None = None,
        request: bool = True,
    ) -> None:
        """Write one gate call: the request event and the verdict event."""
        traj = Trajectory(run_id)

        if request:
            body = {
                "model": "typesafe/jev-1.13",
                "state": json.dumps({"stage": stage, "summary": {"answer": "candidate answer"}}),
                "questions": {
                    "gate": {
                        "type": "noul",
                        "instructions": "does it contain an unsupported claim?",
                    },
                    "claim-01": {
                        "type": "noul",
                        "instructions": f"Claim claim-01 cites: {CHUNK_A}",
                    },
                },
            }
            traj.log_event(
                trace_ctx=TraceCtx(run_id, step, "root"),
                call_type="jev",
                action="jev.decisions",
                model="typesafe/jev-1.13",
                provider="TypeSafe",
                prompt_ref=traj.store_blob(json.dumps(body)),
                completion_ref=traj.store_blob(json.dumps({"answers": {"gate": {"noul": 0.96}}})),
                tokens_in=1234,
                cost_usd=0.0004,
                latency_ms=800,
            )

        traj.log_event(
            trace_ctx=TraceCtx(run_id, step, "root"),
            call_type="note",
            action="gate.verdict",
            stage=stage,
            verdict=verdict,
            confidence=confidence,
            extra={
                "claim_scores": scores or {"claim-01": 0.92},
                "claims_failed": failed if failed is not None else ["claim-01"],
                "criterion": criterion,
                "polarity": "unsupported",
                "tier": "jev",
            },
        )

    def _make_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        caps: dict | None = None,
        updated_at: str = "2026-10-25T10:00:00+00:00",
        answer: str | None = None,
        sources: bool = False,
    ) -> Path:
        """Build a two-turn run with one hydrating tool call in each turn."""
        self._chat(run_id, 0)
        self._chunks(
            run_id,
            0,
            [
                {
                    "chunk_uuid": CHUNK_A,
                    "arxiv_id": "2511.02817",
                    "section_type": "Results",
                    "chunk_text": "chunk one text",
                }
            ],
        )
        self._chat(run_id, 1, tokens_in=500)
        self._chunks(
            run_id,
            1,
            [
                {
                    "chunk_uuid": CHUNK_B,
                    "arxiv_id": "2511.02817",
                    "section_type": "Intro",
                    "chunk_text": "chunk two text",
                }
            ],
        )

        run_dir = self.root / run_id
        manifest = runstore.RunManifest(
            run_id=run_id,
            query="what is evaluated on trec_coarse?",
            status=status,
            created_at="2026-10-25T09:00:00+00:00",
            updated_at=updated_at,
            caps=caps if caps is not None else {"cost_usd": 0.5},
            totals={"cost_usd": 0.02},
            iterations=2,
        )
        runstore.write_manifest(run_dir, manifest)

        if answer is not None:
            (run_dir / "answer.md").write_text(answer, encoding="utf-8")

        if sources:
            (run_dir / "sources.md").write_text("# sources\n", encoding="utf-8")

        return run_dir

    # ------------------------------------------------------- opening a run
    def test_unknown_run_returns_none_and_creates_nothing(self) -> None:
        self.assertIsNone(data.load_run("ghost"))
        self.assertFalse((self.root / "ghost").exists())
        self.assertFalse((self.root / "blobs").exists())

    def test_unsafe_run_id_is_refused(self) -> None:
        self.assertIsNone(data.load_run("../escape"))
        self.assertIsNone(data.load_run("a\\b"))
        self.assertFalse(data.is_run_id(".."))
        self.assertFalse(data.is_run_id(""))

    def test_load_run_returns_turns_and_meters(self) -> None:
        self._make_run("run-a", answer=f"answer citing {CHUNK_A}\n")

        view = data.load_run("run-a")

        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(len(view.turns), 2)
        self.assertEqual(view.tokens_in_series, [100, 500])
        self.assertEqual(view.summary["tool_calls"], 2)
        self.assertEqual(view.missing_refs, 0)
        self.assertAlmostEqual(view.cost_usd, 0.02)
        self.assertAlmostEqual(view.cost_with_fee, 0.02 * 1.055)

    def test_answer_and_sources_are_reported_not_assumed(self) -> None:
        self._make_run("run-a", answer=f"see {CHUNK_A}\n")
        self._make_run("run-b")

        with_answer = data.load_run("run-a")
        without = data.load_run("run-b")

        assert with_answer is not None and without is not None
        self.assertIn(CHUNK_A, with_answer.answer)
        self.assertFalse(with_answer.has_sources)
        self.assertEqual(without.answer, "")
        self.assertEqual(without.citations.counts["cited"], 0)

    def test_citations_resolve_against_the_recorded_payload(self) -> None:
        self._make_run("run-a", answer=f"only chunk one: {CHUNK_A}\n")

        view = data.load_run("run-a")

        assert view is not None
        self.assertEqual(view.citations.counts, {"cited": 1, "resolved": 1, "unresolved": 0})

    # ------------------------------------------------------------- the cap
    def test_cap_comes_from_the_manifest_not_todays_default(self) -> None:
        self._make_run("run-old", caps={"cost_usd": 0.5})

        view = data.load_run("run-old")

        assert view is not None
        self.assertAlmostEqual(view.cap_usd, 0.5)
        self.assertNotAlmostEqual(view.cap_usd, config_module.MAX_COST_USD_PER_QUERY)

    def test_cap_is_none_when_not_recorded(self) -> None:
        self._make_run("run-mystery", caps={})

        view = data.load_run("run-mystery")

        assert view is not None
        self.assertIsNone(view.cap_usd)

    # --------------------------------------------------------- evidence
    def test_evidence_accumulates_with_the_cursor(self) -> None:
        self._make_run("run-a")

        view = data.load_run("run-a")

        assert view is not None
        self.assertEqual([row["chunk_uuid"] for row in data.evidence_upto(view, 0)], [CHUNK_A])
        self.assertEqual(
            [row["chunk_uuid"] for row in data.evidence_upto(view, 1)], [CHUNK_A, CHUNK_B]
        )

    def test_evidence_deduplicates_a_rehydrated_chunk(self) -> None:
        self._chat("run-a", 0)
        self._chunks("run-a", 0, [{"chunk_uuid": CHUNK_A}])
        self._chat("run-a", 1)
        self._chunks("run-a", 1, [{"chunk_uuid": CHUNK_A}, {"chunk_uuid": CHUNK_B}])
        runstore.write_manifest(
            self.root / "run-a", runstore.RunManifest(run_id="run-a", query="q")
        )

        view = data.load_run("run-a")

        assert view is not None
        self.assertEqual(
            [row["chunk_uuid"] for row in data.evidence_upto(view, 9)], [CHUNK_A, CHUNK_B]
        )

    # ---------------------------------------------------------- payloads
    def test_tool_payload_roundtrip_and_missing_ref(self) -> None:
        self._make_run("run-a")

        view = data.load_run("run-a")

        assert view is not None
        ref = view.turns[0].tools[0].result_ref
        payload = data.tool_payload("run-a", ref)

        assert payload is not None
        self.assertIn(CHUNK_A, payload)
        self.assertIsNone(data.tool_payload("run-a", None))
        self.assertIsNone(data.tool_payload("run-a", "sha256:deadbeef"))

    # ------------------------------------------------------------- the list
    def test_run_rows_is_newest_first_and_filters_status(self) -> None:
        self._make_run("run-old", updated_at="2026-10-20T10:00:00+00:00")
        self._make_run("run-new", updated_at="2026-10-25T10:00:00+00:00")
        self._make_run("run-broken", status="failed", updated_at="2026-10-24T10:00:00+00:00")

        rows = data.run_rows()

        self.assertEqual([row["run_id"] for row in rows], ["run-new", "run-broken", "run-old"])
        self.assertEqual([row["run_id"] for row in data.run_rows("failed")], ["run-broken"])
        self.assertAlmostEqual(rows[0]["cost_usd"], 0.02)
        self.assertEqual(rows[0]["iterations"], 2)

    def test_run_rows_skips_directories_that_are_not_runs(self) -> None:
        self._make_run("run-a")
        (self.root / "not-a-run").mkdir()
        (self.root / "blobs").mkdir(exist_ok=True)

        self.assertEqual([row["run_id"] for row in data.run_rows()], ["run-a"])
        self.assertIsNone(data.run_dir("not-a-run"))
        self.assertEqual(data.run_dir("run-a"), self.root / "run-a")

    # ------------------------------------------------------ local display names
    def test_alias_round_trip_and_display_name(self) -> None:
        self._make_run("run-a")

        data.set_alias("run-a", "  Tier-1 anchor  ")

        self.assertEqual(data.load_aliases(), {"run-a": "Tier-1 anchor"})
        self.assertEqual(data.display_name("run-a"), "Tier-1 anchor")
        self.assertEqual(data.display_name("run-b"), "run-b")  # never renamed
        self.assertEqual(data.run_rows()[0]["name"], "Tier-1 anchor")

    def test_alias_is_unique_case_insensitively(self) -> None:
        self._make_run("run-a")
        self._make_run("run-b", updated_at="2026-10-24T10:00:00+00:00")
        data.set_alias("run-a", "Baseline")

        with self.assertRaises(ValueError):
            data.set_alias("run-b", "baseline")

        # The same run may be renamed again, even to a case variant of its own name.
        data.set_alias("run-a", "BASELINE")
        self.assertEqual(data.load_aliases(), {"run-a": "BASELINE"})

    def test_alias_clear_and_length_limit(self) -> None:
        self._make_run("run-a")
        data.set_alias("run-a", "anchor")

        data.set_alias("run-a", "   ")  # empty clears it
        self.assertEqual(data.load_aliases(), {})

        with self.assertRaises(ValueError):
            data.set_alias("run-a", "x" * (data.ALIAS_MAX_CHARS + 1))

    def test_alias_file_lives_outside_the_run_and_survives_corruption(self) -> None:
        self._make_run("run-a")
        data.set_alias("run-a", "anchor")

        path = data.aliases_path()
        self.assertEqual(path.parent, self.root)
        self.assertFalse((self.root / "run-a" / data.ALIASES_FILENAME).exists())

        path.write_text("{not json", encoding="utf-8")
        self.assertEqual(data.load_aliases(), {})
        self.assertEqual(data.load_run("run-a").run_id, "run-a")  # still openable

    # ----------------------------------------------------- cited chunk text
    def test_cited_chunks_resolves_the_answer_s_ids(self) -> None:
        self._make_run("run-cited", answer=f"One claim [{CHUNK_A}], one more [{CHUNK_B}].")

        view = data.load_run("run-cited")
        assert view is not None

        chunks = data.cited_chunks(view)

        self.assertEqual([row["chunk_uuid"] for row in chunks], [CHUNK_A, CHUNK_B])
        self.assertIn("chunk one", chunks[0]["text"])
        self.assertEqual(chunks[0]["chars"], len(chunks[0]["text"]))

    def test_an_id_the_bundle_cannot_explain_is_skipped(self) -> None:
        # The answer cites a chunk no recorded payload hydrated: the citation report
        # already calls it unresolved, and an empty row would look like an empty chunk.
        missing = "99999999-9999-5999-8999-999999999999"
        self._make_run("run-cited-missing", answer=f"One claim [{CHUNK_A}] and [{missing}].")

        view = data.load_run("run-cited-missing")
        assert view is not None

        chunks = data.cited_chunks(view)

        self.assertEqual([row["chunk_uuid"] for row in chunks], [CHUNK_A])

    def test_a_cited_id_is_not_repeated(self) -> None:
        self._make_run("run-cited-twice", answer=f"A [{CHUNK_A}]. B [{CHUNK_A}].")

        view = data.load_run("run-cited-twice")
        assert view is not None

        self.assertEqual(len(data.cited_chunks(view)), 1)

    # ------------------------------------------------------------------ gates
    def test_gate_calls_pair_the_request_with_the_verdict(self) -> None:
        self._make_run("run-gate")
        self._gate("run-gate", 1, criterion="claim-01 (P=0.92): a claim")

        calls = data.gate_calls("run-gate")

        self.assertEqual(sorted(calls), [1])
        call = calls[1][0]
        self.assertEqual((call.stage, call.verdict), ("final", "fail"))
        self.assertAlmostEqual(call.confidence, 0.96)
        self.assertEqual(call.claims_failed, ("claim-01",))
        self.assertAlmostEqual(call.claim_scores["claim-01"], 0.92)
        self.assertIn("unsupported claim", call.question or "")
        self.assertIn("candidate answer", call.state or "")
        self.assertAlmostEqual(call.cost_usd, 0.0004)
        self.assertEqual(call.tokens_in, 1234)

    def test_gate_call_keeps_the_ids_each_claim_was_judged_against(self) -> None:
        self._make_run("run-gate-ids")
        self._gate("run-gate-ids", 2)

        call = data.gate_calls("run-gate-ids")[2][0]

        self.assertIn(CHUNK_A, call.claim_questions["claim-01"])

    def test_a_verdict_without_a_request_is_still_reported(self) -> None:
        # A skipped tier costs nothing and asks nothing; dropping it would hide a
        # gate that ran from the panel entirely.
        self._make_run("run-gate-skip")
        self._gate("run-gate-skip", 3, request=False, verdict="skipped", confidence=None)

        call = data.gate_calls("run-gate-skip")[3][0]

        self.assertEqual(call.verdict, "skipped")
        self.assertIsNone(call.question)
        self.assertIsNone(call.state)
        self.assertEqual(call.cost_usd, 0.0)

    def test_two_gates_in_one_turn_keep_their_order(self) -> None:
        self._make_run("run-gate-two")
        self._gate("run-gate-two", 1, stage="retrieval", verdict="fail", confidence=0.1)
        self._gate("run-gate-two", 1, stage="final", verdict="pass", confidence=0.2)

        calls = data.gate_calls("run-gate-two")[1]

        self.assertEqual([call.stage for call in calls], ["retrieval", "final"])
        self.assertEqual([call.verdict for call in calls], ["fail", "pass"])

    def test_a_run_without_gates_has_no_entries(self) -> None:
        self._make_run("run-no-gate")

        self.assertEqual(data.gate_calls("run-no-gate"), {})

    def test_a_bad_run_id_reads_nothing(self) -> None:
        self.assertEqual(data.gate_calls(".."), {})

    def test_the_state_is_clipped_with_a_marker(self) -> None:
        self._make_run("run-gate-long")
        traj = Trajectory("run-gate-long")
        body = {
            "model": "typesafe/jev-1.13",
            "state": "x" * (data.GATE_STATE_CHARS + 500),
            "questions": {"gate": {"type": "noul", "instructions": "q"}},
        }
        traj.log_event(
            trace_ctx=TraceCtx("run-gate-long", 1, "root"),
            call_type="jev",
            action="jev.decisions",
            prompt_ref=traj.store_blob(json.dumps(body)),
            completion_ref=traj.store_blob("{}"),
            tokens_in=10,
            cost_usd=0.0,
            latency_ms=1,
        )
        traj.log_event(
            trace_ctx=TraceCtx("run-gate-long", 1, "root"),
            call_type="note",
            action="gate.verdict",
            stage="final",
            verdict="fail",
            confidence=0.9,
            extra={"claim_scores": {}, "claims_failed": []},
        )

        state = data.gate_calls("run-gate-long")[1][0].state or ""

        self.assertGreater(len(state), data.GATE_STATE_CHARS)
        self.assertIn("more characters", state)

    def test_the_view_carries_the_gates(self) -> None:
        self._make_run("run-view-gate")
        self._gate("run-view-gate", 1)

        view = data.load_run("run-view-gate")

        self.assertIsNotNone(view)
        self.assertEqual(len(view.gates[1]), 1)


if __name__ == "__main__":
    unittest.main()
