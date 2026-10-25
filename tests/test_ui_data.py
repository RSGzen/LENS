r"""Contract tests for the M10 UI data layer (no Streamlit, no key, no DB, no Docker).

``lens.ui.data`` is the only thing between the Streamlit page and the artifacts, so
the two failures that matter are tested here rather than discovered in the demo:

- **opening a bad id must create nothing on disk** (``Trajectory`` ``mkdir``s), and
- **the cost cap comes from the manifest**, not from today's ``config`` — M9 runs
  recorded ``0.5`` while the default is now ``1.00`` (LOG-120).

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
        self._chunks(run_id, 0, [{"chunk_uuid": CHUNK_A, "arxiv_id": "2511.02817", "section_type": "Results"}])
        self._chat(run_id, 1, tokens_in=500)
        self._chunks(run_id, 1, [{"chunk_uuid": CHUNK_B, "arxiv_id": "2511.02817", "section_type": "Intro"}])

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
        self.assertEqual([row["chunk_uuid"] for row in data.evidence_upto(view, 9)], [CHUNK_A, CHUNK_B])

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


if __name__ == "__main__":
    unittest.main()
