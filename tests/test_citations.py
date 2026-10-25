r"""Contract tests for citation resolution (M9) - no key, no DB, no Docker.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_citations -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import lens.observability.trajectory as trajectory_module
from lens.observability.citations import cited_ids, resolve_citations, write_sources_md
from lens.observability.trajectory import TraceCtx, Trajectory

CHUNK = "11111111-1111-5111-8111-111111111111"
OTHER = "22222222-2222-5222-8222-222222222222"
GHOST = "99999999-9999-5999-8999-999999999999"


class CitationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = self._tmp.name
        self.traj = Trajectory("run-cite")

    def tearDown(self) -> None:
        trajectory_module.LENS_RUNS_ROOT_PATH = self._orig
        self._tmp.cleanup()

    # ---------------------------------------------------------------- helpers
    def _tool(self, step: int, name: str, result: object) -> None:
        self.traj.log_event(
            trace_ctx=TraceCtx("run-cite", step, "tool"),
            call_type="tool",
            action="tool.dispatch",
            tool=name,
            args_hash="hash-" + name,
            args_ref=self.traj.store_blob("{}"),
            result_ref=self.traj.store_blob(json.dumps(result)),
            result_summary=f"{name}: ok",
            latency_ms=10,
        )

    def _answer(self, text: str) -> None:
        (self.traj.run_dir / "answer.md").write_text(text, encoding="utf-8")

    def _bundle(self) -> None:
        """The shape ``rerank_chunks`` returns: {chunks, papers, assets}."""
        self._tool(
            1,
            "rerank_chunks",
            {
                "chunks": [
                    {
                        "chunk_uuid": CHUNK,
                        "arxiv_id": "2512.24601",
                        "section_type": "Methodology",
                    },
                    {
                        "chunk_uuid": OTHER,
                        "arxiv_id": "2511.02817",
                        "section_type": "Result/Discussion",
                    },
                ],
                "papers": [
                    {"arxiv_id": "2512.24601", "title": "Recursive Language Models"},
                    {"arxiv_id": "2511.02817", "title": "OOLONG"},
                ],
                "assets": {},
            },
        )

    # ------------------------------------------------------------------ tests
    def test_resolved_and_unresolved_chunks(self) -> None:
        self._bundle()
        self._answer(f"RLMs recurse on chunks ({CHUNK}). A ghost is cited ({GHOST}).")

        report = resolve_citations("run-cite")

        self.assertEqual(report.counts, {"cited": 2, "resolved": 1, "unresolved": 1})
        self.assertEqual(report.unresolved, [GHOST])

        resolved = [c for c in report.citations if c.resolved][0]
        self.assertEqual(resolved.arxiv_id, "2512.24601")
        self.assertEqual(resolved.section_type, "Methodology")
        self.assertEqual(resolved.title, "Recursive Language Models")
        self.assertEqual(resolved.turn, 1)

    def test_arxiv_ids_resolve_against_retrieved_papers(self) -> None:
        self._bundle()
        self._answer("See 2512.24601 and 2601.99999 for the comparison.")

        report = resolve_citations("run-cite")

        by_token = {c.token: c for c in report.citations}
        self.assertTrue(by_token["2512.24601"].resolved)
        self.assertFalse(by_token["2601.99999"].resolved)
        self.assertEqual(report.unresolved, ["2601.99999"])

    def test_cited_ids_dedupes_and_keeps_order(self) -> None:
        # The final gate needs the ids the answer actually names (LOG-115).
        text = f"First {CHUNK}, then 2512.24601, then {CHUNK} and {GHOST}."

        self.assertEqual(cited_ids(text), [CHUNK, "2512.24601", GHOST])
        self.assertEqual(cited_ids("no ids here"), [])

    def test_duplicate_tokens_count_once(self) -> None:
        self._bundle()
        self._answer(f"{CHUNK} again {CHUNK} and {CHUNK}")

        self.assertEqual(resolve_citations("run-cite").counts["cited"], 1)

    def test_non_hydrating_tools_are_not_evidence(self) -> None:
        # deep_match names candidates; the model could not have read them.
        self._tool(1, "deep_match", [CHUNK])
        self._answer(f"Quoting {CHUNK}")

        self.assertFalse(resolve_citations("run-cite").citations[0].resolved)

    def test_missing_answer_is_an_empty_report(self) -> None:
        # A failed run must still open: no answer.md is not an exception.
        report = resolve_citations("run-cite")

        self.assertEqual(report.counts, {"cited": 0, "resolved": 0, "unresolved": 0})
        self.assertEqual(report.answer_chars, 0)

    def test_damaged_payload_does_not_raise(self) -> None:
        self.traj.log_event(
            trace_ctx=TraceCtx("run-cite", 1, "tool"),
            call_type="tool",
            action="tool.dispatch",
            tool="rerank_chunks",
            args_hash="h",
            result_ref="sha256:deadbeef",  # never stored
            result_summary="boom",
        )
        self._answer(f"Quoting {CHUNK}")

        self.assertEqual(resolve_citations("run-cite").counts["resolved"], 0)

    def test_sources_md_lists_resolved_then_unresolved(self) -> None:
        self._bundle()
        self._answer(f"{CHUNK} and {GHOST}")

        report = resolve_citations("run-cite")
        path = write_sources_md(report, self.traj.run_dir / "sources.md")
        text = path.read_text(encoding="utf-8")

        self.assertIn("2512.24601", text)
        self.assertIn("## Unresolved", text)
        self.assertIn(GHOST, text)
        # answer.md stays the model's words; the trace lives beside it.
        self.assertNotIn("## Unresolved", (self.traj.run_dir / "answer.md").read_text())


if __name__ == "__main__":
    unittest.main()
