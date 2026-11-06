"""Contract tests for the offline claim-ledger script.

No key, no database, no Docker: the script's writers are checked against a
hand-built report, which is the part a campaign reads.
"""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from lens.observability.claims import ClaimFrame, ClaimReport, FrameVerdict

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Import a script by path, because ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


verify_claims = _load("verify_claims")


def _report() -> ClaimReport:
    frame = ClaimFrame(
        unit_index=0,
        text="RLM reaches 56.0 accuracy on OOLONG.",
        predicate="measures",
        subject="RLM",
        benchmark="OOLONG",
        metric="accuracy",
        value="56.0",
        quote="reaches 56.0 accuracy on OOLONG",
        chunk_id="c1",
    )
    return ClaimReport(
        run_id="run-1",
        units=2,
        frames=[frame],
        verdicts=[FrameVerdict("contradicted", "the span states 44.0", "44.0")],
        errors=["one frame was dropped"],
    )


class WriterTests(unittest.TestCase):
    def test_csv_has_one_row_per_frame_and_the_verdict_columns(self) -> None:
        report = _report()

        with tempfile.TemporaryDirectory() as tmp:
            path = verify_claims.write_csv([report], Path(tmp) / "claims.csv")

            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "run-1")
        self.assertEqual(rows[0]["status"], "contradicted")
        self.assertEqual(rows[0]["reason"], "the span states 44.0")
        self.assertEqual(rows[0]["evidence"], "44.0")

    def test_markdown_states_the_totals_and_the_contradiction(self) -> None:
        report = _report()

        with tempfile.TemporaryDirectory() as tmp:
            path = verify_claims.write_markdown([report], Path(tmp) / "claims.md")
            text = path.read_text(encoding="utf-8")

        self.assertIn("coverage 50.0%", text)
        self.assertIn("## Contradicted", text)
        self.assertIn("the span states 44.0", text)
        self.assertIn("## Extractor errors", text)

    def test_totals_sum_units_and_statuses(self) -> None:
        totals = verify_claims._totals([_report(), _report()])

        self.assertEqual(totals["runs"], 2)
        self.assertEqual(totals["units"], 4)
        self.assertEqual(totals["frames"], 2)
        self.assertEqual(totals["contradicted"], 2)
        self.assertEqual(totals["coverage"], 0.5)


class SelectionTests(unittest.TestCase):
    def test_explicit_runs_are_deduplicated_in_order(self) -> None:
        args = type("Args", (), {"run": ["b", "a", "b"], "all": False})()

        self.assertEqual(verify_claims._selected_ids(args), ["b", "a"])


if __name__ == "__main__":
    unittest.main()
