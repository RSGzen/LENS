r"""Tests for the wave runner - no key, no DB, no Docker.

The runner is the only place that decides how many billable cells a night starts
and when a staged-spend limit stops it, so its plan validation and its counting
are checked without a model call.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_run_waves -v
"""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
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


run_waves = _load("run_waves")


def _campaign(root: Path, name: str = "c", *, arms: int = 2, queries: int = 2) -> Path:
    payload = {
        "name": name,
        "arms": [{"name": f"a{index}", "env": {"LENS_RECORDED": "1"}} for index in range(arms)],
        "queries": [{"id": f"q{index}", "text": "x"} for index in range(queries)],
    }
    path = root / "campaigns" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    return path


def _plan(root: Path, *, expect: int = 4, campaign: str = "campaigns/c.json") -> Path:
    path = root / "plan.json"
    path.write_text(
        json.dumps(
            {
                "name": "unit",
                "waves": [{"id": "w1", "campaign": campaign, "expect": expect}],
            }
        ),
        encoding="utf-8",
    )

    return path


class PlanValidationTestCase(unittest.TestCase):
    def test_a_duplicate_cell_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "p",
                        "waves": [
                            {"id": "w", "campaign": "a.json", "expect": 1},
                            {"id": "w", "campaign": "b.json", "expect": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                run_waves.load_plan(path)

    def test_a_missing_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(
                json.dumps({"name": "p", "waves": [{"id": "w", "expect": 1}]}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                run_waves.load_plan(path)


class CellStatusTestCase(unittest.TestCase):
    def test_a_missing_campaign_is_not_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_waves.config, "REPO_ROOT", Path(tmp)):
                status = run_waves.cell_status({"id": "w", "campaign": "missing.json", "expect": 1})

        self.assertFalse(status["ok"])
        self.assertIn("missing", status["reason"])

    def test_a_placeholder_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _campaign(Path(tmp), arms=2, queries=2)

            with patch.object(run_waves.config, "REPO_ROOT", Path(tmp)):
                status = run_waves.cell_status(
                    {"id": "w", "campaign": "campaigns/c.json", "expect": 5}
                )

        self.assertFalse(status["ok"])
        self.assertEqual(status["runs"], 4)
        self.assertIn("expected 5", status["reason"])

    def test_the_expected_count_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _campaign(Path(tmp), arms=2, queries=2)

            with patch.object(run_waves.config, "REPO_ROOT", Path(tmp)):
                status = run_waves.cell_status(
                    {"id": "w", "campaign": "campaigns/c.json", "expect": 4}
                )

        self.assertTrue(status["ok"])
        self.assertEqual(status["runs"], 4)


class SummaryTestCase(unittest.TestCase):
    def test_statuses_and_cost_are_counted_from_the_csv(self) -> None:
        rows = [
            {"status": "completed", "cost_usd": "0.10"},
            {"status": "partial", "cost_usd": "0.05"},
            {"status": "failed", "cost_usd": "0.02"},
            {"status": "missing", "cost_usd": "0"},
        ]

        summary = run_waves.summarise_rows(rows)

        self.assertEqual(summary["ran"], 4)
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["partial"], 1)
        self.assertEqual(summary["failed"], 2)
        self.assertAlmostEqual(summary["cost"], 0.17)

    def test_a_checkpoint_stops_only_after_the_limit(self) -> None:
        self.assertEqual(run_waves.checkpoint_decision(10.0, 20.0), "continue")
        self.assertEqual(run_waves.checkpoint_decision(20.0, 20.0), "continue")
        self.assertEqual(run_waves.checkpoint_decision(20.01, 20.0), "stop")
        self.assertEqual(run_waves.checkpoint_decision(20.01, None), "continue")

    def test_existing_cost_counts_only_finished_runs(self) -> None:
        # A resumed cell is not free: the runs skipped by --skip-existing were
        # paid for on an earlier attempt.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _campaign(root, arms=1, queries=2)
            manifests = {
                "c.a0.q0": SimpleNamespace(status="completed", totals={"cost_usd": 0.10}),
                "c.a0.q1": SimpleNamespace(status="running", totals={"cost_usd": 0.20}),
            }

            with (
                patch.object(run_waves.config, "REPO_ROOT", root),
                patch("lens.agent.runstore.runs_root", return_value=Path("/runs")),
                patch(
                    "lens.agent.runstore.read_manifest",
                    side_effect=lambda path: manifests.get(path.name),
                ),
            ):
                cost = run_waves.cell_existing_cost(
                    {"id": "w", "campaign": "campaigns/c.json", "expect": 2}
                )

        self.assertAlmostEqual(cost, 0.10)


class DryRunTestCase(unittest.TestCase):
    def test_a_missing_cell_returns_one_and_spends_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = _plan(root, campaign="campaigns/missing.json")

            with (
                patch.object(run_waves.config, "REPO_ROOT", root),
                redirect_stdout(io.StringIO()) as out,
            ):
                code = run_waves.main([str(plan), "--dry-run"])

        self.assertEqual(code, 1)
        self.assertIn("FAIL", out.getvalue())

    def test_a_valid_plan_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _campaign(root)
            plan = _plan(root)

            with (
                patch.object(run_waves.config, "REPO_ROOT", root),
                redirect_stdout(io.StringIO()) as out,
            ):
                code = run_waves.main([str(plan), "--dry-run"])

        self.assertEqual(code, 0)
        self.assertIn("runs=4", out.getvalue())

    def test_through_a_cell_ignores_the_cells_after_it(self) -> None:
        # A wave that is not built yet must not block the one that is.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _campaign(root)
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "name": "unit",
                        "waves": [
                            {"id": "w1", "campaign": "campaigns/c.json", "expect": 4},
                            {"id": "w2", "campaign": "campaigns/missing.json", "expect": 9},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(run_waves.config, "REPO_ROOT", root),
                redirect_stdout(io.StringIO()) as out,
            ):
                code = run_waves.main([str(plan), "--dry-run", "--through", "w1"])

        self.assertEqual(code, 0)
        self.assertNotIn("w2", out.getvalue())


class WaveSelectionTests(unittest.TestCase):
    """``--from``/``--through`` bound the slice; a bad bound must not run wider."""

    CELLS = [{"id": ident} for ident in ("w3", "w4", "w5", "w5b", "w6")]

    def test_no_bounds_runs_the_whole_plan(self) -> None:
        self.assertEqual(
            [cell["id"] for cell in run_waves.select_cells(self.CELLS)],
            ["w3", "w4", "w5", "w5b", "w6"],
        )

    def test_through_alone_still_starts_at_the_first_cell(self) -> None:
        self.assertEqual(
            [cell["id"] for cell in run_waves.select_cells(self.CELLS, through="w5")],
            ["w3", "w4", "w5"],
        )

    def test_from_skips_the_placeholders_in_front(self) -> None:
        self.assertEqual(
            [cell["id"] for cell in run_waves.select_cells(self.CELLS, start="w5b", through="w5b")],
            ["w5b"],
        )

    def test_from_and_through_are_inclusive(self) -> None:
        self.assertEqual(
            [cell["id"] for cell in run_waves.select_cells(self.CELLS, start="w4", through="w5b")],
            ["w4", "w5", "w5b"],
        )

    def test_an_unknown_id_names_the_flag_it_came_from(self) -> None:
        with self.assertRaises(ValueError) as caught:
            run_waves.select_cells(self.CELLS, start="nope")

        self.assertIn("--from", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            run_waves.select_cells(self.CELLS, through="nope")

        self.assertIn("--through", str(caught.exception))

    def test_a_start_after_the_stop_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            run_waves.select_cells(self.CELLS, start="w6", through="w5b")

        self.assertIn("not before", str(caught.exception))


class PlaceholderCellTests(unittest.TestCase):
    """A placeholder must fail the count check, not run and die at job 1."""

    def test_a_placeholder_expands_to_fewer_runs_than_expected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "w3.json"
            campaign.write_text(
                json.dumps(
                    {
                        "name": "w3",
                        "repeats": 1,
                        "arms": [{"name": "a0-doc", "env": {"LENS_RECORDED": "1"}}],
                        "queries": [{"id": "bridge-001", "text": "placeholder"}],
                    }
                ),
                encoding="utf-8",
            )
            status = run_waves.cell_status({"id": "w3", "campaign": str(campaign), "expect": 100})

        self.assertFalse(status["ok"])
        self.assertEqual(status["runs"], 1)
        self.assertIn("expected 100", status["reason"])


if __name__ == "__main__":
    unittest.main()
