r"""Contract tests for the cost roll-up (no key, no DB, no Docker).

``scripts/cost_rollup.py`` exists because a campaign's true cost is three artifacts:
the cells, the judge run and any regrade run. These tests pin the arithmetic and the
two rules that are easy to get wrong: a run that does not exist is never constructed
(it would create a directory), and only runs at least as new as the campaign count.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_cost_rollup -v
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from lens import config
from lens.observability import trajectory as trajectory_module
from lens.observability.trajectory import TraceCtx, Trajectory

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cost_rollup = _load("cost_rollup")


class CostRollupTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig = trajectory_module.LENS_RUNS_ROOT_PATH
        self._orig_config = config.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = str(self.root)
        config.LENS_RUNS_ROOT_PATH = self.root
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(setattr, trajectory_module, "LENS_RUNS_ROOT_PATH", self._orig)
        self.addCleanup(setattr, config, "LENS_RUNS_ROOT_PATH", self._orig_config)

    def _run(self, run_id: str, cost: float) -> str:
        trajectory = Trajectory(run_id)
        trajectory.log_event(
            trace_ctx=TraceCtx(trace_id=run_id, step_idx=0, actor="root"),
            call_type="chat",
            action="chat.completion",
            cost_usd=cost,
            tokens_in=1,
            tokens_out=1,
        )
        return str(trajectory.trajectory_path)

    def _campaign(self, created_at: str, cells: list[float]) -> dict:
        rows = [{"run_id": f"cell-{i}", "cost_usd": str(c)} for i, c in enumerate(cells)]
        return {"campaign": "unit", "created_at": created_at, "rows": rows}

    def test_the_total_adds_cells_and_companion_runs(self) -> None:
        # The judge runs after the cells, so its log is newer than the campaign.
        now = datetime.now(timezone.utc).isoformat()
        self._run("judge-1", 0.02)
        campaign = self._campaign(now, [0.25, 0.30])

        extra = cost_rollup.discover_companions(campaign["created_at"])
        rollup = cost_rollup.build_rollup(campaign, extra)

        self.assertEqual(extra, ["judge-1"])
        self.assertAlmostEqual(rollup["cells"], 0.55)
        self.assertAlmostEqual(rollup["companions"][0][1], 0.02)

    def test_a_companion_older_than_the_campaign_is_not_counted(self) -> None:
        # A previous wave's judge run must not be added to this wave's total.
        log = Path(self._run("judge-old", 9.99))
        old = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
        import os

        os.utime(log, (old, old))

        created_at = datetime.now(timezone.utc).isoformat()
        self.assertEqual(cost_rollup.discover_companions(created_at), [])

    def test_a_missing_run_is_none_and_creates_nothing(self) -> None:
        # Constructing a Trajectory mkdirs: a roll-up must not litter the runs root.
        self.assertIsNone(cost_rollup.run_cost("nope-1"))
        self.assertFalse((self.root / "nope-1").exists())

    def test_render_names_the_three_sources_and_the_fee(self) -> None:
        self._run("judge-1", 0.02)
        campaign = self._campaign(datetime.now(timezone.utc).isoformat(), [0.25])

        text = cost_rollup.render(cost_rollup.build_rollup(campaign, ["judge-1"]), "unit")

        self.assertIn("campaign cells", text)
        self.assertIn("judge-1", text)
        self.assertIn("total raw", text)
        self.assertIn(f"x{config.CREDIT_FEE}", text)
        self.assertNotIn("missing real spend", text)

    def test_render_warns_when_only_cells_were_counted(self) -> None:
        campaign = self._campaign(datetime.now(timezone.utc).isoformat(), [0.25])

        text = cost_rollup.render(cost_rollup.build_rollup(campaign, []), "unit")

        self.assertIn("missing real spend", text)

    def test_a_campaign_directory_is_found(self) -> None:
        directory = self.root / "wave"
        directory.mkdir()
        payload = self._campaign(datetime.now(timezone.utc).isoformat(), [0.1])
        (directory / "campaign.json").write_text(json.dumps(payload), encoding="utf-8")

        loaded = cost_rollup.load_campaign(directory)

        self.assertEqual(loaded["campaign"], "unit")
        self.assertEqual(len(loaded["rows"]), 1)

    def test_main_writes_the_rollup(self) -> None:
        self._run("judge-1", 0.02)
        directory = self.root / "wave"
        directory.mkdir()
        payload = self._campaign(datetime.now(timezone.utc).isoformat(), [0.25])
        (directory / "campaign.json").write_text(json.dumps(payload), encoding="utf-8")
        out = self.root / "out"

        with patch("sys.stdout"):
            code = cost_rollup.main([str(directory), "--out", str(out)])

        self.assertEqual(code, 0)
        self.assertTrue((out / "cost_rollup.md").is_file())


if __name__ == "__main__":
    unittest.main()
