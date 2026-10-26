r"""Tests for campaign matrix expansion - no key, no DB, no Docker.

The matrix is what decides how many billable runs a campaign launches, so a wrong
expansion is a money bug: two cells collapsing into one run id spends the budget
twice and reports one result.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_campaign -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lens.experiments.campaign import build_jobs, load_campaign


def _campaign(**overrides: object) -> dict:
    base = {
        "name": "smoke",
        "arms": [{"name": "base", "env": {}}, {"name": "dev", "env": {"LENS_ROOT_MODEL": "m"}}],
        "queries": [{"id": "q1", "text": "one"}, {"id": "q2", "text": "two"}],
    }
    base.update(overrides)
    return base


def _load(payload: dict) -> dict:
    """Write ``payload`` to a temp file and validate it through ``load_campaign``."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        return load_campaign(path)


class CampaignMatrixTestCase(unittest.TestCase):
    def test_one_job_per_cell(self) -> None:
        jobs = build_jobs(_campaign())

        self.assertEqual(len(jobs), 4)
        self.assertEqual(
            {job["run_id"] for job in jobs},
            {"smoke.base.q1", "smoke.base.q2", "smoke.dev.q1", "smoke.dev.q2"},
        )

    def test_repeats_suffix_every_run(self) -> None:
        jobs = build_jobs(_campaign(repeats=3))

        self.assertEqual(len(jobs), 12)
        self.assertIn("smoke.base.q1.r3", {job["run_id"] for job in jobs})

    def test_arm_env_is_stringified(self) -> None:
        # compose env: an int raises on Windows.
        jobs = build_jobs(_campaign(arms=[{"name": "a", "env": {"LENS_GATE": 0}}]))

        self.assertEqual(jobs[0]["env"], {"LENS_GATE": "0"})

    def test_load_and_validate_round_trip(self) -> None:
        self.assertEqual(_load(_campaign())["name"], "smoke")

    def test_duplicate_ids_raise(self) -> None:
        with self.assertRaises(ValueError):
            _load(_campaign(queries=[{"id": "q1", "text": "a"}, {"id": "q1", "text": "b"}]))

    def test_unsafe_id_raises(self) -> None:
        # A space or a slash would silently create a nested run directory.
        with self.assertRaises(ValueError):
            _load(_campaign(arms=[{"name": "my arm", "env": {}}]))

    def test_missing_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            _load({"name": "x", "arms": []})


if __name__ == "__main__":
    unittest.main()
