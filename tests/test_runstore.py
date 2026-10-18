r"""Contract tests for the run store (no key, no DB, no Docker).

``manifest.json`` is identity, ``resume.json`` is a cache, the log is the state.
These tests pin that split and the tolerant-reader rules that keep old runs
readable after a schema change.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_runstore -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lens.agent import runstore
from lens.agent.runstore import RunManifest


class RunStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_run(self, run_id: str, *, status: str = "running", query: str = "a query") -> Path:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True)
        manifest = RunManifest.new(run_id, query)
        manifest.status = status  # type: ignore[assignment]
        runstore.write_manifest(run_dir, manifest)
        return run_dir

    # ------------------------------------------------------------------ model
    def test_new_manifest_is_running_and_versioned(self) -> None:
        manifest = RunManifest.new("run-1", "q")
        self.assertEqual(manifest.status, "running")
        self.assertEqual(manifest.schema_version, runstore.SCHEMA_VERSION)
        self.assertTrue(manifest.created_at)
        self.assertTrue(manifest.updated_at)
        self.assertIsNone(manifest.parent_run_id)

    def test_manifest_round_trip(self) -> None:
        manifest = RunManifest.new(
            "run-1", "q", models={"root": "openai/gpt-5"}, caps={"iterations": 8}
        )
        manifest.totals = {"cost_usd": 0.25}
        restored = RunManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored.run_id, "run-1")
        self.assertEqual(restored.query, "q")
        self.assertEqual(restored.models["root"], "openai/gpt-5")
        self.assertEqual(restored.caps["iterations"], 8)
        self.assertAlmostEqual(restored.totals["cost_usd"], 0.25)
        self.assertEqual(restored.schema_version, runstore.SCHEMA_VERSION)

    def test_unknown_keys_are_preserved(self) -> None:
        # A newer version writes a field this one does not know: it must survive
        # a read+write cycle instead of being silently dropped.
        payload = RunManifest.new("run-1", "q").to_dict()
        payload["future_field"] = {"nested": 1}
        restored = RunManifest.from_dict(payload)
        self.assertEqual(restored.extra["future_field"], {"nested": 1})
        self.assertEqual(restored.to_dict()["future_field"], {"nested": 1})

    # --------------------------------------------------------------- manifest
    def test_write_then_read(self) -> None:
        run_dir = self._make_run("run-1")
        self.assertTrue((run_dir / runstore.MANIFEST_FILENAME).is_file())
        manifest = runstore.read_manifest(run_dir)
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest.run_id, "run-1")

    def test_read_missing_or_corrupt_returns_none(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        self.assertIsNone(runstore.read_manifest(empty))
        bad = self.root / "bad"
        bad.mkdir()
        (bad / runstore.MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
        self.assertIsNone(runstore.read_manifest(bad))

    def test_write_is_atomic_and_leaves_no_temp_file(self) -> None:
        run_dir = self._make_run("run-1")
        names = [p.name for p in run_dir.iterdir()]
        self.assertIn(runstore.MANIFEST_FILENAME, names)
        self.assertFalse([n for n in names if n.endswith(".tmp")], names)

    def test_update_refreshes_timestamp_and_validates(self) -> None:
        run_dir = self._make_run("run-1")
        before = runstore.read_manifest(run_dir)
        assert before is not None
        after = runstore.update_manifest(run_dir, status="completed", iterations=3)
        self.assertEqual(after.status, "completed")
        self.assertEqual(after.iterations, 3)
        self.assertNotEqual(after.updated_at, "")
        with self.assertRaises(ValueError):
            runstore.update_manifest(run_dir, status="bogus")
        with self.assertRaises(TypeError):
            runstore.update_manifest(run_dir, statuss="typo")

    # ------------------------------------------------------------------ index
    def test_list_runs_newest_first_and_skips_junk(self) -> None:
        self._make_run("run-a")
        self._make_run("run-b")
        runstore.update_manifest(self.root / "run-b", notes="newer")  # bumps updated_at
        (self.root / "no-manifest").mkdir()
        corrupt = self.root / "corrupt"
        corrupt.mkdir()
        (corrupt / runstore.MANIFEST_FILENAME).write_text("[]", encoding="utf-8")

        listed = runstore.list_runs(self.root)
        ids = [m.run_id for m in listed]
        self.assertEqual(ids, ["run-b", "run-a"])

    def test_list_runs_filters_by_status(self) -> None:
        self._make_run("run-a", status="completed")
        self._make_run("run-b", status="interrupted")
        listed = runstore.list_runs(self.root, status="interrupted")
        self.assertEqual([m.run_id for m in listed], ["run-b"])

    def test_sweep_marks_only_running_as_interrupted(self) -> None:
        self._make_run("run-live", status="running")
        self._make_run("run-done", status="completed")
        swept = runstore.sweep_interrupted(self.root)
        self.assertEqual(swept, ["run-live"])
        by_id = {m.run_id: m for m in runstore.list_runs(self.root)}
        self.assertEqual(by_id["run-live"].status, "interrupted")
        self.assertEqual(by_id["run-done"].status, "completed")
        self.assertEqual(runstore.sweep_interrupted(self.root), [])  # idempotent

    # ---------------------------------------------------------------- resume
    def test_resume_cache_round_trip(self) -> None:
        run_dir = self._make_run("run-1")
        runstore.write_resume_cache(
            run_dir,
            namespace={"chunks": "sha256:aa", "kg": "sha256:bb"},
            workspace={"AGENTS.md": "cc"},
            namespace_complete=True,
        )
        cache = runstore.read_resume_cache(run_dir)
        self.assertIsNotNone(cache)
        assert cache is not None
        self.assertEqual(cache["namespace"]["chunks"], "sha256:aa")
        self.assertTrue(cache["namespace_complete"])

    def test_resume_cache_missing_returns_none(self) -> None:
        run_dir = self._make_run("run-1")
        self.assertIsNone(runstore.read_resume_cache(run_dir))

    # ------------------------------------------------------------------- fork
    def test_fork_is_provenance_not_continuation(self) -> None:
        source = self._make_run("run-src", status="completed")
        (source / "trajectory.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
        child = runstore.fork_run("run-src", "run-new", path=self.root)
        self.assertEqual(child.parent_run_id, "run-src")
        self.assertEqual(child.query, "a query")
        self.assertEqual(child.status, "running")
        fork_dir = self.root / "run-new"
        self.assertTrue((fork_dir / runstore.MANIFEST_FILENAME).is_file())
        # A fork starts its own log: copying the parent's would put two histories
        # in one trajectory and break replay/fold.
        self.assertFalse((fork_dir / "trajectory.jsonl").exists())
        self.assertTrue((source / "trajectory.jsonl").is_file())
        self.assertEqual((source / "trajectory.jsonl").read_text(encoding="utf-8"), '{"a": 1}\n')
        # ... and it cannot silently overwrite an existing run.
        with self.assertRaises(FileExistsError):
            runstore.fork_run("run-src", "run-new", path=self.root)


if __name__ == "__main__":
    unittest.main()
