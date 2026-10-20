r"""Contract tests for the deterministic task ledger (no key, no DB).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_ledger -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lens.agent.ledger import TaskLedger


class TaskLedgerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "tasks.json"
        self.ledger = TaskLedger(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_create_persists_and_reloads(self) -> None:
        task = self.ledger.create("collect supporting chunks")
        self.assertEqual(task.status, "pending")
        self.assertTrue(self.path.is_file())
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload[0]["task_id"], task.task_id)
        reloaded = TaskLedger(self.path)
        reloaded.load()
        self.assertEqual(len(reloaded.list()), 1)
        self.assertEqual(reloaded.list()[0].requirement, "collect supporting chunks")

    def test_update_status_and_attempts(self) -> None:
        task = self.ledger.create("find evidence")
        updated = self.ledger.update(task.task_id, status="in_progress", attempts=1)
        self.assertEqual(updated.status, "in_progress")
        self.assertEqual(updated.attempts, 1)
        reloaded = TaskLedger(self.path)
        reloaded.load()
        self.assertEqual(reloaded.list()[0].status, "in_progress")

    def test_invalid_status_is_rejected(self) -> None:
        task = self.ledger.create("x")
        with self.assertRaises(ValueError):
            self.ledger.update(task.task_id, status="bogus")

    def test_unknown_task_is_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.ledger.update("missing", status="done")

    def test_unknown_field_is_type_error(self) -> None:
        # A typo must fail loudly: the old whitelist silently ignored every key
        # (``"status" in Literal[...]`` is False), so updates were no-ops.
        task = self.ledger.create("x")
        with self.assertRaises(TypeError):
            self.ledger.update(task.task_id, statuss="done")
        self.assertEqual(self.ledger.list()[0].status, "pending")

    def test_load_populates_tasks_and_restores_the_id_counter(self) -> None:
        first = self.ledger.create("x")
        self.ledger.create("y")
        reloaded = TaskLedger(self.path)
        self.assertEqual(reloaded.list(), [])   # nothing loaded yet
        self.assertEqual(len(reloaded.load()), 2)
        self.assertEqual(len(reloaded.list()), 2)
        # The next id must not collide with an id that already exists.
        third = reloaded.create("z")
        self.assertNotIn(third.task_id, {first.task_id, "t2"})
        self.assertEqual(len(TaskLedger(self.path).load()), 3)

    def test_snapshot_does_not_alias_the_ledger(self) -> None:
        # Callers get dicts, not the stored objects: mutating a snapshot must not
        # change the ledger (the flip side of mutating tasks in place).
        task = self.ledger.create("x")
        self.ledger.add_evidence(task.task_id, ["c1"])
        row = self.ledger.snapshot()[0]
        row["evidence_refs"].append("injected")
        row["status"] = "done"
        self.assertEqual(self.ledger.list()[0].evidence_refs, ["c1"])
        self.assertEqual(self.ledger.list()[0].status, "pending")

    def test_save_leaves_no_temp_file(self) -> None:
        self.ledger.create("x")
        leftovers = [p.name for p in self.path.parent.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_add_evidence_dedupes_and_preserves_order(self) -> None:
        task = self.ledger.create("x")
        self.ledger.add_evidence(task.task_id, ["c2", "c1", "c2"])
        self.assertEqual(self.ledger.list()[0].evidence_refs, ["c2", "c1"])

    def test_snapshot_is_json_serializable(self) -> None:
        self.ledger.create("x")
        json.dumps(self.ledger.snapshot())


if __name__ == "__main__":
    unittest.main()
