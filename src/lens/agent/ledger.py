"""Deterministic task ledger (``tasks.json``).

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §5.

The RLM updates the ledger through host tools (``ledger_*``), never by writing
free-form text. The orchestrator reloads it after a crash; every write is
atomic so a partial file cannot lose completed tasks.

Object identity — read this before touching the CRUD methods
------------------------------------------------------------
``self.tasks`` maps ``task_id -> Task`` and the **same** ``Task`` instance is
handed to every caller. ``update`` / ``add_evidence`` therefore mutate the
*stored* object in place: there is nothing to write back into the dict, only a
ledger to ``save()``. The price of that convenience is aliasing — whatever a
caller receives can mutate the ledger — so every serialisation boundary
(``Task.to_dict`` / ``snapshot``) **copies** the mutable fields
(``evidence_refs``). That is why ``to_dict`` builds a new list instead of
handing out ``task.evidence_refs``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TaskStatus = Literal["pending", "in_progress", "done", "failed"]

VALID_STATUSES: frozenset[str] = frozenset({"pending", "in_progress", "done", "failed"})

# Fields ``update()`` may change. ``task_id`` / ``parent_id`` are identity, the
# rest are derived, so anything outside this set is a programming error rather
# than a silent no-op. (A plain frozenset, not ``typing.Literal``: membership on
# a Literal silently returns False, which is exactly how a whitelist turns into
# a no-op.)
UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {"status", "attempts", "evidence_refs", "tokens", "wall_clock_s", "requirement"}
)


@dataclass
class Task:
    """One ledger row (schema fixed by Tech Stack §5)."""

    task_id: str
    requirement: str
    parent_id: str | None = None
    status: TaskStatus = "pending"
    attempts: int = 0
    evidence_refs: list[str] = field(default_factory=list)
    tokens: int = 0
    wall_clock_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the Tech Stack §5 JSON object for this task.

        ``evidence_refs`` is **copied**: this is a serialisation boundary, so a
        consumer (trajectory event, UI, test) must not be able to mutate the
        ledger through the dict it was given.
        """
        task_dict = {
            "task_id": self.task_id,
            "requirement": self.requirement,
            "parent_id": self.parent_id,
            "status": self.status,
            "attempts": self.attempts,
            "evidence_refs": list(self.evidence_refs),
            "tokens": self.tokens,
            "wall_clock_s": self.wall_clock_s,
        }

        return task_dict

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        """Rebuild a task from ``to_dict()`` output.

        Tolerant reader: a hand-edited or older row may omit optional fields, so
        every non-identity field falls back to its default instead of raising.
        """
        task_instance = Task(
            task_id=payload["task_id"],
            requirement=payload.get("requirement", ""),
            parent_id=payload.get("parent_id"),
            status=payload.get("status", "pending"),
            attempts=payload.get("attempts", 0),
            evidence_refs=list(payload.get("evidence_refs") or []),
            tokens=payload.get("tokens", 0),
            wall_clock_s=payload.get("wall_clock_s", 0.0),
        )

        return task_instance


class TaskLedger:
    """Read/write ``tasks.json`` with atomic replaces.

    Parameters
    ----------
    path:
        Destination file, normally ``<run_dir>/tasks.json``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.tasks: dict[str, Task] = {}
        self._counter = 0

    # ------------------------------------------------------------- lifecycle
    def load(self) -> list[Task]:
        """Load an existing ledger into ``self.tasks`` and return the tasks.

        A missing file is an **empty ledger, not an error** (a fresh run has no
        ``tasks.json``). The id counter is rebuilt from the ids actually present,
        so a resumed or hand-edited file cannot hand out a duplicate ``t<N>``.
        """
        if not self.path.is_file():
            return []

        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.tasks = {
            row["task_id"]: Task.from_dict(row)
            for row in payload
            if isinstance(row, dict) and "task_id" in row
        }
        self._counter = max((self._id_number(tid) for tid in self.tasks), default=0)

        return self.list()

    def save(self) -> None:
        """Atomically write the ledger (temp file + ``os.replace``).

        The temp file is a **sibling** of ``tasks.json`` (``tasks.json.tmp``) —
        ``self.path / ".tmp"`` would try to use the file as a directory. It must
        be on the same filesystem for ``os.replace`` to be atomic, hence a
        sibling and not ``tempfile``.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_file_path = self.path.with_name(self.path.name + ".tmp")

        with open(temp_file_path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())  # force data out of the OS buffers to disk

        # Atomic on one filesystem: readers see either the old or the new file.
        os.replace(temp_file_path, self.path)

    # ----------------------------------------------------------------- CRUD
    def create(
        self,
        requirement: str,
        *,
        parent_id: str | None = None,
        task_id: str | None = None,
    ) -> Task:
        """Create a ``pending`` task, store it, persist, and return it.

        ``task_id`` defaults to a deterministic next id (``t1``, ``t2``, ...).
        An explicit id that already exists is a ``ValueError`` (silent overwrite
        would lose the task's evidence).
        """
        if task_id is None:
            self._counter += 1
            task_id = f"t{self._counter}"
        elif task_id in self.tasks:
            raise ValueError(f"task_id {task_id!r} already exists in the ledger")
        else:
            # Keep the counter ahead of any explicit id so the next ``t<N>``
            # cannot collide with it.
            self._counter = max(self._counter, self._id_number(task_id))

        task_instance = Task(
            task_id=task_id,
            requirement=requirement,
            parent_id=parent_id,
        )
        self.tasks[task_id] = task_instance
        self.save()

        return task_instance

    def update(self, task_id: str, **fields: Any) -> Task:
        """Update whitelisted fields, persist, and return the task.

        Whitelist: ``status``, ``attempts``, ``evidence_refs``, ``tokens``,
        ``wall_clock_s``, ``requirement``.

        Raises
        ------
        KeyError
            Unknown ``task_id``.
        TypeError
            A field outside the whitelist (a typo must not be a silent no-op).
        ValueError
            ``status`` outside :data:`VALID_STATUSES`.

        The returned object **is** ``self.tasks[task_id]``: the mutation is in
        place, so the dict needs no re-assignment — only ``save()``.
        """
        if self.tasks.get(task_id) is None:
            raise KeyError(f"task id: {task_id} invalid and out of range")

        task = self.tasks[task_id]

        unknown = set(fields) - UPDATABLE_FIELDS
        if unknown:
            raise TypeError(f"cannot update {sorted(unknown)}; allowed: {sorted(UPDATABLE_FIELDS)}")

        if "status" in fields and fields["status"] not in VALID_STATUSES:
            raise ValueError(
                f"invalid status {fields['status']!r}; expected one of {sorted(VALID_STATUSES)}"
            )

        for key, value in fields.items():
            setattr(task, key, value)

        self.save()

        return task

    def add_evidence(self, task_id: str, refs: list[str]) -> Task:
        """Append evidence refs (deduplicated, order-preserving), persist, return.

        Mutates the stored task in place like :meth:`update`; the dedupe is a
        linear scan because a task accumulates a handful of refs, and preserving
        first-seen order is what makes the ledger diff readable.
        """
        task = self.tasks[task_id]  # KeyError carries the id

        for ref in refs:
            if ref not in task.evidence_refs:
                task.evidence_refs.append(ref)

        self.save()

        return task

    def list(self) -> list[Task]:
        """Return tasks in insertion order."""
        return list(self.tasks.values())

    # ----------------------------------------------------------------- export
    def snapshot(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable list for the trajectory / UI.

        Built from ``to_dict()``, so each row owns its ``evidence_refs`` list: a
        consumer cannot mutate the ledger through the snapshot.
        """
        return [task.to_dict() for task in self.tasks.values()]

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _id_number(task_id: str) -> int:
        """Return ``N`` for an id of the form ``t<N>``, else 0."""
        suffix = task_id[1:] if task_id.startswith("t") else ""

        return int(suffix) if suffix.isdigit() else 0
