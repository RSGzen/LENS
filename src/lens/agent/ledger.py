"""Deterministic task ledger (``tasks.json``).

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §5.

The RLM updates the ledger through host tools (``ledger_*``), never by writing
free-form text. The orchestrator reloads it after a crash; every write is
atomic so a partial file cannot lose completed tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TaskStatus = Literal["pending", "in_progress", "done", "failed"]

VALID_STATUSES: frozenset[str] = frozenset({"pending", "in_progress", "done", "failed"})


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
        """Return the Tech Stack §5 JSON object for this task."""
        raise NotImplementedError("M8 scaffold: implement Task.to_dict")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        """Rebuild a task from ``to_dict()`` output."""
        raise NotImplementedError("M8 scaffold: implement Task.from_dict")


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

    # ------------------------------------------------------------- lifecycle
    def load(self) -> list[Task]:
        """Load an existing ledger (missing file -> empty ledger, no error)."""
        raise NotImplementedError("M8 scaffold: implement TaskLedger.load")

    def save(self) -> None:
        """Atomically write the ledger (temp file + ``os.replace``)."""
        raise NotImplementedError("M8 scaffold: implement TaskLedger.save")

    # ----------------------------------------------------------------- CRUD
    def create(
        self,
        requirement: str,
        *,
        parent_id: str | None = None,
        task_id: str | None = None,
    ) -> Task:
        """Create a ``pending`` task and persist it.

        ``task_id`` defaults to a deterministic next id (``t1``, ``t2``, ...).
        """
        raise NotImplementedError("M8 scaffold: implement TaskLedger.create")

    def update(self, task_id: str, **fields: Any) -> Task:
        """Update whitelisted fields, persist, and return the task.

        Whitelist: ``status``, ``attempts``, ``evidence_refs``, ``tokens``,
        ``wall_clock_s``, ``requirement``. An invalid status is a ``ValueError``.
        Unknown task ids are a ``KeyError``.
        """
        raise NotImplementedError("M8 scaffold: implement TaskLedger.update")

    def add_evidence(self, task_id: str, refs: list[str]) -> Task:
        """Append evidence refs (deduplicated, order-preserving) and persist."""
        raise NotImplementedError("M8 scaffold: implement TaskLedger.add_evidence")

    def list(self) -> list[Task]:
        """Return tasks in insertion order."""
        raise NotImplementedError("M8 scaffold: implement TaskLedger.list")

    # ----------------------------------------------------------------- export
    def snapshot(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable list for the trajectory / UI."""
        raise NotImplementedError("M8 scaffold: implement TaskLedger.snapshot")
