"""Run store: manifest, resume cache, listing, and forking (M8.5 / P-01b).

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4n; ``MVP Build Roadmap.md`` M8.5.

Why this module exists
----------------------
The harness already writes durable artifacts (``trajectory.jsonl``,
``tasks.json``, ``work/``, ``answer.md``) but nothing answers *which runs exist,
what state are they in, and can I continue one*. This module owns that index and
the two small files that make a run a **session**:

::

    <runs_root>/<run_id>/
      manifest.json     identity + status + index   <- THIS MODULE (authoritative)
      trajectory.jsonl  the event log               <- the session (Trajectory)
      tasks.json        the ledger                  <- TaskLedger
      resume.json       cache: emit map + workspace hashes  <- THIS MODULE
      work/             sandbox workspace (context files, emits/)
      answer.md         the answer

Design (see §4n for the comparison that produced it)
-----------------------------------------------------
- **The log is authoritative.** Messages, budget counters, the next
  ``step_idx``, and the emit map are all derivable from ``trajectory.jsonl``
  (:meth:`lens.observability.trajectory.Trajectory.rebuild`). Nothing here
  duplicates the conversation — that is how Claude Code ends up copying history
  on every resume.
- **``manifest.json`` is identity, not state.** It is written once and updated
  on status changes; it is what ``--list`` and the M10 UI read.
- **``resume.json`` is a cache**, rebuildable from the log: the emit map
  (``variable -> blob ref``) plus workspace file hashes. If it is missing or
  corrupt, the run is still resumable (rebuilt), only slower.
- **Files, not a database.** A directory scan over a manifest handles 10^2-10^3
  runs; SQLite/Postgres would add a dependency and a migration risk for a query
  nobody needs yet.
- **Tolerant readers.** Unknown manifest keys are preserved in ``extra`` and
  re-emitted on write, so adding a field in a later version does not make old
  runs unreadable (the OpenHands lesson: a renamed event class made every old
  conversation unloadable).

Contract tests: ``tests/test_runstore.py`` (no key, no DB, no Docker).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# Bumped only when a reader can no longer understand an older file. Additive
# fields do NOT need a bump (readers keep them in ``extra``).
SCHEMA_VERSION = 1

MANIFEST_FILENAME = "manifest.json"
RESUME_FILENAME = "resume.json"

# ``running`` is the only status that a startup sweep may rewrite: a process that
# died leaves it behind, and calling it ``interrupted`` is honest rather than
# guessing it finished (opencode's "perpetually working" session is the failure
# mode to avoid).
RunStatus = Literal["running", "completed", "partial", "failed", "interrupted"]

VALID_STATUSES: frozenset[str] = frozenset(
    {"running", "completed", "partial", "failed", "interrupted"}
)


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (matches the trajectory)."""
    return datetime.now(timezone.utc).isoformat()


def runs_root(path: str | Path | None = None) -> Path:
    """Resolve the runs root.

    Read from ``lens.config`` **at call time** (not import time) so tests can
    redirect it, and so ``LENS_RUNS_ROOT`` keeps working.
    """
    if path is not None:
        return Path(path)
    from lens import config

    return Path(config.LENS_RUNS_ROOT_PATH)


@dataclass
class RunManifest:
    """One run's identity + status. Written to ``<run_dir>/manifest.json``.

    Everything here is either identity (who/what/when) or a bounded summary.
    Conversation state is **not** stored here; it is folded from the log.
    """

    run_id: str
    query: str
    schema_version: int = SCHEMA_VERSION
    status: RunStatus = "running"
    parent_run_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    # role -> requested model id (+ served snapshot when known), e.g.
    # {"root": "openai/gpt-5", "sub": "openai/gpt-5-mini", "jev": "typesafe/jev-1.13"}
    models: dict[str, str] = field(default_factory=dict)
    # The caps in force for this run (so a resumed run is not judged by today's defaults).
    caps: dict[str, Any] = field(default_factory=dict)
    # From Trajectory.totals(): cost_usd / tokens_in / tokens_out / calls.
    totals: dict[str, Any] = field(default_factory=dict)
    # Provenance for reproducibility (E-04): commit, image digest, python, recorded flag.
    created_by: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    last_seq: int = 0
    # name -> path relative to run_dir (answer.md, trajectory.jsonl, ...)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: str | None = None
    # Unknown keys from a newer (or older) schema. Preserved on rewrite.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON object written to disk.

        Contract: field order as declared, ``extra`` merged at the top level
        (so unknown keys round-trip), and ``extra`` itself not emitted as a
        nested key. No ``None`` value is dropped except where it would collide
        with a declared field.
        """
        raise NotImplementedError("M8.5 scaffold: implement RunManifest.to_dict")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, run_id: str | None = None) -> "RunManifest":
        """Rebuild a manifest, keeping unrecognised keys in ``extra``.

        ``run_id`` may be passed to fill in a missing/renamed id (the directory
        name is the fallback truth). A missing ``schema_version`` means 1.
        """
        raise NotImplementedError("M8.5 scaffold: implement RunManifest.from_dict")

    @classmethod
    def new(
        cls,
        run_id: str,
        query: str,
        *,
        parent_run_id: str | None = None,
        models: dict[str, str] | None = None,
        caps: dict[str, Any] | None = None,
        created_by: dict[str, Any] | None = None,
    ) -> "RunManifest":
        """Create a ``running`` manifest with both timestamps set."""
        raise NotImplementedError("M8.5 scaffold: implement RunManifest.new")


# ------------------------------------------------------------------ manifest io
def write_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    """Atomically write ``manifest.json`` (temp file + ``os.replace``).

    Atomic matters: a crash mid-write must not leave a truncated manifest, which
    would make the run unlistable *and* unresumable.
    """
    raise NotImplementedError("M8.5 scaffold: implement write_manifest")


def read_manifest(run_dir: str | Path) -> RunManifest | None:
    """Read ``manifest.json``; ``None`` when it is missing or unreadable.

    Tolerant by design: a corrupt manifest must not break ``--list`` for every
    other run (skipping one run is the correct failure).
    """
    raise NotImplementedError("M8.5 scaffold: implement read_manifest")


def update_manifest(run_dir: str | Path, **fields: Any) -> RunManifest:
    """Read-modify-write one manifest and refresh ``updated_at``.

    Unknown ``fields`` raise ``TypeError`` (a typo must not silently do nothing);
    an invalid ``status`` raises ``ValueError``. Missing manifest -> ``FileNotFoundError``.
    """
    raise NotImplementedError("M8.5 scaffold: implement update_manifest")


def list_runs(path: str | Path | None = None, *, status: str | None = None) -> list[RunManifest]:
    """Return manifests newest-first (``updated_at`` descending).

    Skip anything that is not a run directory (no manifest, or unreadable).
    ``status`` filters exactly (e.g. ``"interrupted"``).
    """
    raise NotImplementedError("M8.5 scaffold: implement list_runs")


def sweep_interrupted(path: str | Path | None = None) -> list[str]:
    """Mark every ``running`` run as ``interrupted``; return the affected ids.

    Call once at process start, before any new run. A run left ``running`` by a
    killed process is exactly what the resume path needs to find.
    """
    raise NotImplementedError("M8.5 scaffold: implement sweep_interrupted")


# ------------------------------------------------------------- resume cache io
def write_resume_cache(
    run_dir: str | Path,
    *,
    namespace: dict[str, str],
    workspace: dict[str, str],
    namespace_complete: bool,
) -> Path:
    """Write ``resume.json`` (atomic). A **cache**, not the source of truth.

    ``namespace`` maps a sandbox variable name -> ``emit()`` ref (``sha256:...``);
    ``workspace`` maps a file name -> sha256 of its bytes. ``namespace_complete``
    is ``False`` when the last turn left a variable that was never emitted, so a
    resume must tell the model to re-derive it instead of pretending it exists.
    """
    raise NotImplementedError("M8.5 scaffold: implement write_resume_cache")


def read_resume_cache(run_dir: str | Path) -> dict[str, Any] | None:
    """Read ``resume.json``; ``None`` when missing/corrupt (caller rebuilds)."""
    raise NotImplementedError("M8.5 scaffold: implement read_resume_cache")


# ---------------------------------------------------------------------- fork
def fork_run(
    source_run_id: str,
    new_run_id: str,
    *,
    path: str | Path | None = None,
    query: str | None = None,
) -> RunManifest:
    """Create a new run that records ``parent_run_id = source_run_id``.

    A fork is **provenance, not continuation**: the new run gets its own
    ``trajectory.jsonl`` and its own ``work/`` (context files are rewritten from
    the query), so replay/fold never see two histories in one log. Use
    ``--resume`` to *continue* a run; use ``--fork`` to run the same review under
    a different arm (caps, ``KG_ENABLED``, depth) and keep the lineage.

    Inherits the query (unless overridden), ``models`` and ``created_by``.
    Refuses to overwrite an existing ``new_run_id`` (``FileExistsError``).
    """
    raise NotImplementedError("M8.5 scaffold: implement fork_run")


__all__ = [
    "MANIFEST_FILENAME",
    "RESUME_FILENAME",
    "SCHEMA_VERSION",
    "VALID_STATUSES",
    "RunManifest",
    "RunStatus",
    "fork_run",
    "list_runs",
    "now_iso",
    "read_manifest",
    "read_resume_cache",
    "runs_root",
    "sweep_interrupted",
    "update_manifest",
    "write_manifest",
    "write_resume_cache",
]
