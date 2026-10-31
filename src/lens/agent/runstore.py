"""Run store: manifest, resume cache, listing, and forking.

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

Design
-----------------------------------------------------
- **The log is authoritative.** Messages, budget counters, the next
  ``step_idx``, and the emit map are all derivable from ``trajectory.jsonl``
  (:meth:`lens.observability.trajectory.Trajectory.rebuild`). Nothing here
  duplicates the conversation — that is how Claude Code ends up copying history
  on every resume.
- **``manifest.json`` is identity, not state.** It is written once and updated
  on status changes; it is what ``--list`` and the UI read.
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

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
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
    # Provenance for reproducibility : commit, image digest, python, recorded flag.
    created_by: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    last_seq: int = 0
    # Why the loop stopped. ``iterations``/``tokens``/``cost``/``backtracks`` are
    # the cap that bound it, ``flat`` means the baseline had no loop to bound, and
    # ``error`` means it failed. A report needs this next to any accuracy number:
    # a cap-bound run is a different result from a finished one.
    stop_reason: str | None = None
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
        declared: dict[str, Any] = {
            "run_id": self.run_id,
            "query": self.query,
            "schema_version": self.schema_version,
            "status": self.status,
            "parent_run_id": self.parent_run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "models": self.models,
            "caps": self.caps,
            "totals": self.totals,
            "created_by": self.created_by,
            "iterations": self.iterations,
            "last_seq": self.last_seq,
            "stop_reason": self.stop_reason,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }

        # Unknown keys first, declared keys win: a newer field that happens to
        # share a name with an old one must not shadow the value we just set.
        return {**self.extra, **declared}

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, run_id: str | None = None) -> "RunManifest":
        """Rebuild a manifest, keeping unrecognised keys in ``extra``.

        ``run_id`` may be passed to fill in a missing/renamed id (the directory
        name is the fallback truth). A missing ``schema_version`` means 1, and a
        missing ``status`` is ``running`` — reading tolerates a partial file, so
        a hand-edited manifest does not have to be complete.
        """
        if not isinstance(payload, dict):
            raise TypeError(f"manifest payload must be a dict, got {type(payload).__name__}")

        declared = {f.name for f in dataclass_fields(cls)} - {"extra"}
        extra = {key: value for key, value in payload.items() if key not in declared}

        return cls(
            run_id=run_id or payload.get("run_id") or "",
            query=payload.get("query") or "",
            schema_version=payload.get("schema_version") or SCHEMA_VERSION,
            status=payload.get("status") or "running",
            parent_run_id=payload.get("parent_run_id"),
            created_at=payload.get("created_at") or "",
            updated_at=payload.get("updated_at") or "",
            models=payload.get("models") or {},
            caps=payload.get("caps") or {},
            totals=payload.get("totals") or {},
            created_by=payload.get("created_by") or {},
            iterations=payload.get("iterations") or 0,
            last_seq=payload.get("last_seq") or 0,
            stop_reason=payload.get("stop_reason"),
            artifacts=payload.get("artifacts") or {},
            notes=payload.get("notes"),
            extra=extra,
        )

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
        """Create a ``running`` manifest with both timestamps set.

        Also records ``extra["blob_format"]``: the blob-store rule this run writes
        under. It cannot be upgraded later without rewriting the bytes it points
        at, which is why it is set once, here.
        """
        timestamp = now_iso()

        from lens import config

        return cls(
            run_id=run_id,
            query=query,
            parent_run_id=parent_run_id,
            created_at=timestamp,
            updated_at=timestamp,
            models=models or {},
            caps=caps or {},
            created_by=created_by or {},
            extra={"blob_format": config.BLOB_FORMAT_VERSION},
        )


#: Fields ``update_manifest`` may change. ``extra`` is **not** updatable: it is
#: the round-trip bucket for keys this version does not know, and letting a caller
#: write it would erase them (two ways to write one thing). ``run_id`` is excluded
#: too — it is the directory name, so a rename is a directory move, not a field.
UPDATABLE: frozenset[str] = frozenset(
    {f.name for f in dataclass_fields(RunManifest)} - {"extra", "run_id"}
)


# ------------------------------------------------------------------ manifest io
def write_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    """Atomically write ``manifest.json`` (temp file + ``os.replace``).

    Atomic matters: a crash mid-write must not leave a truncated manifest, which
    would make the run unlistable *and* unresumable.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    temp_file_path = run_dir / "manifest.json.tmp"

    with open(temp_file_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())  # force data out of the OS buffers to disk

    # Atomic on one filesystem: readers see either the old or the new file.
    os.replace(temp_file_path, manifest_path)

    return manifest_path


def read_manifest(run_dir: str | Path) -> RunManifest | None:
    """Read ``manifest.json``; ``None`` when it is missing or unreadable.

    Tolerant by design: a corrupt manifest must not break ``--list`` for every
    other run (skipping one run is the correct failure). "Unreadable" covers a
    truncated file, a JSON document that is not an object, and an OS error — all
    of them mean *this directory is not a run*, not *the store is broken*.
    """
    manifest_path = Path(run_dir) / MANIFEST_FILENAME

    if not manifest_path.is_file():
        return None

    try:
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)

        return RunManifest.from_dict(payload, run_id=Path(run_dir).name)
    except (OSError, ValueError, TypeError):
        return None


def update_manifest(run_dir: str | Path, **fields: Any) -> RunManifest:
    """Read-modify-write one manifest and refresh ``updated_at``.

    Unknown ``fields`` raise ``TypeError`` (a typo must not silently do nothing);
    an invalid ``status`` raises ``ValueError``. Missing manifest ->
    ``FileNotFoundError``.

    Validate first, then mutate, then write: there is exactly **one** return path,
    which is what keeps "every branch returns a manifest" true without a pile of
    ``try``/``except`` blocks.
    """
    manifest = read_manifest(run_dir)

    if manifest is None:
        raise FileNotFoundError(f"no readable {MANIFEST_FILENAME} in {run_dir}")

    for key in fields:
        if key not in UPDATABLE:
            raise TypeError(f"unknown manifest field {key!r}")

    if "status" in fields and fields["status"] not in VALID_STATUSES:
        raise ValueError(
            f"invalid status {fields['status']!r}; expected one of {sorted(VALID_STATUSES)}"
        )

    for key, value in fields.items():
        setattr(manifest, key, value)

    manifest.updated_at = now_iso()
    write_manifest(run_dir, manifest)

    return manifest


def list_runs(path: str | Path | None = None, *, status: str | None = None) -> list[RunManifest]:
    """Return manifests newest-first (``updated_at`` descending).

    Skip anything that is not a run directory (no manifest, or unreadable).
    ``status`` filters exactly (e.g. ``"interrupted"``).

    Ordering compares the ISO strings, not parsed datetimes: every timestamp is
    written by :func:`now_iso` in one format (so lexicographic order is
    chronological), while ``fromisoformat`` raises on an empty value and cannot
    compare a naive with an aware datetime.
    """
    root = runs_root(path)

    if not root.is_dir():
        return []

    manifests: list[RunManifest] = []

    for entry in root.iterdir():
        if not entry.is_dir():
            continue

        manifest = read_manifest(entry)

        if manifest is None:
            continue

        if status is not None and manifest.status != status:
            continue

        manifests.append(manifest)

    return sorted(manifests, key=lambda manifest: manifest.updated_at, reverse=True)


def sweep_interrupted(
    path: str | Path | None = None,
    *,
    stale_after_s: float | None = None,
) -> list[str]:
    """Mark stale ``running`` runs as ``interrupted``; return the affected ids.

    Call once at process start, before any new run. A run left ``running`` by a
    killed process is exactly what the resume path needs to find.

    **The log is the heartbeat.** A run whose ``trajectory.jsonl`` was written
    more recently than ``stale_after_s`` (``config.RUN_HEARTBEAT_STALE_S``,
    five minutes by default) is left alone: it may be a live process in another
    terminal, and marking it interrupted would let a second ``--resume`` append to
    a log something else is still writing. A run with no log at all has nothing to
    resume, so it is swept immediately.

    ``updated_at`` is **not** touched: a sweep is not activity, and bumping it
    would reorder ``--list`` on every startup so the ordering stops meaning
    "when this run last did something".
    """
    root = runs_root(path)

    if not root.is_dir():
        return []

    from lens import config

    if stale_after_s is None:
        stale_after_s = config.RUN_HEARTBEAT_STALE_S

    affected: list[str] = []

    for entry in root.iterdir():
        if not entry.is_dir():
            continue

        manifest = read_manifest(entry)

        if manifest is None or manifest.status != "running":
            continue

        log = entry / config.TRAJECTORY_FILENAME
        if log.is_file() and (time.time() - log.stat().st_mtime) < stale_after_s:
            continue

        manifest.status = "interrupted"
        write_manifest(entry, manifest)
        affected.append(manifest.run_id)

    return affected


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
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_path = run_dir / "resume.json"
    temp_file_path = run_dir / "resume.json.tmp"

    resume_dict = {
        "schema_version": SCHEMA_VERSION,
        "written_at": now_iso(),
        "namespace": namespace,
        "workspace": workspace,
        "namespace_complete": bool(namespace_complete),
    }

    with open(temp_file_path, "w", encoding="utf-8") as f:
        json.dump(resume_dict, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())  # force data out of the OS buffers to disk

    os.replace(temp_file_path, resume_path)

    return resume_path


def read_resume_cache(run_dir: str | Path) -> dict[str, Any] | None:
    """Read ``resume.json``; ``None`` when missing/corrupt (caller rebuilds).

    It is a **cache**: any doubt means "rebuild from the log", never an
    exception. A payload without a ``namespace`` mapping is not this schema.
    """
    resume_path = Path(run_dir) / RESUME_FILENAME

    if not resume_path.is_file():
        return None

    try:
        with open(resume_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None

    if not isinstance(payload, dict) or not isinstance(payload.get("namespace"), dict):
        return None

    return payload


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
    root = runs_root(path)

    source_manifest = read_manifest(root / source_run_id)

    if source_manifest is None:
        raise FileNotFoundError(f"no readable run {source_run_id!r} under {root}")

    new_run_path = root / new_run_id

    if new_run_path.exists():
        raise FileExistsError(f"{new_run_path} already exists")

    # ``caps`` is deliberately **not** inherited: a fork is usually the same
    # review under a different arm (different caps, KG on/off, depth), so copying
    # the parent's caps would silently run the same configuration twice.
    new_manifest = RunManifest.new(
        new_run_id,
        query if query is not None else source_manifest.query,
        parent_run_id=source_run_id,
        models=dict(source_manifest.models),
        created_by=dict(source_manifest.created_by),
    )

    write_manifest(new_run_path, new_manifest)

    return new_manifest


# ------------------------------------------------------------------- bundle
# A bundle is a **self-contained, copyable directory** (``<run_id>.lens``) with a
# run's artifacts and only the blobs its events reference. Blobs live in one
# global ``<runs_root>/blobs/`` store (deduped across runs), so copying a run
# directory alone leaves every ref dangling — that is why this exists.
#
# ``<run_id>.lens/``
#   bundle.json        index: schema_version, run_id, created_at, blob list +
#                      sizes, trajectory sha256, source manifest snapshot, missing
#   manifest.json      copied verbatim
#   trajectory.jsonl   copied verbatim
#   tasks.json, answer.md   copied when present
#   work/              the sandbox workspace minus ``emits/`` (those are blobs);
#                      ``query.json`` travels here, where it actually lives
#   blobs/<sha256>     only the referenced, content-addressed payloads
#
# Import re-hashes every blob against its filename, so a truncated or edited
# bundle is rejected instead of silently rendering wrong evidence.

BUNDLE_EXT = ".lens"
BUNDLE_INDEX = "bundle.json"
BUNDLE_RUN_FILES: tuple[str, ...] = (
    "manifest.json",
    "trajectory.jsonl",
    "tasks.json",
    "answer.md",
    "resume.json",
)
# Every event field that holds a blob ref. Keep in sync with ``log_event``.
REF_FIELDS: tuple[str, ...] = (
    "prompt_ref",
    "completion_ref",
    "result_ref",
    "code_ref",
    "stdout_ref",
    "stderr_ref",
    "args_ref",
)


def _sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes (16 MiB chunks: blobs can be large)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_refs(trajectory_path: str | Path) -> list[str]:
    """Return every distinct blob ref used by the events in ``trajectory_path``.

    Also includes the refs in a sibling ``resume.json`` (the namespace map), so a
    resumed run exports everything its restore block needs. Order: first
    appearance. Malformed lines are skipped (a damaged tail must not block an
    export of the rest).

    Order matters for a deterministic bundle: the same trajectory must produce the
    same ref list, so the export's blob set is reproducible.
    """
    path = Path(trajectory_path)
    refs: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value.startswith("sha256:") and value not in seen:
            seen.add(value)
            refs.append(value)

    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(event, dict):
                continue

            for field_name in REF_FIELDS:
                add(event.get(field_name))

    cache = read_resume_cache(path.parent)

    if cache is not None:
        for ref in (cache.get("namespace") or {}).values():
            add(ref)

    return refs


def export_run(
    run_id: str,
    destination: str | Path,
    *,
    path: str | Path | None = None,
) -> Path:
    """Write ``<destination>/<run_id>.lens`` and return its path.

    Copies the artifacts in ``BUNDLE_RUN_FILES`` (when present), ``work/`` minus
    ``emits/`` (emits are blobs and would double the size), and the **referenced**
    blobs only. ``bundle.json`` records each blob's size plus the trajectory's
    sha256, and lists refs that were already missing in the source (``missing``)
    instead of failing.

    Contract: deterministic (same input -> same blob set), no network, and the
    source run is never modified. Refuses an existing destination bundle
    (``FileExistsError``) so a re-export cannot silently merge two runs.

    ``bundle.json`` is written **last**: a bundle without an index is incomplete,
    and a half-written one fails ``verify_bundle`` instead of looking importable.
    """
    root = runs_root(path)
    run_dir = root / run_id

    if not run_dir.is_dir():
        raise FileNotFoundError(f"no run {run_id!r} under {root}")

    bundle = Path(destination) / f"{run_id}{BUNDLE_EXT}"

    if bundle.exists():
        raise FileExistsError(f"{bundle} already exists")

    bundle.mkdir(parents=True)
    (bundle / "blobs").mkdir()

    for name in BUNDLE_RUN_FILES:
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, bundle / name)

    work_dir = run_dir / "work"
    if work_dir.is_dir():
        shutil.copytree(work_dir, bundle / "work", ignore=shutil.ignore_patterns("emits"))

    blobs: dict[str, int] = {}
    missing: list[str] = []
    total_bytes = 0

    for ref in referenced_refs(run_dir / "trajectory.jsonl"):
        digest = ref.split(":", 1)[1]
        source = root / "blobs" / digest

        if not source.is_file():
            # Already missing in the source (e.g. an aborted run): record it, do
            # not fail — the bundle is still worth more than nothing.
            missing.append(ref)
            continue

        shutil.copy2(source, bundle / "blobs" / digest)
        size = (bundle / "blobs" / digest).stat().st_size
        blobs[digest] = size
        total_bytes += size

    trajectory = run_dir / "trajectory.jsonl"
    manifest = read_manifest(run_dir)

    index = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": now_iso(),
        "trajectory_sha256": _sha256_file(trajectory) if trajectory.is_file() else None,
        "blobs": blobs,
        "blob_bytes": total_bytes,
        "missing": missing,
        "manifest": manifest.to_dict() if manifest is not None else None,
    }

    with open(bundle / BUNDLE_INDEX, "w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())

    return bundle


def verify_bundle(bundle: str | Path) -> dict[str, Any]:
    """Re-hash a bundle and return ``{"ok", "errors", "blobs", "bytes", "run_id"}``.

    An error (not an exception) per damaged item: missing blob, hash mismatch,
    index/trajectory sha mismatch, unreadable ``bundle.json``. This is what makes
    "tamper-evident recording" a true statement in the demo rather than a claim.

    The blob's **filename is its hash**, so verification is just "hash the bytes
    and compare with the name" — no separate manifest of hashes to trust.

    The directory listing must also equal the index: a file under ``blobs/`` that
    the index does not name is unverified bytes travelling inside a bundle that
    claims to be verified.
    """
    bundle = Path(bundle)
    index_path = bundle / BUNDLE_INDEX

    if not index_path.is_file():
        return {
            "ok": False,
            "errors": [f"missing {BUNDLE_INDEX}"],
            "blobs": 0,
            "bytes": 0,
            "run_id": None,
        }

    try:
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, ValueError) as exc:
        return {
            "ok": False,
            "errors": [f"unreadable {BUNDLE_INDEX}: {exc}"],
            "blobs": 0,
            "bytes": 0,
            "run_id": None,
        }

    errors: list[str] = []
    blob_index: dict[str, Any] = index.get("blobs") or {}
    total_bytes = 0
    verified = 0

    for digest, expected_size in blob_index.items():
        blob_path = bundle / "blobs" / digest

        if not blob_path.is_file():
            errors.append(f"missing blob {digest}")
            continue

        if _sha256_file(blob_path) != digest:
            errors.append(f"blob {digest} does not match its name")
            continue

        size = blob_path.stat().st_size
        if expected_size is not None and size != expected_size:
            errors.append(f"blob {digest} is {size} bytes, index says {expected_size}")
            continue

        verified += 1
        total_bytes += size

    expected_trajectory = index.get("trajectory_sha256")
    if expected_trajectory:
        trajectory = bundle / "trajectory.jsonl"
        actual = _sha256_file(trajectory) if trajectory.is_file() else None
        if actual != expected_trajectory:
            errors.append("trajectory.jsonl does not match the index")

    blob_dir = bundle / "blobs"
    if blob_dir.is_dir():
        for extra in sorted(path.name for path in blob_dir.iterdir() if path.is_file()):
            if extra not in blob_index:
                errors.append(f"unexpected file blobs/{extra} is not in the index")

    return {
        "ok": not errors,
        "errors": errors,
        "blobs": verified,
        "bytes": total_bytes,
        "run_id": index.get("run_id"),
    }


def import_run(bundle: str | Path, *, path: str | Path | None = None) -> RunManifest:
    """Install a bundle into ``<runs_root>/<run_id>/`` and return its manifest.

    Steps: ``verify_bundle`` first (any error -> ``ValueError`` listing them, no
    partial install), then blobs into ``<runs_root>/blobs/<sha>`` (skip ones
    already present — content-addressed, so identical bytes), then the run files
    and ``work/``. An existing ``run_id`` is a ``FileExistsError``.

    The imported manifest keeps its original ``created_at`` and marks itself
    imported via ``extra["imported_at"]`` (provenance, not a rewrite).
    """
    bundle = Path(bundle)
    report = verify_bundle(bundle)

    if not report["ok"]:
        # Nothing is copied before this point, so a bad bundle leaves no residue.
        raise ValueError("bundle failed verification: " + "; ".join(report["errors"]))

    with open(bundle / BUNDLE_INDEX, encoding="utf-8") as handle:
        index = json.load(handle)

    run_id = index.get("run_id")
    if not run_id:
        raise ValueError(f"{BUNDLE_INDEX} has no run_id")

    root = runs_root(path)
    run_dir = root / run_id

    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists")

    blobs_root = root / "blobs"
    blobs_root.mkdir(parents=True, exist_ok=True)

    for digest in index.get("blobs") or {}:
        destination = blobs_root / digest

        if destination.exists():
            continue  # content-addressed: the bytes are identical by definition

        shutil.copy2(bundle / "blobs" / digest, destination)

    run_dir.mkdir(parents=True)

    for name in BUNDLE_RUN_FILES:
        source = bundle / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)

    work_dir = bundle / "work"
    if work_dir.is_dir():
        shutil.copytree(work_dir, run_dir / "work")

    manifest = read_manifest(run_dir) or RunManifest.new(run_id, "")
    manifest.run_id = run_id
    manifest.extra["imported_at"] = now_iso()

    write_manifest(run_dir, manifest)

    return manifest


__all__ = [
    "BUNDLE_EXT",
    "BUNDLE_INDEX",
    "MANIFEST_FILENAME",
    "RESUME_FILENAME",
    "SCHEMA_VERSION",
    "VALID_STATUSES",
    "RunManifest",
    "RunStatus",
    "export_run",
    "fork_run",
    "import_run",
    "list_runs",
    "now_iso",
    "read_manifest",
    "read_resume_cache",
    "referenced_refs",
    "runs_root",
    "sweep_interrupted",
    "update_manifest",
    "verify_bundle",
    "write_manifest",
    "write_resume_cache",
]
