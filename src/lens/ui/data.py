"""Read-only data access for the M10 UI — no Streamlit, no key, no DB, no Docker.

Spec: ``MVP Build Roadmap.md`` §4 M10a/M10b; contract: ``Tech Stack (Draft).md`` §4p.

Why this module exists
----------------------
The UI adds **no pipeline logic** (T-06). Everything it shows is already on disk
in ``runs/<run_id>/``: the manifest (identity + status), ``trajectory.jsonl``
(folded by :func:`lens.observability.replay.replay_turns`), ``answer.md``, and the
content-addressed blobs behind every tool result. This module is the one place
that reads them, so the Streamlit script stays presentation only and the whole
read path is testable with no browser and no Streamlit installed.

Two rules that are easy to get wrong and expensive to debug:

1. **Never build a :class:`~lens.observability.trajectory.Trajectory` for an id
   that is not a run.** Its constructor ``mkdir``s ``runs/<id>/`` and
   ``runs/blobs/``, so a stale URL parameter would silently *create* an empty run.
   :func:`load_run` reads ``manifest.json`` first and answers ``None``.
2. **The cap comes from the manifest, not from today's ``config``.** Runs
   recorded before LOG-120 carry ``caps.cost_usd = 0.5`` while the default is now
   ``1.00``; reading the config would make an M9 run look half spent.

Contract tests: ``tests/test_ui_data.py`` (no streamlit, no key, no DB, no Docker).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lens.agent.runstore import (
    RunManifest,
    list_runs,
    read_manifest,
    runs_root,
)
from lens.observability.citations import CitationReport, resolve_citations
from lens.observability.replay import Turn, replay_summary, replay_turns
from lens.observability.trajectory import Trajectory

#: The 5.5 % OpenRouter credit-load fee. A *reported* ``$/query`` includes it
#: (R4); the trajectory's ``cost_usd`` is raw. Shown as a second column, never
#: folded into the raw number the log holds.
CREDIT_FEE = 1.055

#: Tools whose payload hydrates chunks. The same pair ``citations.py`` trusts: an
#: id list from ``deep_match`` names candidates, not evidence the model could read.
CHUNK_TOOLS = frozenset({"rerank_chunks", "fetch_chunks"})

#: Local display names live beside the blob store, not inside a run directory.
ALIASES_FILENAME = "aliases.json"

#: Long enough for "Tier-1 anchor, gpt-5, aggregation query", short enough to render.
ALIAS_MAX_CHARS = 64


def is_run_id(run_id: str | None) -> bool:
    """Return ``True`` when ``run_id`` is a plain directory name under the runs root.

    The run id travels through a URL query parameter, so it is untrusted input.
    A ``.``/``..``/path component is refused here rather than resolved, because
    every reader below joins it onto the runs root.
    """
    if not run_id or len(run_id) > 200:
        return False

    return not any(char in run_id for char in ("/", "\\", "\x00")) and run_id not in {".", ".."}


def cap_of(manifest: RunManifest) -> float | None:
    """Return the per-query cost cap **recorded** for the run, or ``None``.

    ``None`` means the manifest predates the cap being recorded: the meter must
    then say "cap not recorded" instead of comparing against today's default,
    which would be a claim about a budget the run never had (R4).
    """
    recorded = (manifest.caps or {}).get("cost_usd")

    try:
        return float(recorded)
    except (TypeError, ValueError):
        return None


@dataclass
class RunView:
    """Everything one run shows: identity, turns, meters, citations, answer."""

    run_id: str
    manifest: RunManifest
    turns: list[Turn]
    summary: dict[str, Any]
    citations: CitationReport
    answer: str
    #: turn number -> chunk rows hydrated by that turn's tool calls.
    evidence: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    has_sources: bool = False
    has_report: bool = False

    @property
    def cap_usd(self) -> float | None:
        """The run's recorded cost cap (``None`` = not recorded)."""
        return cap_of(self.manifest)

    @property
    def cost_usd(self) -> float:
        """Chat + gate spend (``replay_summary``'s total, matching ``totals()``)."""
        return float(self.summary.get("cost_usd") or 0.0)

    @property
    def cost_with_fee(self) -> float:
        """The reported ``$/query`` (raw + the 5.5 % credit-load fee)."""
        return self.cost_usd * CREDIT_FEE

    @property
    def tokens_in_series(self) -> list[int]:
        """Prompt tokens per turn — the context-growth chart (§4n.1)."""
        return [turn.tokens_in or 0 for turn in self.turns]

    @property
    def missing_refs(self) -> int:
        """Blobs this bundle cannot resolve (0 for an intact recording)."""
        return int(self.summary.get("missing_refs") or 0)


def run_rows(status: str | None = None) -> list[dict[str, Any]]:
    """Return one flat row per run for the session-list table, newest first.

    ``status`` filters exactly (``completed`` / ``partial`` / ``failed`` /
    ``interrupted`` / ``running``); ``None`` means every run. A failed or partial
    run is a row like any other — it is labelled, never hidden. ``name`` is the
    local display name (empty when the run has never been renamed).
    """
    aliases = load_aliases()
    rows: list[dict[str, Any]] = []

    for manifest in list_runs(status=status):
        rows.append(
            {
                "name": aliases.get(manifest.run_id, ""),
                "run_id": manifest.run_id,
                "status": manifest.status,
                "updated_at": manifest.updated_at,
                "query": manifest.query,
                "cost_usd": float((manifest.totals or {}).get("cost_usd") or 0.0),
                "iterations": manifest.iterations,
            }
        )

    return rows


# --------------------------------------------------------- local display names
def aliases_path() -> Path:
    """Path of the local rename file (``<runs_root>/aliases.json``).

    Deliberately **outside** the run directory: a display name is a *viewer's*
    convenience, not run identity, so it never enters ``manifest.json`` (which is
    provenance and is written once), never travels inside a ``.lens`` bundle, and
    cannot change what the M11 campaign reports.
    """
    return runs_root() / ALIASES_FILENAME


def load_aliases() -> dict[str, str]:
    """Return ``{run_id: display_name}``; a missing or corrupt file is ``{}``.

    Tolerance is the contract: a rename must never be the reason a run cannot be
    opened (the same rule as a damaged blob in ``replay_turns``).
    """
    path = aliases_path()

    if not path.is_file():
        return {}

    try:
        loaded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return {}

    if not isinstance(loaded, dict):
        return {}

    return {
        str(run_id): str(name)
        for run_id, name in loaded.items()
        if isinstance(name, str) and name.strip()
    }


def set_alias(run_id: str, name: str) -> dict[str, str]:
    """Rename ``run_id`` locally and return the whole map.

    Contract:
      1. ``name`` is trimmed; an empty name **clears** the alias (back to the id).
      2. 1–``ALIAS_MAX_CHARS`` characters, and unique across runs
         **case-insensitively** — two runs called "baseline" would defeat the
         point of a label, and the run id underneath stays the unique key.
      3. The file is written through a temp file and ``os.replace``: a crash
         mid-rename must not leave a half-written alias map.
    """
    clean = (name or "").strip()
    aliases = load_aliases()

    if not clean:
        aliases.pop(run_id, None)
        _write_aliases(aliases)
        return aliases

    if len(clean) > ALIAS_MAX_CHARS:
        raise ValueError(f"a display name is at most {ALIAS_MAX_CHARS} characters")

    taken = {
        other: existing
        for other, existing in aliases.items()
        if other != run_id and existing.casefold() == clean.casefold()
    }

    if taken:
        raise ValueError(f"the name {clean!r} already belongs to {next(iter(taken))!r}")

    aliases[run_id] = clean
    _write_aliases(aliases)

    return aliases


def display_name(run_id: str, aliases: dict[str, str] | None = None) -> str:
    """Return the local name for ``run_id``, falling back to the run id itself."""
    resolved = aliases if aliases is not None else load_aliases()

    return resolved.get(run_id) or run_id


def _write_aliases(aliases: dict[str, str]) -> None:
    """Write the alias map atomically (temp file in the same directory, then replace)."""
    path = aliases_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(dict(sorted(aliases.items())), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_run(run_id: str) -> RunView | None:
    """Load one run for display, or ``None`` when it is not a run.

    Contract:
      1. An unsafe id (:func:`is_run_id`) or a directory without ``manifest.json``
         answers ``None`` **before** any :class:`Trajectory` is built, so opening
         a bad id creates nothing on disk.
      2. A missing/corrupt log or blob never raises: ``replay_turns`` and
         ``resolve_citations`` are tolerant by design (a damaged bundle must open
         and say what is missing), so a half-written run is still viewable.
      3. ``answer.md`` is the model's own words, verbatim; ``sources.md`` is the
         derived citation join. Their presence is reported, not assumed.
    """
    if not is_run_id(run_id):
        return None

    run_dir = runs_root() / str(run_id)
    manifest = read_manifest(run_dir)

    if manifest is None:
        return None

    trajectory = Trajectory(str(run_id))
    turns = replay_turns(trajectory)
    answer_path = run_dir / "answer.md"
    answer = (
        answer_path.read_text(encoding="utf-8", errors="replace") if answer_path.is_file() else ""
    )

    return RunView(
        run_id=str(run_id),
        manifest=manifest,
        turns=turns,
        summary=replay_summary(turns),
        citations=resolve_citations(str(run_id)),
        answer=answer,
        evidence=_evidence_by_turn(turns, trajectory),
        has_sources=(run_dir / "sources.md").is_file(),
        has_report=(run_dir / "report.md").is_file(),
    )


def evidence_upto(view: RunView, cursor: int) -> list[dict[str, Any]]:
    """Return the chunk rows hydrated by turns ``<= cursor``, deduplicated.

    This is the "evidence grows" half of the M10b claim: the panel fills as the
    playback advances while the context chart is driven by ``tokens_in`` alone.
    A chunk re-hydrated by a later turn is listed once, at its first appearance.
    """
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    for turn in view.turns:
        if turn.turn > cursor:
            break

        for row in view.evidence.get(turn.turn, []):
            chunk_id = str(row.get("chunk_uuid"))

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            rows.append(row)

    return rows


def tool_payload(run_id: str, ref: str | None) -> str | None:
    """Return the raw text of a tool result blob, or ``None`` if it is gone.

    The caller (the ``Show full payload`` toggle) decides how much to render: a
    ``build_kg`` node-link is the largest blob in a run, and a collapsed
    ``st.expander`` still builds its DOM, so this stays opt-in.
    """
    if not ref or not is_run_id(run_id):
        return None

    try:
        return Trajectory(str(run_id)).load_blob(ref)
    except (OSError, KeyError, ValueError, IndexError):
        return None


def _evidence_by_turn(turns: list[Turn], trajectory: Trajectory) -> dict[int, list[dict[str, Any]]]:
    """Map each turn to the chunk rows its hydrating tool calls returned."""
    evidence: dict[int, list[dict[str, Any]]] = {}

    for turn in turns:
        rows: list[dict[str, Any]] = []

        for tool in turn.tools:
            if tool.name not in CHUNK_TOOLS or not tool.result_ref:
                continue

            try:
                payload = json.loads(trajectory.load_blob(tool.result_ref))
            except (OSError, KeyError, ValueError, IndexError):
                continue  # a damaged payload is reported by replay_turns, not here

            if isinstance(payload, dict):
                payload = payload.get("chunks") or []

            if not isinstance(payload, list):
                continue

            rows.extend(row for row in payload if isinstance(row, dict) and row.get("chunk_uuid"))

        if rows:
            evidence[turn.turn] = rows

    return evidence


def run_dir(run_id: str) -> Path | None:
    """Return ``runs/<run_id>`` when it holds a manifest, else ``None``."""
    if not is_run_id(run_id):
        return None

    candidate = runs_root() / str(run_id)

    return candidate if (candidate / "manifest.json").is_file() else None


__all__ = [
    "ALIASES_FILENAME",
    "ALIAS_MAX_CHARS",
    "CHUNK_TOOLS",
    "CREDIT_FEE",
    "RunView",
    "aliases_path",
    "cap_of",
    "display_name",
    "evidence_upto",
    "is_run_id",
    "load_aliases",
    "load_run",
    "run_dir",
    "run_rows",
    "set_alias",
    "tool_payload",
]
