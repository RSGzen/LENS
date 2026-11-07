"""Read-only data access for the UI — no Streamlit, no key, no DB, no Docker.

Why this module exists
----------------------
The UI adds **no pipeline logic**. Everything it shows is already on disk
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
   recorded before carry ``caps.cost_usd = 0.5`` while the default is now
   ``1.00``; reading the config would make an run look half spent.

Contract tests: ``tests/test_ui_data.py`` (no streamlit, no key, no DB, no Docker).
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from lens import config
from lens.agent.runstore import (
    RunManifest,
    list_runs,
    read_manifest,
    runs_root,
)
from lens.observability.citations import (
    CitationReport,
    cited_ids,
    expand_short_ids,
    resolve_citations,
)
from lens.observability.replay import Turn, replay_summary, replay_turns
from lens.observability.trajectory import Trajectory

#: The 5.5 % OpenRouter credit-load fee, defined once in ``lens.config``. A
#: *reported* ``$/query`` includes it; the trajectory's ``cost_usd`` is raw. Shown
#: as a second column, never folded into the raw number the log holds.
CREDIT_FEE = config.CREDIT_FEE

#: Tools whose payload hydrates chunks. The same set ``citations.py`` trusts: an
#: id list names candidates, not evidence the model could read. ``rerank_chunks``
#: stays for runs recorded before the funnel became one tool.
CHUNK_TOOLS = frozenset({"retrieval_funnel", "rerank_chunks", "fetch_chunks"})

#: Local display names live beside the blob store, not inside a run directory.
ALIASES_FILENAME = "aliases.json"

#: Long enough for "Tier-1 anchor, gpt-5, aggregation query", short enough to render.
ALIAS_MAX_CHARS = 64


def _clip(text: str, limit: int) -> str:
    """Return ``text`` capped at ``limit`` characters, with an explicit marker.

    A silently shortened string reads as the whole thing. The marker says how much
    was dropped, so a reader knows to open the blob rather than trust the panel.
    """
    if len(text) <= limit:
        return text

    return text[:limit] + f"\n... [{len(text) - limit:,} more characters]"


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
    which would claim a budget the run never had.
    """
    recorded = (manifest.caps or {}).get("cost_usd")

    try:
        return float(recorded)
    except (TypeError, ValueError):
        return None


#: Characters of the gate's state kept for display. The state is the whole prompt
#: payload (answer, evidence excerpts, sample chunks); the UI shows enough to audit
#: what the verdict was based on, not enough to re-read the run.
GATE_STATE_CHARS = 4000

#: Characters of one gate question kept for display.
GATE_QUESTION_CHARS = 900


@dataclass(frozen=True)
class GateCall:
    """One gate call: what it was asked, what it answered, and what it cost.

    The verdict alone is not auditable. A reader needs the **question** (which
    claim, which citations), the **state** it was judged against, and the
    probability, because "fail 0.96" says nothing about whether the rejection was
    right.
    """

    stage: str | None
    verdict: str
    confidence: float | None
    criterion: str | None
    polarity: str | None
    #: Which threshold produced the verdict: ``aggregate``, ``claim`` or ``both``.
    #: ``None`` on runs recorded before the gate logged it, and on a pass.
    decision: str | None
    #: The thresholds in force when the call was made (``None`` on older runs).
    tau_agg: float | None
    tau_claim: float | None
    claims_failed: tuple[str, ...]
    #: claim id -> probability the claim is unsupported.
    claim_scores: dict[str, float]
    #: claim id -> the question text sent for it (clipped).
    claim_questions: dict[str, str]
    #: claim id -> the claim itself, as the gate was told it. The state carries only
    #: ids and citations, so without this a reader sees "claim-03" and no claim.
    claim_texts: dict[str, str]
    question: str | None
    state: str | None
    cost_usd: float
    latency_ms: int | None
    tokens_in: int | None
    #: Blob refs for the request and the reply, kept so the page can load the whole
    #: payload on demand instead of only showing the clipped text.
    prompt_ref: str | None
    completion_ref: str | None


@dataclass
class ClaimLedger:
    """The recorded claim ledger (``claims.json``) as the viewer reads it."""

    run_id: str
    units: int = 0
    coverage: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    tokens_out: int = 0
    errors: list[str] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)


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
    #: arxiv_id -> paper metadata (title, authors, year) from the recorded payloads.
    papers: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: the recorded claim ledger, when the run has one.
    claims: ClaimLedger | None = None
    #: turn number -> the gate calls made in that turn, in the order they ran.
    gates: dict[int, list[GateCall]] = field(default_factory=dict)
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
        """Prompt tokens per turn — the context-growth chart."""
        return [turn.tokens_in or 0 for turn in self.turns]

    @property
    def cached_series(self) -> list[int]:
        """Cached prompt tokens per turn — the provider-served share of the prompt.

        A subset of :attr:`tokens_in_series`, not a second prompt: the two lines
        sit on the same token axis, and the gap between them is the uncached input.
        """
        return [turn.cached_tokens or 0 for turn in self.turns]

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
    cannot change what the campaign reports.
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
        papers=_papers_by_turn(turns, trajectory),
        claims=load_claims(str(run_id)),
        gates=gate_calls(str(run_id)),
        has_sources=(run_dir / "sources.md").is_file(),
        has_report=(run_dir / "report.md").is_file(),
    )


def load_claims(run_id: str) -> ClaimLedger | None:
    """Read ``claims.json`` for one run; ``None`` when it is absent or damaged.

    A missing ledger is normal for a run recorded before the claim pass existed,
    and a damaged one must not stop the answer from opening.
    """
    if not is_run_id(run_id):
        return None

    path = runs_root() / str(run_id) / "claims.json"

    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    frames = [frame for frame in payload.get("frames") or [] if isinstance(frame, dict)]
    counts = {
        str(key): int(value)
        for key, value in (payload.get("counts") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }

    return ClaimLedger(
        run_id=str(payload.get("run_id") or run_id),
        units=int(payload.get("units") or 0),
        coverage=float(payload.get("coverage") or 0.0),
        counts=counts,
        cost_usd=float(payload.get("cost_usd") or 0.0),
        tokens_out=int(payload.get("tokens_out") or 0),
        errors=[str(error) for error in payload.get("errors") or []],
        frames=frames,
    )


def claim_rows(ledger: ClaimLedger) -> list[dict[str, Any]]:
    """Return one display row per frame, plus an ``uncovered`` row per empty unit."""
    rows: list[dict[str, Any]] = []
    covered: set[int] = set()

    for index, frame in enumerate(ledger.frames, start=1):
        verdict = frame.get("verdict") or {}
        unit = frame.get("unit_index")

        if isinstance(unit, int):
            covered.add(unit)

        rows.append(
            {
                "#": index,
                "status": str(verdict.get("status") or ""),
                "claim": " ".join(str(frame.get("text") or "").split())[:220],
                "predicate": frame.get("predicate"),
                "source": frame.get("quote_source"),
                "quote": " ".join(str(frame.get("quote") or "").split())[:160],
                "chunk": str(frame.get("chunk_id") or "")[:8],
            }
        )

    for unit in range(ledger.units):
        if unit in covered:
            continue

        rows.append(
            {
                "#": len(rows) + 1,
                "status": "uncovered",
                "claim": f"unit {unit} produced no frame",
                "predicate": None,
                "source": None,
                "quote": "",
                "chunk": "",
            }
        )

    return rows


def gate_calls(run_id: str) -> dict[int, list[GateCall]]:
    """Return one run's gate calls, grouped by turn, in the order they ran.

    A gate call is two log events: ``jev.decisions`` carries the request (state and
    questions) and its cost; ``gate.verdict`` carries the answer that was acted on.
    They are paired within a turn because neither names the other: the request holds
    no stage, the verdict holds no ref. A verdict with no request (a skipped tier)
    is still returned, with the question fields empty -- dropping it would hide a
    gate that ran and was not paid for.

    Every field is bounded, and a damaged blob degrades to ``None`` rather than
    raising: a half-written run must still open.
    """
    if not is_run_id(run_id):
        return {}

    log = runs_root() / str(run_id) / "trajectory.jsonl"

    if not log.is_file():
        return {}

    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}

    trajectory = Trajectory(str(run_id))
    requests: dict[int, list[dict[str, Any]]] = {}
    calls: dict[int, list[GateCall]] = {}

    for line in lines:
        if not line.strip():
            continue

        try:
            event = json.loads(line)
        except ValueError:
            continue

        step = int(event.get("step_idx") or 0)
        action = event.get("action")

        if action == "jev.decisions":
            requests.setdefault(step, []).append(_gate_request(trajectory, event))
        elif action == "gate.verdict":
            pending = requests.get(step) or []
            request = pending.pop(0) if pending else {}
            calls.setdefault(step, []).append(_gate_call(event, request))

    return calls


def _gate_request(trajectory: Trajectory, event: dict[str, Any]) -> dict[str, Any]:
    """Return what one gate call was asked: state, questions, cost."""
    request: dict[str, Any] = {
        "cost_usd": float(event.get("cost_usd") or 0.0),
        "latency_ms": event.get("latency_ms"),
        "tokens_in": event.get("tokens_in"),
        "prompt_ref": event.get("prompt_ref"),
        "completion_ref": event.get("completion_ref"),
    }

    payload = _blob_json(trajectory, event.get("prompt_ref"))

    if not isinstance(payload, dict):
        return request

    state = payload.get("state")

    if isinstance(state, str):
        request["state"] = _clip(state, GATE_STATE_CHARS)
    elif state is not None:
        request["state"] = _clip(json.dumps(state, ensure_ascii=False, indent=2), GATE_STATE_CHARS)

    questions = payload.get("questions")

    if isinstance(questions, dict):
        aggregate = questions.get("gate")

        if isinstance(aggregate, dict):
            request["question"] = _clip(
                str(aggregate.get("instructions") or ""), GATE_QUESTION_CHARS
            )

        for key, value in questions.items():
            if not str(key).startswith("claim-") or not isinstance(value, dict):
                continue

            instructions = str(value.get("instructions") or "")
            request.setdefault("claim_questions", {})[str(key)] = _clip(
                instructions, GATE_QUESTION_CHARS
            )

            # The claim is the tail of the question ("... answer false.\nClaim: <text>"),
            # which is the only place it appears: the state lists ids, not claims.
            marker = "\nClaim: "
            if marker in instructions:
                request.setdefault("claim_texts", {})[str(key)] = instructions.split(marker, 1)[
                    1
                ].strip()

    return request


def _gate_call(event: dict[str, Any], request: dict[str, Any]) -> GateCall:
    """Join one ``gate.verdict`` event with its request."""
    extra = event.get("extra") or {}
    scores = extra.get("claim_scores") or {}

    return GateCall(
        stage=event.get("stage"),
        verdict=str(event.get("verdict") or "unknown"),
        confidence=event.get("confidence"),
        criterion=extra.get("criterion"),
        polarity=extra.get("polarity"),
        decision=extra.get("decision"),
        tau_agg=extra.get("tau_agg"),
        tau_claim=extra.get("tau_claim"),
        claims_failed=tuple(str(name) for name in extra.get("claims_failed") or ()),
        claim_scores={
            str(key): float(value)
            for key, value in scores.items()
            if isinstance(value, (int, float))
        },
        claim_questions=request.get("claim_questions") or {},
        claim_texts=request.get("claim_texts") or {},
        question=request.get("question"),
        state=request.get("state"),
        cost_usd=float(request.get("cost_usd") or 0.0),
        latency_ms=request.get("latency_ms"),
        tokens_in=request.get("tokens_in"),
        prompt_ref=request.get("prompt_ref"),
        completion_ref=request.get("completion_ref"),
    )


def _blob_json(trajectory: Trajectory, ref: str | None) -> Any:
    """Return a JSON blob, or ``None`` when it is missing or unparseable."""
    if not ref:
        return None

    try:
        return json.loads(trajectory.load_blob(ref))
    except (OSError, KeyError, ValueError, IndexError):
        return None


def cited_chunks(view: RunView) -> list[dict[str, Any]]:
    """Return the chunk rows behind the ids the answer cites, in answer order.

    The answer names ``chunk_uuid`` values; the text behind each one is already in
    the run, hydrated by the tool call that returned it, so this is a join and not a
    query. Ids the answer cites that no recorded payload explains are skipped here --
    :attr:`RunView.citations` already reports them as unresolved, and inventing an
    empty row would make a missing blob look like an empty chunk.
    """
    rows_by_id: dict[str, dict[str, Any]] = {}

    for rows in view.evidence.values():
        for row in rows:
            chunk_id = str(row.get("chunk_uuid") or "")

            if chunk_id:
                rows_by_id.setdefault(chunk_id, row)

    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    for token in cited_ids(view.answer):
        row = rows_by_id.get(token)

        if row is None:
            # A truncated id (and some payloads log the short form) still resolves
            # when exactly one hydrated chunk starts with it.
            candidates = [chunk_id for chunk_id in rows_by_id if chunk_id.startswith(token)]
            row = rows_by_id[candidates[0]] if len(candidates) == 1 else None

        if row is None:
            continue

        chunk_id = str(row.get("chunk_uuid"))

        if chunk_id in seen:
            continue

        text = str(row.get("chunk_text") or "")
        seen.add(chunk_id)
        found.append(
            {
                "chunk_uuid": chunk_id,
                "arxiv_id": row.get("arxiv_id"),
                "section_type": row.get("section_type"),
                "chars": len(text),
                "text": text,
            }
        )

    return found


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


def blob_text(run_id: str, ref: str | None) -> str | None:
    """Return the raw text of one content-addressed blob, or ``None`` if it is gone.

    Callers decide how much to render: a ``build_kg`` node-link or a gate state is
    the largest thing in a run, and a collapsed ``st.expander`` still builds its DOM,
    so the full payload stays behind a toggle.
    """
    if not ref or not is_run_id(run_id):
        return None

    try:
        return Trajectory(str(run_id)).load_blob(ref)
    except (OSError, KeyError, ValueError, IndexError):
        return None


def tool_payload(run_id: str, ref: str | None) -> str | None:
    """Return the raw text of a tool result blob (the tool-panel name for a blob)."""
    return blob_text(run_id, ref)


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


def _papers_by_turn(turns: list[Turn], trajectory: Trajectory) -> dict[str, dict[str, Any]]:
    """Return ``arxiv_id -> metadata`` from the recorded paper payloads.

    The funnel's ``papers`` dict already carries title, authors and year; keeping
    them on the view lets the answer show a human label without a database read,
    which is what a bundle on a laptop needs.
    """
    papers: dict[str, dict[str, Any]] = {}

    for turn in turns:
        for tool in turn.tools:
            if tool.name not in {"retrieval_funnel", "fetch_papers", "rerank_papers"}:
                continue

            if not tool.result_ref:
                continue

            try:
                payload = json.loads(trajectory.load_blob(tool.result_ref))
            except (OSError, KeyError, ValueError, IndexError):
                continue

            rows = payload.get("papers") if isinstance(payload, dict) else payload

            if isinstance(rows, dict):
                for arxiv_id, meta in rows.items():
                    if isinstance(meta, dict):
                        papers.setdefault(str(arxiv_id), meta)
            elif isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("arxiv_id"):
                        papers.setdefault(str(row["arxiv_id"]), row)

    return papers


#: How long a citation label may run before it is ellipsised; the full title is
#: in the hover text and the citation table.
CITE_LABEL_CHARS = 36


def short_title(title: str | None, limit: int = CITE_LABEL_CHARS) -> str:
    """Return a one-line label for a paper title."""
    body = " ".join(str(title or "").split())

    return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


#: Chunk and arXiv ids in an answer, with an optional surrounding backtick pair.
#: The backticks are consumed so the chip is not rendered as inline code; one
#: alternation means a token can never be replaced twice.
_CITE_ANY = re.compile(
    r"`?(?P<token>"
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    r"|\b\d{4}\.\d{4,5}(?:v\d+)?\b"
    r")`?"
)


def citation_labels(view: RunView) -> dict[str, str]:
    """Return ``token -> short label`` for every citation in the answer.

    Besides the resolved citations, every chunk the run hydrated gets a label:
    an 8-character prefix in the answer expands to a full id, and the chip still
    has a title even when ``sources.md`` did not list it as a citation.
    """
    labels: dict[str, str] = {}

    for citation in view.citations.citations:
        if not citation.resolved:
            continue

        meta = view.papers.get(citation.arxiv_id or "", {})
        title = citation.title or meta.get("title") or citation.arxiv_id or "source"
        label = short_title(title)

        if citation.kind == "chunk" and citation.section_type:
            label = f"{label} · {citation.section_type}"

        labels[citation.token] = label

    for rows in view.evidence.values():
        for row in rows:
            token = str(row.get("chunk_uuid") or "")

            if not token or token in labels:
                continue

            meta = view.papers.get(str(row.get("arxiv_id") or ""), {})
            title = meta.get("title") or str(row.get("arxiv_id") or "") or "source"
            label = short_title(str(title))

            if row.get("section_type"):
                label = f"{label} · {row['section_type']}"

            labels[token] = label

    return labels


def citation_tip(view: RunView, token: str) -> str:
    """Return the hover text for one citation token."""
    citation = next((c for c in view.citations.citations if c.token == token), None)

    if citation is not None:
        meta = view.papers.get(citation.arxiv_id or "", {})
        title = citation.title or meta.get("title") or citation.arxiv_id or "source"
        lines = [str(title)]

        if meta.get("authors"):
            lines.append(str(meta["authors"]))
        if meta.get("year"):
            lines.append(f"year {meta['year']}")
        if citation.arxiv_id:
            lines.append(f"arXiv {citation.arxiv_id}")
        if citation.kind == "chunk":
            if citation.section_type:
                lines.append(f"section {citation.section_type}")
            lines.append(str(citation.token))

        return "\n".join(lines)

    row = next(
        (
            row
            for rows in view.evidence.values()
            for row in rows
            if str(row.get("chunk_uuid")) == token
        ),
        None,
    )

    if row is None:
        return "not resolved against the recorded evidence"

    meta = view.papers.get(str(row.get("arxiv_id") or ""), {})
    title = meta.get("title") or str(row.get("arxiv_id") or "") or "source"
    lines = [str(title)]

    if meta.get("authors"):
        lines.append(str(meta["authors"]))
    if meta.get("year"):
        lines.append(f"year {meta['year']}")
    if row.get("arxiv_id"):
        lines.append(f"arXiv {row['arxiv_id']}")
    if row.get("section_type"):
        lines.append(f"section {row['section_type']}")
    lines.append(token)

    return "\n".join(lines)


def render_answer_html(view: RunView) -> str:
    """Replace citation ids with readable chips, keeping the id in the hover text.

    The answer itself is unchanged on disk; this is display only. A short prefix
    is expanded against the chunks the run actually hydrated, a backticked id is
    consumed with its backticks (otherwise Markdown renders the chip as code),
    and a fenced code block is left untouched. An unresolved id stays visible.
    """
    labels = citation_labels(view)
    resolved = {c.token for c in view.citations.citations if c.resolved} | set(labels)
    known = {
        str(row.get("chunk_uuid"))
        for rows in view.evidence.values()
        for row in rows
        if row.get("chunk_uuid")
    }
    expanded = expand_short_ids(view.answer or "", known)

    def replace(match: re.Match[str]) -> str:
        token = match.group("token")

        if token not in resolved:
            return token

        label = labels.get(token) or short_title(token)
        tip = citation_tip(view, token)

        return (
            f'<span class="lens-cite" data-tip="{html.escape(tip, quote=True)}">'
            f"{html.escape(label)}</span>"
        )

    parts = re.split(r"(```.*?```)", expanded, flags=re.DOTALL)

    for index, part in enumerate(parts):
        if part.startswith("```"):
            continue

        parts[index] = _CITE_ANY.sub(replace, part)

    return "".join(parts)


#: How long one lookup may take. The query is a single indexed row or a list of ids;
#: a slow answer means the database is unreachable, and a page should say so.
LOOKUP_TIMEOUT_S = 30


def _run_lookup(*args: str) -> dict[str, Any]:
    """Run one lookup in a child process and parse its JSON envelope.

    A child process rather than a connection in this module: the UI process must not
    hold a database pool open for the lifetime of a browser tab, which is the same
    reason the Live page shells out to the CLI. Everything below returns an envelope
    -- ``{"ok": False, "error": ...}`` -- because a page must render the failure, not
    raise it.
    """
    command = [sys.executable, "-m", "lens.lookup", *args]

    try:
        finished = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=LOOKUP_TIMEOUT_S,
            cwd=config.REPO_ROOT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": f"lookup did not run: {type(exc).__name__}"}

    try:
        payload = json.loads(finished.stdout or "")
    except ValueError:
        last = (finished.stderr or "").strip().splitlines()

        return {
            "ok": False,
            "error": "lookup returned no JSON" + (f": {last[-1][:200]}" if last else ""),
        }

    return payload if isinstance(payload, dict) else {"ok": False, "error": "unexpected reply"}


def lookup_paper(arxiv_id: str) -> dict[str, Any]:
    """Return ``{"ok", "found", "missing"}`` for one arXiv id."""
    return _run_lookup("paper", str(arxiv_id or "").strip())


def lookup_chunks(chunk_ids: Sequence[str]) -> dict[str, Any]:
    """Return ``{"ok", "found", "missing"}`` for a list of chunk ids."""
    cleaned = [str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()]

    if not cleaned:
        return {"ok": False, "error": "no chunk ids given"}

    return _run_lookup("chunk", *cleaned)


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
