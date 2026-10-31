"""Citation resolution: does the answer's evidence actually exist?

What this answers
-----------------
``answer.md`` cites ``chunk_uuid`` (and sometimes an ``arxiv_id``). That is only a
checkable claim if the cited ids can be found again in the evidence the run
actually retrieved. This module joins the answer against the **recorded tool
payloads** (``rerank_chunks`` / ``fetch_chunks`` result blobs) and reports three
numbers: cited, resolved, unresolved.

It is deliberately read-only and offline: no key, no DB, no Docker, no GPU. That
is what lets it run inside ``scripts/check_run.py``, behind M10a's answer view,
and over an imported bundle on a laptop.

An **unresolved citation is not necessarily a hallucination** - the model may cite
a chunk it fetched in a turn whose payload was dropped from an incomplete bundle.
The report therefore keeps the token and the turn, and the reader decides. What it
never does is silently count an unknown id as supported.

Contract tests: ``tests/test_citations.py`` (no key, no DB, no Docker).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lens.observability.replay import Turn, replay_turns
from lens.observability.trajectory import Trajectory

# uuid5 chunk ids as written by ``lens.dataset.chunks.chunk_uuid``.
_CHUNK_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
# arXiv ids: YYMM.NNNN or YYMM.NNNNN (an optional version suffix is ignored).
_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
# Abbreviated chunk ids as a model writes them: [3b4365f9]. Six to twelve hex
# characters in brackets, which a hyphenated uuid cannot match.
_SHORT_ID_RE = re.compile(r"\[([0-9a-fA-F]{6,12})\]")

# Tools whose payload hydrates chunks / papers. Only these carry the evidence map.
_CHUNK_TOOLS = frozenset({"rerank_chunks", "fetch_chunks"})
_PAPER_TOOLS = frozenset({"rerank_papers", "fetch_papers"})


@dataclass
class Citation:
    """One citation in the answer, joined to the evidence that backs it."""

    token: str
    kind: str  # "chunk" | "arxiv"
    resolved: bool
    arxiv_id: str | None = None
    section_type: str | None = None
    title: str | None = None
    # The turn whose tool payload supplied the mapping (None = never retrieved).
    turn: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape used by ``sources.md`` and a campaign row."""
        return {
            "token": self.token,
            "kind": self.kind,
            "resolved": self.resolved,
            "arxiv_id": self.arxiv_id,
            "section_type": self.section_type,
            "title": self.title,
            "turn": self.turn,
        }


@dataclass
class CitationReport:
    """The answer's citations, resolved against the recorded evidence."""

    run_id: str
    answer_chars: int = 0
    citations: list[Citation] = field(default_factory=list)
    # Ids present in the answer that no recorded payload explains.
    unresolved: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        """Return ``{cited, resolved, unresolved}`` (unique tokens per kind)."""
        return {
            "cited": len(self.citations),
            "resolved": sum(1 for c in self.citations if c.resolved),
            "unresolved": len(self.unresolved),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape written by ``check_run``/``run_campaign``."""
        return {
            "run_id": self.run_id,
            "answer_chars": self.answer_chars,
            "counts": self.counts,
            "citations": [c.to_dict() for c in self.citations],
            "unresolved": list(self.unresolved),
        }


def _evidence_index(turns: list[Turn], trajectory: Trajectory) -> tuple[dict, dict, dict]:
    """Return ``(chunks, papers, provenance)`` built from recorded tool payloads.

    ``chunks`` maps ``chunk_uuid -> {arxiv_id, section_type, title}``,
    ``papers`` maps ``arxiv_id -> title``, ``provenance`` maps a cited id to the
    turn it was first seen in. Only the hydrating tools count: an id list from
    ``deep_match`` names candidates, not evidence the model could have read.
    """
    chunks: dict[str, dict[str, Any]] = {}
    papers: dict[str, str] = {}
    provenance: dict[str, int] = {}

    for turn in turns:
        for tool in turn.tools:
            if not tool.result_ref:
                continue

            if tool.name not in _CHUNK_TOOLS and tool.name not in _PAPER_TOOLS:
                continue

            try:
                payload = json.loads(trajectory.load_blob(tool.result_ref))
            except (OSError, KeyError, ValueError, IndexError):
                continue  # a damaged bundle is reported by replay_turns, not here

            if tool.name in _PAPER_TOOLS:
                rows, paper_rows = [], payload if isinstance(payload, list) else []
            elif isinstance(payload, list):
                rows, paper_rows = payload, []
            elif isinstance(payload, dict):
                rows = payload.get("chunks") or []
                paper_rows = payload.get("papers") or []
            else:
                continue

            for paper in paper_rows:
                if isinstance(paper, dict) and paper.get("arxiv_id"):
                    papers.setdefault(paper["arxiv_id"], paper.get("title") or "")

            for row in rows:
                if not isinstance(row, dict) or not row.get("chunk_uuid"):
                    continue

                arxiv_id = row.get("arxiv_id")

                chunks.setdefault(
                    row["chunk_uuid"],
                    {
                        "arxiv_id": arxiv_id,
                        "section_type": row.get("section_type"),
                        "title": papers.get(arxiv_id or "", "") or None,
                    },
                )
                provenance.setdefault(row["chunk_uuid"], turn.turn)

    return chunks, papers, provenance


def cited_ids(text: str) -> list[str]:
    """Return the unique ``chunk_uuid`` / arXiv ids cited in ``text``, in order.

    Cheap and pure, for callers that need the ids but not the evidence join: the
    **final groundedness gate** used to be handed ``"citations": []`` hard-coded,
    so it was asked "is every claim supported by the cited chunks" about an answer
    whose citations it had never seen - a blind check.
    """
    tokens: list[str] = []

    # Sorted by position so the list reads in the answer's own order: a chunk id
    # and an arXiv id interleaved in one sentence must not be reported grouped by
    # kind (two passes would do exactly that).
    matches = [(match.start(), match.group(0)) for match in _CHUNK_RE.finditer(text or "")]
    matches += [(match.start(), match.group(1)) for match in _ARXIV_RE.finditer(text or "")]

    for _, token in sorted(matches):
        if token not in tokens:
            tokens.append(token)

    return tokens


def resolve_citations(run_id: str) -> CitationReport:
    """Join ``answer.md`` against the evidence the run recorded.

    Contract:
      1. No ``answer.md`` (or no log) -> an empty report, never an exception: a
         failed run is still openable.
      2. Each **unique** ``chunk_uuid`` in the answer is resolved against the
         ``rerank_chunks``/``fetch_chunks`` payloads; an id in no payload is
         *unresolved* (listed in ``report.unresolved``), never silently dropped.
      3. Each unique arXiv id is resolved when it is one of the retrieved papers.
      4. Resolution is a set membership test over recorded blobs - nothing is
         re-queried, so the verdict describes the run as it happened.
    """
    trajectory = Trajectory(run_id)
    report = CitationReport(run_id=run_id)

    answer_path = trajectory.run_dir / "answer.md"

    if answer_path.is_file():
        answer = answer_path.read_text(encoding="utf-8", errors="replace")
    else:
        answer = ""

    report.answer_chars = len(answer)

    if not answer.strip():
        return report

    turns = replay_turns(trajectory)
    chunks, papers, provenance = _evidence_index(turns, trajectory)

    seen: set[str] = set()

    for token in _CHUNK_RE.findall(answer):
        if token in seen:
            continue

        seen.add(token)
        evidence = chunks.get(token)

        if evidence is None:
            report.unresolved.append(token)
            report.citations.append(Citation(token=token, kind="chunk", resolved=False))
            continue

        arxiv_id = evidence.get("arxiv_id")
        report.citations.append(
            Citation(
                token=token,
                kind="chunk",
                resolved=True,
                arxiv_id=arxiv_id,
                section_type=evidence.get("section_type"),
                title=evidence.get("title") or papers.get(arxiv_id or "") or None,
                turn=provenance.get(token),
            )
        )

    # Abbreviated ids: [3b4365f9]. A model that writes the first eight characters
    # of a chunk uuid is citing the chunk, and a strict matcher scored those
    # answers as uncited — which under-reported the flat baseline's grounding.
    # A prefix resolves only when exactly one recorded id matches it; zero and
    # many stay unresolved and are listed.
    for token in _SHORT_ID_RE.findall(answer):
        key = token.lower()

        if key in seen:
            continue

        matches = [full for full in chunks if full.lower().startswith(key)]

        if len(matches) != 1 or matches[0] in seen:
            if len(matches) != 1:
                seen.add(key)
                report.unresolved.append(token)
                report.citations.append(Citation(token=token, kind="chunk", resolved=False))
            continue

        full = matches[0]
        seen.add(full)
        evidence = chunks[full]
        arxiv_id = evidence.get("arxiv_id")
        report.citations.append(
            Citation(
                token=full,
                kind="chunk",
                resolved=True,
                arxiv_id=arxiv_id,
                section_type=evidence.get("section_type"),
                title=evidence.get("title") or papers.get(arxiv_id or "") or None,
                turn=provenance.get(full),
            )
        )

    for token in _ARXIV_RE.findall(answer):
        if token in seen:
            continue

        seen.add(token)
        title = papers.get(token)

        if title is None:
            report.unresolved.append(token)
            report.citations.append(Citation(token=token, kind="arxiv", resolved=False))
            continue

        report.citations.append(
            Citation(
                token=token,
                kind="arxiv",
                resolved=True,
                arxiv_id=token,
                title=title or None,
            )
        )

    return report


def write_sources_md(report: CitationReport, path: str | Path | None = None) -> Path:
    """Write ``sources.md`` beside ``answer.md`` and return its path.

    ``answer.md`` stays the model's own words; this is the derived trace, so the
    two cannot be confused when someone quotes the answer. Unresolved ids are
    listed **last and explicitly** - hiding them would make a partial run look
    fully grounded.
    """
    from lens import config  # local: keep module import cheap for --help

    if path is None:
        path = Path(config.LENS_RUNS_ROOT_PATH) / report.run_id / "sources.md"
    else:
        path = Path(path)

    resolved = [c for c in report.citations if c.resolved]
    lines = [
        f"# Sources - {report.run_id}",
        "",
        f"cited {report.counts['cited']} · "
        f"resolved {report.counts['resolved']} · "
        f"unresolved {report.counts['unresolved']}",
        "",
        "| id | paper | section | turn |",
        "|---|---|---|---|",
    ]

    for citation in resolved:
        title = (citation.title or "").replace("|", "/")[:70]
        turn = citation.turn if citation.turn is not None else "-"
        lines.append(
            f"| `{citation.token}` | {citation.arxiv_id or '-'} {title} "
            f"| {citation.section_type or '-'} | {turn} |"
        )

    if report.unresolved:
        lines += [
            "",
            "## Unresolved",
            "",
            "Not found in any recorded tool payload - unsupported, or the payload "
            "is missing from this run/bundle:",
            "",
        ]
        lines += [f"- `{token}`" for token in report.unresolved]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return path


__all__ = [
    "Citation",
    "CitationReport",
    "cited_ids",
    "resolve_citations",
    "write_sources_md",
]
