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
from collections.abc import Iterable, Mapping
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
# The same prefix written anywhere: ``chunk_uuid=bc4e8dcb``, ``[92287ee9, 2603.15653]``,
# ``a861854f…``. A neighbouring letter, digit or hyphen means it is part of a longer
# token (a full uuid, a word), so it is not a prefix.
_LOOSE_SHORT_ID_RE = re.compile(r"(?<![0-9A-Za-z-])([0-9a-fA-F]{6,12})(?![0-9A-Za-z-])")

# Tools whose payload hydrates chunks / papers. Only these carry the evidence map.
# ``rerank_chunks`` / ``rerank_papers`` stay listed so runs recorded before the
# funnel became one tool still resolve.
_CHUNK_TOOLS = frozenset(
    {"retrieval_funnel", "rerank_chunks", "fetch_chunks", "fetch_paper_chunks"}
)
_PAPER_TOOLS = frozenset({"rerank_papers", "fetch_papers"})

#: Sentence and list-line boundaries. A claim unit is what sits between two of them.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


#: Any id we understand, for stripping a unit down to its words.
_ID_TOKEN_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|\d{4}\.\d{4,5}(?:v\d+)?"
)

#: A unit whose words are shorter than this is an id dump, not a claim.
MIN_CLAIM_WORDS = 12

#: An **uncited** claim needs this many words to be worth a question. Fifteen is the
#: observed length of the sentences that matter here ("From the 15 cited chunks we
#: inspected, drawn from 100 reranked chunks across 19 papers, the text supports?"),
#: and it keeps headings and one-line labels out.
MIN_UNCITED_WORDS = 12

#: A line that is a bullet, a heading, a table row or a quote. Bullets are handled as
#: whole units (above), and the rest are structure rather than prose.
_STRUCTURE = re.compile(r"^(?:[-*\u2022#|>]|\d+\.)\s")

#: Punctuation left behind when the ids are removed.
_RESIDUE = re.compile(r"[\s\-–—•*:;,.()\[\]/|<>\"'`]+")

#: A prose line ends with sentence-final punctuation, possibly behind a closing
#: quote or bracket. A line that does not is a heading or a label however many
#: words it holds: measured on a recorded answer, `Q2 — How these methods differ in
#: where aggregation happens (grounded by quotes)` was asked about as an uncited
#: claim, scored 0.50 for asserting nothing, and was then named as the culprit.
_TERMINAL = re.compile(r"[.!?][\"'”’)\]]*$")


#: A top-level bullet and everything under it, up to the next bullet.
_BULLET_BLOCK = re.compile(r"(?m)^(?:[-*\u2022]\s+).*?(?=^[-*\u2022]\s+|\Z)", re.DOTALL)


def _bullet_blocks(text: str) -> list[str]:
    """Return one string per top-level bullet, continuation lines included."""
    return [match.group(0) for match in _BULLET_BLOCK.finditer(text)]


def uncited_units(text: str) -> list[tuple[str, list[str]]]:
    """Return prose lines that assert something with **no citation at all**.

    The aggregate question asks whether the answer contains a claim its citations do
    not support, and it judges the whole text. The per-claim questions were built only
    from **cited** bullets, so an answer could be rejected for an uncited sentence that
    no claim question ever mentioned ? measured on a recorded answer whose opening
    scope sentence ("From the 15 cited chunks we inspected?") carried three numbers and
    no id.

    Only non-structure lines qualify: a bullet, heading, table row or quote is handled
    elsewhere, and a bullet's claim is already covered by its own unit, citations and
    all. A line that ends with ``:`` is a lead-in to what follows, not a claim, and a
    line without sentence-final punctuation is a heading or a label: a claim is
    something the answer *states*, so it ends like a sentence.
    """
    units: list[tuple[str, list[str]]] = []

    for line in (text or "").splitlines():
        unit = " ".join(line.split())

        if not unit or _STRUCTURE.match(unit):
            continue

        # A line ending in ":" introduces what follows ("How they differ in where
        # aggregation happens:"). It scored highest of every unit because it cites
        # nothing, so the gate's feedback kept naming headings for the model to fix.
        if unit.endswith(":"):
            continue

        # A heading asserts nothing, so asking whether an excerpt supports it can only
        # produce an ambiguous score that lands on the aggregate threshold. Requiring
        # sentence-final punctuation is the cheapest test that separates a sentence
        # from a heading; it also drops table rows and bare labels.
        if not _TERMINAL.search(unit):
            continue

        if cited_ids(unit):
            continue

        if len(unit.split()) < MIN_UNCITED_WORDS:
            continue

        units.append((unit, []))

    return units


def claim_units(text: str, *, include_uncited: bool = False) -> list[tuple[str, list[str]]]:
    """Split an answer into claim-sized units with the ids each one cites.

    A *claim* is a sentence (or a list line) with **words and** a source. Two kinds
    of unit are skipped:

    - units with no citation — they are prose, and the aggregate question already
      fails an answer whose facts are uncited;
    - units whose text is only ids and punctuation. A bullet like
      ``- 85849030-…@2510.04532`` asserts nothing, so asking whether an excerpt
      supports it can only produce a rejection that names a non-claim. Measured on
      the pilot: without this filter the per-claim gate rejected answers on their
      id lists, and the named "failing claim" was a citation dump.

    Returns ``[(unit_text, [id, ...]), ...]`` with the bullets first and then the
    prose outside them. An answer with no claim at all collapses to one unit with an
    empty id list, so it is still judged once rather than silently skipped.

    ``include_uncited`` appends :func:`uncited_units` after the cited ones: those are
    the claims the aggregate can see and the per-claim questions could not, which is
    why a coverage failure used to name nothing.
    """
    units: list[tuple[str, list[str]]] = []

    # A bulleted answer puts the claim, its explanation and its citations on
    # separate lines, so sentence splitting tears them apart: the claim line has no
    # ids and the ``Citations:`` line has no words, and both are dropped. Measured
    # on a recorded answer with nine cited bullets, that produced **one** claim unit.
    # When there are bullets, each top-level bullet is the unit.
    blocks = _bullet_blocks(text or "")
    bulleted = len(blocks) > 1

    if bulleted:
        for block in blocks:
            unit = " ".join(block.split())
            ids = cited_ids(unit)

            if not ids:
                continue

            residue = _RESIDUE.sub(" ", _ID_TOKEN_RE.sub(" ", unit)).strip()

            if len(residue) < MIN_CLAIM_WORDS:
                continue

            units.append((unit, ids))

    # A mixed answer — prose findings plus a bulleted gap list — lost its prose when
    # the bullet path replaced the sentence path. The prose is the text outside the
    # bullets, so both are read and no bullet sentence is counted twice.
    prose = _BULLET_BLOCK.sub("", text or "") if bulleted else (text or "")

    for piece in _SENTENCE_SPLIT.split(prose):
        unit = " ".join(piece.split())

        if not unit:
            continue

        ids = cited_ids(unit)

        if not ids:
            continue

        residue = _RESIDUE.sub(" ", _ID_TOKEN_RE.sub(" ", unit)).strip()

        if len(residue) < MIN_CLAIM_WORDS:
            continue

        units.append((unit, ids))

    # The whole-answer fallback exists so an answer with no claim is still judged
    # once. With uncited units requested it would double-count them, and an answer
    # whose every line is structure or two words has nothing to ask about.
    if not units and not include_uncited:
        body = " ".join((text or "").split())

        if body:
            units.append((body, []))

    if include_uncited:
        units.extend(uncited_units(text))

    return units


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
    # ``(written, resolved)`` pairs: a full uuid whose suffix was invented but
    # whose eight-character prefix names exactly one recorded chunk.
    merged: list[tuple[str, str]] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        """Return ``{cited, resolved, merged, unresolved}`` (unique tokens per kind)."""
        return {
            "cited": len(self.citations),
            "resolved": sum(1 for c in self.citations if c.resolved),
            "merged": len(self.merged),
            "unresolved": len(self.unresolved),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape written by ``check_run``/``run_campaign``."""
        return {
            "run_id": self.run_id,
            "answer_chars": self.answer_chars,
            "counts": self.counts,
            "citations": [c.to_dict() for c in self.citations],
            "merged": [{"written": written, "chunk_uuid": full} for written, full in self.merged],
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

                # The funnel's ``papers`` is ``{arxiv_id: {title, ...}}``, not rows.
                if isinstance(paper_rows, dict):
                    paper_rows = [
                        {"arxiv_id": key, **value}
                        for key, value in paper_rows.items()
                        if isinstance(value, dict)
                    ]
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


#: A quoted span in an answer. The straight quote is both opening and closing, so a
#: boundary is required: an opener is not preceded by a letter/digit and a closer is
#: not followed by one. Without it, a short quoted word's closing quote opened a
#: false span that ran to the next quote (measured on a recorded answer). Curly
#: quotes are unambiguous.
_QUOTED = re.compile(
    r'(?<![0-9A-Za-z])"(?=\S)([^"\n]{25,400}?)(?<=\S)"(?![0-9A-Za-z])'
    r"|\u201c(?=\S)([^\u201d\n]{25,400}?)(?<=\S)\u201d"
)


def _quoted_text(match: re.Match[str]) -> str:
    """Return the content of a ``_QUOTED`` match (straight or curly)."""
    return match.group(1) if match.group(1) is not None else match.group(2)


#: Characters either side of a quote in which a citation counts as "the one the
#: answer attaches to it" — ids may precede or follow the span depending on style.
_CITATION_WINDOW = 300


def normalize_span(text: str) -> str:
    """Return ``text`` lowercased with every non-alphanumeric character collapsed.

    One definition for every span comparison: a quote check and a claim check
    must agree on what "the same words" means.
    """
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


_words = normalize_span


def _span_segments(span: str) -> list[str]:
    """Return the parts of a quoted span, split on an elision.

    An elided quote is legitimate — the model is allowed to leave words out and mark
    the gap — so each part is checked on its own, and a span passes only when every
    part is found. Splitting before normalising is what keeps ``…`` from being
    swallowed by the punctuation strip.
    """
    parts = re.split(r"\u2026|\.{2,}", span)

    return [segment for segment in (_words(part) for part in parts) if len(segment.split()) >= 4]


def verify_quoted_spans(
    answer: str, texts: Mapping[str, str], *, window: int = _CITATION_WINDOW
) -> tuple[int, int, int]:
    """Return ``(in_cited_chunk, elsewhere_only, total)`` for evidence quotes.

    Grounding is measured at two strengths, because they fail differently: a span
    found under a *different* chunk than the one cited is a mis-citation (the reader
    who follows the id does not find the words), while a span found nowhere is a
    paraphrase wearing quotation marks. The first is cheap to fix, the second is the
    failure the answer contract exists to prevent.

    ``texts`` maps ``chunk_uuid -> chunk text`` for everything the run retrieved, so
    this is a read-only check over recorded evidence: no model call, no database.
    Short ids are expanded first, and a span with **no chunk id in the window** is
    not counted: a ledger task name or an emphasis quote is not an evidence quote.
    """
    answer = expand_short_ids(answer or "", texts)
    answer, _ = resolve_merged_ids(answer, texts)
    normalized = {key: _words(value) for key, value in texts.items()}
    in_cited = elsewhere = total = 0

    for match in _QUOTED.finditer(answer):
        segments = _span_segments(_quoted_text(match))

        if not segments:
            continue

        nearby = set(
            re.findall(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                answer[max(0, match.start() - window) : match.end() + window],
            )
        )

        if not nearby:
            continue

        total += 1
        holders = {key for key, body in normalized.items() if all(s in body for s in segments)}

        if holders & nearby:
            in_cited += 1
        elif holders:
            elsewhere += 1

    return in_cited, elsewhere, total


def document_texts(trajectory: Trajectory) -> dict[str, str]:
    """Return ``record_id -> text`` from a document run's evidence file.

    A document record is the evidence for that run, so it plays the role a
    chunk payload plays in a corpus run: the claim pass and the gate need the
    text behind a cited id, and the run's own file is where it lives. A missing
    or malformed file returns nothing rather than raising, because a damaged
    bundle must still open and report what is missing.

    Record ids must be uuid-shaped for the citation scanner to see them; the
    instrument builders generate them that way.
    """
    path = trajectory.run_dir / "work" / "documents.jsonl"

    if not path.is_file():
        return {}

    texts: dict[str, str] = {}

    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                if isinstance(record, dict) and record.get("id") is not None:
                    texts[str(record["id"])] = str(record.get("text") or "")
    except (OSError, ValueError):
        return {}

    return texts


def recorded_chunk_texts(trajectory: Trajectory) -> dict[str, str]:
    """Return ``chunk_uuid -> chunk text`` from the run's recorded tool payloads.

    The same payloads :func:`resolve_citations` reads, for the fidelity check: a
    quote can only be verified against text the run actually held. A document
    run holds no tool payloads, so its evidence file is the fallback.
    """
    texts: dict[str, str] = {}

    try:
        turns = {turn.turn: turn for turn in replay_turns(trajectory)}
    except Exception:  # noqa: BLE001 - a damaged bundle yields no texts, not a crash
        return texts

    for turn in turns.values():
        for tool in turn.tools:
            if not tool.result_ref or tool.name not in _CHUNK_TOOLS:
                continue

            try:
                payload = json.loads(trajectory.load_blob(tool.result_ref))
            except (OSError, KeyError, ValueError, IndexError):
                continue

            rows = payload if isinstance(payload, list) else (payload or {}).get("chunks") or []

            for row in rows:
                if isinstance(row, dict) and row.get("chunk_uuid") and row.get("chunk_text"):
                    texts.setdefault(str(row["chunk_uuid"]), str(row["chunk_text"]))

    if not texts:
        texts.update(document_texts(trajectory))

    return texts


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


def expand_short_ids(text: str, known_ids: Iterable[str]) -> str:
    """Return ``text`` with every abbreviated chunk id replaced by its full uuid.

    A model that prints ``c["chunk_uuid"][:8]`` in its digests cites the prefix it
    saw. Every id-based check (:func:`cited_ids`, the final gate's evidence fetch,
    the per-claim questions) recognises only the full uuid, so such an answer
    reached the gate with **no evidence at all** and was rejected four times for a
    reason that had nothing to do with its claims.

    A token expands only when it is unambiguous: exactly one of ``known_ids`` (the
    chunks this run retrieved) starts with it. It must also mix letters and digits,
    so ordinary words ("decade", "facade") and plain numbers stay as written.
    """
    known = [str(full) for full in known_ids if full]

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)

        # shortcut: an all-letter or all-digit prefix (~2 % of uuids) stays
        # unexpanded; widen this if answers are seen citing one.
        if token.isdigit() or not any(char.isdigit() for char in token):
            return token

        key = token.lower()
        matches = [full for full in known if full.lower().startswith(key)]

        return matches[0] if len(matches) == 1 else token

    return _LOOSE_SHORT_ID_RE.sub(replace, text or "")


def resolve_merged_ids(text: str, known_ids: Iterable[str]) -> tuple[str, list[tuple[str, str]]]:
    """Return ``text`` with an invented full uuid replaced by the chunk it names.

    A model can write a full uuid whose first eight characters are right and whose
    suffix was reconstructed from memory. That token is neither a real id nor a
    short prefix, so the prefix expansion leaves it alone and the answer is
    reported as citing a chunk that was never retrieved. It is repaired here only
    when its prefix names **exactly one** recorded chunk; anything else stays as
    written and is reported unresolved by the caller.

    The returned pairs are ``(written, resolved)``, so a report can show what the
    answer said without rewriting the answer itself.
    """
    known = {str(full).lower(): str(full) for full in known_ids if full}
    repairs: list[tuple[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        key = token.lower()

        if key in known:
            return known[key]

        matches = [full for low, full in known.items() if low.startswith(key[:8])]

        if len(matches) != 1:
            return token

        repairs.append((token, matches[0]))
        return matches[0]

    return _CHUNK_RE.sub(replace, text or ""), repairs


def recorded_chunk_ids(trajectory: Trajectory) -> set[str]:
    """Return every chunk uuid a hydrating tool returned in this run's log."""
    ids: set[str] = set()

    if not trajectory.trajectory_path.is_file():
        return ids

    for line in trajectory.trajectory_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue

        if not isinstance(event, dict) or event.get("tool") not in _CHUNK_TOOLS:
            continue

        if not event.get("result_ref"):
            continue

        try:
            payload = json.loads(trajectory.load_blob(event["result_ref"]))
        except (OSError, KeyError, ValueError, IndexError):
            continue

        rows = payload.get("chunks") if isinstance(payload, dict) else payload

        for row in rows or []:
            if isinstance(row, dict) and row.get("chunk_uuid"):
                ids.add(str(row["chunk_uuid"]))

    return ids


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

    # A document run has no tool payloads: its evidence file is the record set.
    if not chunks:
        chunks = {
            record_id: {"chunk_text": text}
            for record_id, text in document_texts(trajectory).items()
        }

    answer = expand_short_ids(answer, chunks)
    answer, report.merged = resolve_merged_ids(answer, chunks)

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
    "normalize_span",
    "resolve_citations",
    "write_sources_md",
]
