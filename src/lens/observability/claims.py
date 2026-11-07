"""Claim frames and mechanical verification.

A review answer is a sequence of sentences, and a sentence is a poor unit to
check: "Method A scores 56.0 on benchmark B" binds a value to a benchmark, and
splitting it into two claims loses that binding. A :class:`ClaimFrame` keeps the
predicate, its roles, the value, the verbatim span and the chunk the span came
from.

The verifier is code, not a model. It checks that the span exists in the cited
chunk, that a value parses and appears in the span, and that a comparison's
direction matches the words between the two entities. When it cannot decide it
returns ``unverifiable``; it never returns ``contradicted`` for missing
evidence. The asymmetry is deliberate: a false ``grounded`` is worse than an
honest "not checked", because the report presents the first as support.

The extractor that turns a sentence into frames calls the sub model, but it is
host-side: the root never writes the ledger, and a frame whose id or quote the
host cannot verify is dropped before it is counted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from lens.observability.citations import _QUOTED, _quoted_text, normalize_span

#: Predicates a frame may carry. The first four have a mechanical check; the
#: rest are kept so an answer is not forced into a shape the text does not have,
#: and they are reported as ``unverifiable``.
PREDICATES: tuple[str, ...] = (
    "reports",
    "measures",
    "compares",
    "uses",
    "evaluated_on",
    "addresses",
    "other",
)

GROUNDED = "grounded"
CONTRADICTED = "contradicted"
UNVERIFIABLE = "unverifiable"
STATUSES: tuple[str, ...] = (GROUNDED, CONTRADICTED, UNVERIFIABLE)

#: Words that carry no claim content when matching an object against a span.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)

_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

#: Direction cues. A comparison is grounded only when one side of the pair is
#: described by a positive cue, and contradicted only when the cue points the
#: other way. Two cues in the same window are ambiguous, not evidence.
_POSITIVE = (
    "outperform",
    "surpass",
    "exceed",
    "better",
    "higher",
    "above",
    "improve",
    "more accurate",
)
_NEGATIVE = ("underperform", "worse", "lower", "below", "less accurate", "inferior")


@dataclass(frozen=True)
class ClaimFrame:
    """One checkable claim: a predicate, its bound roles, and its evidence."""

    unit_index: int
    text: str
    predicate: str
    subject: str
    object: str | None = None
    benchmark: str | None = None
    metric: str | None = None
    value: str | None = None
    unit: str | None = None
    direction: str | None = None
    quote: str = ""
    chunk_id: str | None = None
    #: Where the quote came from: ``"answer"`` (the span the answer displays) or
    #: ``"chunk"`` (a span the extractor chose when the answer has no quotation).
    quote_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape written to the claim ledger."""
        return {
            "unit_index": self.unit_index,
            "text": self.text,
            "predicate": self.predicate,
            "subject": self.subject,
            "object": self.object,
            "benchmark": self.benchmark,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "direction": self.direction,
            "quote": self.quote,
            "chunk_id": self.chunk_id,
            "quote_source": self.quote_source,
        }


@dataclass(frozen=True)
class FrameVerdict:
    """The result of one mechanical check."""

    status: str
    reason: str = ""
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape written beside the frame."""
        return {"status": self.status, "reason": self.reason, "evidence": self.evidence}


@dataclass
class ClaimReport:
    """Every frame in one answer, with its verdict and the coverage summary."""

    run_id: str = ""
    units: int = 0
    frames: list[ClaimFrame] = field(default_factory=list)
    verdicts: list[FrameVerdict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Spend attributable to the claim extractor for this answer.
    cost_usd: float = 0.0
    tokens_out: int = 0
    #: Whether a trailing self-reported-gap section was excluded before extraction.
    #: The section is not a finding, and the full answer keeps it on disk.
    self_reported_gaps_removed: bool = False

    @property
    def counts(self) -> dict[str, int]:
        """Return ``{grounded, contradicted, unverifiable}`` over the frames."""
        counts = {status: 0 for status in STATUSES}

        for verdict in self.verdicts:
            counts[verdict.status] = counts.get(verdict.status, 0) + 1

        return counts

    @property
    def covered_units(self) -> int:
        """Return how many claim units produced at least one checked frame."""
        covered = {
            frame.unit_index
            for frame, verdict in zip(self.frames, self.verdicts)
            if verdict.status != UNVERIFIABLE
        }

        return len(covered)

    @property
    def coverage(self) -> float:
        """Return the share of claim units that produced a checked frame."""
        return 0.0 if self.units < 1 else round(self.covered_units / self.units, 4)

    def to_dict(self) -> dict[str, Any]:
        """Return the artifact shape written to ``claims.json``."""
        return {
            "run_id": self.run_id,
            "units": self.units,
            "coverage": self.coverage,
            "counts": self.counts,
            "cost_usd": round(self.cost_usd, 6),
            "tokens_out": self.tokens_out,
            "self_reported_gaps_removed": self.self_reported_gaps_removed,
            "errors": list(self.errors),
            "frames": [
                {**frame.to_dict(), "verdict": verdict.to_dict()}
                for frame, verdict in zip(self.frames, self.verdicts)
            ],
        }

    def rows(self) -> list[dict[str, Any]]:
        """Return one flat row per frame, for the campaign CSV."""
        return [
            {
                "run_id": self.run_id,
                **frame.to_dict(),
                "status": verdict.status,
                "reason": verdict.reason,
                "evidence": verdict.evidence,
            }
            for frame, verdict in zip(self.frames, self.verdicts)
        ]


def _content_tokens(text: str) -> list[str]:
    """Return the non-stopword tokens of ``text`` after normalisation."""
    return [token for token in normalize_span(text).split() if token and token not in _STOPWORDS]


def _terms_present(span: str, text: str, threshold: float = 0.6) -> bool:
    """Return true when enough content terms of ``text`` occur in ``span``.

    ``span`` is already normalised. The threshold allows a paraphrase of the
    object while still refusing a quote that shares only a word or two.
    """
    wanted = _content_tokens(text)

    if not wanted:
        return False

    available = set(span.split())
    present = sum(1 for token in wanted if token in available)

    return present / len(wanted) >= threshold


def _numbers(text: str) -> list[float]:
    """Return every number in ``text`` as a float, commas removed."""
    values: list[float] = []

    for match in _NUMBER.finditer(text or ""):
        try:
            values.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue

    return values


def _value_number(value: str | None) -> float | None:
    """Return the first number in ``value``, or ``None``."""
    numbers = _numbers(value or "")

    return numbers[0] if numbers else None


def _contains(span: str, text: str | None) -> bool:
    """Return true when the normalised ``text`` occurs in the normalised span."""
    wanted = normalize_span(text or "")

    return bool(wanted) and wanted in span


def _verify_measurement(frame: ClaimFrame, span: str, raw: str) -> FrameVerdict:
    """Check a value against the numbers the span states.

    The numbers come from the raw quote, not the normalised span: normalisation
    turns ``56.0`` into ``56 0``, which would lose the decimal the check exists
    to compare.
    """
    number = _value_number(frame.value)

    if number is None:
        return FrameVerdict(UNVERIFIABLE, "the frame carries no numeric value")

    numbers = _numbers(raw)

    if not numbers:
        return FrameVerdict(UNVERIFIABLE, "the span states no number")

    if not any(abs(number - seen) <= 1e-6 for seen in numbers):
        return FrameVerdict(
            CONTRADICTED,
            f"the span states {sorted(set(numbers))} but the claim states {number}",
            evidence=" ".join(str(seen) for seen in numbers),
        )

    for role in (frame.benchmark, frame.metric):
        if role and not _contains(span, role):
            return FrameVerdict(UNVERIFIABLE, f"{role!r} is not stated in the span")

    return FrameVerdict(GROUNDED, "the value and its roles are stated in the span")


def _verify_attribution(frame: ClaimFrame, span: str) -> FrameVerdict:
    """Check that the cited span names the subject and states the object."""
    if not frame.subject or not _contains(span, frame.subject):
        return FrameVerdict(UNVERIFIABLE, "the subject is not stated in the span")

    if frame.object and not _terms_present(span, frame.object):
        return FrameVerdict(UNVERIFIABLE, "the object's terms are not stated in the span")

    return FrameVerdict(GROUNDED, "the subject and object are stated in the span")


def _between(span: str, left: str, right: str) -> tuple[str, bool] | None:
    """Return the text between two entities and whether ``left`` came first."""
    left_text = normalize_span(left)
    right_text = normalize_span(right)

    if not left_text or not right_text:
        return None

    left_at = span.find(left_text)
    right_at = span.find(right_text)

    if left_at < 0 or right_at < 0:
        return None

    if left_at <= right_at:
        start = left_at + len(left_text)
        return span[start:right_at], True

    start = right_at + len(right_text)

    return span[start:left_at], False


def _verify_comparison(frame: ClaimFrame, span: str) -> FrameVerdict:
    """Check a directional comparison against the words between its entities.

    This handles the ordinary "A outperforms B" form. A sentence that nests
    clauses, negates the cue, or compares through a pronoun is unverifiable, not
    contradicted: the check is deliberately narrow so its ``grounded`` verdict
    means something.
    """
    if not frame.subject or not frame.object:
        return FrameVerdict(UNVERIFIABLE, "a comparison needs both entities")

    if not _contains(span, frame.subject) or not _contains(span, frame.object):
        return FrameVerdict(UNVERIFIABLE, "both entities are not stated in the span")

    direction = frame.direction or ">"

    if direction not in {">", "<"}:
        return FrameVerdict(UNVERIFIABLE, f"direction {direction!r} has no mechanical check")

    between = _between(span, frame.subject, frame.object)

    if between is None:
        return FrameVerdict(UNVERIFIABLE, "the entities have no ordered relation in the span")

    body, subject_first = between
    positive = any(cue in body for cue in _POSITIVE)
    negative = any(cue in body for cue in _NEGATIVE)

    if positive == negative:
        return FrameVerdict(UNVERIFIABLE, "the span states no single direction between them")

    # A positive cue between the two names means the first name is the stronger.
    first_is_stronger = positive
    agrees = first_is_stronger == (subject_first if direction == ">" else not subject_first)

    if agrees:
        return FrameVerdict(GROUNDED, "the direction matches the span")

    return FrameVerdict(
        CONTRADICTED,
        f"the span describes the opposite direction for {frame.subject!r} and {frame.object!r}",
        evidence=body.strip(),
    )


def verify_frame(frame: ClaimFrame, texts: Mapping[str, str]) -> FrameVerdict:
    """Verify one frame against the chunk text the run recorded.

    ``texts`` maps ``chunk_uuid -> chunk_text``. A missing chunk is
    ``unverifiable``: the run may have cited an id it never hydrated, and that is
    a citation problem, not a contradiction.
    """
    if not frame.chunk_id:
        return FrameVerdict(UNVERIFIABLE, "the frame cites no chunk")

    text = texts.get(frame.chunk_id)

    if not text:
        return FrameVerdict(UNVERIFIABLE, "the cited chunk has no recorded text")

    span = normalize_span(frame.quote)

    if not span:
        return FrameVerdict(UNVERIFIABLE, "the frame carries no quote")

    if span not in normalize_span(text):
        return FrameVerdict(UNVERIFIABLE, "the quote is not verbatim in the cited chunk")

    if frame.predicate == "measures":
        return _verify_measurement(frame, span, frame.quote)

    if frame.predicate == "compares":
        return _verify_comparison(frame, span)

    if frame.predicate in {"reports", "uses", "evaluated_on", "addresses"}:
        return _verify_attribution(frame, span)

    return FrameVerdict(UNVERIFIABLE, f"predicate {frame.predicate!r} has no mechanical check")


def verify_units(
    units: Sequence[tuple[str, Sequence[str]]],
    frames: Sequence[ClaimFrame],
    texts: Mapping[str, str],
    *,
    run_id: str = "",
) -> ClaimReport:
    """Verify every frame and return the report for one answer.

    ``units`` is ``[(claim_text, [chunk_id, ...]), ...]`` in the answer's order,
    as produced by :func:`lens.observability.citations.claim_units`.
    """
    report = ClaimReport(run_id=run_id, units=len(units))
    report.frames = list(frames)
    report.verdicts = [verify_frame(frame, texts) for frame in frames]

    return report


#: Phrases that mark a line as the heading of the answer's own limitations section.
#: Substring matching, not a whitelist of headings: three recorded runs produced
#: three different wordings ("Claims I could not support with the verified chunks in
#: this pass", "Claims we could not support from the inspected chunks", "Unsupported
#: with current evidence (not claimed above)"), so any list of exact strings goes
#: stale on the next run.
GAP_PHRASES: tuple[str, ...] = (
    "unsupported",
    "could not support",
    "cannot support",
    "could not verify",
    "not supported by",
)

#: Longest line treated as a heading rather than as prose that happens to match.
#: Recorded headings run 18-84 characters, so the bound has to clear the longest of
#: them. What separates a heading from prose that happens to contain the words is
#: the first letter: a heading starts a line with a capital.
GAP_HEADING_MAX_CHARS = 120


def strip_self_reported_gaps(answer: str) -> str:
    """Return ``answer`` without a trailing section that lists unsupported claims.

    An answer that ends with "claims I could not support: ..." is being honest about
    its own limits. A list of disclaimed items is not a finding: the aggregate gate
    reads it as unsupported text, and the claim pass would count each line as a
    claim that no chunk can state. Measured on three recorded answers, two were
    rejected by the aggregate alone while every claim unit scored 0.26-0.56.

    Nothing is deleted from the run: this only decides what the gate and the claim
    pass are asked about. The full answer stays in ``answer.md`` and in the
    trajectory.
    """
    lines = answer.splitlines()

    for index, line in enumerate(lines):
        raw = line.strip()
        bold = re.match(r"^\*\*(.+?)\*\*", raw)
        candidate = bold.group(1) if bold else raw
        heading = candidate.strip().strip("*_#`: ").strip()

        if not heading or len(heading) > GAP_HEADING_MAX_CHARS:
            continue

        # A heading, not a sentence: short, starts a line with a capital, and does
        # not end in a full stop -- unless it is marked as a heading (bold or #),
        # where a trailing period is common: `**Could not support.**`. The model
        # also writes the gap text on the same line as the bold heading, so the
        # bold span at the start is enough.
        marked = bool(bold) or raw.startswith(("#", "__")) or raw.endswith("**")

        if (heading.endswith(".") and not marked) or not heading[:1].isupper():
            continue

        lowered = heading.lower()

        if any(phrase in lowered for phrase in GAP_PHRASES):
            return "\n".join(lines[:index]).rstrip()

    return answer


def quoted_spans(text: str) -> list[str]:
    """Return the quoted spans in one claim unit, in reading order."""
    return [_quoted_text(match) for match in _QUOTED.finditer(text or "")]


def split_long_units(
    units: Sequence[tuple[str, Sequence[str]]],
    *,
    max_words: int = 60,
    min_words: int = 8,
) -> list[tuple[str, list[str]]]:
    """Split a long bullet into sentence-sized claim units.

    A review bullet can hold several claims and a dozen ids; treating it as one
    unit makes coverage a paragraph measure and gives the gate one question for
    ten assertions. Sentences that name an id keep that id; a sentence that names
    none inherits the parent's ids, so a split never drops evidence.
    """
    result: list[tuple[str, list[str]]] = []

    for text, ids in units:
        body = " ".join(text.split())

        if len(body.split()) <= max_words:
            result.append((body, [str(value) for value in ids]))
            continue

        for sentence in re.split(r"(?<=[.!?])\s+", body):
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence.split()) < min_words:
                if result:
                    previous, previous_ids = result[-1]
                    result[-1] = (f"{previous} {sentence}", previous_ids)
                else:
                    result.append((sentence, [str(value) for value in ids]))
                continue

            named = [str(value) for value in ids if str(value) in sentence]
            result.append((sentence, named or [str(value) for value in ids]))

    return result


def _closest_quote(quote: str, candidates: Sequence[str]) -> str:
    """Return the candidate that shares the most content with ``quote``."""
    if len(candidates) == 1:
        return candidates[0]

    wanted = set(_content_tokens(quote))

    return max(
        candidates,
        key=lambda candidate: len(wanted & set(_content_tokens(candidate))) / max(1, len(wanted)),
    )


def align_quotes(
    frames: Sequence[ClaimFrame],
    answer_quotes: Mapping[int, Sequence[str]],
) -> list[ClaimFrame]:
    """Point every frame at the span the answer displays, when it quotes one.

    The extractor chooses a span from the chunk; the answer may display a
    different one. Left alone, the ledger can call a claim grounded while the
    reader cannot find the displayed quote in the chunk. This replaces the
    frame's quote with the answer's own span and records where it came from.
    A unit with no quotation keeps the extractor's chunk span.
    """
    aligned: list[ClaimFrame] = []

    for frame in frames:
        candidates = list(answer_quotes.get(frame.unit_index) or ())

        if not candidates:
            aligned.append(replace(frame, quote_source="chunk"))
            continue

        aligned.append(
            replace(
                frame,
                quote=_closest_quote(frame.quote, candidates),
                quote_source="answer",
            )
        )

    return aligned


def parse_frames(
    payload: Mapping[str, Any] | None,
    units: Sequence[tuple[str, Sequence[str]]],
    *,
    allowed_units: set[int] | None = None,
) -> tuple[list[ClaimFrame], list[str]]:
    """Return ``(frames, errors)`` from one extractor reply.

    The host validates the two things the model cannot be trusted with: the
    ``unit_index`` must address a real claim unit, and the ``chunk_id`` must be
    one the unit cited. A frame that fails either is dropped, never repaired.

    ``allowed_units`` narrows the check to the units one batched call was shown,
    so a frame cannot address a unit the model never saw.
    """
    errors: list[str] = []
    raw = payload.get("frames") if isinstance(payload, Mapping) else None

    if not isinstance(raw, list):
        return [], ["reply had no frames array"]

    frames: list[ClaimFrame] = []

    for entry in raw:
        if not isinstance(entry, Mapping):
            errors.append("a frame was not an object")
            continue

        try:
            unit_index = int(entry.get("unit_index"))
        except (TypeError, ValueError):
            errors.append("a frame had no integer unit_index")
            continue

        if not 0 <= unit_index < len(units):
            errors.append(f"frame addressed unit {unit_index}, which is out of range")
            continue

        if allowed_units is not None and unit_index not in allowed_units:
            errors.append(f"frame addressed unit {unit_index}, which this call was not shown")
            continue

        _text, ids = units[unit_index]
        allowed = {str(chunk_id) for chunk_id in ids}
        chunk_id = entry.get("chunk_id")
        chunk_id = None if chunk_id is None else str(chunk_id)

        if chunk_id is not None and chunk_id not in allowed:
            errors.append(f"frame cited {chunk_id!r}, which its unit did not cite")
            continue

        if chunk_id is None and allowed:
            # A unit with one cited chunk is unambiguous; stamp it so the
            # verifier does not lose a frame over a field the model omitted.
            chunk_id = next(iter(allowed)) if len(allowed) == 1 else None

        predicate = str(entry.get("predicate") or "other")

        if predicate not in PREDICATES:
            predicate = "other"

        frames.append(
            ClaimFrame(
                unit_index=unit_index,
                text=str(entry.get("text") or units[unit_index][0]),
                predicate=predicate,
                subject=str(entry.get("subject") or ""),
                object=None if entry.get("object") is None else str(entry.get("object")),
                benchmark=None if entry.get("benchmark") is None else str(entry.get("benchmark")),
                metric=None if entry.get("metric") is None else str(entry.get("metric")),
                value=None if entry.get("value") is None else str(entry.get("value")),
                unit=None if entry.get("unit") is None else str(entry.get("unit")),
                direction=None if entry.get("direction") is None else str(entry.get("direction")),
                quote=str(entry.get("quote") or ""),
                chunk_id=chunk_id,
            )
        )

    return frames, errors


__all__ = [
    "CONTRADICTED",
    "GROUNDED",
    "PREDICATES",
    "STATUSES",
    "UNVERIFIABLE",
    "ClaimFrame",
    "ClaimReport",
    "FrameVerdict",
    "align_quotes",
    "parse_frames",
    "quoted_spans",
    "split_long_units",
    "verify_frame",
    "verify_units",
]
