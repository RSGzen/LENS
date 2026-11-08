"""Score an answer against an instrument's gold claim set.

The I5/I6 gold is a list of claims, each with a paper id, a chunk id and a value.
The metrics here are deliberately **text-level**: they read `answer.md` and the
run's recorded chunk text, never the claim ledger. The ledger can be stale when
a later draft replaces the answer (a partial run may keep the previous draft's
`claims.json`), so it is reported separately and never used as evidence.

Value matching is a normalised containment: a number matches within 0.5 % (or
0.01 absolute), a phrase matches when its content words are present. That is a
weak rule and the thesis states it; it is the weakest rule that still has a
consumer, not a semantic judgement.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[a-z0-9]+")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def normalise_value(value: Any) -> str:
    """Lowercase, collapse whitespace, drop commas and edge punctuation."""
    text = " ".join(str(value or "").split()).lower()
    return text.replace(",", "").strip(".!?;:()[]\"'")


def _as_float(value: str) -> float | None:
    match = _NUMBER.search(value)
    return float(match.group(0).replace(",", "")) if match else None


def _close(expected: float, observed: float) -> bool:
    return abs(expected - observed) <= max(0.01, abs(expected) * 0.005)


def value_matches(gold: Any, text: str) -> bool:
    """Return whether ``text`` states ``gold`` under the normalised rule."""
    expected = normalise_value(gold)
    observed = normalise_value(text)

    if not expected:
        return False

    number = _as_float(expected)

    if number is not None:
        return any(
            _close(number, value)
            for match in _NUMBER.finditer(observed)
            if (value := _as_float(match.group(0))) is not None
        )

    if expected in observed:
        return True

    words = [word for word in _WORD.findall(expected) if len(word) > 2]

    if not words:
        return False

    present = set(_WORD.findall(observed))
    return all(word in present for word in words)


def claim_values(claim: Mapping[str, Any]) -> list[str]:
    """Return the value plus any aliases, dropping blanks and duplicates."""
    values: list[str] = []
    raw = [claim.get("value"), *(claim.get("aliases") or [])]

    for value in raw:
        text = normalise_value(value)

        if text and text not in values:
            values.append(text)

    return values


def claim_matches(claim: Mapping[str, Any], text: str) -> bool:
    """Return whether any of the claim's values is stated in ``text``."""
    return any(value_matches(value, text) for value in claim_values(claim))


def split_claims(answer: str) -> list[str]:
    """Split an answer into non-empty sentence/line units."""
    return [part.strip() for part in _SENTENCE.split(answer or "") if part.strip()]


def cited_ids(sentence: str) -> list[str]:
    """Return the unique chunk and arXiv ids cited in one sentence."""
    seen: list[str] = []
    for match in [*_UUID.finditer(sentence), *_ARXIV.finditer(sentence)]:
        token = match.group(0)
        if token not in seen:
            seen.append(token)
    return seen


def claim_recall(gold_claims: Sequence[Mapping[str, Any]], answer: str) -> tuple[int, list[int]]:
    """Return ``(matched, matched_indexes)`` for the gold claims stated in the answer."""
    units = split_claims(answer)
    matched: list[int] = []

    for index, claim in enumerate(gold_claims):
        if any(claim_matches(claim, unit) for unit in units):
            matched.append(index)

    return len(matched), matched


def citation_precision(
    gold_claims: Sequence[Mapping[str, Any]],
    answer: str,
    texts: Mapping[str, str],
) -> tuple[int, int]:
    """Return ``(supported, judged)`` for cited sentences that state a gold value.

    A sentence is judged when it carries an id **and** states at least one gold
    value. It is supported when one of its cited chunks contains that value.
    """
    judged = 0
    supported = 0

    for unit in split_claims(answer):
        ids = cited_ids(unit)

        if not ids or not any(claim_matches(claim, unit) for claim in gold_claims):
            continue

        judged += 1
        cited_texts = [texts.get(token, "") for token in ids if token in texts]

        if any(
            claim_matches(claim, cited_text) for claim in gold_claims for cited_text in cited_texts
        ):
            supported += 1

    return supported, judged


def gold_chunk_recall(
    gold_claims: Sequence[Mapping[str, Any]], recorded_ids: Iterable[str]
) -> tuple[int, int]:
    """Return ``(present, gold)`` for the gold chunk ids in the funnel's chunks."""
    recorded = {str(chunk_id) for chunk_id in recorded_ids}
    gold = {str(claim["chunk_uuid"]) for claim in gold_claims if claim.get("chunk_uuid")}

    return len(gold & recorded), len(gold)


def citation_recall(gold_claims: Sequence[Mapping[str, Any]], answer: str) -> tuple[int, int]:
    """Return ``(cited, gold)`` for the gold chunk ids named in the answer."""
    cited = set(_UUID.findall(answer or ""))
    gold = {str(claim["chunk_uuid"]) for claim in gold_claims if claim.get("chunk_uuid")}

    return len(gold & cited), len(gold)


def ledger_is_stale(answer: str, payload: Mapping[str, Any] | None) -> bool:
    """Return whether ``claims.json`` describes something other than ``answer``.

    A partial run can keep the ledger of an earlier draft. If no frame's unit
    text appears in the final answer, the ledger is not about this answer.
    """
    if not payload:
        return False

    if int(payload.get("units") or 0) < 1:
        return False

    if not answer.strip():
        return True

    normalised = " ".join(answer.split()).lower()
    frames = payload.get("frames") or []

    for frame in frames:
        text = " ".join(str(frame.get("text") or "").split())

        if text and text[:80].lower() in normalised:
            return False

    return True
