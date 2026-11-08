"""Score a four-bucket segmentation against a gold annotation.

The comparison is deliberately **lexical**: the segmenter is instructed to copy
the user's own wording, so a normalised containment rule is a fair test of
whether a clause is present. It does not judge whether a paraphrase is
semantically equivalent; near-matches are for a human to review.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

BUCKETS = ("objectives", "problems", "constraints", "requirements")
_WORD = re.compile(r"[a-z0-9]+")


def normalise_clause(text: str) -> str:
    """Lowercase, collapse whitespace, strip edge punctuation."""
    return " ".join(str(text or "").split()).lower().strip(".!?;:()[]\"'")


def clause_matches(gold: str, produced: str) -> bool:
    """Return whether ``produced`` expresses the ``gold`` clause.

    A normalised containment passes; otherwise every content word (longer than
    two characters) of the gold must be present in the produced clause. That
    tolerates the segmenter adding a word, not replacing one.
    """
    expected = normalise_clause(gold)
    observed = normalise_clause(produced)

    if not expected or not observed:
        return False

    if expected in observed:
        return True

    words = [word for word in _WORD.findall(expected) if len(word) > 2]

    if not words:
        return False

    present = set(_WORD.findall(observed))
    return all(word in present for word in words)


def bucket_scores(gold: Sequence[str], produced: Sequence[str]) -> dict[str, Any]:
    """Return recall/precision counts for one bucket."""
    gold = [str(item) for item in gold if str(item).strip()]
    produced = [str(item) for item in produced if str(item).strip()]

    matched_gold = sum(
        1 for expected in gold if any(clause_matches(expected, item) for item in produced)
    )
    matched_produced = sum(
        1 for item in produced if any(clause_matches(expected, item) for expected in gold)
    )

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    return {
        "gold": len(gold),
        "produced": len(produced),
        "matched_gold": matched_gold,
        "matched_produced": matched_produced,
        "recall": rate(matched_gold, len(gold)),
        "precision": rate(matched_produced, len(produced)),
    }


def score_buckets(
    gold_buckets: Mapping[str, Any], produced: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return one score dict per bucket in ``BUCKETS``."""
    result: dict[str, dict[str, Any]] = {}

    for bucket in BUCKETS:
        gold = gold_buckets.get(bucket) or []
        got = produced.get(bucket) or []
        result[bucket] = bucket_scores(
            gold if isinstance(gold, list) else [gold],
            got if isinstance(got, list) else [got],
        )

    return result
