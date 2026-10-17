r"""M7.3 / Phase 1.5 — KG quality layer: span filter, heuristic coref, direction verifier.

Spec: ``MVP Build Roadmap.md`` §4 (M7), ``Tech Stack (Draft).md`` §4l. Decisions:
LOG-78 (Phase 1.5 approved), LOG-79 (lossless preprocessing + KG on/off).

Stages (all pure except the injected JEV call):

1. **Span filter** — drop triples whose head/tail is a pronoun, determiner, or a
   lowercase adjective-only/generic span (``novels``, ``long-text scenarios``,
   ``target model``). Capitalised names (``Full Context``, ``spaCy``) survive.
2. **Conservative heuristic coref** — resolve pronominal/demonstrative mentions
   (``it``, ``they``, ``this method``, ``our approach``) to the nearest preceding
   **compatible-type** entity in the same chunk. Unresolved mentions are removed
   by the span filter. No fastcoref (deliberate, LOG-78).
3. **Direction verifier (three tiers)** — cue/SVO check → **budgeted JEV `choice`**
   for ambiguous cases → soft flags. Ambiguity is a **multi-signal** gate
   (small ``direction_margin``, cue disagreement, non-decisive types, negation,
   sentence distance), not the raw GLiREL score.

No text rewriting: the lossless normalizer is unchanged (no lowercasing,
stopword removal, punctuation stripping, lemmatization, or negation rewriting).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from lens import config
from lens.models import JevResult, Question

Triple = dict[str, Any]

# ------------------------------------------------------------------ lexicons
PRONOUNS = frozenset({
    "it", "its", "they", "them", "their", "theirs", "he", "him", "his",
    "she", "her", "hers", "we", "us", "our", "ours", "i", "me", "my", "you",
    "your", "yours", "this", "that", "these", "those", "which", "who", "whom",
    "what", "such", "same", "other", "another", "some", "any", "all", "both",
    "each", "either", "neither", "none", "one", "ones",
})
DETERMINERS = frozenset({"the", "a", "an", "this", "that", "these", "those"})
GENERIC_ADJECTIVES = frozenset({
    "novel", "novels", "new", "proposed", "existing", "current", "previous",
    "baseline", "baselines", "state-of-the-art", "sota", "long-text",
    "academic", "exact", "target", "generic", "various", "different",
    "significant", "better", "worse", "best", "worst", "full", "entire",
    "whole", "same", "similar", "multiple", "several",
})
GENERIC_NOUNS = frozenset({
    "scenario", "scenarios", "term", "terms", "name", "names", "value",
    "values", "information", "setting", "settings", "performance", "result",
    "results", "score", "scores", "approach", "approaches", "method", "methods",
    "model", "models", "technique", "techniques", "framework", "frameworks",
    "system", "systems", "algorithm", "algorithms", "task", "tasks", "benchmark",
    "benchmarks", "dataset", "datasets", "corpus", "corpora", "data",
})
GENERIC_SPANS = PRONOUNS | DETERMINERS | GENERIC_ADJECTIVES | GENERIC_NOUNS

# Cue phrases per asymmetric relation (used for the first tier).
RELATION_CUES = {
    "outperforms": r"outperform\w*|better than|surpass\w*|exceed\w*|improve\w* over|beats?\b|outclass\w*",
    "extends": r"extend\w*|build(?:s|ing)? (?:up)?on|improve\w* upon|based on|follow\w* up",
    "evaluated_on": r"evaluat\w* (?:on|with|using)|test\w* (?:on|with)|benchmark\w* (?:on|with)|experiment\w* on",
    "measures": r"measur\w*|report\w*|evaluat\w* (?:with|using|by)|score\w*",
    "uses": r"\buse[sd]?\b|using|employ\w*|leverag\w*|adopt\w*|utiliz\w*",
    "addresses": r"address\w*|tackl\w*|solv\w*|target\w*|focus\w* on",
    "explores": r"explor\w*|investigat\w*|stud(?:y|ies)|examin\w*",
    "compares_to": r"compar\w*|\bversus\b|\bvs\.?\b|\bagainst\b",
}
NEGATION = re.compile(
    r"\b(?:not|no|never|without|fail\w*|unable|worse|inferior|less than|rather than)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-\.]*")


def _words(text: str) -> list[str]:
    return _WORD.findall(text or "")


def is_pronoun_span(text: str) -> bool:
    """True for a pronoun/demonstrative/determiner-headed mention."""
    words = _words(text)
    if not words:
        return True
    folded = [w.casefold().strip(".") for w in words]
    if len(folded) == 1 and folded[0] in PRONOUNS:
        return True
    if folded[0] in {"this", "these", "those", "that"} and len(folded) <= 4:
        return True
    if folded[0] in {"the", "our", "its", "their"} and len(folded) <= 4:
        return True
    return False


def is_generic_span(text: str) -> bool:
    """True for pronoun/determiner/adjective-only/lowercase generic mentions."""
    words = _words(text)
    if not words:
        return True
    folded = [w.casefold().strip(".") for w in words]

    # Pronouns/determiners are always dropped.
    if is_pronoun_span(text):
        return True

    # A capitalised or numeric token signals a proper name: keep it.
    has_proper = any(w[0].isupper() for w in words) or any(ch.isdigit() for ch in text)
    if has_proper:
        return False

    if len(folded) == 1 and folded[0] in GENERIC_SPANS:
        return True
    if all(token in GENERIC_SPANS for token in folded):
        return True
    if folded[-1] in GENERIC_NOUNS:
        return True
    return False


def filter_spans(triples: Sequence[Triple]) -> list[Triple]:
    """Drop triples with a pronoun/determiner/adjective-only/generic endpoint."""
    return [
        triple
        for triple in triples
        if not is_generic_span(triple["head"]) and not is_generic_span(triple["tail"])
    ]


# ------------------------------------------------------------------ coref
def resolve_coref(
    triples: Sequence[Triple], chunk_texts: Mapping[str, str]
) -> list[Triple]:
    """Resolve pronominal mentions to the nearest preceding compatible entity.

    Heuristic only: same chunk, earlier position, same entity type. Unresolved
    mentions are left for :func:`filter_spans` to remove.
    """
    by_chunk: dict[str, list[Triple]] = defaultdict(list)
    for triple in triples:
        by_chunk[triple.get("chunk_uuid")].append(triple)

    out: list[Triple] = []
    for chunk_id, group in by_chunk.items():
        text = chunk_texts.get(chunk_id, "")
        lowered = text.casefold()

        candidates: list[tuple[int, str, str]] = []
        for triple in group:
            for side in ("head", "tail"):
                name = triple[side]
                if is_pronoun_span(name):
                    continue
                position = lowered.find(name.casefold())
                if position >= 0:
                    candidates.append((position, name, triple[f"{side}_type"]))

        for triple in group:
            for side in ("head", "tail"):
                if not is_pronoun_span(triple[side]):
                    continue
                position = lowered.find(triple[side].casefold())
                compatible = [
                    candidate
                    for candidate in candidates
                    if candidate[0] < position and candidate[2] == triple[f"{side}_type"]
                ]
                if compatible:
                    _pos, name, _type = max(compatible, key=lambda c: c[0])
                    triple[side] = name
            out.append(triple)
    return out


# ------------------------------------------------------------------ direction
def sentence_for(text: str, head: str, tail: str) -> str:
    """Return the sentence that contains both mentions (or the best available)."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]
    if not sentences:
        return text or ""
    for sentence in sentences:
        low = sentence.casefold()
        if head.casefold() in low and tail.casefold() in low:
            return sentence
    for sentence in sentences:
        low = sentence.casefold()
        if head.casefold() in low or tail.casefold() in low:
            return sentence
    return sentences[0]


def cue_check(sentence: str, head: str, tail: str, relation: str) -> str:
    """Return ``agree`` / ``disagree`` / ``unknown`` from cue order vs entity order."""
    pattern = RELATION_CUES.get(relation)
    if not pattern:
        return "unknown"
    low = sentence.casefold()
    head_pos = low.find(head.casefold())
    tail_pos = low.find(tail.casefold())
    if head_pos < 0 or tail_pos < 0:
        return "unknown"
    for match in re.finditer(pattern, sentence, re.IGNORECASE):
        if head_pos < match.start() and tail_pos > match.end():
            return "agree"
        if tail_pos < match.start() and head_pos > match.end():
            return "disagree"
    return "unknown"


def has_negation(sentence: str, relation: str) -> bool:
    pattern = RELATION_CUES.get(relation)
    if not pattern:
        return False
    for match in re.finditer(pattern, sentence, re.IGNORECASE):
        window = sentence[max(0, match.start() - 40) : match.end() + 40]
        if NEGATION.search(window):
            return True
    return False


def ambiguity(triple: Triple, sentence: str) -> tuple[bool, list[str]]:
    """Multi-signal gate: is the predicted direction uncertain?"""
    relation = triple.get("relation", "")
    if relation not in RELATION_CUES:
        return (False, [])

    reasons: list[str] = []
    cue = cue_check(sentence, triple["head"], triple["tail"], relation)
    if cue == "disagree":
        reasons.append("cue_disagree")

    if has_negation(sentence, relation):
        reasons.append("negation")

    margin = abs(float(triple.get("direction_margin", 1.0)))
    if margin < config.KG_VERIFY_MARGIN:
        reasons.append("small_margin")

    if (
        triple.get("head_type") == triple.get("tail_type") == "Method"
        and cue == "unknown"
    ):
        reasons.append("non_decisive_types")

    # If the cue explicitly agrees and there is no negation, trust it.
    if cue == "agree" and "negation" not in reasons:
        return (False, [])
    return (bool(reasons), reasons)


def _flip(triple: Triple) -> None:
    """Swap head/tail (and their type metadata) for a reversed relation."""
    for suffix in ("", "_type", "_type_source", "_type_confidence"):
        head_key, tail_key = f"head{suffix}", f"tail{suffix}"
        head_value, tail_value = triple.get(head_key), triple.get(tail_key)
        if head_value is None and tail_value is None:
            continue
        triple[head_key], triple[tail_key] = tail_value, head_value


@dataclass
class VerifyStats:
    budget: int = 0
    ambiguous: int = 0
    verified: int = 0
    flipped: int = 0
    dropped: int = 0
    kept: int = 0
    cost_usd: float = 0.0
    tokens_in: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "ambiguous": self.ambiguous,
            "verified": self.verified,
            "flipped": self.flipped,
            "dropped": self.dropped,
            "kept_after_verify": self.kept,
            "cost_usd": round(self.cost_usd, 8),
            "tokens_in": self.tokens_in,
        }


class DirectionVerifier:
    """Budgeted three-tier direction verifier (cue -> JEV -> soft flag).

    ``jev_call`` is an injected callable ``(state, questions) -> JevResult`` so the
    verifier is testable without the network and can be wired to ``LensClient.jev``
    in the probe/grid.
    """

    def __init__(
        self,
        jev_call: Callable[[str, Mapping[str, Question]], JevResult],
        *,
        budget: int = config.KG_VERIFY_BUDGET,
        high: float = config.KG_VERIFY_HIGH,
        low: float = config.KG_VERIFY_LOW,
        max_state_chars: int = config.KG_VERIFY_MAX_STATE_CHARS,
    ) -> None:
        self.jev_call = jev_call
        self.budget = budget
        self.high = high
        self.low = low
        self.max_state_chars = max_state_chars

    @property
    def fingerprint(self) -> str:
        """Stable id for the cache key (budget + confidence thresholds)."""
        return f"jev:{self.budget}:{self.high}:{self.low}:{self.max_state_chars}"

    def verify(
        self, triples: Sequence[Triple], chunk_texts: Mapping[str, str]
    ) -> tuple[list[Triple], VerifyStats]:
        stats = VerifyStats(budget=self.budget)
        pending: list[tuple[Triple, str]] = []

        for triple in triples:
            sentence = sentence_for(
                chunk_texts.get(triple.get("chunk_uuid"), ""),
                triple["head"],
                triple["tail"],
            )
            is_ambiguous, reasons = ambiguity(triple, sentence)
            triple.setdefault("direction_verified", True)
            triple.setdefault("direction_source", "cue")
            if is_ambiguous:
                stats.ambiguous += 1
                triple["direction_ambiguous"] = True
                triple["direction_reasons"] = reasons
                pending.append((triple, sentence))
            else:
                triple["direction_ambiguous"] = False

        # Most ambiguous first (smallest margin); verify up to the budget.
        pending.sort(key=lambda item: abs(float(item[0].get("direction_margin", 1.0))))
        for triple, sentence in pending[: self.budget]:
            state = sentence[: self.max_state_chars]
            questions = {
                "direction": Question(
                    type="choice",
                    instructions=(
                        "Given the sentence, which direction does it support? "
                        "Use 'neither' if the relation is not stated or is ambiguous."
                    ),
                    criteria={
                        "ab": f"{triple['head']} -> {triple['tail']}",
                        "ba": f"{triple['tail']} -> {triple['head']}",
                        "neither": "not stated or ambiguous",
                    },
                )
            }
            result = self.jev_call(state, questions)
            stats.verified += 1
            stats.cost_usd += result.cost_usd
            stats.tokens_in += result.tokens_in

            answer = result.answers.get("direction", {})
            choice = str(answer.get("choice", ""))
            confidence = float(answer.get("confidence") or 0.0)
            triple["direction_confidence"] = confidence

            if choice == "ab" and confidence >= self.high:
                triple["direction_verified"] = True
                triple["direction_source"] = "jev"
            elif choice == "ba" and confidence >= self.high:
                _flip(triple)
                triple["direction_verified"] = True
                triple["direction_source"] = "jev"
                stats.flipped += 1
            elif choice == "neither" or confidence < self.low:
                triple["_drop"] = True
                stats.dropped += 1
            else:
                triple["direction_verified"] = False
                triple["direction_source"] = "jev_uncertain"

        # Budget-exhausted ambiguous triples stay unverified but are kept.
        for triple, _sentence in pending[self.budget :]:
            triple["direction_verified"] = False
            triple["direction_source"] = "unverified"

        out = [triple for triple in triples if not triple.get("_drop")]
        for triple in out:
            triple.pop("_drop", None)
        stats.kept = len(out)
        return out, stats
