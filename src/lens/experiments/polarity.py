"""Polarity probe: four JEV calls that decide whether the gate still reads `noul`.

Why this exists
---------------
The gate maps `noul` at 0.5: a proposition the model believes true scores high, one
it believes false scores low. That mapping is the *only* reason a threshold of 0.5
means anything, and it is a property of a third-party alpha endpoint — a shape
change, a flipped sign or a renamed field turns every verdict into noise while the
pipeline keeps running and reports confidences.

Four probes cost about $0.00002 and answer the question before a wave spends money.
Each probe is one obvious case (a true claim, a false claim, an answer its excerpt
supports, an answer its excerpt contradicts), so "skewed" means the layer is not
reading the shape the thresholds assume — not that the model got something hard
wrong.

Deliberately **not** a calibration: it does not fit a threshold, it checks polarity.
Used by ``scripts/run_campaign.py`` as the wave preflight, and it must stay cheap
enough to run every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from lens.models import JevResult, Question

#: A proposition nobody can argue with, and its mirror image.
TRUE_STATE = "Water boils at 100 degrees Celsius at sea level."
FALSE_STATE = "Water boils at 40 degrees Celsius at sea level."
TRUE_CLAIM = "Water boils at 100 degrees Celsius at sea level."
FALSE_CLAIM = "Water boils at 40 degrees Celsius at sea level."

#: An answer that its excerpt supports, and one its excerpt contradicts.
EXCERPT = "The reported cost was 0.25 USD per query over five runs. No cache hits were observed."
SUPPORTED_ANSWER = "The cost was 0.25 USD per query."
CONTRADICTED_ANSWER = "The cost was 4.80 USD per query."

#: Below this the layer reads "false", above it "true". The gate's own mapping.
MIDPOINT = 0.5


@dataclass
class Probe:
    """One polarity case: what was asked, what it should score, what it scored."""

    name: str
    expect_high: bool
    noul: float | None
    error: str = ""

    @property
    def ok(self) -> bool:
        if self.noul is None:
            return False
        return self.noul >= MIDPOINT if self.expect_high else self.noul < MIDPOINT

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expect_high": self.expect_high,
            "noul": self.noul,
            "ok": self.ok,
            "error": self.error,
        }


@dataclass
class PolarityResult:
    """The four probes plus what they cost."""

    probes: list[Probe] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens_in: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.probes) and all(probe.ok for probe in self.probes)

    @property
    def skewed(self) -> list[str]:
        return [probe.name for probe in self.probes if not probe.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "skewed": self.skewed,
            "cost_usd": round(self.cost_usd, 8),
            "tokens_in": self.tokens_in,
            "probes": [probe.to_dict() for probe in self.probes],
        }


def _statement_question(claim: str) -> Question:
    """Ask whether a bare claim is true — the cleanest test of the `noul` sign."""
    return Question(
        type="noul",
        instructions=f"Is the following statement true?\n\n{claim}",
        criteria={"true": "the statement is factually true", "false": "it is not"},
    )


def _support_question(answer: str, excerpt: str) -> Question:
    """Ask whether an answer is supported by a fixed excerpt."""
    return Question(
        type="noul",
        instructions=(
            "Is every factual claim in the answer supported by the excerpt? "
            f"Excerpt: {excerpt} Answer: {answer}"
        ),
        criteria={"true": "the answer is supported", "false": "it is not"},
    )


def cases() -> list[tuple[str, bool, str, Question]]:
    """Return the four probes as ``(name, expect_high, state, question)``."""
    return [
        ("true-claim", True, TRUE_STATE, _statement_question(TRUE_CLAIM)),
        ("false-claim", False, FALSE_STATE, _statement_question(FALSE_CLAIM)),
        ("supported-answer", True, EXCERPT, _support_question(SUPPORTED_ANSWER, EXCERPT)),
        (
            "contradicted-answer",
            False,
            EXCERPT,
            _support_question(CONTRADICTED_ANSWER, EXCERPT),
        ),
    ]


def _read_noul(result: JevResult, name: str) -> tuple[float | None, str]:
    """Return ``(noul, error)`` from one answer, without guessing a shape."""
    answer: Mapping[str, Any] = result.answers.get("polarity") or {}

    if not isinstance(answer, Mapping):
        return None, f"answer is {type(answer).__name__}, not a mapping"

    try:
        return float(answer["noul"]), ""
    except (KeyError, TypeError, ValueError):
        return None, f"no numeric `noul` in answer keys {sorted(answer)}"


def probe_polarity(
    jev_call: Callable[[str, Mapping[str, Question]], JevResult],
) -> PolarityResult:
    """Run the four probes through ``jev_call`` and return what they scored.

    ``jev_call`` is injected as ``(state, questions) -> JevResult``, the same shape
    the direction verifier takes, so the probe is testable without the network and
    the caller owns the client, the trace context and the accounting.
    """
    result = PolarityResult()

    for name, expect_high, state, question in cases():
        call = jev_call(state, {"polarity": question})
        result.cost_usd += call.cost_usd
        result.tokens_in += call.tokens_in

        noul, error = _read_noul(call, name)
        result.probes.append(Probe(name, expect_high, noul, error))

    return result


__all__ = [
    "MIDPOINT",
    "PolarityResult",
    "Probe",
    "cases",
    "probe_polarity",
]
