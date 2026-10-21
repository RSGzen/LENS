"""Per-stage confidence gate and the final groundedness check (Decision Layer §3.x.4b).

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4m/§6; ``Decision Layer (Draft).md``
§3.x.3 (the stage → decision table), §3.x.4b (the three tiers).

Who runs it
-----------
**The orchestrator, automatically — not the RLM.** The draft calls the per-stage
gate "a control rule, not a classifier": it drives the F-01 bounded backtracking,
so its input and its cost have to be decided host-side. The RLM's only lever is
``backtrack(stage, reason)``; it cannot re-run a gate, because a model that decides
whether it passed will pass itself — and that would confound the decision-layer
ablation.

Two gates, same machinery (the "double" gate):

1. **Per stage** — after the stage-completing tool, once per stage per run:
   ``rerank_chunks`` → ``retrieval``, ``build_kg`` → ``kg``.
2. **Final** — on ``FINAL_ANSWER``, stage ``"final"``: is every claim supported by
   the cited chunks? A reject hands the failing criterion back for one more turn
   (inside ``MAX_ITERATIONS``), so a fabricated answer is not silently accepted.

Tiers (cheapest first, §3.x.4b)
-------------------------------
1. a calibrated confidence threshold on ``artifact["score"]`` — ~free;
2. a JEV ``noul`` sufficiency check — one prefill pass, only in the ambiguous band;
3. the root RLM's judgment — **no extra call**: the verdict rides in the RPC
   envelope and the model's next turn *is* the judgment. Tier 3 is not a tool call.

The artifact
------------
The gate never sees a raw tool payload: ``build_kg``'s node-link graph would blow
JEV's 32 K state cap. ``Runtime._dispatch_tool`` passes

    {"score": float | None, "summary": <bounded dict>}

``score`` is ``None`` where a stage has no calibrated score (retrieval keeps only
an ordering, the KG has no confidence) — then tier 1 is skipped and tier 2 decides.
``summary`` is what goes into the JEV state, so it stays small.

Ablation
--------
``config.GATE_ENABLED = 0`` (``LENS_GATE=0``) is the decision-layer-OFF arm:
:func:`evaluate_stage` returns ``None`` **before** any client call, and the
orchestrator then attaches no ``gate`` key at all. Off must look like absent, not
like a gate that always passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from lens import config
from lens.clients import LensClient
from lens.models import Question
from lens.observability.trajectory import TraceCtx

Verdict = Literal["pass", "fail", "uncertain"]

# Calibrated thresholds are filled from the Decision Layer draft / M11 data.
# Do not invent numbers here; an empty map forces the caller to pass one.
# Keys are stage names: "retrieval" / "kg" / "final".
STAGE_THRESHOLDS: dict[str, float] = {}

# Where the gate runs (M8). Map **stage-completing** tool -> stage. Do not add the
# earlier tools of a stage: ``shallow_match`` returns a bare ``list[str]`` of ids
# and ``rerank_papers`` only ids too, so there is nothing to evaluate — the first
# version of this map gated ``shallow_match`` and asked JEV whether 100 ids were
# "sufficient". One gate per stage per run: rerunning it would only add cost.
GATED_STAGES: dict[str, str] = {
    "rerank_chunks": "retrieval",
    "build_kg": "kg",
}


@dataclass
class StageVerdict:
    """Outcome of one stage gate."""

    verdict: Verdict
    confidence: float | None = None
    failing_criterion: str | None = None
    tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable verdict for the trajectory."""
        raise NotImplementedError("M8 scaffold: implement StageVerdict.to_dict")


def build_sufficiency_question(stage: str, artifact: Mapping[str, Any]) -> Question:
    """Build the JEV ``noul`` question used by tier 2 for a **stage**.

    ``criteria`` must be exactly ``{"true": ..., "false": ...}`` (see
    :class:`lens.models.Question`). The question is about the *stage requirement*
    ("does this output satisfy what the stage was for?"), not about the stage's
    name — ``instructions`` names the stage so the verdict is auditable.

    Keep the state small: it is built from ``artifact["summary"]``, never the raw
    payload (JEV has a 32 K cap).
    """
    raise NotImplementedError("M8 scaffold: implement build_sufficiency_question")


def build_groundedness_question(answer: str, citations: Sequence[str]) -> Question:
    """Build the JEV ``noul`` question used by the **final** gate.

    "Is every claim in the answer supported by the cited chunks?" — ``citations``
    are the ``chunk_uuid`` values the answer relies on. This is the guardrail row
    of §3.x.3 ("final guardrail / hallucination check"), so a ``false`` verdict
    means "do not accept this answer yet", not "the run failed".

    Same ``criteria == {"true", "false"}`` contract; keep the state to the answer
    plus the citation ids (the chunk text is already in the sandbox).
    """
    raise NotImplementedError("M8 scaffold: implement build_groundedness_question")


def evaluate_stage(
    stage: str,
    artifact: Mapping[str, Any],
    *,
    client: LensClient,
    trace_ctx: TraceCtx,
    threshold: float | None = None,
) -> StageVerdict | None:
    """Run the three-tier gate and return a :class:`StageVerdict`.

    Returns ``None`` when ``config.GATE_ENABLED`` is off (the ablation arm) —
    **before** any client call — so the caller attaches no ``gate`` at all.

    Tier 1: compare ``artifact["score"]`` with ``threshold`` (or
    ``STAGE_THRESHOLDS[stage]``). **Skip tier 1 when the score is missing or no
    threshold is calibrated** — do not invent one, and do not treat a missing
    score as a failure. Tier 2: JEV ``noul`` with
    :func:`build_sufficiency_question` (or :func:`build_groundedness_question`
    when ``stage == "final"``). Tier 3: leave ``verdict="uncertain"`` with the
    failing criterion; the orchestrator puts it in the envelope and the root's
    next turn is the judgment. Log ``stage``, ``confidence`` and ``verdict`` on
    the trajectory.
    """
    raise NotImplementedError("M8 scaffold: implement evaluate_stage")


__all__ = [
    "GATED_STAGES",
    "STAGE_THRESHOLDS",
    "StageVerdict",
    "Verdict",
    "build_groundedness_question",
    "build_sufficiency_question",
    "evaluate_stage",
]
