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

import json

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from lens import config
from lens.clients import LensClient
from lens.models import Question
from lens.observability.trajectory import TraceCtx

Verdict = Literal["pass", "fail", "uncertain"]

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
        """Return a JSON-serializable verdict for the trajectory.

        The shape is **stable** (all four keys, ``None`` where unknown) because the
        M10b timeline and the M11 report both read it; a key that appears only for
        some verdicts forces every reader to branch on the verdict first.
        """
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "failing_criterion": self.failing_criterion,
            "tier": self.tier,
        }


def build_sufficiency_question(stage: str, artifact: Mapping[str, Any]) -> Question:
    """Build the JEV ``noul`` question used by tier 2 for a **stage**.

    ``criteria`` must be exactly ``{"true": ..., "false": ...}`` (see
    :class:`lens.models.Question`). The question is about the *stage requirement*
    ("does this output satisfy what the stage was for?"), not about the stage's
    name — ``instructions`` names the stage so the verdict is auditable.

    Keep the state small: it is built from ``artifact["summary"]``, never the raw
    payload (JEV has a 32 K cap).

    ``artifact`` contributes only its **summary size**: the question text must not
    depend on the score, or tier 1 and tier 2 would be judging different things.
    """
    requirements = {
        "retrieval": (
            "the retrieved chunks contain the evidence needed to answer the query, "
            "covering every objective and problem the query names"
        ),
        "kg": (
            "the knowledge graph's entities and relations reflect what the selected "
            "chunks actually say, with no invented links and no merged distinct entities"
        ),
    }
    requirement = requirements.get(
        stage, f"the output fulfils the purpose of the {stage} stage"
    )
    fields = len(artifact.get("summary") or {})

    return Question(
        type="noul",
        instructions=(
            f"Stage under review: {stage}. Decide whether {requirement}. "
            f"The state holds a bounded summary of the stage output ({fields} fields). "
            "Answer true only if the output clearly satisfies the stage requirement; "
            "answer false if something needed is missing, unrelated or unsupported, "
            "and name the requirement that failed."
        ),
        criteria={
            "true": f"the output satisfies the {stage} stage requirement",
            "false": f"the output does not satisfy the {stage} stage requirement",
        },
    )


def build_groundedness_question(answer: str, citations: Sequence[str]) -> Question:
    """Build the JEV ``noul`` question used by the **final** gate.

    "Is every claim in the answer supported by the cited chunks?" — ``citations``
    are the ``chunk_uuid`` values the answer relies on. This is the guardrail row
    of §3.x.3 ("final guardrail / hallucination check"), so a ``false`` verdict
    means "do not accept this answer yet", not "the run failed".

    Same ``criteria == {"true", "false"}`` contract; keep the state to the answer
    plus the citation ids (the chunk text is already in the sandbox).
    """
    citation_list = ", ".join(str(citation) for citation in citations) or "(none cited)"
    excerpt = (answer or "").strip()

    if len(excerpt) > 2000:
        excerpt = excerpt[:2000] + " …"

    return Question(
        type="noul",
        instructions=(
            "Decide whether every factual claim in the answer is supported by the cited "
            f"chunks. Citations given: {citation_list}. Answer false if any claim is "
            "uncited, contradicted by its citation, or supported by a chunk that does "
            "not contain it - a number, dataset or comparison with no source counts as "
            "unsupported. An answer that cites nothing at all - placeholders such as "
            "'[see above]' or 'the retrieved evidence' are not citations - is "
            f"unsupported. Answer under review (truncated): {excerpt}"
        ),
        criteria={
            "true": "every factual claim is supported by the cited chunks",
            "false": "at least one claim is uncited, contradicted, or unsupported",
        },
    )


def _read_answer(answer: Mapping[str, Any]) -> tuple[Verdict, float | None, str | None]:
    """Map one JEV answer object to ``(verdict, confidence, failing_criterion)``.

    The transport contract promises "typed answers with confidence/probabilities"
    but never pins the key names, so every read is defensive: an unrecognised shape
    becomes ``uncertain`` (tier 3 takes over) instead of raising. A strict parser
    here would turn a schema drift into a silent "always passes" gate.
    """
    raw = answer.get("answer", answer.get("value", answer.get("verdict")))

    # The JEV ``noul`` primitive answers with ``{"type": "noul", "noul": 0.03}`` -
    # no ``answer`` key at all. Reading only ``answer``/``value``/``verdict`` made
    # **every** gate verdict ``uncertain``, and since only ``fail`` rejects, the
    # final groundedness gate silently accepted a template answer that cited
    # nothing (M9 smoke, LOG-115). ``noul`` is a probability that the proposition
    # holds (0.96 on a plainly true example in the JEV contract, 0.03 on that
    # template answer), so it maps at 0.5.
    if raw is None and isinstance(answer.get("noul"), (int, float)) and not isinstance(
        answer.get("noul"), bool
    ):
        noul = float(answer["noul"])
        raw = noul >= 0.5
        answer = {**answer, "confidence": noul}
    confidence = answer.get("confidence")

    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = None
    else:
        confidence = float(confidence)

    if isinstance(raw, bool):
        verdict: Verdict = "pass" if raw else "fail"
    elif isinstance(raw, str):
        normalised = raw.strip().lower()

        if normalised in {"true", "yes", "pass"}:
            verdict = "pass"
        elif normalised in {"false", "no", "fail"}:
            verdict = "fail"
        else:
            verdict = "uncertain"
    else:
        verdict = "uncertain"

    criterion = (
        answer.get("failing_criterion")
        or answer.get("criterion")
        or answer.get("reason")
    )

    return verdict, confidence, str(criterion)[:300] if criterion else None


def _log_verdict(
    client: LensClient, trace_ctx: TraceCtx, stage: str, verdict: StageVerdict
) -> None:
    """Log one ``gate.verdict`` event (best-effort: observability must not decide).

    The JEV call itself is logged by the transport (``jev.decisions``); this event
    carries the **decision** — stage, verdict, confidence, tier — which is what the
    M11 report counts.
    """
    trajectory = getattr(client, "trajectory", None)

    if trajectory is None:
        return

    try:
        trajectory.log_event(
            trace_ctx=trace_ctx,
            call_type="note",
            action="gate.verdict",
            stage=stage,
            confidence=verdict.confidence,
            verdict=verdict.verdict,
            extra={"tier": verdict.tier, "criterion": verdict.failing_criterion},
        )
    except Exception:  # noqa: BLE001 - a log failure must not change the verdict
        pass


def evaluate_stage(
    stage: str,
    artifact: Mapping[str, Any],
    *,
    client: LensClient,
    trace_ctx: TraceCtx,
    threshold: float | None = None,
    budget: Any | None = None,
) -> StageVerdict | None:
    """Run the three-tier gate and return a :class:`StageVerdict`.

    Returns ``None`` when ``config.GATE_ENABLED`` is off (the ablation arm) —
    **before** any client call — so the caller attaches no ``gate`` at all.

    Tier 1: compare ``artifact["score"]`` with ``threshold`` (or
    ``config.STAGE_THRESHOLDS[stage]``). **Skip tier 1 when the score is missing or no
    threshold is calibrated** — do not invent one, and do not treat a missing
    score as a failure. Tier 2: JEV ``noul`` with
    :func:`build_sufficiency_question` (or :func:`build_groundedness_question`
    when ``stage == "final"``). Tier 3: leave ``verdict="uncertain"`` with the
    failing criterion; the orchestrator puts it in the envelope and the root's
    next turn is the judgment. Log ``stage``, ``confidence`` and ``verdict`` on
    the trajectory.

    ``budget`` is optional and only used for accounting: the decision layer is
    cheap, not free, and H₃ counts it (:meth:`Budget.record_jev`).
    """
    if not config.GATE_ENABLED:
        # First statement: the ablation arm must not touch the client at all.
        return None

    summary = artifact.get("summary") or {}
    score = artifact.get("score")
    limit = threshold if threshold is not None else config.STAGE_THRESHOLDS.get(stage)

    # Tier 1 — only ever *accepts*. A calibrated τ_pass says "good enough", not
    # "bad": without a lower τ_fail, a low score means "ask tier 2".
    if score is not None and limit is not None:
        try:
            passed = float(score) >= float(limit)
        except (TypeError, ValueError):
            passed = False

        if passed:
            verdict = StageVerdict(
                verdict="pass", confidence=float(score), tier="threshold"
            )
            _log_verdict(client, trace_ctx, stage, verdict)

            return verdict

    if stage == "final":
        question = build_groundedness_question(
            str(summary.get("answer") or ""), list(summary.get("citations") or [])
        )
    else:
        question = build_sufficiency_question(stage, artifact)

    state = json.dumps(
        {"stage": stage, "summary": summary}, ensure_ascii=False, default=str
    )

    try:
        result = client.jev(state, {"gate": question}, trace_ctx=trace_ctx)
    except Exception as exc:  # noqa: BLE001 - a dead decision layer must not end the run
        verdict = StageVerdict(
            verdict="uncertain",
            failing_criterion=f"decision layer unavailable: {type(exc).__name__}",
            tier="jev",
        )
        _log_verdict(client, trace_ctx, stage, verdict)

        return verdict

    if budget is not None:
        try:
            budget.record_jev(result)
        except Exception:  # noqa: BLE001 - accounting must not change the verdict
            pass

    verdict_value, confidence, criterion = _read_answer(
        (getattr(result, "answers", None) or {}).get("gate") or {}
    )
    verdict = StageVerdict(
        verdict=verdict_value,
        confidence=confidence,
        # A criterion is only meaningful when the gate did not pass.
        failing_criterion=criterion if verdict_value != "pass" else None,
        tier="jev",
    )
    _log_verdict(client, trace_ctx, stage, verdict)

    return verdict


__all__ = [
    "GATED_STAGES",
    "StageVerdict",
    "Verdict",
    "build_groundedness_question",
    "build_sufficiency_question",
    "evaluate_stage",
]
