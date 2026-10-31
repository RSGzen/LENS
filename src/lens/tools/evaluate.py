"""Per-stage confidence gate and the final groundedness check.

Who runs it
-----------
**The orchestrator, automatically — not the RLM.** The draft calls the per-stage
gate "a control rule, not a classifier": it drives the bounded backtracking,
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

Tiers
-------------------------------
1. a calibrated confidence threshold on ``artifact["score"]`` — ~free;
2. a JEV ``noul`` sufficiency check — one prefill pass, only in the ambiguous band;
3. the root RLM's judgment — **no extra call**: the verdict rides in the RPC
   envelope and the model's next turn *is* the judgment. Tier 3 is not a tool call.

The artifact
------------
The gate never sees a raw tool payload: ``build_kg``'s node-link graph would blow
the graph itself. ``Runtime._dispatch_tool`` passes

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
from lens.observability.citations import claim_units
from lens.observability.trajectory import TraceCtx

Verdict = Literal["pass", "fail", "uncertain"]

# Where the gate runs. Map **stage-completing** tool -> stage. Do not add the
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
        M10b timeline and the report both read it; a key that appears only for
        some verdicts forces every reader to branch on the verdict first.
        """
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "failing_criterion": self.failing_criterion,
            "tier": self.tier,
        }


def build_sufficiency_question(
    stage: str,
    artifact: Mapping[str, Any],
    *,
    query: str | None = None,
) -> Question:
    """Build the JEV ``noul`` question used by tier 2 for a **stage**.

    ``criteria`` must be exactly ``{"true": ..., "false": ...}`` (see
    :class:`lens.models.Question`). The question is about the *stage requirement*
    ("does this output satisfy what the stage was for?"), not about the stage's
    name — ``instructions`` names the stage so the verdict is auditable.

    ``query`` is the research question the stage was working on. Without it the
    retrieval requirement — "covering every objective and problem the query
    names" — is a test of something the gate cannot see, so its verdict is
    arbitrary by construction. Pass it.

    Keep the state small: it is built from ``artifact["summary"]``, never the raw
    payload (the gate reads a summary, never the whole graph).

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
    requirement = requirements.get(stage, f"the output fulfils the purpose of the {stage} stage")
    fields = len(artifact.get("summary") or {})
    question_text = (query or "").strip() or "(not provided)"

    return Question(
        type="noul",
        instructions=(
            f"Stage under review: {stage}. The research query is: {question_text}\n"
            f"Decide whether {requirement}. "
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


def build_unsupported_claim_question(
    answer: str,
    citations: Sequence[str],
    evidence: Sequence[str] | Mapping[str, str] = (),
) -> Question:
    """Build the JEV ``noul`` question used by the **final** gate.

    Polarity: ``noul`` is **P(the answer contains an unsupported or uncited
    factual claim)** — a *failure-detection* proposition, so a high score means
    reject. The earlier question ("is **every** claim supported") is a conjunction
    over the claims: with eleven well-supported claims its probability collapses
    even when nothing is wrong, which is why measured verdicts sat at 0.33-0.49
    for answers whose evidence was present and correct. An existential keeps a
    fabrication detectable without penalising an answer for being long.

    ``evidence`` carries a bounded excerpt of the cited chunk text
    (``chunk_uuid -> text``). It is not optional: with ids alone the gate can only
    judge plausibility, and it rejected a cited, correct answer on every call.
    """
    citation_list = ", ".join(str(citation) for citation in citations) or "(none cited)"
    excerpt = (answer or "").strip()

    if len(excerpt) > ANSWER_MAX_CHARS:
        excerpt = excerpt[:ANSWER_MAX_CHARS] + " …"

    evidence_text = _format_evidence(evidence)

    return Question(
        type="noul",
        instructions=(
            "Decide whether this answer contains a factual claim that is NOT supported "
            "by the cited chunks, or a factual claim with no citation at all. "
            f"Citations given: {citation_list}. "
            "Answer true only if at least one such claim exists — a number, dataset, "
            "comparison or attribution that its citation does not contain, or that "
            "cites nothing. Answer false if every factual claim is backed by the "
            "excerpts below. An answer that only reports what it inspected, and says "
            "so, has no unsupported claim. "
            f"Cited excerpts: {evidence_text} "
            f"Answer under review:\n{excerpt}"
        ),
        criteria={
            "true": "at least one factual claim is uncited or not supported by its citation",
            "false": "every factual claim is supported by the cited excerpts",
        },
    )


def build_groundedness_question(
    answer: str,
    citations: Sequence[str],
    evidence: Sequence[str] | Mapping[str, str] = (),
) -> Question:
    """Build the JEV ``noul`` question used by the **final** gate.

    "Is every claim in the answer supported by the cited chunks?" — ``citations``
    are the ``chunk_uuid`` values the answer relies on, and ``evidence`` carries a
    **bounded** excerpt of the cited chunk text (``chunk_uuid -> text``, or bare
    excerpts). This is the guardrail row of.x.3 ("final guardrail /
    hallucination check"), so a ``false`` verdict means "do not accept this answer
    yet", not "the run failed".

    **The excerpt is not optional in practice.** Without it the gate sees ids and
    nothing else, so it can only judge plausibility — measured on the first `gpt-5`
    run: nine gate calls returned ``noul`` 0.03-0.35 and rejected a *cited, correct*
    answer until the iteration cap, because "is claim X supported by chunk
    ``33384c38``" is unanswerable from the id alone. The polarity was verified
    separately (true statement 0.92, false 0.03, grounded 0.89, ungrounded 0.01),
    so the rejections were substantive, not a parsing artefact. Keep the total
    bounded (``EVIDENCE_MAX_CHARS``) — JEV is input-priced at about $0.04 per
    million tokens, so the budget is generous rather than tight.
    """
    citation_list = ", ".join(str(citation) for citation in citations) or "(none cited)"
    excerpt = (answer or "").strip()

    if len(excerpt) > ANSWER_MAX_CHARS:
        excerpt = excerpt[:ANSWER_MAX_CHARS] + " …"

    evidence_text = _format_evidence(evidence)

    return Question(
        type="noul",
        instructions=(
            "Decide whether every factual claim in the answer is supported by the cited "
            f"chunks. Citations given: {citation_list}. Answer false if any claim is "
            "uncited, contradicted by its citation, or supported by a chunk that does "
            "not contain it - a number, dataset or comparison with no source counts as "
            "unsupported. An answer that cites nothing at all - placeholders such as "
            "'[see above]' or 'the retrieved evidence' are not citations - is "
            "unsupported. A claim that something is **absent** from the evidence is "
            "supported when the excerpt shows the inspection (a count of what was "
            "searched) rather than a hunch. "
            f"Cited evidence (truncated): {evidence_text} "
            f"Answer under review (truncated): {excerpt}"
        ),
        criteria={
            "true": "every factual claim is supported by the cited chunks",
            "false": "at least one claim is uncited, contradicted, or unsupported",
        },
    )


#: Total characters of cited chunk text the final gate is shown. The decision layer
#: is input-priced and measured at roughly $0.04 per million tokens, while one
#: avoided revision costs a chat turn 30-50 times more — so the budget is set to
#: carry the whole cited set (the longest observed answer cited 37 chunks, about
#: 66 000 characters) rather than to save fractions of a cent.
#:
#: An earlier version capped this at 2 000 characters, on the belief that the
#: endpoint had a 32 K context limit. It does not: the model window is 262 144
#: tokens for state plus questions. That false constraint made most claims
#: unsupported by construction and every RLM run ended at the iteration cap.
EVIDENCE_MAX_CHARS = 120000

#: Characters of the answer itself the gate is shown. An answer may not be larger
#: than the evidence that is supposed to prove it.
ANSWER_MAX_CHARS = 8000

#: Claims asked about individually in one decision call. Observed answers carry at
#: most nineteen cited units; the cap only stops a pathological answer from
#: building an enormous question map.
MAX_GATE_CLAIMS = 40

#: A claim at or above this probability of being unsupported fails the gate.
#: **Not calibrated yet.** Every per-claim probability is logged on the
#: ``gate.verdict`` event, so a later reliability curve can re-fit this threshold
#: over recorded answers without re-running a single query.
CLAIM_FAIL_PROBABILITY = 0.5


def _format_evidence(evidence: Sequence[str] | Mapping[str, str]) -> str:
    """Render cited excerpts as ``id: text`` lines within ``EVIDENCE_MAX_CHARS``."""
    items = evidence.items() if isinstance(evidence, Mapping) else enumerate(evidence)
    parts: list[str] = []
    used = 0

    for key, text in items:
        body = " ".join(str(text or "").split())

        if not body:
            continue

        line = f"[{key}] {body}"
        remaining = EVIDENCE_MAX_CHARS - used

        if remaining <= 0:
            break

        parts.append(line[:remaining])
        used += len(parts[-1])

    return " | ".join(parts) or "(no chunk text available - judge from the citations)"


def _read_answer(
    answer: Mapping[str, Any],
    *,
    polarity: str = "supported",
) -> tuple[Verdict, float | None, str | None]:
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
    # nothing. ``noul`` is a probability that the proposition holds (0.96 on a
    # plainly true example in the JEV contract, 0.03 on that template answer).
    #
    # ``polarity`` says which way that probability points. "supported" means the
    # proposition is "the answer is supported" (high = pass); "unsupported" means
    # it is "the answer contains an unsupported claim" (high = fail), which is what
    # the final gate asks so that a long, well-cited answer is not penalised for
    # being a conjunction of many claims.
    if (
        raw is None
        and isinstance(answer.get("noul"), (int, float))
        and not isinstance(answer.get("noul"), bool)
    ):
        noul = float(answer["noul"])
        raw = (noul < 0.5) if polarity == "unsupported" else (noul >= 0.5)
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

    criterion = answer.get("failing_criterion") or answer.get("criterion") or answer.get("reason")

    return verdict, confidence, str(criterion)[:300] if criterion else None


def build_claim_question(claim_id: str, text: str, citations: Sequence[str]) -> Question:
    """Build the per-claim ``noul`` question used by the **final** gate.

    Polarity matches the aggregate question: ``noul`` is P(this claim is **not**
    supported). One question per claim means the verdict can name the claim that
    failed, which is what turns "revise the answer" into "fix claim 07".

    The excerpts are not repeated here — they live in the shared state, and JEV
    evaluates every question against that same state.
    """
    cited = ", ".join(str(citation) for citation in citations) or "(none cited)"
    body = (text or "").strip()

    if len(body) > 600:
        body = body[:600] + " …"

    return Question(
        type="noul",
        instructions=(
            f"Claim {claim_id} cites: {cited}. Decide whether that citation supports it, "
            "using the excerpt for each cited id in the state. Answer true only if the "
            "claim is NOT supported — it states a number, dataset, comparison or "
            "attribution the excerpt does not contain, or it cites an id with no excerpt. "
            "Answer false if the excerpt supports it. A claim that the evidence does not "
            "contain something is supported when the excerpts show what was inspected, and "
            "a statement about the process (what was inspected, searched or scanned) is not "
            "a factual claim about the evidence — answer false for those.\n"
            f"Claim: {body}"
        ),
        criteria={
            "true": "the claim is not supported by the excerpt of the ids it cites",
            "false": "the claim is supported by the excerpt of the ids it cites",
        },
    )


def build_scope_question(answer: str) -> Question:
    """Build the JEV ``noul`` question about **scope honesty** (final gate, truncated).

    Asked only when the evidence budget bound before every cited chunk was shown.
    A gate holding a third of the evidence cannot tell "unsupported" from "not
    shown", so the second question judges the one thing that is still observable
    from the answer alone: does it claim less than it can prove?

    Its verdict is **recorded, not enforced**: the acceptance rule is the
    groundedness verdict, and silently adding a second rejection rule would change
    what the gate means without measuring it first.
    """
    excerpt = (answer or "").strip()

    if len(excerpt) > ANSWER_MAX_CHARS:
        excerpt = excerpt[:ANSWER_MAX_CHARS] + " …"

    return Question(
        type="noul",
        instructions=(
            "Decide whether this answer limits its claims of absence to the material "
            "it says it inspected. Answer true if every 'not found' or 'no evidence' "
            "statement is scoped to what was read ('in the 12 chunks inspected'); "
            "answer false if it states a global absence ('nothing in the corpus', "
            "'none of the papers') from a partial look, or offers no scope at all. "
            f"Answer:\n{excerpt}"
        ),
        criteria={
            "true": "the answer's absences are scoped to what it inspected",
            "false": "the answer claims a global absence it cannot have checked",
        },
    )


def _log_verdict(
    client: LensClient,
    trace_ctx: TraceCtx,
    stage: str,
    verdict: StageVerdict,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Log one ``gate.verdict`` event (best-effort: observability must not decide).

    The JEV call itself is logged by the transport (``jev.decisions``); this event
    carries the **decision** — stage, verdict, confidence, tier — which is what the
     report counts. ``extra`` carries secondary signals (a scope verdict) that are
     recorded but do not change the decision.
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
            extra={
                "tier": verdict.tier,
                "criterion": verdict.failing_criterion,
                **dict(extra or {}),
            },
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
    query: str | None = None,
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

    ``query`` is the research question the stage was working on, and it is put in
    the **state** as well as the question: a sufficiency verdict about coverage
    cannot be audited, or calibrated, if the state does not say what was asked.

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
            verdict = StageVerdict(verdict="pass", confidence=float(score), tier="threshold")
            _log_verdict(client, trace_ctx, stage, verdict)

            return verdict

    if stage == "final":
        question = build_unsupported_claim_question(
            str(summary.get("answer") or ""),
            list(summary.get("citations") or []),
            summary.get("evidence") or (),
        )
        polarity = "unsupported"
        # One claim per question, in the same call: JEV evaluates a question map
        # against one state, so per-claim scoring costs a few hundred input tokens
        # and returns *which* claim failed. The aggregate question stays as a
        # cross-check; the two disagree in the log rather than being averaged.
        claims = claim_units(str(summary.get("answer") or ""))[:MAX_GATE_CLAIMS]
        summary = {
            **summary,
            "claims": [
                {"id": f"claim-{index:02d}", "citations": ids}
                for index, (_text, ids) in enumerate(claims, start=1)
            ],
        }
    else:
        claims = []
        question = build_sufficiency_question(stage, artifact, query=query)
        polarity = "supported"

    state = json.dumps(
        {"query": (query or "").strip() or None, "stage": stage, "summary": summary},
        ensure_ascii=False,
        default=str,
    )

    questions: dict[str, Question] = {"gate": question}

    for index, (claim_text, claim_ids) in enumerate(claims, start=1):
        name = f"claim-{index:02d}"
        questions[name] = build_claim_question(name, claim_text, claim_ids)

    if stage == "final" and summary.get("evidence_truncated"):
        # The groundedness verdict is being asked about evidence that is not all
        # there, so the answer's own scope claim is the only other observable
        # signal. Recorded, not enforced.
        questions["scope"] = build_scope_question(str(summary.get("answer") or ""))

    try:
        result = client.jev(state, questions, trace_ctx=trace_ctx)
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
        (getattr(result, "answers", None) or {}).get("gate") or {},
        polarity=polarity,  # type: ignore[arg-type]
    )

    # Per-claim verdicts. They can only *tighten* the aggregate: a claim named as
    # unsupported fails the gate even if the answer as a whole scored below the
    # threshold, and the criterion then names that claim so the next turn has
    # something to fix instead of "revise".
    answers = getattr(result, "answers", None) or {}
    claim_scores: dict[str, float] = {}
    claim_text_by_id: dict[str, str] = {}

    for index, (claim_text, _ids) in enumerate(claims, start=1):
        name = f"claim-{index:02d}"
        _verdict, probability, _criterion = _read_answer(
            answers.get(name) or {}, polarity="unsupported"
        )

        if probability is not None:
            claim_scores[name] = probability
            claim_text_by_id[name] = claim_text

    failed_claims = {
        name: score for name, score in claim_scores.items() if score >= CLAIM_FAIL_PROBABILITY
    }

    if failed_claims:
        worst = max(failed_claims, key=lambda name: failed_claims[name])
        criterion = (
            f"{worst} (P={failed_claims[worst]:.2f}): {claim_text_by_id.get(worst, '')[:160]}"
        )

        if verdict_value != "fail":
            verdict_value = "fail"

    verdict = StageVerdict(
        verdict=verdict_value,
        confidence=confidence,
        # A criterion is only meaningful when the gate did not pass.
        failing_criterion=criterion if verdict_value != "pass" else None,
        tier="jev",
    )
    scope_extra: dict[str, Any] = {}

    if claim_scores:
        scope_extra.update(
            {
                "claims": len(claim_scores),
                "claim_scores": claim_scores,
                "claims_failed": sorted(failed_claims),
            }
        )

    if "scope" in questions:
        scope_verdict, scope_confidence, _criterion = _read_answer(
            (getattr(result, "answers", None) or {}).get("scope") or {}
        )
        scope_extra = {
            "scope_verdict": scope_verdict,
            "scope_confidence": scope_confidence,
            **scope_extra,
        }

    _log_verdict(
        client,
        trace_ctx,
        stage,
        verdict,
        extra={"polarity": polarity, **scope_extra},
    )

    return verdict


__all__ = [
    "GATED_STAGES",
    "StageVerdict",
    "Verdict",
    "build_groundedness_question",
    "build_scope_question",
    "build_sufficiency_question",
    "evaluate_stage",
]
