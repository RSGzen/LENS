"""Per-stage confidence gate (Decision Layer §3.x.4b).

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §6; ``Decision Layer (Draft).md``.

Tier order:
  1. calibrated threshold on a stage score;
  2. JEV ``noul`` sufficiency check;
  3. root-model judgment.

The gate returns a verdict **and** the failing criterion so the ledger and the
trajectory explain why a stage was rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from lens.clients import LensClient
from lens.models import Question
from lens.observability.trajectory import TraceCtx

Verdict = Literal["pass", "fail", "uncertain"]

# Calibrated thresholds are filled from the Decision Layer draft / M11 data.
# Do not invent numbers here; an empty map forces the caller to pass one.
STAGE_THRESHOLDS: dict[str, float] = {}


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


def build_sufficiency_question(stage: str, artifact: Any) -> Question:
    """Build the JEV ``noul`` question used by tier 2.

    ``criteria`` must be exactly ``{"true": ..., "false": ...}`` (see
    :class:`lens.models.Question`). Keep the state small — JEV has a 32 K cap.
    """
    raise NotImplementedError("M8 scaffold: implement build_sufficiency_question")


def evaluate_stage(
    stage: str,
    artifact: Any,
    *,
    client: LensClient,
    trace_ctx: TraceCtx,
    threshold: float | None = None,
) -> StageVerdict:
    """Run the three-tier gate and return a :class:`StageVerdict`.

    Tier 1: compare ``artifact["score"]`` with ``threshold`` (or
    ``STAGE_THRESHOLDS[stage]``). Tier 2: JEV ``noul`` with
    :func:`build_sufficiency_question`. Tier 3: root judgment is the caller's
    fallback when the first two are inconclusive. Log ``stage``, ``confidence``,
    and ``verdict`` on the trajectory.
    """
    raise NotImplementedError("M8 scaffold: implement evaluate_stage")
