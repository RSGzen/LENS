"""Central RLM budget: caps, counters, and a read-only snapshot.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a (budget) and §6 (R-05/M-06).

The orchestrator owns the only mutable :class:`Budget`. The sandbox receives
``snapshot()`` through ``rlm_api.budget()`` and cannot change any counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lens import config
from lens.models import CallResult, JevResult


@dataclass(frozen=True)
class BudgetLimits:
    """Frozen caps for one query (override via ``config`` / the environment)."""

    max_iterations: int = config.MAX_ITERATIONS
    max_tokens: int = config.MAX_TOKENS_PER_QUERY
    max_cost_usd: float = config.MAX_COST_USD_PER_QUERY
    max_exec_block_s: float = config.MAX_EXEC_BLOCK_S
    max_backtracks: int = config.MAX_BACKTRACKS


@dataclass
class Budget:
    """Mutable accounting for one query.

    ``record_*`` methods are called by the orchestrator only. ``can_continue``
    answers whether another root turn is allowed. ``stop_reason`` names the cap
    that was hit (for the partial-answer report), or ``None``.

    Two jobs, two sources (R4): this object **enforces** the caps, and
    :meth:`lens.observability.trajectory.Trajectory.totals` is what a reported
    ``$/query`` is read from. They can differ — the caps only see what the
    orchestrator records, while the transport logs every call itself.
    """

    limits: BudgetLimits = field(default_factory=BudgetLimits)
    iterations: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    exec_seconds: float = 0.0
    backtracks: int = 0

    # ------------------------------------------------------------- recording
    def record_iteration(self) -> None:
        """Count one completed root turn."""
        raise NotImplementedError("M8 scaffold: implement Budget.record_iteration")

    def record_chat(self, result: CallResult) -> None:
        """Add one chat call's tokens and cost to the running totals."""
        raise NotImplementedError("M8 scaffold: implement Budget.record_chat")

    def record_jev(self, result: JevResult) -> None:
        """Add one JEV (decision-layer) call's tokens and cost.

        JEV bills on **input** tokens only (``usage.cost`` is the per-call cost),
        but it is still spend: leaving it out would understate the per-query cost
        the caps enforce. ``JevResult`` has ``tokens_in`` / ``cost_usd`` and no
        ``tokens_out`` — count output as zero.
        """
        raise NotImplementedError("M8 scaffold: implement Budget.record_jev")

    def record_exec(self, seconds: float) -> None:
        """Add one sandbox execution block's wall-clock seconds."""
        raise NotImplementedError("M8 scaffold: implement Budget.record_exec")

    def record_backtrack(self) -> None:
        """Count one backtracking retry (F-01)."""
        raise NotImplementedError("M8 scaffold: implement Budget.record_backtrack")

    # ------------------------------------------------------------- decisions
    def can_continue(self) -> bool:
        """``True`` while every cap still has room for one more root turn."""
        raise NotImplementedError("M8 scaffold: implement Budget.can_continue")

    def stop_reason(self) -> str | None:
        """Return the first exhausted cap name, or ``None``.

        Names: ``"iterations"`` / ``"tokens"`` / ``"cost"`` / ``"backtracks"``.
        """
        raise NotImplementedError("M8 scaffold: implement Budget.stop_reason")

    # --------------------------------------------------------------- readout
    def snapshot(self) -> dict:
        """Return a JSON-serializable, read-only view for ``rlm_api.budget()``.

        Contract::

            {
              "limits": {"iterations", "tokens", "cost_usd", "exec_block_s",
                         "backtracks"},
              "used":   {"iterations", "tokens_in", "tokens_out", "cost_usd",
                         "exec_seconds", "backtracks"},
              "left":   {"iterations", "tokens", "cost_usd", "exec_seconds",
                         "backtracks"},
              "stop_reason": str | None,
            }

        ``left`` values are clamped at zero. The snapshot must never expose the
        API key or any credential.
        """
        raise NotImplementedError("M8 scaffold: implement Budget.snapshot")
