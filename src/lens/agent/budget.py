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
    # Per block, enforced by the runner's SIGALRM — NOT a cumulative budget.
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
        self.iterations += 1

    def record_chat(self, result: CallResult) -> None:
        """Add one chat call's tokens and cost to the running totals.

        Both token counts accumulate because the token cap bounds **total work**;
        they stay separate so a snapshot can show which half is expensive (output
        is 8x input on gpt-5).
        """
        self.tokens_in += result.tokens_in
        self.tokens_out += result.tokens_out

        self.cost_usd += result.cost_usd

    def record_jev(self, result: JevResult) -> None:
        """Add one JEV (decision-layer) call's tokens and cost.

        JEV bills on **input** tokens only (``usage.cost`` is the per-call cost),
        but it is still spend: leaving it out would understate the per-query cost
        the caps enforce. ``JevResult`` has ``tokens_in`` / ``cost_usd`` and no
        ``tokens_out`` — count output as zero.
        """
        self.tokens_in += result.tokens_in

        self.cost_usd += result.cost_usd

    def record_exec(self, seconds: float) -> None:
        """Add one sandbox execution block's wall-clock seconds."""
        self.exec_seconds += seconds

    def record_backtrack(self) -> None:
        """Count one backtracking retry (F-01)."""
        self.backtracks += 1

    # ------------------------------------------------------------- decisions
    def can_continue(self) -> bool:
        """``True`` while every cap still has room for one more root turn.

        The caps live in exactly **one** place, :meth:`stop_reason`, and this is a
        thin alias for it. Two functions used to enumerate the same four caps;
        if they ever drift (one edited, the other not) the loop continues and
        then labels the partial answer with a *different* reason than the one that
        actually stopped it — so the duplication is removed rather than
        disciplined.

        Caps: iterations, tokens (**input + output** — output is the expensive
        half), cost, backtracks. Deliberately **not** a cap: ``exec_seconds``.
        The 120 s limit is *per block* and the runner enforces it, so summing
        blocks must never end a run early (5 x 25 s would otherwise "exhaust" a
        120 s budget with the token budget barely touched).
        """
        return self.stop_reason() is None

    def stop_reason(self) -> str | None:
        """Return the first exhausted cap name, or ``None``.

        Names, in the order they are checked: ``"iterations"`` / ``"tokens"`` /
        ``"cost"`` / ``"backtracks"``. This set is the vocabulary that
        ``answer.md`` and ``manifest.json`` use to label a partial answer
        (``RunResult.stop_reason``), so do not invent new values here without
        updating those artifacts.
        """
        if self.iterations >= self.limits.max_iterations:
            return "iterations"

        if (self.tokens_in + self.tokens_out) >= self.limits.max_tokens:
            return "tokens"

        if self.cost_usd >= self.limits.max_cost_usd:
            return "cost"

        if self.backtracks >= self.limits.max_backtracks:
            return "backtracks"

        return None

    # --------------------------------------------------------------- readout
    def snapshot(self) -> dict:
        """Return a JSON-serializable, read-only view for ``rlm_api.budget()``.

        Contract::

            {
              "limits": {"iterations", "tokens", "cost_usd", "exec_block_s",
                         "backtracks"},
              "used":   {"iterations", "tokens_in", "tokens_out", "cost_usd",
                         "exec_seconds", "backtracks"},
              "left":   {"iterations", "tokens", "cost_usd", "backtracks"},
              "stop_reason": str | None,
            }

        ``used.tokens_in`` / ``used.tokens_out`` stay separate (the model can see
        which half is expensive) while ``left.tokens`` is their sum, because the
        cap bounds total work.

        There is deliberately **no** ``left.exec_block_s``: the 120 s cap is
        *per block* (runner-enforced), so "seconds remaining" would read as a
        run-level budget that does not exist — and a zero there would push the
        model to wrap up while it still had budget. ``used.exec_seconds`` remains
        as a diagnostic (it feeds the M10b meters).

        ``left`` values are clamped at zero. The snapshot must never expose the
        API key or any credential.
        """
        snapshot_dict = {
            "limits": {
                "iterations": self.limits.max_iterations,
                "tokens": self.limits.max_tokens,
                "cost_usd": self.limits.max_cost_usd,
                "exec_block_s": self.limits.max_exec_block_s,
                "backtracks": self.limits.max_backtracks,
            },
            "used": {
                "iterations": self.iterations,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "cost_usd": self.cost_usd,
                "exec_seconds": self.exec_seconds,
                "backtracks": self.backtracks,
            },
            "left": {
                "iterations": max(0, (self.limits.max_iterations - self.iterations)),
                "tokens": max(0, (self.limits.max_tokens - (self.tokens_in + self.tokens_out))),
                "cost_usd": max(0, (self.limits.max_cost_usd - self.cost_usd)),
                "backtracks": max(0, (self.limits.max_backtracks - self.backtracks)),
            },
            "stop_reason": self.stop_reason(),
        }

        return snapshot_dict
