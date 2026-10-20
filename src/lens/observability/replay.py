"""Turn-level replay: turn the JSONL log into what a viewer shows (M10b / M8.5+).

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4n (recorded replay) and the
``MVP Build Roadmap.md`` M10b block.

Why this module exists
----------------------
:meth:`lens.observability.trajectory.Trajectory.rebuild` answers *"what state do I
need to continue?"* (messages, counters, namespace). This module answers a
different question: *"what happened, turn by turn, so a human can step through
it?"* Both are read-only folds of the same log; neither invents data.

One **turn** = one ``step_idx`` (one root iteration). A turn contains, in file
order: the root ``chat`` (prompt + completion), the ``exec`` block (code + stdout),
any ``tool`` calls made by that block, ``jev`` calls, ``emit`` events, and the
turn's ``budget`` snapshot. Grouping is by ``step_idx`` — **never** by event
count, because one iteration deliberately logs several events.

The playback modes (do not conflate them — claim discipline):

- **A. Scrubbed playback** (this module): renders recorded turns from blobs.
  No LLM, no DB, no Docker, no GPU. It is a *viewer over a recording*.
- **B. Deterministic harness replay** (``Runtime(replay=True)``): re-runs the loop
  serving recorded tool results. Needs Docker, no key, no GPU.
- **C. Live run**: the real pipeline.

Tolerance is a feature: a missing or corrupt blob yields ``None`` for that field
and is recorded in :attr:`Turn.missing_refs`, so one damaged artifact cannot make
the whole run unopenable (the OpenHands lesson). Nothing here raises on bad data.

Contract tests: ``tests/test_replay.py`` (no key, no DB, no Docker).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lens.observability.trajectory import Trajectory


@dataclass
class ToolStep:
    """One host-tool call inside a turn."""

    name: str
    args: dict[str, Any] | None = None
    args_hash: str | None = None
    result_summary: str | None = None
    result_ref: str | None = None
    latency_ms: int | None = None
    error: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    replayed: bool = False


@dataclass
class Turn:
    """Everything one root iteration did, ready to render."""

    turn: int
    model: str | None = None
    provider: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    # The model's full message (reasoning + the ```python block).
    completion: str | None = None
    # The code that actually ran (exec.code_ref); usually the block from completion.
    code: str | None = None
    # What the model saw next (the truncated observation) vs the raw capture.
    observation: str | None = None
    stdout_full: str | None = None
    stderr_full: str | None = None
    ok: bool | None = None
    error: dict[str, Any] | None = None
    tools: list[ToolStep] = field(default_factory=list)
    emits: list[str] = field(default_factory=list)
    gate: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    # Refs that could not be resolved from the bundle (damaged/incomplete export).
    missing_refs: list[str] = field(default_factory=list)


def replay_turns(trajectory: Trajectory) -> list[Turn]:
    """Group the log by ``step_idx`` and resolve every blob into a :class:`Turn`.

    Contract:
      1. Read events in file order; group by ``step_idx`` ascending; keep the
         in-group order as written. Empty/missing log -> ``[]``.
      2. A ``chat`` event (actor ``root`` or ``sub``) fills the turn's model,
         provider, tokens, cost, latency and ``completion`` (``completion_ref``);
         a ``sub_call`` therefore appears as its own turn only when it has its own
         ``step_idx`` — otherwise it is attached to the surrounding turn.
      3. An ``exec`` event fills ``code`` (``code_ref``), ``stdout_full``
         (``stdout_ref``), ``stderr_full`` (``stderr_ref`` if present), ``ok`` and
         ``error``. The **observation** the model saw is the ``result_summary``
         field when present, else the stdout truncated to
         ``config.TOOL_RESULT_SUMMARY_CHARS`` — never the full capture, so the
         view matches what actually entered the context.
      4. A ``tool`` event appends a :class:`ToolStep` (name, ``args`` from
         ``args_ref``, hash, summary, ``result_ref``, latency, error, gate from
         ``extra.gate``, ``replayed`` from ``extra.replayed``).
      5. ``emit`` events append ``extra.var`` to ``emits``; the last ``budget``
         snapshot of the turn sets ``budget``; ``note`` events are ignored here
         (they are for the trajectory view, not the turn cards).
      6. Any ref that fails to load is left ``None`` and appended to
         ``missing_refs`` — **never** raise. A demo viewer must open a partially
         damaged bundle and say what is missing.
      7. The last turn may be incomplete (crash): ``code`` without
         ``stdout_full`` and ``ok is None`` means "interrupted", and the viewer
         must render it as such rather than as success.
    """
    raise NotImplementedError("M8.5 scaffold: implement replay_turns")


def replay_summary(turns: list[Turn]) -> dict[str, Any]:
    """Return the header numbers a viewer shows above the timeline.

    Contract: ``{"turns", "tokens_in", "tokens_out", "cost_usd", "tool_calls",
    "emits", "gates_passed", "gates_failed", "missing_refs"}`` — all summed or
    counted from ``turns``, missing values counted as zero, and the totals
    matching :meth:`Trajectory.totals` for the same log (both read the same
    fields; if they disagree, the bug is here).
    """
    raise NotImplementedError("M8.5 scaffold: implement replay_summary")


__all__ = ["ToolStep", "Turn", "replay_summary", "replay_turns"]
