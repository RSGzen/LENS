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

import json
from dataclasses import dataclass, field
from typing import Any

from lens import config
from lens.agent.context import truncate_observation
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
    # Gate spend, kept apart from ``cost_usd``: the context-growth chart must show
    # **chat** tokens only (a JEV call's ~500 input tokens are not the growing
    # context), while a cost meter must show total spend. Reported separately so
    # `replay_summary`'s cost can equal `Trajectory.totals()` without distorting
    # the chart (M9 S5, LOG-116).
    jev_cost_usd: float | None = None
    jev_tokens_in: int | None = None
    ok: bool | None = None
    error: dict[str, Any] | None = None
    tools: list[ToolStep] = field(default_factory=list)
    emits: list[str] = field(default_factory=list)
    gate: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    # Refs that could not be resolved from the bundle (damaged/incomplete export).
    missing_refs: list[str] = field(default_factory=list)


def _load(trajectory: Trajectory, ref: str | None, missing: list[str]) -> str | None:
    """Return the blob text for ``ref``; record it in ``missing`` instead of raising.

    A viewer must open a partially damaged bundle and *say* what is missing. The
    trajectory itself stays strict (a resume needs its evidence), so the tolerance
    lives here, on the read-only path.
    """
    if not ref:
        return None

    try:
        return trajectory.load_blob(ref)
    except (OSError, KeyError, ValueError, IndexError):
        missing.append(ref)
        return None


def _load_json(trajectory: Trajectory, ref: str | None, missing: list[str]) -> Any:
    """Return the parsed JSON blob, or ``None`` when it is absent/unreadable."""
    text = _load(trajectory, ref, missing)

    if text is None:
        return None

    try:
        return json.loads(text)
    except ValueError:
        missing.append(ref or "")
        return None


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
    path = trajectory.trajectory_path

    if not path.is_file():
        return []

    turns: dict[int, Turn] = {}

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        try:
            event = json.loads(line)
        except ValueError:
            continue  # a damaged tail must not hide the turns before it

        if not isinstance(event, dict) or "step_idx" not in event:
            continue

        try:
            step = int(event["step_idx"])
        except (TypeError, ValueError):
            continue

        turn = turns.get(step)

        if turn is None:
            turn = Turn(turn=step)
            turns[step] = turn

        _apply(turn, event, trajectory)

    # Ascending by step_idx; a dict preserves the insertion (file) order inside a
    # step, which is what "keep the in-group order as written" asks for.
    return [turns[step] for step in sorted(turns)]


def _apply(turn: Turn, event: dict[str, Any], trajectory: Trajectory) -> None:
    """Fold one event into ``turn`` (in file order)."""
    extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
    call_type = event.get("call_type")

    # Any event may carry a budget snapshot (the runtime logs it as a note); the
    # last one in the turn wins, which is the turn's closing state.
    if extra.get("budget") is not None:
        turn.budget = extra["budget"]

    if call_type == "chat":
        # Accumulate: a turn can hold the root chat *and* a sub_call, and the
        # summary must still match ``Trajectory.totals()`` for the same log.
        turn.tokens_in = (turn.tokens_in or 0) + (event.get("tokens_in") or 0)
        turn.tokens_out = (turn.tokens_out or 0) + (event.get("tokens_out") or 0)
        turn.cost_usd = (turn.cost_usd or 0.0) + (event.get("cost_usd") or 0.0)
        turn.latency_ms = (turn.latency_ms or 0) + (event.get("latency_ms") or 0)
        turn.model = turn.model or event.get("model")
        turn.provider = turn.provider or event.get("provider")

        # The root completion is the turn's message; a sub_call sharing the step
        # must not overwrite it (its text belongs to that call, not the turn).
        if event.get("completion_ref") and (
            event.get("actor") == "root" or turn.completion is None
        ):
            completion = _load(trajectory, event.get("completion_ref"), turn.missing_refs)

            if completion is not None:
                turn.completion = completion

    elif call_type == "jev":
        # Gate calls cost money (H3 counts the decision layer) but are not context:
        # keep them out of ``tokens_in`` so the growth chart stays honest, and sum
        # them into the summary's cost so it matches ``Trajectory.totals()``.
        turn.jev_cost_usd = (turn.jev_cost_usd or 0.0) + (event.get("cost_usd") or 0.0)
        turn.jev_tokens_in = (turn.jev_tokens_in or 0) + (event.get("tokens_in") or 0)

    elif call_type == "exec":
        code = _load(trajectory, event.get("code_ref"), turn.missing_refs)
        stdout = _load(trajectory, event.get("stdout_ref"), turn.missing_refs)
        stderr = _load(trajectory, event.get("stderr_ref"), turn.missing_refs)

        if code is not None:
            turn.code = code

        if stdout is not None:
            turn.stdout_full = stdout

        if stderr is not None:
            turn.stderr_full = stderr

        if event.get("error") is not None:
            turn.ok = False
            turn.error = event["error"]
        elif event.get("stdout_ref"):
            turn.ok = True
        else:
            # No stdout and no error: the block never reported back (crash or a
            # killed bundle). Render "interrupted", never "success".
            turn.ok = None

        # The observation is what entered the context, so an explicit summary wins;
        # the truncated capture is only a fallback for events logged without one.
        summary = event.get("result_summary")

        if summary:
            turn.observation = summary
        elif turn.observation is None and stdout is not None:
            turn.observation = truncate_observation(stdout, config.TOOL_RESULT_SUMMARY_CHARS)

    elif call_type == "tool":
        args = _load_json(trajectory, event.get("args_ref"), turn.missing_refs)
        gate = extra.get("gate") if isinstance(extra.get("gate"), dict) else None

        step = ToolStep(
            name=event.get("tool") or "unknown",
            args=args if isinstance(args, dict) else None,
            args_hash=event.get("args_hash"),
            result_summary=event.get("result_summary"),
            result_ref=event.get("result_ref"),
            latency_ms=event.get("latency_ms"),
            error=event.get("error"),
            gate=gate,
            replayed=bool(extra.get("replayed")),
        )
        turn.tools.append(step)

        if gate is not None:
            turn.gate = gate  # the last gate of the turn is the turn's verdict

    elif call_type == "emit":
        label = extra.get("var")

        if isinstance(label, str) and label:
            turn.emits.append(label)

    # ``note`` and ``jev`` add nothing to a turn card beyond ``extra.budget`` above.


def replay_summary(turns: list[Turn]) -> dict[str, Any]:
    """Return the header numbers a viewer shows above the timeline.

    Contract: ``{"turns", "tokens_in", "tokens_out", "jev_tokens_in", "cost_usd",
    "chat_cost_usd", "tool_calls", "emits", "gates_passed", "gates_failed",
    "missing_refs"}`` — all summed or counted from ``turns``, missing values
    counted as zero.

    ``cost_usd`` (chat + gate calls) matches ``Trajectory.totals()["cost_usd"]``
    for the same log; ``chat_cost_usd`` and ``tokens_in/tokens_out`` stay chat-only
    because they drive the context-growth chart (a JEV call's input tokens are not
    the growing context). The two used to be one number, so a viewer's cost meter
    was short by the whole decision layer (M9 S5, LOG-116).
    """
    gates = [tool.gate for turn in turns for tool in turn.tools if tool.gate]

    return {
        "turns": len(turns),
        "tokens_in": sum(turn.tokens_in or 0 for turn in turns),
        "tokens_out": sum(turn.tokens_out or 0 for turn in turns),
        "jev_tokens_in": sum(turn.jev_tokens_in or 0 for turn in turns),
        # Total spend = chat + gate calls: the cost meter compares against the cap,
        # so it must not hide the decision layer. ``tokens_in`` stays chat-only
        # because it drives the context-growth chart.
        "cost_usd": sum((turn.cost_usd or 0.0) + (turn.jev_cost_usd or 0.0) for turn in turns),
        "chat_cost_usd": sum(turn.cost_usd or 0.0 for turn in turns),
        "tool_calls": sum(len(turn.tools) for turn in turns),
        "emits": sum(len(turn.emits) for turn in turns),
        "gates_passed": sum(1 for gate in gates if gate.get("verdict") == "pass"),
        "gates_failed": sum(1 for gate in gates if gate.get("verdict") == "fail"),
        "missing_refs": sum(len(turn.missing_refs) for turn in turns),
    }


__all__ = ["ToolStep", "Turn", "replay_summary", "replay_turns"]
