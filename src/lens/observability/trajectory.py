"""Observability sink: content-addressed blobs + one-schema JSONL trajectory.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §5 (trajectory fields + payloads,
LOG-31) and §4a (Observability layer).

Design: :class:`Trajectory` is the *injected* sink — ``clients.py`` receives one
in its constructor and calls :meth:`Trajectory.store_blob` /
:meth:`Trajectory.log_event` for every LLM/JEV call, so no call escapes logging.
One instance = one run, writing to ``<runs_root>/<run_id>/trajectory.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import math
import copy

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# Import central path config. A missing config is a setup error, not a value to
# paper over — fail loudly instead of writing into the current directory.
try:
    from lens.config import (
        BLOBS_DIR_NAME,
        LENS_RUNS_ROOT_PATH,
        TRAJECTORY_FILENAME,
    )
except ModuleNotFoundError as e:
    raise RuntimeError(
        "lens.config not importable — run from the repo root or install the "
        "package with `uv sync`."
    ) from e

# Who performed the action (Tech Stack §5 `actor`).
Actor = Literal["root", "sub", "tool"]

# What the event is (Tech Stack 5). One schema, several kinds:
# - ``chat`` / ``jev`` : the two LLM transports (LOG-31)
# - ``tool``          : one host-tool dispatch (ToolRegistry.dispatch)
# - ``exec``          : one sandbox code block (Runtime._run_loop) — code + stdout blobs
# - ``emit``          : one ``rlm_api.emit()`` offload (variable -> blob ref)
# - ``note``          : a free-form RLM/operator annotation (also ``run.resume``)
# Keeping them in one enum is what lets H3 cost aggregation separate model spend
# from tool spend; labelling a tool call ``chat`` would corrupt that number.
#
# ``chat`` + ``exec`` + ``emit`` make the log self-sufficient: the root message
# list is a fold of them (``Trajectory.rebuild``), so there is no second copy of
# the conversation to keep in sync.
CallType = Literal["chat", "jev", "tool", "exec", "emit", "note"]


@dataclass(frozen=True)
class TraceCtx:
    """Identity of one step within one query's trajectory.

    ``trace_id`` is stable for a whole query/run, ``step_idx`` increments per
    step, and ``actor`` says who acted. Frozen so it can be passed around and
    logged safely.
    """

    trace_id: str
    step_idx: int
    actor: Actor


@dataclass
class RebuiltState:
    """Everything the log can tell us about a run (M8.5 resume).

    Returned by :meth:`Trajectory.rebuild`. If it is in here, it is derived —
    never stored twice, so it cannot diverge from the log.
    """

    # The root message list as the model last saw it (system + turns).
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Last ``budget`` snapshot event payload (``{}`` when none was logged).
    budget: dict[str, Any] = field(default_factory=dict)
    # Sandbox variable name -> emit ref (``sha256:...``); the namespace contract.
    emits: dict[str, str] = field(default_factory=dict)
    # ``(tool, args_hash)`` -> result ref, for deterministic recorded replay.
    tool_results: dict[str, str] = field(default_factory=dict)
    # Next free step index (max logged step_idx + 1).
    last_seq: int = 0


class Trajectory:
    """Single JSONL sink + content-addressed blob store for one run.

    Layout (MVP Roadmap §3)::

        <runs_root>/<run_id>/trajectory.jsonl   # metadata via log_event()
        <runs_root>/blobs/<hexdigest>           # prompt / completion payloads
    """

    def __init__(self, run_id: str) -> None:
        root = Path(LENS_RUNS_ROOT_PATH)
        self.run_id = run_id
        self.runs_root = root
        self.run_dir = root / run_id
        self.blobs_dir = root / BLOBS_DIR_NAME
        self.trajectory_path = self.run_dir / TRAJECTORY_FILENAME

        # Created eagerly so the run workspace always exists.
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ blobs
    def store_blob(self, text: str) -> str:
        """Store ``text`` once under a content-derived name; return its ref.

        Contract:
          - SHA-256 of the UTF-8 bytes; the **filename is the hex digest itself**
            (no extension, no timestamp).
          - **Idempotent**: if the file already exists, do not rewrite it.
          - Return ``"sha256:<hexdigest>"`` — the value used as ``prompt_ref`` /
            ``completion_ref`` in trajectory events.
        """

        hexdigest = hashlib.sha256(text.encode("utf-8")).hexdigest()

        blob_path = self.blobs_dir / hexdigest

        if not blob_path.exists():
            blob_path.write_text(text, encoding="utf-8")

        return f"sha256:{hexdigest}"

    # --------------------------------------------------------------- events
    def log_event(
        self,
        *,
        trace_ctx: TraceCtx,
        call_type: CallType,
        action: str,
        model: str | None = None,
        provider: str | None = None,
        model_snapshot: str | None = None,
        prompt_ref: str | None = None,
        completion_ref: str | None = None,
        result_ref: str | None = None,
        code_ref: str | None = None,
        stdout_ref: str | None = None,
        stderr_ref: str | None = None,
        args_ref: str | None = None,
        result_summary: str | None = None,
        args_hash: str | None = None,
        tool: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: int | None = None,
        cost_usd: float | None = None,
        stage: str | None = None,
        confidence: float | None = None,
        verdict: str | None = None,
        error: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append exactly one JSON object as one line to ``trajectory.jsonl``.

        Contract:
          - Merge ``trace_id`` / ``step_idx`` / ``actor`` from ``trace_ctx`` and
            add a UTC ISO-8601 ``timestamp``; keep every other argument.
          - **Omit ``None`` fields** so events stay compact.
          - Append one UTF-8 JSON line per call (never buffer or batch here).
          - ``code_ref`` / ``stdout_ref`` are the ``exec`` event's payload refs
            (the code that ran and what it printed); they are what makes the root
            context reconstructible (:meth:`rebuild`).
          - ``args_ref`` is the blobbed **arguments** of a ``tool`` event. The
            ``args_hash`` is enough to *match* a recorded call for replay, but a
            viewer cannot render a hash — the args blob is what makes a turn
            displayable (M10b) and a bundle self-describing.
        """
        iso_string = datetime.now(timezone.utc).isoformat()

        info_dict = {"trace_id": trace_ctx.trace_id,
                     "step_idx": trace_ctx.step_idx,
                     "timestamp": iso_string,
                     "actor": trace_ctx.actor,
                     "call_type": call_type,
                     "action": action,
                     "model": model,
                     "provider": provider,
                     "model_snapshot": model_snapshot,
                     "prompt_ref": prompt_ref,
                     "completion_ref": completion_ref,
                     "result_ref": result_ref,
                     "code_ref": code_ref,
                     "stdout_ref": stdout_ref,
                     "stderr_ref": stderr_ref,
                     "args_ref": args_ref,
                     "result_summary": result_summary,
                     "args_hash": args_hash,
                     "tool": tool,
                     "tokens_in": tokens_in,
                     "tokens_out": tokens_out,
                     "latency_ms": latency_ms,
                     "cost_usd": cost_usd,
                     "stage": stage,
                     "confidence": confidence,
                     "verdict": verdict,
                     "error": error,
                     "extra": extra
                     }

        info_dict = {k: v for k, v in info_dict.items() if v is not None}

        with open(self.trajectory_path, "a", encoding="utf-8") as f:
            json.dump(info_dict, f, ensure_ascii=False)
            f.write("\n")

    # ---------------------------------------------------------------- totals
    def totals(self) -> dict[str, Any]:
        """Sum this run's cost and tokens from ``trajectory.jsonl``.

        Contract::

            {
              "cost_usd":   <float>,   # sum over every event (tools add 0)
              "tokens_in":  <int>,     # idem
              "tokens_out": <int>,
              "calls":      <int>,     # BILLABLE model calls: chat|jev, no "error"
              "tool_calls": <int>,     # call_type == "tool"
              "by_action":  {<action>: {"events": <int>, "cost_usd": <float>,
                                        "tokens_in": <int>, "tokens_out": <int>}},
            }

        Vocabulary (Tech Stack §5) — do not conflate the levels:

        - **run** = one review (this file, one ``run_id``);
        - **turn** = one loop iteration (one ``step_idx``);
        - **call** = one API request/response (``chat`` or ``jev``);
        - **tool call** = one host-tool dispatch (``call_type="tool"``), which
          costs no API money and is therefore counted **separately**.

        ``calls`` is the number that sits next to cost in the thesis table
        (H₃), so it must not move when retrieval does more work. It also must not
        move on a retry: ``clients._post`` logs each failed attempt as a
        ``chat.retry`` event **carrying an ``error``**, so only the successful
        completion (no ``error`` key) counts. ``by_action`` keeps every action
        visible, keyed by ``action`` — its buckets count ``events`` because
        "call" is now a specific word.

        Why the log and not ``Budget``: ``Budget`` is the **cap enforcer** and
        only sees what the orchestrator tells it; JEV calls and host tools have
        their own paths. The trajectory is written by the transport itself, so
        it is the authoritative record for a reported ``$/query`` (R4). Missing
        fields count as zero; a missing/empty file returns zeroed totals;
        unreadable lines are skipped (a crash mid-append must not make a run
        unreportable). Use ``math.fsum`` for ``cost_usd`` so the reported total
        is correctly rounded and independent of event order.

        This is read-only and must never mutate or rewrite the log (F-02: the
        audit trail is append-only).
        """
        totals_dict = {
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "calls": 0,
            "by_action": {}
        }

        if not self.trajectory_path.is_file():
            return totals_dict

        with open(self.trajectory_path, 'r', encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                
                if not clean_line:
                    continue

                try:
                    data = json.loads(clean_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue

                action = data.get("action")
                
                bucket = {action: {"events": 0, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}}

                bucket[action]["events"] += 1
                bucket[action]["cost_usd"] += float(data.get("cost_usd", 0.0))
                bucket[action]["tokens_in"] += data.get("cost_usd", 0.0)
                bucket[action]["tokens_out"] += data.get("tokens_out", 0)

                cur_cost =  float(data.get("cost_usd", 0.0))
                totals_dict["tokens_in"] += data.get("tokens_in", 0)
                totals_dict["tokens_out"] += data.get("tokens_out", 0)

                totals_dict["cost_usd"] = math.fsum([totals_dict["cost_usd"], cur_cost])

                if data.get("call_type") in ("chat", "jev") and "error" not in data:
                    totals_dict["calls"] += 1

                totals_dict["by_action"].update(bucket)

        return totals_dict

    # --------------------------------------------------------------- rebuild
    def load_blob(self, ref: str) -> str:
        """Return the text stored under ``ref`` (``"sha256:<hex>"``).

        The inverse of :meth:`store_blob`. Raises ``FileNotFoundError`` when the
        blob is gone — a resume that cannot rebuild its evidence must fail
        loudly, not silently produce a thinner context.
        """
        hexdigest = ref.split(":")[1]

        blob_path = self.blobs_dir / hexdigest

        if not blob_path.exists():
            raise FileNotFoundError(f"Blob not found at {blob_path}")
        else:
            with open(blob_path, 'r', encoding="utf-8") as file:
                content = file.read()

                return content

    def rebuild(self) -> "RebuiltState":
        """Fold ``trajectory.jsonl`` back into the state a resume needs.

        Contract:
          1. Read the events in file order (append-only). Empty/missing log ->
             zeroed :class:`RebuiltState`.
          2. ``last_seq`` = max ``step_idx`` + 1 (0 when empty).
          3. ``budget`` = the last event whose ``extra["budget"]`` is a dict
             (the per-turn snapshot the orchestrator logs).
          4. ``emits`` = for every ``call_type == "emit"`` event, ``extra["var"]``
             -> ``result_ref`` (last write wins).
          5. ``tool_results`` = ``f"{tool}:{args_hash}"`` -> ``result_ref`` for
             every ``call_type == "tool"`` event that has both.
          6. ``messages`` = **fold**, in order:
             - on a root ``chat`` completion: load ``prompt_ref``, parse it as the
               request body, take ``payload["messages"]`` as the current list;
               remember the completion text (``load_blob(completion_ref)``);
             - on the following ``exec``: append the assistant message (that
               completion) and then a ``user`` message with the block's stdout
               (``load_blob(stdout_ref)``), or, when the block failed, the error
               text. This is exactly the sequence the loop built.
             - a trailing ``exec`` with no ``stdout_ref`` (crash mid-block) still
               appends a ``user`` message saying the block was interrupted and
               that its effects are unknown — the resume must not pretend it ran.
          7. Deterministic: two calls on the same log return equal state.
        """
        if not self.trajectory_path.is_file():
            raise FileNotFoundError(f"Trajectory file not found at {self.trajectory_path}")

        messages = []
        budget = {}
        emits = {}
        tool_results = {}
        last_seq = 0

        with open(self.trajectory_path, 'r', encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                
                if not clean_line:
                    continue
                
                try:
                    data = json.loads(clean_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue

                trace_ctx = data.get("trace_ctx")

                if trace_ctx is not None:
                    last_seq = max(trace_ctx.step_idx+1, last_seq)

                extra_budget_dict = data.get("extra", {}).get("budget")

                if extra_budget_dict is not None:
                    budget = copy.deepcopy(extra_budget_dict)

                if data.get("call_type") == "emit" and "error" not in data:
                    var_name = data.get("extra", {}).get("var")
                    result_ref = data.get("result_ref")

                    if var_name is not None and result_ref is not None:
                        emits[var_name] = result_ref

                if data.get("call_type") == "tool" and "error" not in data:
                    tool = data.get("tool")
                    args_hash = data.get("args_hash")
                    result_ref = data.get("result_ref")

                    if tool is not None and args_hash is not None and result_ref is not None:
                        key_name = f"{tool}:{args_hash}"
                        tool_results[key_name] = result_ref

                if data.get("call_type") == "chat" and "error" not in data:
                    prompt_ref = data.get("prompt_ref")
        
        rebuild_obj = RebuiltState(
            messages=messages,
            budget=budget,
            emits=emits,
            tool_results=tool_results,
            last_seq=last_seq
        )
        
        return rebuild_obj

