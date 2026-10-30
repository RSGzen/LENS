"""Observability sink: content-addressed blobs + one-schema JSONL trajectory.

Design: :class:`Trajectory` is the *injected* sink — ``clients.py`` receives one
in its constructor and calls :meth:`Trajectory.store_blob` /
:meth:`Trajectory.log_event` for every LLM/JEV call, so no call escapes logging.
One instance = one run, writing to ``<runs_root>/<run_id>/trajectory.jsonl``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
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
        "lens.config not importable — run from the repo root or install the package with `uv sync`."
    ) from e

# Who performed the action.
Actor = Literal["root", "sub", "tool"]

# What the event is. One schema, several kinds:
# - ``chat`` / ``jev`` : the two LLM transports
# - ``tool``          : one host-tool dispatch (ToolRegistry.dispatch)
# - ``exec``          : one sandbox code block (Runtime._run_loop) — code + stdout blobs
# - ``emit``          : one ``rlm_api.emit()`` offload (variable -> blob ref)
# - ``note``          : a free-form RLM/operator annotation (also ``run.resume``)
# Keeping them in one enum is what lets cost aggregation separate model spend
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
    """Everything the log can tell us about a run.

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

    Layout ::

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
            # **Bytes**, not text: ``write_text`` translates "\n" to "\r\n" on
            # Windows, so the file on disk would not be the bytes that were hashed.
            # Content-addressing then silently lies — the filename is not the hash of
            # the file — which is exactly what bundle verification caught (three
            # blobs "do not match its name", ). Byte-exact or not at
            # all is the only honest option for a content-addressed store.
            blob_path.write_bytes(text.encode("utf-8"))

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
        cached_tokens: int | None = None,
        reasoning_tokens: int | None = None,
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

        info_dict = {
            "trace_id": trace_ctx.trace_id,
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
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "stage": stage,
            "confidence": confidence,
            "verdict": verdict,
            "error": error,
            "extra": extra,
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
              "cached_tokens":    <int>,  # subset of tokens_in, billed cheaper
              "reasoning_tokens": <int>,  # subset of tokens_out (reasoning models)
              "calls":      <int>,     # BILLABLE model calls: chat|jev, no "error"
              "tool_calls": <int>,     # call_type == "tool"
              "by_action":  {<action>: {"events": <int>, "cost_usd": <float>,
                                        "tokens_in": <int>, "tokens_out": <int>}},
            }

        Vocabulary — do not conflate the levels:

        - **run** = one review (this file, one ``run_id``);
        - **turn** = one loop iteration (one ``step_idx``);
        - **call** = one API request/response (``chat`` or ``jev``);
        - **tool call** = one host-tool dispatch (``call_type="tool"``), which
          costs no API money and is therefore counted **separately**.

        ``calls`` is the number that sits next to cost in the table
        (H₃), so it must not move when retrieval does more work. It also must not
        move on a retry: ``clients._post`` logs each failed attempt as a
        ``chat.retry`` event **carrying an ``error``**, so only the successful
        completion (no ``error`` key) counts. ``by_action`` keeps every action
        visible, keyed by ``action`` — its buckets count ``events`` because
        "call" is now a specific word.

        Why the log and not ``Budget``: ``Budget`` is the **cap enforcer** and
        only sees what the orchestrator tells it; JEV calls and host tools have
        their own paths. The trajectory is written by the transport itself, so
        it is the authoritative record for a reported ``$/query``. Missing
        fields count as zero; a missing/empty file returns zeroed totals;
        unreadable lines are skipped (a crash mid-append must not make a run
        unreportable). Use ``math.fsum`` for ``cost_usd`` so the reported total
        is correctly rounded and independent of event order.

        This is read-only and must never mutate or rewrite the log (: the
        audit trail is append-only).
        """
        totals_dict = {
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "calls": 0,
            "tool_calls": 0,
            "by_action": {},
        }

        if not self.trajectory_path.is_file():
            return totals_dict

        with open(self.trajectory_path, "r", encoding="utf-8") as f:
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

                action = data.get("action") or "unknown"

                # One bucket PER ACTION, reused across events: building a fresh
                # dict per event and `.update()`-ing it in would keep only the
                # last event of each action.
                bucket = totals_dict["by_action"].setdefault(
                    action,
                    {"events": 0, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0},
                )

                bucket["events"] += 1
                bucket["cost_usd"] += float(data.get("cost_usd", 0.0))
                bucket["tokens_in"] += int(data.get("tokens_in") or 0)
                bucket["tokens_out"] += int(data.get("tokens_out") or 0)

                cur_cost = float(data.get("cost_usd", 0.0))
                totals_dict["tokens_in"] += int(data.get("tokens_in") or 0)
                totals_dict["tokens_out"] += int(data.get("tokens_out") or 0)
                # Both are subsets of tokens_in / tokens_out, kept separately so the
                # cost model can be checked (cache hit rate, reasoning share) rather
                # than assumed. Budget caps stay on the totals they belong to.
                totals_dict["cached_tokens"] += int(data.get("cached_tokens") or 0)
                totals_dict["reasoning_tokens"] += int(data.get("reasoning_tokens") or 0)

                totals_dict["cost_usd"] = math.fsum([totals_dict["cost_usd"], cur_cost])

                if data.get("call_type") in ("chat", "jev") and "error" not in data:
                    totals_dict["calls"] += 1

                # Every dispatch counts, failed or not: a tool that errored was
                # still called. (There are no tool.retry events, so no inflation.)
                if data.get("call_type") == "tool":
                    totals_dict["tool_calls"] += 1

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

        # Read **bytes** and decode: a text-mode read normalises newlines, which
        # would hide the fact that older blobs were written with CRLF (see
        # ``store_blob``). The explicit normalisation keeps those legacy blobs
        # readable while new ones are byte-exact.
        return blob_path.read_bytes().decode("utf-8").replace("\r\n", "\n")

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
        messages = []
        budget = {}
        emits = {}
        tool_results = {}
        last_seq = 0

        if not self.trajectory_path.is_file():
            rebuild_obj = RebuiltState(
                messages=messages,
                budget=budget,
                emits=emits,
                tool_results=tool_results,
                last_seq=last_seq,
            )
            return rebuild_obj

        # Must live OUTSIDE the loop: the chat and its exec are two different lines.
        pending_completion = None

        with open(self.trajectory_path, "r", encoding="utf-8") as f:
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

                step_idx = data.get("step_idx")

                if step_idx is not None:
                    last_seq = max(int(step_idx) + 1, last_seq)

                extra = data.get("extra")
                if isinstance(extra, dict) and extra.get("budget") is not None:
                    budget = copy.deepcopy(extra["budget"])

                call_type = data.get("call_type")

                # To play the /sub_call/ part of the request, the body is an API
                # request with `messages`; a `sub_call` also logs call_type="chat"
                # but with actor="sub" and a small, unrelated history, so the fold
                # must match actor == "root" or it would destroy the context.
                if call_type == "chat" and data.get("actor") == "root" and "error" not in data:
                    # prompt_ref holds the WHOLE request body, so its `messages`
                    # is the authoritative history so far: REPLACE, never extend
                    # (extending would re-append every earlier turn, quadratically).
                    payload = json.loads(self.load_blob(data["prompt_ref"]))
                    messages = list(payload.get("messages") or [])
                    pending_completion = self.load_blob(data["completion_ref"])

                # `jev` (and `tool`/`emit`/`note`) never contribute to the root
                # context: a JEV request is `state` + `questions`, not messages.
                elif call_type == "exec":
                    if pending_completion is not None:
                        messages.append({"role": "assistant", "content": pending_completion})
                        pending_completion = None

                    if data.get("stdout_ref"):
                        messages.append(
                            {
                                "role": "user",
                                "content": self.load_blob(data["stdout_ref"]),
                            }
                        )
                    elif data.get("error"):
                        messages.append(
                            {
                                "role": "user",
                                "content": f"[block failed: {data['error']}]",
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": "[block interrupted; effects unknown]",
                            }
                        )

                elif call_type == "emit" and "error" not in data:
                    var_name = extra.get("var") if isinstance(extra, dict) else None
                    result_ref = data.get("result_ref")

                    if var_name is not None and result_ref is not None:
                        # name -> REF (prefix kept), not the content:
                        # Runtime._restore_state calls load_blob(ref) itself.
                        emits[var_name] = result_ref

                elif call_type == "tool" and "error" not in data:
                    tool = data.get("tool")
                    args_hash = data.get("args_hash")
                    result_ref = data.get("result_ref")

                    if tool is not None and args_hash is not None and result_ref is not None:
                        tool_results[f"{tool}:{args_hash}"] = result_ref

        # A chat whose exec never happened (crash between the two): the block it
        # wrote is still part of the state the resumed model must see.
        if pending_completion is not None:
            messages.append({"role": "assistant", "content": pending_completion})

        return RebuiltState(
            messages=messages,
            budget=budget,
            emits=emits,
            tool_results=tool_results,
            last_seq=last_seq,
        )
