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
from dataclasses import dataclass
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

# The two transports sharing the one trajectory schema (LOG-31).
CallType = Literal["chat", "jev"]


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
