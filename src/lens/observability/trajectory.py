# For self-refencing in TraceCtx dataclass params
from __future__ import annotations

import json
import hashlib

from datetime import datetime, timezone
from dataclasses import dataclass
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

# Who performed the action
Actor = Literal["root", "sub", "tool"]

# Normal chat (GPT-5, GPT-5-mini, Qwen3-Coder) vs JEV have different log types
CallType = Literal["chat", "jev"]

@dataclass(frozen=True)
class TraceCtx:
    """
    Identity of one step within one query's trajectory.

    trace_id  --> ID for prompts that belong to the same LLM
    step_idx  --> Current number of prompts under the same LLM
    actor     --> Who performed this action
    """

    trace_id: str
    step_idx: int
    actor: Actor

class Trajectory:
    """
    Single JSONL sink + content-addressed blob store for one run.

    <run_id>/trajectory.jsonl --> For saving trajectory metadata via log_event()
    blobs/<hexdigest>         --> For saving prompt / answer text via hexdigest naming for idempotent operations
    """

    def __init__(self, run_id: str) -> None:
        root = Path(LENS_RUNS_ROOT_PATH)
        self.run_id = run_id
        self.runs_root = root
        self.run_dir = root / run_id
        self.blobs_dir = root / BLOBS_DIR_NAME
        self.trajectory_path = self.run_dir / TRAJECTORY_FILENAME

        # Create if folder does not exists
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ blobs
    def store_blob(self, text: str) -> str:
        """
        Store ``text`` once under a content-derived name; return its ref.

        1. Hash content with text to obtain hexdigest that acts as filename
        - **Idempotent**: if the file already exists, do not rewrite it.

        2. Save <hexdigest> with input text
        3. Return hexdigest as text reference ID
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
        """
        Append exactly one JSON object as one line to ``trajectory.jsonl``.

        1. Create information dictionary with all params
        2. Inject dictionary with current UTC timestamp
        3. Check for <None> value fields and exclude that key value pair
        4. Save as JSONL file
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
