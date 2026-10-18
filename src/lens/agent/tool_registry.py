"""Host tool registry: manifest, dispatch, and per-call logging.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a/§4i and §5.

The registry is the single source of truth for what the RLM may call. The
manifest exposes names, signatures, and docstrings only — never source code —
and every dispatch stores the full result as a content-addressed blob while
returning a compact summary for the trajectory.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from lens import config
from lens.observability.trajectory import TraceCtx, Trajectory


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool: callable + the contract shown to the RLM."""

    name: str
    func: Callable[..., Any]
    signature: str
    description: str


@dataclass
class ToolCallResult:
    """Outcome of one dispatch (also the sandbox RPC envelope)."""

    ok: bool
    result: Any = None
    result_ref: str | None = None
    result_summary: str | None = None
    error: dict[str, Any] | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable RPC response."""
        raise NotImplementedError("M8 scaffold: implement ToolCallResult.to_dict")


def args_hash(args: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 over canonical JSON of ``args``."""
    raise NotImplementedError("M8 scaffold: implement args_hash")


class ToolRegistry:
    """Name -> :class:`ToolSpec` map used by the orchestrator and RPC server."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        *,
        description: str | None = None,
    ) -> None:
        """Register ``func`` under ``name``; derive the signature with ``inspect``."""
        raise NotImplementedError("M8 scaffold: implement ToolRegistry.register")

    def get(self, name: str) -> ToolSpec:
        """Return one spec; unknown names are a ``KeyError``."""
        raise NotImplementedError("M8 scaffold: implement ToolRegistry.get")

    def names(self) -> list[str]:
        """Return registered tool names in registration order."""
        raise NotImplementedError("M8 scaffold: implement ToolRegistry.names")

    def manifest(self) -> str:
        """Render the manifest as plain text for the system prompt."""
        raise NotImplementedError("M8 scaffold: implement ToolRegistry.manifest")

    def dispatch(
        self,
        name: str,
        args: Mapping[str, Any],
        *,
        trace_ctx: TraceCtx,
        trajectory: Trajectory,
    ) -> ToolCallResult:
        """Call ``name`` with ``args``, log one event, and return the envelope.

        Log fields: ``tool``, ``args_hash``, ``result_ref``, ``result_summary``,
        ``latency_ms``, and ``error`` when the call failed. The full result is
        stored with ``trajectory.store_blob``. Exceptions are captured as a
        structured error, never re-raised with credentials in the message.
        """
        raise NotImplementedError("M8 scaffold: implement ToolRegistry.dispatch")

    def summarize_result(self, result: Any) -> str:
        """Return a compact, stable summary of ``result`` for the trajectory."""
        raise NotImplementedError("M8 scaffold: implement ToolRegistry.summarize_result")
