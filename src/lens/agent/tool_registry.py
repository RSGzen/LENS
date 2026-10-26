"""Host tool registry: manifest, dispatch, and per-call logging.

The registry is the single source of truth for what the RLM may call. The
manifest exposes names, signatures, and docstrings only — never source code or a
function ``repr`` (a repr carries a memory address, which would make the system
prompt differ on every run and break prefix caching) — and every dispatch stores
the full result as a content-addressed blob while returning a compact summary for
the trajectory.

Two objects, two jobs
---------------------
``ToolCallResult`` is what :meth:`ToolRegistry.dispatch` returns **and** what
crosses the RPC boundary (:meth:`ToolCallResult.to_dict`). The **full** ``result``
goes back to the sandbox and lands as a variable; ``result_summary`` is the bounded
one-liner kept for the trajectory and the UI. That split is the context
discipline: nothing large ever enters the root prompt by accident.
"""

from __future__ import annotations

import hashlib
import inspect
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
        """Return a JSON-serializable RPC response.

        ``ok`` and ``result`` are always present — ``rlm_api._post`` unwraps
        ``result`` and raises on ``ok: false``, so a missing key would be a crash
        in the sandbox rather than an error the model can read. The optional
        fields are omitted when ``None`` to keep the envelope small.
        """
        envelope: dict[str, Any] = {"ok": self.ok, "result": self.result}

        for key in ("result_ref", "result_summary", "error", "duration_ms"):
            value = getattr(self, key)
            if value is not None:
                envelope[key] = value

        return envelope


def args_hash(args: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 over **canonical** JSON of ``args``.

    Canonical = sorted keys + compact separators, so ``{"a":1,"b":2}`` and
    ``{"b":2,"a":1}`` hash the same. That is what makes recorded replay work:
    :meth:`Trajectory.rebuild` indexes recorded results by
    ``f"{tool}:{args_hash}"`` and a live call must produce the identical string.
    """
    json_str = json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    hex_digest = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    return hex_digest


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
        """Register ``func`` under ``name``; derive the signature with ``inspect``.

        Registration **publishes** a callable — it never calls it. ``func`` is a
        host function the RLM must not see the internals of: a closure that binds
        ``conn`` / ``embedder`` / ``reranker`` (so the visible signature is only
        the RLM-facing parameters) or a runtime closure that needs orchestrator
        state (``backtrack``, ``ledger_*``). ``functools.partial`` is deliberately
        not used: it loses ``__doc__`` and exposes the bound parameters.

        ``description`` defaults to the first line of the function's docstring —
        that line *is* the manifest entry, so a tool without one is undocumented
        by construction.

        Raises
        ------
        ValueError
            ``name`` is already registered (a wiring bug must fail at startup,
            not silently replace a tool), or ``func`` is not callable.
        """
        if not callable(func):
            raise TypeError(f"tool {name!r} is not callable: {func!r}")

        if name in self._specs:
            raise ValueError(f"tool {name!r} is already registered")

        doc = inspect.getdoc(func) or ""
        first_line = doc.splitlines()[0].strip() if doc else ""

        self._specs[name] = ToolSpec(
            name=name,
            func=func,
            signature=str(inspect.signature(func)),
            description=description or first_line,
        )

    def get(self, name: str) -> ToolSpec:
        """Return one spec; unknown names are a ``KeyError``."""
        if name not in self._specs:
            raise KeyError(f"unknown tool {name!r}; registered: {sorted(self._specs)}")

        return self._specs[name]

    def names(self) -> list[str]:
        """Return registered tool names in **registration** order.

        Registration order is the order ``build_runtime`` wires the funnel in
        (shallow → rerank → deep → rerank_chunks), so the manifest reads in
        pipeline order. Sorting alphabetically would scramble that and would also
        disagree with the order :meth:`manifest` prints.
        """
        return list(self._specs.keys())

    def parameters(self) -> dict[str, list[str]]:
        """Return each tool's positional parameter names, in call order.

        Used to bind the tools as plain namespace functions in the sandbox
        (``shallow_match(buckets=..., k=100)``) — the form the manifest prints.
        Without the binding the RLM reads ``shallow_match(buckets)`` in the
        manifest and calls a name that does not exist: the exact ``NameError`` the
         smoke hit. Only positional kinds are listed, so a
        keyword-only parameter must still be passed by name and ``**kwargs`` adds
        nothing to the mapping.
        """
        bound: dict[str, list[str]] = {}

        for name, spec in self._specs.items():
            bound[name] = [
                parameter.name
                for parameter in inspect.signature(spec.func).parameters.values()
                if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.POSITIONAL_ONLY)
            ]

        return bound

    def manifest(self) -> str:
        """Render the manifest as plain text for the system prompt.

        One entry per tool: number, name, signature (which carries the return
        type when the function is annotated), then the first docstring line
        indented. No ``repr``, no source, no bound infrastructure — the model must
        be able to *call* the tool, not read it.
        """
        lines: list[str] = []

        for index, spec in enumerate(self._specs.values(), start=1):
            lines.append(f"{index}. {spec.name}{spec.signature}")
            if spec.description:
                lines.append(f"   {spec.description}")

        return "\n".join(lines)

    def dispatch(
        self,
        name: str,
        args: Mapping[str, Any],
        *,
        trace_ctx: TraceCtx,
        trajectory: Trajectory,
    ) -> ToolCallResult:
        """Call ``name`` with ``args``, log one event, and return the envelope.

        Log fields: ``tool``, ``action="tool.dispatch"``, ``args_hash`` (the
        canonical fingerprint used for replay matching), ``args_ref`` (the args
        blob), ``result_ref``, ``result_summary``, ``latency_ms``, and ``error``
        when the call failed. The **full** result is stored with
        ``trajectory.store_blob``.

        Semantics that matter:

        - **No exceptions escape.** An unknown tool, a bad argument, or a tool that
          raises all become ``ok=False`` with ``error={"type","message"}``, so the
          model reads a structured failure and can retry — a crashed turn loses
          the whole run.
        - **``latency_ms`` is int milliseconds of the call** (args serialisation
          included; blob store and logging excluded). Those are observability
          overhead, and their cost depends on payload size — including them would
          make latency drift with output size and would not be comparable with a
          replayed call, which serves from a blob.
        - ``action`` is ``"tool.dispatch"``, never the tool name: ``action`` says
          *what happened* and ``tool`` says *which* tool, and
          ``Trajectory.totals()["by_action"]`` aggregates on ``action``.
        """
        args_str: str | None = None
        args_fingerprint: str | None = None
        start = time.perf_counter()

        try:
            spec = self.get(name)
            args_str = json.dumps(args, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            args_fingerprint = args_hash(args)

            result = spec.func(**args)
            latency_ms = int((time.perf_counter() - start) * 1000)

            result_str = json.dumps(
                result,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                # A tool may return a DB row: ``date``/``Decimal``/``UUID`` are not
                # JSON types, and an unserialisable result used to abort the whole
                # turn (``fetch_papers`` -> "Object of type date is not JSON
                # serializable", smoke, ). The blob and the RPC envelope
                # are JSON boundaries; ``str`` is the honest narrowing.
                default=str,
            )
            result_ref = trajectory.store_blob(result_str)
            result_summary = self.summarize_result(result)

            trajectory.log_event(
                trace_ctx=trace_ctx,
                call_type="tool",
                action="tool.dispatch",
                tool=name,
                args_hash=args_fingerprint,
                args_ref=trajectory.store_blob(args_str),
                result_ref=result_ref,
                result_summary=result_summary,
                latency_ms=latency_ms,
            )

            return ToolCallResult(
                ok=True,
                result=result,
                result_ref=result_ref,
                result_summary=result_summary,
                duration_ms=latency_ms,
            )

        except Exception as exc:  # noqa: BLE001 - a tool failure must not end the run
            latency_ms = int((time.perf_counter() - start) * 1000)
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}

            trajectory.log_event(
                trace_ctx=trace_ctx,
                call_type="tool",
                action="tool.dispatch",
                tool=name,
                args_hash=args_fingerprint,
                args_ref=trajectory.store_blob(args_str) if args_str is not None else None,
                error=error,
                latency_ms=latency_ms,
            )

            return ToolCallResult(ok=False, error=error, duration_ms=latency_ms)

    def summarize_result(self, result: Any) -> str:
        """Return a compact, stable summary of ``result`` for the trajectory.

        A **structural digest**, not ``str(result)``: collections become their
        size (plus a short preview), so a 400 KB KG payload and a 5-chunk bundle
        both fit in one line. Keys are sorted so two runs of the same tool produce
        the same string (the trajectory is diffed).

        This is **not** the model's view of the result — the model holds the full
        value as a variable and prints its own digest. It exists so an auditor or
        the M10b timeline can read one line without loading the blob.
        """
        if isinstance(result, Mapping):
            parts: list[str] = []
            for key in sorted(result):
                value = result[key]
                if isinstance(value, (list, tuple, set, Mapping)):
                    parts.append(f"{key}={len(value)}")
                else:
                    parts.append(f"{key}={value!r}"[:40])
            summary = "{" + ", ".join(parts) + "}"
        elif isinstance(result, (list, tuple)):
            head = ", ".join(repr(item)[:40] for item in list(result)[:3])
            summary = f"[{len(result)}] {head}" + (" …" if len(result) > 3 else "")
        else:
            summary = repr(result)[:120]

        summary = " ".join(summary.split())  # one line: the field is JSONL
        limit = config.TOOL_RESULT_SUMMARY_CHARS

        return summary if len(summary) <= limit else summary[:limit] + "…"
