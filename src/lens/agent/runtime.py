"""The LENS orchestrator: host-owned RLM loop.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a (host drives, sandbox executes)
and §5 (ledger + trajectory).

One :class:`Runtime` instance owns one query. It starts the sandbox, seeds the
workspace, calls the root model, sends generated code to the passive runner,
services RPCs from that code, appends compact observations, checks the budget,
and returns a grounded answer or a labelled partial answer.

Invariants
----------
- The host owns the API key, DB connection, budget, ledger, and trajectory.
- The sandbox holds no key and no DB driver; every effect is an RPC here.
- Large data stays in sandbox variables or ``emit()`` files. Only summaries
  enter the root context.
- Depth 1: ``sub_call`` is a plain GPT-5-mini call with no tools and no REPL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from lens import config
from lens.agent.budget import Budget
from lens.agent.ledger import TaskLedger
from lens.agent.sandbox_session import SandboxSession
from lens.agent.tool_registry import ToolRegistry
from lens.clients import LensClient
from lens.observability.trajectory import TraceCtx, Trajectory


@dataclass
class RunResult:
    """Outcome of one query, also the handoff to M9/M10."""

    answer: str
    partial: bool
    stop_reason: str | None
    iterations: int
    cost_usd: float
    trajectory_path: Path
    tasks_path: Path
    answer_path: Path


class Runtime:
    """Own the loop for one query.

    Dependencies are injected so unit tests can use fakes (no key, no DB, no
    Docker). ``run.py`` wires the real objects.
    """

    def __init__(
        self,
        query: str,
        run_id: str,
        *,
        client: LensClient,
        trajectory: Trajectory,
        conn: Any,
        embedder: Any,
        reranker: Any,
        extractor: Any,
        sandbox: SandboxSession,
        registry: ToolRegistry,
        budget: Budget | None = None,
        ledger: TaskLedger | None = None,
        recorded: bool = config.LENS_RECORDED,
    ) -> None:
        self.query = query
        self.run_id = run_id
        self.client = client
        self.trajectory = trajectory
        self.conn = conn
        self.embedder = embedder
        self.reranker = reranker
        self.extractor = extractor
        self.sandbox = sandbox
        self.registry = registry
        self.budget = budget if budget is not None else Budget()
        self.ledger = ledger if ledger is not None else TaskLedger(
            trajectory.run_dir / "tasks.json"
        )
        self.recorded = recorded
        self.run_dir: Path = trajectory.run_dir
        self.workspace: Path = sandbox.workspace
        self._step_idx = 0
        self._trace_id = run_id
        self._query_buckets: dict[str, list[str]] = {}
        self._messages: list[dict[str, Any]] = []
        self._ontology_summary: dict[str, Any] | None = None

    # ------------------------------------------------------------------- run
    def run(self) -> RunResult:
        """Execute one query end-to-end.

        Sequence:
          1. ``segment_query`` -> ``query.json``.
          2. Start the sandbox and seed its namespace/context files.
          3. Loop: build context, call root, execute code, service RPCs, observe,
             compact, check budget — until ``FINAL_ANSWER:`` or a cap.
          4. Stop the sandbox and write ``answer.md`` + ``RunResult``.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime.run")

    # ------------------------------------------------------------------- RPC
    def handle_rpc(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Route one sandbox RPC to the host implementation.

        ``endpoint`` is one of ``/tool``, ``/sub_call``, ``/budget``,
        ``/emit``, ``/note``. Return ``{"ok": True, "result": ...}`` or
        ``{"ok": False, "error": {...}}``. Never include credentials.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime.handle_rpc")

    # --------------------------------------------------------------- internal
    def _segment_query(self) -> dict[str, list[str]]:
        """Segment the query with M6 and write ``<run_dir>/query.json``."""
        raise NotImplementedError("M8 scaffold: implement Runtime._segment_query")

    def _seed_workspace(self) -> None:
        """Write context files and seed the sandbox namespace.

        The first sandbox block should define at least ``query``, ``RUN_ID``,
        and ``WORKSPACE`` so later code does not have to re-read ``query.json``.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime._seed_workspace")

    def _build_context(self) -> list[dict[str, Any]]:
        """Assemble the root message list, compacting when over the threshold."""
        raise NotImplementedError("M8 scaffold: implement Runtime._build_context")

    def _run_loop(self) -> tuple[str, bool, str | None]:
        """Return ``(answer, partial, stop_reason)``."""
        raise NotImplementedError("M8 scaffold: implement Runtime._run_loop")

    def _dispatch_tool(
        self,
        name: str,
        args: Mapping[str, Any],
        *,
        trace_ctx: TraceCtx,
    ) -> dict[str, Any]:
        """Dispatch one host tool and return the RPC envelope."""
        raise NotImplementedError("M8 scaffold: implement Runtime._dispatch_tool")

    def _sub_call(self, prompt: str, context: str | None = None) -> str:
        """Depth-1 plain GPT-5-mini call. No tools, no REPL, no recursion."""
        raise NotImplementedError("M8 scaffold: implement Runtime._sub_call")

    def _emit(self, value: Any, label: str | None = None) -> dict[str, Any]:
        """Offload a value to ``<run_dir>/emits/`` and return its ref/path."""
        raise NotImplementedError("M8 scaffold: implement Runtime._emit")

    def _note(self, text: str) -> None:
        """Append one free-form note event to the trajectory."""
        raise NotImplementedError("M8 scaffold: implement Runtime._note")

    def _finalize(self, answer: str, *, partial: bool, stop_reason: str | None) -> RunResult:
        """Write ``answer.md``, stop the sandbox, and build the result."""
        raise NotImplementedError("M8 scaffold: implement Runtime._finalize")
