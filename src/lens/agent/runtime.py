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
- **The sandbox workspace is ``<run_dir>/work``** — the only directory mounted
  into the container. ``trajectory.jsonl``, ``tasks.json`` and ``answer.md`` live
  one level up in ``<run_dir>`` and are *never* visible to the RLM, so a
  generated block cannot rewrite its own evidence. Anything the RLM must ``open``
  has a container path under ``config.SANDBOX_WORKSPACE_PATH`` (``/workspace``).
- Large data stays in sandbox variables or ``emit()`` files. Only **printed
  digests** enter the root context (the tool result itself never does).
- Depth 1: ``sub_call`` is a plain GPT-5-mini call with no tools and no REPL.
- Two cost sources: :class:`~lens.agent.budget.Budget` enforces the caps, and
  :meth:`lens.observability.trajectory.Trajectory.totals` is what the reported
  ``$/query`` is read from (R4).
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
    """Outcome of one query, also the handoff to M9/M10.

    ``cost_usd`` is the **trajectory** total (all chat + JEV calls, cached-token
    pricing included), not the budget counter — that is the number R4 asks for.
    ``partial`` says the answer was cut by a cap or a backtrack limit, and
    ``stop_reason`` names which one.
    """

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
        # The mounted, RLM-writable directory. Never ``run_dir`` itself: the
        # artifacts there are the audit trail (see the module invariants).
        self.workspace: Path = sandbox.workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._step_idx = 0
        self._trace_id = run_id
        self._query_buckets: dict[str, list[str]] = {}
        self._messages: list[dict[str, Any]] = []
        self._ontology_summary: dict[str, Any] | None = None
        # Stage gate bookkeeping: one gate per stage per run (see
        # ``tools.evaluate.GATED_STAGES``).
        self._gated_stages: set[str] = set()
        self._gate_verdicts: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------- run
    def run(self) -> RunResult:
        """Execute one query end-to-end.

        Sequence:
          1. ``_segment_query`` -> ``<workspace>/query.json`` (M6 owns the file).
          2. ``_register_runtime_tools`` -> ledger tools + ``backtrack``.
          3. Start the sandbox, ``_seed_workspace`` (context files + a first block
             defining ``query`` / ``RUN_ID`` / ``WORKSPACE``).
          4. Loop: build context (compacting; appending the wrap-up nudge when a
             cap is near), call the root with ``max_tokens`` =
             ``ROOT_MAX_OUTPUT_TOKENS``, execute code, service RPCs, truncate the
             observation, log a ``budget`` event, check the budget — until
             ``FINAL_ANSWER:`` or a cap.
          5. ``_finalize``: ``answer.md`` + ``RunResult`` (cost from
             ``trajectory.totals()``), sandbox stopped **even on failure**.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime.run")

    # ------------------------------------------------------------------- RPC
    def handle_rpc(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Route one sandbox RPC to the host implementation.

        ``endpoint`` is one of ``/tool``, ``/sub_call``, ``/budget``,
        ``/emit``, ``/note``. Return ``{"ok": True, "result": ...}`` or
        ``{"ok": False, "error": {...}}``. Never include credentials.

        ``/tool`` returns the **full** tool result: that is the point of the
        design — the payload lands in a sandbox variable and never in the root
        context. Only the code's printed digest becomes an observation. A stage
        gate verdict, when produced, rides along as ``"gate"`` on the same
        envelope.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime.handle_rpc")

    # --------------------------------------------------------------- internal
    def _register_runtime_tools(self) -> None:
        """Register the host-runtime operations the RLM is allowed to call.

        LOG-31: these are **not** sandbox-local functions — they touch host
        state (budget, ledger) and are reached through the façade like any tool.
        Register them as small closures with a one-line docstring each (the
        docstring *is* the manifest entry the root reads; a ``functools.partial``
        would show up with no docstring and with its bound parameters leaked
        into the visible signature):

        - ``ledger_create(requirement, parent_id=None) -> {"task_id": ...}``
        - ``ledger_update(task_id, status) -> task`` (status in
          ``ledger.VALID_STATUSES``; invalid -> structured error)
        - ``ledger_add_evidence(task_id, refs) -> task``
        - ``backtrack(stage, reason) -> {"backtracks_left": int}`` — counts one
          backtrack, marks the stage's task ``failed`` with the reason, and
          refuses past ``MAX_BACKTRACKS``.

        Without these the ledger stays empty and ``tasks.json`` is decorative.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime._register_runtime_tools")

    def _segment_query(self) -> dict[str, list[str]]:
        """Segment the query with M6 and write ``<workspace>/query.json``.

        Use :func:`lens.agent.segmentation.segment_query` then
        :func:`lens.agent.segmentation.write_query_json` — M6 owns the file's
        format, and the workspace copy **is** the artifact (it sits under
        ``<run_dir>/work``). Do not write a second copy into ``run_dir``.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime._segment_query")

    def _seed_workspace(self) -> None:
        """Write context files and seed the sandbox namespace.

        Context files via :func:`lens.agent.context.write_context_files`
        (``AGENTS.md``, ``tools.md``, ``budget.md``, ``ontology.md`` when the KG is
        on) — **not** ``query.json``, which M6 already wrote here.

        Then run one block that defines at least ``query``, ``RUN_ID`` and
        ``WORKSPACE`` so later code does not re-read ``query.json`` on every turn.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime._seed_workspace")

    def _build_context(self) -> list[dict[str, Any]]:
        """Assemble the root message list, compacting when over the threshold.

        Compaction must keep: the system message (it holds the
        ``FINAL_ANSWER:`` contract), a one-line note that the sandbox namespace
        persists across turns, and the most recent turns. The namespace is the
        state — dropping the *code* that built a variable must not make the model
        believe the variable is gone.

        Also append :func:`lens.agent.context.budget_warning` when it returns
        text: one system line asking for ``FINAL_ANSWER`` before a cap bites.
        With no warning, the prompt stays byte-identical so the static prefix
        remains cacheable.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime._build_context")

    def _run_loop(self) -> tuple[str, bool, str | None]:
        """Return ``(answer, partial, stop_reason)``.

        Per turn: call the root (``max_tokens=config.ROOT_MAX_OUTPUT_TOKENS``,
        record the chat in the budget), extract one ```python block or the
        ``FINAL_ANSWER:`` text, send code to the sandbox, truncate the printed
        observation with :func:`lens.agent.context.truncate_observation`, append
        it as a ``user`` message, then log one ``budget`` event with
        ``self.budget.snapshot()`` (this is what M11 reads for the cap-binding
        rate) and check :meth:`Budget.can_continue`.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime._run_loop")

    def _dispatch_tool(
        self,
        name: str,
        args: Mapping[str, Any],
        *,
        trace_ctx: TraceCtx,
    ) -> dict[str, Any]:
        """Dispatch one host tool and return the **full** RPC envelope.

        Call ``self.registry.dispatch(name, args, trace_ctx=trace_ctx,
        trajectory=self.trajectory)`` and return ``result.to_dict()`` (the
        registry already stored the blob and logged the event).

        If ``name`` is in ``tools.evaluate.GATED_STAGES`` and that stage has not
        been gated yet in this run, call
        :func:`lens.tools.evaluate.evaluate_stage` on the successful result and
        add ``"gate": verdict.to_dict()`` to the envelope. Record the verdict so
        ``evaluate_stage`` is called **once per stage per run** (tier 2 is a JEV
        call; repeating it on every rerank would only add cost).
        """
        raise NotImplementedError("M8 scaffold: implement Runtime._dispatch_tool")

    def _sub_call(self, prompt: str, context: str | None = None) -> str:
        """Depth-1 plain GPT-5-mini call. No tools, no REPL, no recursion.

        Record it in the budget (:meth:`Budget.record_chat`) — sub-calls are
        cheap but they are not free, and H₃ counts them.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime._sub_call")

    def _emit(self, value: Any, label: str | None = None) -> dict[str, Any]:
        """Offload a value to ``<workspace>/emits/`` and return its ref + paths.

        Return shape (the RLM's contract)::

            {"ref": "sha256:<hex>", "path": "/workspace/emits/<name>",
             "host_path": "D:\\...", "bytes": <int>}

        Path mapping rule for the whole harness: host ``self.workspace`` <->
        container ``config.SANDBOX_WORKSPACE_PATH``. **``path`` must be the
        container form** — a host path is unopenable inside the sandbox — and the
        blob is stored through the trajectory so the artifact is auditable.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime._emit")

    def _note(self, text: str) -> None:
        """Append one free-form note event to the trajectory (``call_type="note"``)."""
        raise NotImplementedError("M8 scaffold: implement Runtime._note")

    def _finalize(self, answer: str, *, partial: bool, stop_reason: str | None) -> RunResult:
        """Write ``answer.md``, stop the sandbox, and build the result.

        ``answer.md`` goes to ``run_dir`` (never the workspace). Cost comes from
        ``self.trajectory.totals()["cost_usd"]``. Stop the sandbox in a
        ``finally``-safe path so a crash mid-loop does not leave a container
        behind.
        """
        raise NotImplementedError("M8 scaffold: implement Runtime._finalize")
