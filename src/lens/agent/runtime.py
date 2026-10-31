"""The LENS orchestrator: host-owned RLM loop.

One :class:`Runtime` instance owns one query. It starts the sandbox, seeds the
workspace, calls the root model, sends generated code to the passive runner,
services RPCs from that code, appends compact observations, checks the budget,
and returns a grounded answer or a labelled partial answer.

Invariants
----------
- The host owns the API key, DB connection, budget, ledger, and trajectory.
- The sandbox holds no key and no DB driver; every effect is an RPC here.
- **The sandbox workspace is ``<run_dir>/work``** — the only directory mounted
  into the container. ``trajectory.jsonl``, ``tasks.json``, ``manifest.json`` and
  ``answer.md`` live one level up in ``<run_dir>`` and are *never* visible to the
  RLM, so a generated block cannot rewrite its own evidence.
- Large data stays in sandbox variables or ``emit()`` files. Only **printed
  digests** enter the root context (the tool result itself never does).
- Depth 1: ``sub_call`` is a plain GPT-5-mini call with no tools and no REPL.
- Two cost sources: :class:`~lens.agent.budget.Budget` enforces the caps, and
  :meth:`lens.observability.trajectory.Trajectory.totals` is what the reported
  ``$/query`` is read from.

Durability
------------------------
The **log is the session**: every turn logs a root ``chat`` event (its
``prompt_ref`` holds the exact request body), an ``exec`` event (code + stdout
blobs), and a ``budget`` snapshot, so the root context and the counters can be
rebuilt by :meth:`Trajectory.rebuild` instead of being duplicated into a
checkpoint. What survives outside the log:

- ``manifest.json`` — identity + status (:mod:`lens.agent.runstore`), written at
  start and on every terminal transition;
- ``resume.json`` — a **cache** of the emit map + workspace hashes;
- ``work/`` — context files and ``emits/`` on disk.

Resume therefore means *rebuild*, never *restore a container*: fresh sandbox,
context files re-read, variables restored from ``emit`` blobs only, and a
``run.resume`` note appended (history is never rewritten).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from lens import config
from lens.agent.budget import Budget
from lens.agent.context import (
    budget_usage_line,
    budget_warning,
    build_messages,
    build_system_prompt,
    compact_messages,
    count_tokens,
    extract_final_answer,
    extract_python_block,
    tool_bindings,
    truncate_observation,
    write_context_files,
)
from lens.agent.ledger import VALID_STATUSES, TaskLedger
from lens.agent.runstore import (
    RunManifest,
    read_manifest,
    update_manifest,
    write_manifest,
    write_resume_cache,
)
from lens.agent.sandbox_session import SandboxSession
from lens.agent.segmentation import segment_query, write_query_json
from lens.agent.tool_registry import ToolRegistry, args_hash
from lens.clients import LensClient
from lens.experiments.arm import arm_fingerprint
from lens.observability.citations import cited_ids
from lens.observability.trajectory import TraceCtx, Trajectory
from lens.tools.evaluate import GATED_STAGES, evaluate_stage

#: How much cited chunk text the final gate is shown: how many chunks to fetch,
#: how many characters of each, and the total. ``EVIDENCE_FETCH_CHARS`` is larger
#: than ``evaluate.EVIDENCE_MAX_CHARS`` on purpose — the formatting budget is the
#: binding one, so the fetch never silently cuts a chunk the format would have
#: carried.
MAX_EVIDENCE_CHUNKS = 64
EVIDENCE_CHUNK_CHARS = 2000
EVIDENCE_FETCH_CHARS = 120000


def _error_envelope(kind: str, message: str) -> dict[str, Any]:
    """Return the RPC failure shape the sandbox façade raises on."""
    return {"ok": False, "error": {"type": kind, "message": message[:300]}}


def _sha256_text(text: str) -> str:
    """Return the hex sha256 of ``text`` (workspace-file fingerprints)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class RunResult:
    """Outcome of one query, also the handoff to /.

        ``cost_usd`` is the **trajectory** total (all chat + JEV calls, cached-token
        pricing included), not the budget counter: the transport is the only place that
    sees every call.
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
        resume: bool = False,
        parent_run_id: str | None = None,
        replay: bool = True,
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
        self.ledger = (
            ledger if ledger is not None else TaskLedger(trajectory.run_dir / "tasks.json")
        )
        self.recorded = recorded
        # : continue an existing run (``--resume``) vs start a new one.
        self.resume = resume
        self.parent_run_id = parent_run_id
        # Recorded replay: serve a tool result from the log when
        # ``(tool, args_hash)`` was already recorded, so a resumed or replayed run
        # is deterministic even if the corpus changed. Every replayed envelope
        # carries ``"replayed": True``; nothing is faked silently.
        self.replay = replay
        # ``None`` = "not built yet": the index is folded from the log on first
        # use, and reused afterwards (``_restore_state`` seeds it on resume).
        self._replayed: dict[str, str] | None = None
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
        self._manifest: RunManifest | None = None
        # Stage gate bookkeeping: one gate per stage per run (see
        # ``tools.evaluate.GATED_STAGES``).
        self._gated_stages: set[str] = set()
        self._gate_verdicts: dict[str, dict[str, Any]] = {}
        # Namespace map (label -> emit ref) and its completeness claim, kept in
        # memory so ``_save_checkpoint`` does not re-fold the log every turn.
        self._emits: dict[str, str] = {}
        self._namespace_complete: bool = True
        # Where a resume picked up (for the ``run.resume`` note).
        self._from_seq: int = 0

    # ------------------------------------------------------------------- run
    def run(self) -> RunResult:
        """Execute one query end-to-end (or continue one, when ``resume``).

        Sequence:
          1. ``_open_manifest`` -> ``manifest.json`` (status ``running``; a resumed
             run keeps its existing ``created_at`` and ``parent_run_id``).
          2. If ``resume``: ``_restore_state`` (log -> messages/budget/emits, see
             :meth:`Trajectory.rebuild`), then append a ``run.resume`` note and
             continue at ``last_seq``. Otherwise: ``_segment_query`` ->
             ``<workspace>/query.json`` and ``_register_runtime_tools``.
          3. Start the sandbox, ``_seed_workspace`` (context files + a first block
             defining ``query`` / ``RUN_ID`` / ``WORKSPACE``).
          4. Loop: build context (compacting; appending the wrap-up nudge when a
             cap is near), call the root with ``max_tokens`` =
             ``ROOT_MAX_OUTPUT_TOKENS``, execute code, service RPCs, truncate the
             observation, log ``exec`` + ``budget`` events, ``_save_checkpoint``,
             check the budget — until ``FINAL_ANSWER:`` or a cap.
          5. ``_finalize``: ``answer.md`` + manifest status + ``RunResult`` (cost
             from ``trajectory.totals()``), sandbox stopped **even on failure**.
        """
        self._open_manifest()

        if self.resume:
            # Host-only work: the fold, the budget counters, the query.json guard.
            # The namespace is restored in ``_seed_workspace`` instead, because it
            # needs a running container.
            self._restore_state()
        elif config.RUN_MODE != "document":
            # A document run has no corpus to retrieve from, so segmentation has
            # nothing to feed: the question is the whole input.
            self._segment_query()

        self._register_runtime_tools()

        answer, partial, stop_reason = "", True, "error"

        try:
            self.sandbox.start()
            self._seed_workspace()
            answer, partial, stop_reason = self._run_loop()
        except Exception as exc:  # noqa: BLE001 - a crash must still finalize
            # Label it, never leave the manifest ``running`` and never lose the
            # partial answer: the run is evidence even when it broke.
            self._note(f"run failed: {type(exc).__name__}: {exc}")
            answer = answer or f"Run failed before an answer was produced: {exc}"
            partial, stop_reason = True, "error"

        return self._finalize(answer, partial=partial, stop_reason=stop_reason)

    # ------------------------------------------------------- flat baseline
    def run_flat(self) -> RunResult:
        """Run the flat-prompt baseline: one retrieval, one model turn, no loop.

        This is the orchestration-off arm, not a degraded run. It uses the same
        query segmentation and the same retrieval funnel as the RLM arm — so the
        evidence set is identical — then answers in a single chat call with the
        chunks inline. The funnel dispatches through the tool registry, so the
        trajectory records the same hydrating payload and citations resolve
        against it.

        ``stop_reason`` is ``"flat"``: no cap bound the run, and reporting
        ``"iterations"`` would make the baseline look like a loop that ran out.
        """
        self._open_manifest()
        answer = ""

        try:
            self._segment_query()
            chunks, _dropped = self._flat_evidence()
            answer = self._flat_answer(chunks)
        except Exception as exc:  # noqa: BLE001 - a crash must still finalize
            self._note(f"run failed: {type(exc).__name__}: {exc}")
            return self._finalize(
                answer or f"Run failed before an answer was produced: {exc}",
                partial=True,
                stop_reason="error",
            )

        return self._finalize(answer, partial=False, stop_reason="flat")

    def _dispatch_flat(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch one funnel tool for the flat arm and return its result."""
        outcome = self.registry.dispatch(
            name,
            args,
            trace_ctx=TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="tool"),
            trajectory=self.trajectory,
        )

        if not outcome.ok:
            error = outcome.error or {}
            raise RuntimeError(f"{name} failed: {error.get('message') or error}")

        return outcome.result

    def _flat_evidence(self) -> tuple[list[dict[str, Any]], int]:
        """Return ``(chunks, dropped)`` for the flat prompt.

        The same four funnel calls the RLM arm runs, in the same order, so the
        baseline is handed the same evidence. A chunk that would push the prompt
        past the context limit is dropped **and counted** — never silently cut.
        """
        buckets = self._query_buckets or {"query": self.query}

        ids = self._dispatch_flat("shallow_match", {"buckets": buckets})
        papers = self._dispatch_flat("rerank_papers", {"arxiv_ids": ids, "buckets": buckets})
        arxiv_ids = [row["arxiv_id"] for row in papers if row.get("arxiv_id")]
        chunk_ids = self._dispatch_flat("deep_match", {"arxiv_ids": arxiv_ids, "buckets": buckets})
        bundle = self._dispatch_flat("rerank_chunks", {"chunk_ids": chunk_ids, "buckets": buckets})

        chunks: list[dict[str, Any]] = []
        dropped = 0
        budget = config.CONTEXT_TOKEN_LIMIT

        for row in bundle.get("chunks") or []:
            cost = count_tokens(str(row.get("chunk_text") or "")) + 8

            if cost > budget:
                dropped += 1
                continue

            budget -= cost
            chunks.append(row)

        self._note(f"flat evidence: {len(chunks)} chunks, {dropped} dropped for size")

        return chunks, dropped

    def _flat_answer(self, chunks: list[dict[str, Any]]) -> str:
        """One root call over the inline evidence; return the completion text."""
        system_prompt = build_system_prompt(
            self.registry.manifest(),
            {},
            self.budget.snapshot()["limits"],
            variant="naive",
        )
        evidence = "\n\n".join(
            f"[{row.get('chunk_uuid')}] {row.get('chunk_text') or ''}" for row in chunks
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Evidence ({len(chunks)} items):\n\n{evidence}\n\nQuestion: {self.query}"
                ),
            },
        ]
        result = self.client.chat(
            "root",
            messages,
            trace_ctx=TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="root"),
        )
        self.budget.record_iteration()
        self.budget.record_chat(result)

        return result.text

    # ------------------------------------------------------------------ RPC
    def handle_rpc(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Route one sandbox RPC to the host implementation.

        ``endpoint`` is one of ``/tool``, ``/sub_call``, ``/budget``, ``/emit``,
        ``/note``. Return ``{"ok": True, "result": ...}`` or
        ``{"ok": False, "error": {...}}``. Never include credentials.

        ``/tool`` returns the **full** tool result: that is the point of the
        design — the payload lands in a sandbox variable and never in the root
        context. Only the code's printed digest becomes an observation. A stage
        gate verdict, when produced, rides along as ``"gate"`` on the same
        envelope; a recorded replay carries ``"replayed": True``.

        This is **not** an HTTP call: ``ToolServer`` already owns the transport and
        calls this method with a parsed payload. A request from here would call the
        server that is currently inside ``handle_request``.
        """
        trace_ctx = TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="tool")

        try:
            if endpoint == "/tool":
                name = payload.get("name")

                if not isinstance(name, str) or not name:
                    return _error_envelope("BadRequest", "/tool needs a tool name")

                args = payload.get("args") or {}

                if not isinstance(args, Mapping):
                    return _error_envelope("BadRequest", "args must be an object")

                return self._dispatch_tool(name, args, trace_ctx=trace_ctx)

            if endpoint == "/sub_call":
                prompt = payload.get("prompt")

                if not isinstance(prompt, str) or not prompt:
                    return _error_envelope("BadRequest", "/sub_call needs a prompt")

                return {"ok": True, "result": self._sub_call(prompt, payload.get("context"))}

            if endpoint == "/budget":
                return {"ok": True, "result": self.budget.snapshot()}

            if endpoint == "/emit":
                return {
                    "ok": True,
                    "result": self._emit(payload.get("value"), payload.get("label")),
                }

            if endpoint == "/note":
                text = payload.get("text")

                if not isinstance(text, str):
                    return _error_envelope("BadRequest", "/note needs text")

                self._note(text)
                return {"ok": True, "result": None}

        except Exception as exc:  # noqa: BLE001 - the model must read the failure
            return _error_envelope(type(exc).__name__, str(exc))

        # The HTTP layer 404s unknown paths, so reaching this is a wiring bug.
        return _error_envelope("NotFound", f"unknown endpoint {endpoint!r}")

    # --------------------------------------------------------------- internal
    def _open_manifest(self) -> RunManifest:
        """Create or load ``<run_dir>/manifest.json`` and keep it on the runtime.

        New run: ``RunManifest.new(...)`` with the caps in force, the model roles
        from ``config.MODEL_REGISTRY`` (+ ``config.JEV_MODEL_ID``), the provenance
        dict (commit / image digest / python / ``recorded``) and the artifact
        names. Resumed run: keep the existing ``created_at`` and
        ``parent_run_id``, set ``status="running"``.

        The manifest is *identity*, never state: no messages, no counters. Those
        are folded from the log.
        """
        manifest: RunManifest | None = None

        if (self.run_dir / config.MANIFEST_FILENAME).is_file():
            manifest = read_manifest(self.run_dir)

        if manifest is None:
            manifest = RunManifest.new(
                self.run_id,
                self.query,
                parent_run_id=self.parent_run_id,
                # The registry maps a role to a ModelSpec (a dataclass, so not
                # JSON-serialisable): the manifest records the **ids** only.
                # JEV is deliberately outside the registry.
                models={role: spec.id for role, spec in config.MODEL_REGISTRY.items()}
                | {"jev": config.JEV_MODEL_ID},
                caps=self.budget.snapshot()["limits"],
                created_by=self._provenance(),
            )
            # The arm is written once, at creation, and deliberately **not**
            # updatable (``UPDATABLE`` excludes ``extra``): a resumed run keeps the
            # arm it started with, which is what makes an old run attributable.
            manifest.extra["arm"] = arm_fingerprint(recorded=self.recorded, replay=self.replay)
        else:
            # Resume: identity and lineage survive, status does not.
            manifest.status = "running"

        write_manifest(self.run_dir, manifest)
        self._manifest = manifest

        return manifest

    def _provenance(self) -> dict[str, Any]:
        """Return the reproducibility stamp for the manifest.

        ``commit`` comes from ``LENS_COMMIT`` when the launcher knows it and is
        ``None`` otherwise — never ``0``: a provenance field that looks like a
        real short hash but is a placeholder is worse than an explicit null.
        """
        return {
            "commit": os.environ.get("LENS_COMMIT") or None,
            "image": config.SANDBOX_IMAGE,
            # The tag alone is not reproducible (it is rebuilt in place); the digest
            # is what wants. The launcher resolves it (`docker inspect`) and
            # passes it in, so this stays a cheap env read with an honest ``None``.
            "image_digest": os.environ.get("LENS_SANDBOX_DIGEST") or None,
            "python": platform.python_version(),
            "recorded": self.recorded,
        }

    def _restore_state(self) -> None:
        """Rebuild this run's in-memory state from the log, for ``resume``.

        Use :meth:`Trajectory.rebuild` and then:
          1. ``self._messages`` = the folded messages; ``self._step_idx`` =
             ``last_seq``; restore the budget counters from the folded ``budget``
             snapshot (the caps stay today's — the manifest records the originals
             for reporting, but the resume must not silently raise them).
          2. Re-write the context files into ``work/`` (they are already on disk;
             rewriting makes a moved/copied run directory work). ``query.json`` is
             **read from disk** — re-segmenting would spend a call and could
             produce different buckets, which would silently change the retrieval
             the run is built on. Re-segment only if the file is missing, and say
             so in the resume note.
          3. Restore the namespace from ``emits``: one block that reads each
             ``ref`` back into a variable named after the emit label. Variables
             that were never emitted are *gone* — set ``namespace_complete``
             accordingly and have the block print what is missing so the root
             knows to re-derive it.
          4. Append one ``note`` event (``action="run.resume"``) with
             ``extra={"from_seq": ..., "namespace_complete": ...}``. Never rewrite
             the existing history.

        Steps 2's context files and 3's namespace block are executed by
        :meth:`_seed_workspace`, which runs **after** the sandbox starts; this
        method is host-only on purpose (it must be callable before a container
        exists).
        """
        rebuilt = self.trajectory.rebuild()

        self._messages = rebuilt.messages
        self._step_idx = rebuilt.last_seq
        self._from_seq = rebuilt.last_seq
        self._emits = dict(rebuilt.emits)
        self._replayed = dict(rebuilt.tool_results)

        # query.json is an input: read it, never re-segment (new buckets would
        # silently change the retrieval the run was built on).
        query_path = self.workspace / "query.json"

        if query_path.is_file():
            try:
                self._query_buckets = json.loads(query_path.read_text(encoding="utf-8"))
            except ValueError:
                self._query_buckets = {}

        used = (rebuilt.budget or {}).get("used") or {}

        for name in (
            "iterations",
            "tokens_in",
            "tokens_out",
            "cost_usd",
            "exec_seconds",
            "backtracks",
        ):
            if name in used:
                setattr(self.budget, name, used[name])

        # ``query.json`` is an input, not a derivation: re-segmenting would spend a
        # call and could produce different buckets, silently changing retrieval the
        # run is already built on. A missing file is noted, never re-derived here.
        if not (self.workspace / "query.json").is_file():
            self._note(
                "query.json missing on resume; continuing without buckets "
                "(re-segmenting would change the retrieval this run was built on)"
            )

    def _save_checkpoint(self) -> None:
        """Persist the small resume cache after a turn (``resume.json``).

        Write ``namespace`` (emit label -> ref), ``workspace`` (file -> sha256) and
        ``namespace_complete`` via :func:`lens.agent.runstore.write_resume_cache`,
        then update the manifest (``iterations``, ``last_seq``, ``totals``). This
        is a **cache of the log**, so a failure here must degrade to a rebuild,
        not abort the run.
        """
        workspace_hashes: dict[str, str] = {}

        for path in sorted(self.workspace.rglob("*")):
            if path.is_file():
                try:
                    workspace_hashes[path.relative_to(self.workspace).as_posix()] = _sha256_text(
                        path.read_text(encoding="utf-8", errors="replace")
                    )
                except OSError:
                    continue

        try:
            write_resume_cache(
                self.run_dir,
                namespace=dict(self._emits),
                workspace=workspace_hashes,
                namespace_complete=self._namespace_complete,
            )
        except OSError:
            # The cache is an optimization; the log is the source of truth.
            return

        update_manifest(
            self.run_dir,
            iterations=self.budget.iterations,
            last_seq=self._step_idx,
            totals=self.trajectory.totals(),
        )

    def _replay_recorded(self, name: str, args: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return a recorded tool result for ``(name, args_hash)``, or ``None``.

        Build the index **lazily from the log on first use** (and reuse the one
        ``_restore_state`` already built on resume): ``Trajectory.rebuild()``
        gives ``tool_results``. Used by ``_dispatch_tool`` when ``self.replay`` is
        on: a replay must reproduce the evidence byte-for-byte, so the corpus is
        not queried twice. The returned envelope is the recorded one plus
        ``"replayed": True``; a missing blob is ``None`` (fall through to a live
        call) rather than an error.
        """
        if not self.replay:
            return None

        if self._replayed is None:
            self._replayed = dict(self.trajectory.rebuild().tool_results)

        ref = self._replayed.get(f"{name}:{args_hash(args)}")

        if ref is None:
            return None

        try:
            result = json.loads(self.trajectory.load_blob(ref))
        except (OSError, KeyError, ValueError):
            # A dangling ref is not a failure: fall through to a live call.
            return None

        return {"ok": True, "result": result, "result_ref": ref, "replayed": True}

    def _register_runtime_tools(self) -> None:
        """Register the host-runtime operations the RLM is allowed to call.

        : these are **not** sandbox-local functions — they touch host
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

        def ledger_create(requirement: str, parent_id: str | None = None) -> dict[str, Any]:
            """Create one ledger task for a requirement; returns {"task_id": "<id>"}."""
            task = self.ledger.create(requirement, parent_id=parent_id)
            return {"task_id": task.task_id}

        def ledger_update(task_id: str, status: str) -> dict[str, Any]:
            """Set a task's status (pending / in_progress / done / failed); returns the task."""
            if status not in VALID_STATUSES:
                raise ValueError(
                    f"invalid status {status!r}; expected one of {sorted(VALID_STATUSES)}"
                )
            return self.ledger.update(task_id, status=status).to_dict()

        def ledger_add_evidence(task_id: str, refs: list[str]) -> dict[str, Any]:
            """Attach evidence refs (chunk_uuid or sha256 refs) to a task."""
            return self.ledger.add_evidence(task_id, list(refs)).to_dict()

        def backtrack(stage: str, reason: str) -> dict[str, Any]:
            """Record one backtrack for a stage; refused past MAX_BACKTRACKS."""
            if self.budget.backtracks >= config.MAX_BACKTRACKS:
                raise RuntimeError(
                    f"backtrack budget exhausted ({config.MAX_BACKTRACKS}); "
                    "answer with what you have and emit FINAL_ANSWER:"
                )

            self.budget.record_backtrack()
            # The stage's task, when the model made one, records why it went back.
            for task in self.ledger.list():
                if stage in task.requirement and task.status != "done":
                    self.ledger.update(task.task_id, status="failed", attempts=task.attempts + 1)
                    self.ledger.add_evidence(task.task_id, [])
                    break

            self._note(f"backtrack on {stage}: {reason}")
            return {"backtracks_left": config.MAX_BACKTRACKS - self.budget.backtracks}

        for name, func in (
            ("ledger_create", ledger_create),
            ("ledger_update", ledger_update),
            ("ledger_add_evidence", ledger_add_evidence),
            ("backtrack", backtrack),
        ):
            if name not in self.registry.names():
                self.registry.register(name, func)

    def _segment_query(self) -> dict[str, list[str]]:
        """Segment the query with and write ``<workspace>/query.json``.

        Use :func:`lens.agent.segmentation.segment_query` then
        :func:`lens.agent.segmentation.write_query_json` — owns the file's
        format, and the workspace copy **is** the artifact (it sits under
        ``<run_dir>/work``). Do not write a second copy into ``run_dir``.
        """
        trace_ctx = TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="root")
        buckets = segment_query(self.client, self.query, trace_ctx=trace_ctx)

        write_query_json(buckets, self.workspace / "query.json")
        self._query_buckets = buckets

        return buckets

    def _seed_workspace(self) -> None:
        """Write context files and seed the sandbox namespace.

        Context files via :func:`lens.agent.context.write_context_files`
        (``AGENTS.md``, ``tools.md``, ``budget.md``, ``ontology.md`` when the KG is
        on) — **not** ``query.json``, which already wrote here.

        Then run one block that defines at least ``query``, ``RUN_ID`` and
        ``WORKSPACE`` so later code does not re-read ``query.json`` on every turn.

        On resume the same block also restores the emitted namespace, and the
        ``run.resume`` note is appended here (after the restore) so it can carry
        the real ``namespace_complete``.
        """
        write_context_files(
            self.workspace,
            tool_manifest=self.registry.manifest(),
            budget_snapshot=self.budget.snapshot(),
            ontology_summary=self._ontology_summary,
        )

        lines = [
            "import json",
            f"query = {self.query!r}",
            f"RUN_ID = {self.run_id!r}",
            f"WORKSPACE = {config.SANDBOX_WORKSPACE_PATH!r}",
        ]

        if self._query_buckets:
            # The segmented buckets as a ready variable: the funnel tools want
            # exactly this mapping, and a model that has to guess the shape hands
            # over ``{"query": "..."}`` instead (which the tools now tolerate, but
            # the real segmentation is better evidence than a re-typed query).
            lines.append(
                "buckets = json.loads("
                + repr(json.dumps(self._query_buckets, ensure_ascii=True))
                + ")"
            )

        lines.append(
            # Bind every host tool as a plain function (the form the manifest
            # prints) — see context.tool_bindings.
            tool_bindings(self.registry.parameters())
        )
        missing: list[str] = []

        if self.resume:
            emit_dir = config.SANDBOX_WORKSPACE_PATH.rstrip("/") + "/emits"

            for label, _ref in sorted(self._emits.items()):
                if (self.workspace / "emits" / f"{label}.json").is_file():
                    lines.append(f"{label} = json.load(open({emit_dir}/{label}.json))")
                else:
                    missing.append(label)

            lines.append(f"print('namespace restored; missing: {missing}')")

        # Not logged as an exec event: this is setup, not a turn. A logged event
        # would inject a user message into the folded context before any assistant
        # turn, which the resume fold reads as a turn that never happened.
        result = self.sandbox.exec("\n".join(lines))

        if not result.ok:
            raise RuntimeError(f"seeding the sandbox failed: {result.error}")

        self._namespace_complete = not missing

        if self.resume:
            self.trajectory.log_event(
                trace_ctx=TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="tool"),
                call_type="note",
                action="run.resume",
                extra={"from_seq": self._from_seq, "namespace_complete": self._namespace_complete},
            )

    def _build_context(self) -> list[dict[str, Any]]:
        """Assemble the root message list, compacting when over the threshold.

        Compaction must keep: the system message (it holds the
        ``FINAL_ANSWER:`` contract), a one-line note that the sandbox namespace
        persists across turns, and the most recent turns. The namespace is the
        state — dropping the *code* that built a variable must not make the model
        believe the variable is gone.

        Also append :func:`lens.agent.context.budget_usage_line` (the live counters)
        and :func:`lens.agent.context.budget_warning` when it returns text (the
        wrap-up nudge). Both ride at the **end**, never inside the system prompt:
        they change every turn, and a changing system prompt would invalidate the
        provider's prefix cache for the whole conversation.
        """
        system_prompt = build_system_prompt(
            self.registry.manifest(),
            self._query_buckets,
            self.budget.snapshot()["limits"],
            question=self.query,
        )
        messages = build_messages(system_prompt=system_prompt, memory=self._messages)
        messages = compact_messages(messages)

        # The tail, always: the usage line (it changes every turn, which is why it
        # must not live in the system prompt) and, near a cap, the wrap-up nudge.
        # Both are appended rather than merged into history, so the prompt up to
        # this point is the previous turn's prompt plus what that turn produced —
        # exactly what a provider can cache.
        tail = [{"role": "user", "content": budget_usage_line(self.budget.snapshot())}]
        warning = budget_warning(self.budget.snapshot())

        if warning:
            tail.append({"role": "user", "content": warning})

        return [*messages, *tail]

    def _run_loop(self) -> tuple[str, bool, str | None]:
        """Return ``(answer, partial, stop_reason)``.

        Per turn: call the root (``max_tokens=config.ROOT_MAX_OUTPUT_TOKENS``,
        record the chat in the budget), extract one ```python block or the
        ``FINAL_ANSWER:`` text (**code wins**: a turn containing both is still
        working, and the sentinel may just be quoted), send code to the sandbox,
        truncate the printed observation with
        :func:`lens.agent.context.truncate_observation`, append it as a ``user``
        message, then log one ``budget`` event with ``self.budget.snapshot()``
         and check
        :meth:`Budget.can_continue`.

        **Final gate** (the second half of the double gate): on ``FINAL_ANSWER``
        call ``evaluate_stage("final", {"score": None, "summary": {"answer":
        …, "citations": [...]}})``. A ``fail`` hands the ``failing_criterion``
        back as one more turn **inside the iteration cap**; ``None`` (gate off)
        or ``pass``/``uncertain`` accepts the answer. If the cap is reached first,
        return the answer as ``partial`` with ``stop_reason="iterations"`` — a
        labelled partial answer, never a silently accepted fabrication.
        """
        answer = ""

        while self.budget.can_continue():
            trace_ctx = TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="root")
            result = self.client.chat(
                "root",
                self._build_context(),
                trace_ctx=trace_ctx,
                max_tokens=config.ROOT_MAX_OUTPUT_TOKENS,
            )
            self.budget.record_chat(result)
            self.budget.record_iteration()

            completion = result.text or ""
            self._messages.append({"role": "assistant", "content": completion})

            code = extract_python_block(completion)
            # Fence-aware: the sentinel counts only outside a code block, so a
            # turn may run one last block *and* answer.
            final = extract_final_answer(completion)

            if code is not None:
                observation = self._execute_block(code)
                self._messages.append({"role": "user", "content": observation})

                if final is not None:
                    # The answer was written in the same message as the block, so
                    # it cannot have used that block's output: a model that
                    # aggregates in code then answers blind reports the *names* of
                    # its variables. Run the block, then ask for the answer again.
                    # The blind text is kept as ``answer`` so a cap on the next turn
                    # still leaves something labelled rather than nothing.
                    answer = final
                    self._messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The block above ran. If its output answers the question, "
                                "re-emit that answer now as FINAL_ANSWER: <the values, not "
                                "the variable names>. Otherwise continue."
                            ),
                        }
                    )
                    self._after_turn()
                    continue

            if final is None:
                if code is None:
                    self._messages.append(
                        {
                            "role": "user",
                            "content": (
                                "No ```python block and no FINAL_ANSWER: marker in that turn. "
                                "Return executable code, or finish with FINAL_ANSWER: <answer>."
                            ),
                        }
                    )

                self._after_turn()
                continue

            citations = cited_ids(final)

            if config.RUN_MODE == "document":
                # Document mode has no corpus behind it: the cited ids are record
                # ids in a supplied file, so there is no chunk text to verify and
                # the gate can only answer "unsupported". Skipping is logged, not
                # silent — a correct aggregate must not be rejected eight times
                # because the evidence source does not exist in this mode.
                self._note(
                    "final gate skipped: document mode has no corpus evidence to verify citations"
                )
                verdict = None
            else:
                cited_chunks = [citation for citation in citations if "-" in citation]
                evidence, evidence_truncated = self._citation_evidence(citations, answer=final)
                verdict = self._evaluate(
                    "final",
                    {
                        "score": None,
                        "summary": {
                            "answer": final,
                            # Real ids: the gate was previously handed a hard-coded
                            # empty list, so "is every claim supported by the cited
                            # chunks" was asked about citations it never saw.
                            "citations": citations,
                            # And the *text* of those chunks: an id alone cannot be
                            # verified, so every gate call came back "unsupported" and a
                            # correct answer was rejected until the iteration cap.
                            "evidence": evidence,
                            # True when the evidence budget bound before every cited
                            # chunk was shown, so a fail is not read as a bad answer.
                            "evidence_truncated": evidence_truncated,
                            "evidence_dropped": max(0, len(cited_chunks) - len(evidence)),
                        },
                    },
                    trace_ctx=trace_ctx,
                )

            answer = final

            if verdict is not None and verdict.verdict == "fail":
                self._messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The final gate rejected the answer: "
                            f"{verdict.failing_criterion or 'unsupported claim'}. "
                            "Revise it and emit FINAL_ANSWER: again."
                        ),
                    }
                )
                self._after_turn()
                continue

            self._after_turn()

            return (answer, False, None)

        # The cap bit: the answer is labelled, never silently accepted.
        return (answer, True, self.budget.stop_reason() or "iterations")

    def _after_turn(self) -> None:
        """Advance the step, log the budget snapshot, and persist the cache."""
        self._step_idx += 1
        self._log_budget()
        self._save_checkpoint()

    def _log_budget(self) -> None:
        """Log one ``budget`` snapshot."""
        self.trajectory.log_event(
            trace_ctx=TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="tool"),
            call_type="note",
            action="budget.snapshot",
            extra={"budget": self.budget.snapshot()},
        )

    def _execute_block(self, code: str) -> str:
        """Run one block in the sandbox, log the ``exec`` event, return the digest.

        The returned text is exactly what the model sees **and** what the fold
        reads back from ``stdout_ref`` — the two must not diverge, or a resumed
        run would remember a turn differently from the run that produced it.
        """
        trace_ctx = TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="tool")
        result = self.sandbox.exec(code)

        if result.duration_ms:
            self.budget.record_exec(result.duration_ms / 1000)

        chunks = [result.stdout or ""]

        if result.stderr and not result.ok:
            # The runner puts the traceback in stderr: the model needs it to fix
            # the block, and the fold needs the same text.
            chunks.append(result.stderr)

        observation = truncate_observation("\n".join(part for part in chunks if part).strip())

        self.trajectory.log_event(
            trace_ctx=trace_ctx,
            call_type="exec",
            action="exec.block",
            code_ref=self.trajectory.store_blob(code),
            stdout_ref=self.trajectory.store_blob(observation),
            error=result.error,
            extra={"budget": self.budget.snapshot()},
        )

        return observation

    def _evaluate(
        self,
        stage: str,
        artifact: Mapping[str, Any],
        *,
        trace_ctx: TraceCtx,
    ) -> Any:
        """Call the stage gate, recording its cost against the budget.

        The decision layer is cheap but not free, so the call is accounted for
        (H₃). A gate failure is **not** caught here: ``evaluate_stage`` already
        degrades a dead decision layer to ``uncertain``, and anything else is a
        harness bug that should surface.
        """
        return evaluate_stage(
            stage,
            artifact,
            client=self.client,
            trace_ctx=trace_ctx,
            budget=self.budget,
            query=self.query,
        )

    def _dispatch_tool(
        self,
        name: str,
        args: Mapping[str, Any],
        *,
        trace_ctx: TraceCtx,
    ) -> dict[str, Any]:
        """Dispatch one host tool and return the **full** RPC envelope.

        Order: (1) if ``self.replay`` and :meth:`_replay_recorded` has this
        ``(name, args_hash)``, return the recorded envelope (evidence is
        reproduced, not re-queried); (2) otherwise call
        ``self.registry.dispatch(name, args, trace_ctx=trace_ctx,
        trajectory=self.trajectory)`` and return ``result.to_dict()`` (the
        registry already stored the blob and logged the event).

        **Stage gate.** If ``name`` is in ``tools.evaluate.GATED_STAGES`` and that
        stage has not been gated yet in this run, build the bounded gate artifact
        — ``{"score": <calibrated score or None>, "summary": <small dict>}``, never
        the raw payload (``build_kg``'s graph is large, and the gate only needs a
        summary of it) — and call :func:`lens.tools.evaluate.evaluate_stage`. If it returns a
        verdict, add ``"gate": verdict.to_dict()`` to the envelope; if it returns
        ``None`` (``GATE_ENABLED=0``, the ablation arm), add **nothing** — an arm
        with no gate must not look like a gate that always passes.

        Only the **stage-completing** tools are mapped (``rerank_chunks``,
        ``build_kg``). Earlier tools of the same stage return ids only, so gating
        them would ask JEV whether a list of ``arxiv_id``s is "sufficient".
        """
        recorded = self._replay_recorded(name, args)

        if recorded is not None:
            return recorded

        result = self.registry.dispatch(name, args, trace_ctx=trace_ctx, trajectory=self.trajectory)
        envelope = result.to_dict()

        stage = GATED_STAGES.get(name)

        if stage is not None and stage not in self._gated_stages and envelope.get("ok"):
            self._gated_stages.add(stage)  # one gate per stage per run
            summary = (
                _retrieval_gate_summary(result.result)
                if stage == "retrieval"
                else _bounded_artifact(result.result)["summary"]
            )
            verdict = self._evaluate(
                stage, {"score": None, "summary": summary}, trace_ctx=trace_ctx
            )

            if verdict is not None:
                envelope["gate"] = verdict.to_dict()
                self._gate_verdicts[stage] = envelope["gate"]

        return envelope

    def _sub_call(self, prompt: str, context: str | None = None) -> str:
        """Depth-1 plain GPT-5-mini call. No tools, no REPL, no recursion.

        Record it in the budget (:meth:`Budget.record_chat`) — sub-calls are
        cheap but they are not free, and H₃ counts them.
        """
        messages: list[dict[str, str]] = []

        if context:
            messages.append({"role": "system", "content": context})

        messages.append({"role": "user", "content": prompt})

        trace_ctx = TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="sub")
        result = self.client.chat("sub", messages, trace_ctx=trace_ctx)
        self.budget.record_chat(result)

        return result.text

    def _citation_evidence(
        self, citations: list[str], *, answer: str = ""
    ) -> tuple[dict[str, str], bool]:
        """Return ``(chunk_uuid -> chunk_text, truncated)`` for the cited ids.

        The final gate is asked whether the citations support the claims; without
        the text it can only guess, and it guessed "unsupported" on a correct answer
        until the iteration cap. Host-side and read-only: no model call,
        no tool event — the root context stays untouched (this text never becomes an
        observation).

        **Every** cited chunk is fetched, not the first five, and at full length: an
        answer citing 37 chunks is the longest seen, about 66 000 characters. When
        the budget does bind, the chunks the answer cites **most often** are the
        ones kept — dropping them by fetch order would hide exactly the evidence
        the claim leans on. ``truncated`` is reported so a ``fail`` can be read as
        "insufficient input" rather than "bad answer".
        """
        ids = [citation for citation in citations if "-" in citation][:MAX_EVIDENCE_CHUNKS]

        if not ids or self.conn is None:
            return {}, bool(ids)

        # Most-cited first, first appearance as the tie-break.
        position = {chunk_id: index for index, chunk_id in enumerate(ids)}
        ranked = sorted(
            ids, key=lambda chunk_id: (-(answer or "").count(chunk_id), position[chunk_id])
        )

        from lens.tools.retrieval import fetch_chunks

        try:
            rows = fetch_chunks(self.conn, ranked)
        except Exception:  # noqa: BLE001 - a gate with no evidence still runs
            return {}, True

        by_id = {
            str(row["chunk_uuid"]): str(row.get("chunk_text") or "")
            for row in rows
            if isinstance(row, dict) and row.get("chunk_uuid")
        }

        evidence: dict[str, str] = {}
        used = 0
        truncated = False

        for chunk_id in ranked:
            body = by_id.get(chunk_id)

            if body is None:
                continue

            remaining = EVIDENCE_FETCH_CHARS - used

            if remaining <= 0:
                truncated = True
                break

            evidence[chunk_id] = body[: min(EVIDENCE_CHUNK_CHARS, remaining)]
            used += len(evidence[chunk_id])

        return evidence, truncated

    def _emit(self, value: Any, label: str | None = None) -> dict[str, Any]:
        """Offload a value to ``<workspace>/emits/`` and return its ref + paths.

        Return shape (the RLM's contract)::

            {"ref": "sha256:<hex>", "path": "/workspace/emits/<name>",
             "host_path": "D:\\...", "bytes": <int>}

        Path mapping rule for the whole harness: host ``self.workspace`` <->
        container ``config.SANDBOX_WORKSPACE_PATH``. **``path`` must be the
        container form** — a host path is unopenable inside the sandbox — and the
        blob is stored through the trajectory so the artifact is auditable.

        **The label is the variable name** : ``emit(x, label="chunks")`` is
        how ``chunks`` survives a resume, because the restore block reads the blob
        back into a variable with that name. Also log one ``call_type="emit"``
        event with ``result_ref`` and ``extra={"var": label}`` — that event is how
        :meth:`Trajectory.rebuild` recovers the namespace map. An emit without a
        label is still stored, but it cannot be restored by name.

        The file and the blob are the **same bytes** (``json.dumps`` of the value),
        so the restore block's ``json.load`` reproduces the variable exactly.
        """
        text = json.dumps(value, ensure_ascii=False, default=str)
        ref = self.trajectory.store_blob(text)
        name = label or f"emit-{ref.split(':', 1)[1][:12]}"

        emits_dir = self.workspace / "emits"
        emits_dir.mkdir(parents=True, exist_ok=True)
        host_path = emits_dir / f"{name}.json"
        host_path.write_text(text, encoding="utf-8")

        self.trajectory.log_event(
            trace_ctx=TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="tool"),
            call_type="emit",
            action="emit.value",
            result_ref=ref,
            extra={"var": label} if label else None,
        )

        if label:
            self._emits[label] = ref

        return {
            "ref": ref,
            "path": f"{config.SANDBOX_WORKSPACE_PATH.rstrip('/')}/emits/{name}.json",
            "host_path": str(host_path),
            "bytes": len(text.encode("utf-8")),
        }

    def _note(self, text: str) -> None:
        """Append one free-form note event to the trajectory (``call_type="note"``)."""
        self.trajectory.log_event(
            trace_ctx=TraceCtx(trace_id=self._trace_id, step_idx=self._step_idx, actor="tool"),
            call_type="note",
            action="note",
            extra={"text": text},
        )

    def _finalize(self, answer: str, *, partial: bool, stop_reason: str | None) -> RunResult:
        """Write ``answer.md``, close the manifest, stop the sandbox, build the result.

        ``answer.md`` goes to ``run_dir`` (never the workspace). Cost comes from
        ``self.trajectory.totals()["cost_usd"]``. Manifest status is one of
        ``completed`` (a clean ``FINAL_ANSWER``), ``partial`` (a cap or the backtrack
        limit stopped it), or ``failed`` (an exception escaped the loop) — never
        left ``running``. Stop the sandbox in a ``finally``-safe path so a crash
        mid-loop does not leave a container behind.
        """
        answer_path = self.run_dir / "answer.md"
        answer_path.write_text(answer.rstrip() + "\n", encoding="utf-8")

        totals = self.trajectory.totals()

        if stop_reason == "error":
            status = "failed"
        else:
            status = "partial" if partial else "completed"

        artifacts = {
            "answer": "answer.md",
            "trajectory": "trajectory.jsonl",
            "tasks": "tasks.json",
            "manifest": config.MANIFEST_FILENAME,
            "resume": "resume.json",
            "query": "work/query.json",
        }

        try:
            update_manifest(
                self.run_dir,
                status=status,
                totals=totals,
                iterations=self.budget.iterations,
                last_seq=self._step_idx,
                artifacts=artifacts,
                stop_reason=stop_reason,
            )
        finally:
            # Never leave a container behind, even when the manifest write failed.
            try:
                self.sandbox.stop()
            except Exception:  # noqa: BLE001 - cleanup must not mask the outcome
                pass

        return RunResult(
            answer=answer,
            partial=partial,
            stop_reason=stop_reason,
            iterations=self.budget.iterations,
            cost_usd=float(totals.get("cost_usd") or 0.0),
            trajectory_path=self.trajectory.trajectory_path,
            tasks_path=Path(getattr(self.ledger, "path", self.run_dir / "tasks.json")),
            answer_path=answer_path,
        )


def _bounded_artifact(result: Any) -> dict[str, Any]:
    """Return ``{"score": None, "summary": {...}}`` for a **stage-completing** tool.

    The gate must never receive the raw payload: ``build_kg``'s node-link graph
    would drown the useful signal and make the call expensive for a reason that
    has nothing to do with the stage's quality.

    A tool that returns its own ``summary`` hands that over instead of a count:
    ``{"summary": {...}}`` used to become ``{"summary": 4}`` (the number of keys),
    which told the gate nothing about what the graph contains.
    """
    summary: dict[str, Any] = {}

    if isinstance(result, Mapping):
        own = result.get("summary")

        if isinstance(own, Mapping):
            summary.update(dict(list(own.items())[:20]))

        for key, value in list(result.items())[:12]:
            if key == "summary":
                continue

            if isinstance(value, (list, tuple, set, Mapping)):
                summary[key] = len(value)
            elif isinstance(value, str):
                summary[key] = value[:80]
            elif isinstance(value, (int, float, bool)) or value is None:
                summary[key] = value
    else:
        summary["type"] = type(result).__name__

    # ``score`` stays None: no calibrated threshold exists yet, and a made-up score
    # would make tier 1 decide on a number the cannot defend.
    return {"score": None, "summary": summary}


#: Chunks whose text the retrieval gate is shown, and how much of each.
GATE_SAMPLE_CHUNKS = 6
GATE_SAMPLE_CHARS = 300


def _retrieval_gate_summary(result: Any) -> dict[str, Any]:
    """Return a summary a *coverage* verdict can actually be made from.

    ``{"chunks": 100, "papers": 19}`` cannot answer "does this cover every
    objective the query names": counts carry no content. The gate gets the paper
    ids plus a few real excerpts, bounded, so a sufficiency judgement has evidence
    in front of it.
    """
    if not isinstance(result, Mapping):
        return {"type": type(result).__name__}

    chunks = result.get("chunks") or []
    papers = result.get("papers") or []
    paper_ids = [
        str(row.get("arxiv_id"))
        for row in papers
        if isinstance(row, Mapping) and row.get("arxiv_id")
    ][:8]
    samples = [
        {
            "id": str(row.get("chunk_uuid") or ""),
            "paper": str(row.get("arxiv_id") or ""),
            "section": str(row.get("section_type") or ""),
            "text": " ".join(str(row.get("chunk_text") or "").split())[:GATE_SAMPLE_CHARS],
        }
        for row in chunks[:GATE_SAMPLE_CHUNKS]
        if isinstance(row, Mapping)
    ]

    return {
        "chunks": len(chunks),
        "papers": len(papers),
        "paper_ids": paper_ids,
        "chunk_samples": samples,
    }
