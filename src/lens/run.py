"""CLI entry point for one LENS query, plus run listing/resume.

Usage::

    python -m lens.run "How do RLM systems aggregate long contexts?"
    python -m lens.run --list                     # saved runs, newest first
    python -m lens.run --show <run_id>            # read-only: manifest + answer
    python -m lens.run --resume <run_id>          # continue an interrupted run
    python -m lens.run --fork <run_id> --run-id new "same query, other arm"
    python -m lens.run --export <run_id> --destination D:\\demo   # portable bundle
    python -m lens.run --import D:\\demo\\<run_id>.lens           # install a bundle

``--list`` / ``--show`` / ``--export`` / ``--import`` are **read-only** (the last
two touch only the destination / runs root): they never build a runtime, call a
model, or start a container. That is what makes the UI and the laptop demo
cheap.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from lens import config
from lens.agent.runtime import Runtime
from lens.agent.sandbox_session import SandboxSession
from lens.agent.tool_registry import ToolRegistry
from lens.agent.tool_server import ToolServer
from lens.clients import LensClient
from lens.db.connect import connect
from lens.embeddings import Embedder
from lens.kg_extract import GlinerExtractor
from lens.observability.trajectory import Trajectory
from lens.reranking import Reranker
from lens.tools.kg import build_kg
from lens.tools.retrieval import (
    deep_match,
    fetch_chunks,
    fetch_papers,
    rerank_chunks,
    rerank_papers,
    shallow_match,
)
from lens.tools.sql import query_sql, sql_is_allowed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m lens.run",
        description="Run one LENS query end-to-end, or inspect saved runs.",
    )
    # Optional: --list / --show take no query.
    parser.add_argument("query", nargs="?", help="natural-language research query")
    parser.add_argument(
        "--run-id",
        default=None,
        help="run id / artifact directory name (default: generated timestamp)",
    )
    parser.add_argument(
        "--recorded",
        action=argparse.BooleanOptionalAction,
        default=config.LENS_RECORDED,
        help="disable dev-only escape hatches (T3 SQL) for a recorded run",
    )
    parser.add_argument(
        "--allow-sql",
        action=argparse.BooleanOptionalAction,
        default=config.ALLOW_SQL,
        help="allow the gated T3 query(sql) tool (dev only; ignored when recorded)",
    )
    # --- session management (no runtime for --list / --show) -------------
    parser.add_argument(
        "--list",
        action="store_true",
        help="list saved runs (id, status, date, query) and exit",
    )
    parser.add_argument(
        "--show",
        metavar="RUN_ID",
        default=None,
        help="print a saved run's manifest summary and answer, then exit",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="resume an interrupted run (rebuilds from the log; never rewrites it)",
    )
    parser.add_argument(
        "--fork",
        metavar="RUN_ID",
        default=None,
        help="start a new run that records --run-id as a child of RUN_ID",
    )
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="do not serve recorded tool results on resume (re-query the corpus)",
    )
    parser.add_argument(
        "--export",
        metavar="RUN_ID",
        default=None,
        help="write a portable <run_id>.lens bundle into --destination and exit",
    )
    parser.add_argument(
        "--import",
        dest="import_bundle",
        metavar="BUNDLE",
        default=None,
        help="install a .lens bundle into the runs root and exit (hash-verified)",
    )
    parser.add_argument(
        "--destination",
        default=".",
        help="output directory for --export (default: current directory)",
    )
    return parser


def build_runtime(
    query: str,
    run_id: str,
    *,
    recorded: bool = config.LENS_RECORDED,
    allow_sql: bool = config.ALLOW_SQL,
    resume: bool = False,
    parent_run_id: str | None = None,
    replay: bool = True,
) -> tuple[Any, Any, Any]:
    """Build the real dependency graph; return ``(runtime, server, sandbox)``.

    Kept as one function so the heavy imports stay out of module import time
    (``--help`` must not load torch) and so tests can patch this single seam
    (``tests/test_run_wiring.py``).

    Outline:
      1. ``workspace = Trajectory(run_id).run_dir / "work"`` — the sandbox mounts
         only the workspace; ``trajectory.jsonl`` / ``tasks.json`` /
         ``manifest.json`` / ``answer.md`` stay in ``run_dir`` (the RLM must not be
         able to rewrite them).
      2. ``client = LensClient(trajectory)``; ``conn = db.connect()`` (admin);
         a second ``db.connect(readonly=True)`` **only** if ``allow_sql`` — the T3
         tool needs ``lens_ro`` with ``autocommit=False``, and nothing else uses
         it.
      3. ``Embedder`` / ``Reranker`` / ``GlinerExtractor`` (all lazy).
      4. ``registry = ToolRegistry()`` — register **closures**, not
         ``functools.partial``: bind ``conn`` / ``embedder`` / ``reranker`` /
         ``extractor`` outside the visible signature and give each closure its
         own docstring (the docstring is the manifest entry; ``partial`` has none
         and would expose the bound parameters as if the RLM had to supply them):
         ``shallow_match`` · ``rerank_papers`` · ``deep_match`` · ``rerank_chunks``
         · ``fetch_papers`` · ``fetch_chunks`` · ``build_kg`` · ``query(sql)``
         (only when ``sql_is_allowed(recorded=..., allow_sql=...)``).
      5. ``token = secrets.token_urlsafe(32)``; ``sandbox = SandboxSession(run_id,
         workspace, token=token)``; ``server = ToolServer(runtime, token=token)``;
         ``server.start()`` — **then** set ``sandbox.host_rpc_port =
         server.port``, because the server may bind port 0. The container never
         sees that port: it dials the relay, and the relay is retargeted through
         ``LENS_RELAY_TARGET_PORT``. The sandbox is started by ``Runtime.run``.
      6. Pass ``resume`` / ``parent_run_id`` / ``replay`` straight through to
         ``Runtime``. On resume, the trajectory and manifest already exist;
         do not truncate or recreate them.
    """
    trajectory = Trajectory(run_id=run_id)
    workspace = trajectory.run_dir / "work"
    client = LensClient(trajectory=trajectory)

    conn = connect()
    embedder, reranker = Embedder(), Reranker()
    extractor = GlinerExtractor()

    registry = ToolRegistry()

    def shallow_match_tool(buckets: dict[str, Any]) -> list[str]:
        """Stage A: HNSW candidate arxiv ids for the query buckets."""
        return shallow_match(conn, buckets, embedder)

    def rerank_papers_tool(arxiv_ids: list[str], buckets: dict[str, Any]) -> list[dict[str, Any]]:
        """Stage A: cross-encoder rerank of candidate papers; top n rows."""
        return rerank_papers(conn, arxiv_ids, buckets, reranker)

    def deep_match_tool(arxiv_ids: list[str], buckets: dict[str, Any]) -> list[str]:
        """Stage B: chunk candidates inside the selected papers."""
        return deep_match(conn, arxiv_ids, buckets, embedder)

    def rerank_chunks_tool(chunk_ids: list[str], buckets: dict[str, Any]) -> dict[str, Any]:
        """Stage B: rerank chunks and hydrate {chunks, papers, assets}."""
        return rerank_chunks(conn, chunk_ids, buckets, reranker)

    def fetch_papers_tool(arxiv_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch full paper rows by arxiv_id."""
        return fetch_papers(conn, arxiv_ids)

    def fetch_chunks_tool(chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch chunk rows by chunk_uuid."""
        return fetch_chunks(conn, chunk_ids)

    def build_kg_tool(chunk_ids: list[str]) -> dict[str, Any]:
        """Build the knowledge graph for the selected chunks (node-link payload)."""
        return build_kg(conn, chunk_ids, extractor)

    registry.register("shallow_match", shallow_match_tool)
    registry.register("rerank_papers", rerank_papers_tool)
    registry.register("deep_match", deep_match_tool)
    registry.register("rerank_chunks", rerank_chunks_tool)
    registry.register("fetch_papers", fetch_papers_tool)
    registry.register("fetch_chunks", fetch_chunks_tool)
    registry.register("build_kg", build_kg_tool)

    if sql_is_allowed(recorded=recorded, allow_sql=allow_sql):
        sql_conn = connect(readonly=True, autocommit=False)

        def query_sql_tool(sql: str) -> list[dict[str, Any]]:
            """Run one read-only SQL query (dev only; disabled in recorded runs)."""
            return query_sql(sql_conn, sql)

        registry.register("query", query_sql_tool)

    token = secrets.token_urlsafe(32)

    sandbox = SandboxSession(run_id=run_id, workspace=workspace, token=token)

    runtime = Runtime(
        query=query,
        run_id=run_id,
        client=client,
        trajectory=trajectory,
        conn=conn,
        embedder=embedder,
        reranker=reranker,
        extractor=extractor,
        sandbox=sandbox,
        registry=registry,
        recorded=recorded,
        resume=resume,
        parent_run_id=parent_run_id,
        replay=replay,
    )

    server = ToolServer(runtime=runtime, token=token)
    server.start()

    # The relay forwards to the port the server actually bound; the container's
    # URL stays the constant relay address (config.SANDBOX_RELAY_URL).
    sandbox.host_rpc_port = server.port

    return (runtime, server, sandbox)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args, dispatch, and return an exit code.

    Dispatch order matters:
      1. ``--list`` -> :func:`lens.agent.runstore.list_runs` (sweep ``running`` ->
         ``interrupted`` first), print one line per run (id, status, date, query,
         cost), return 0. **No runtime.**
      2. ``--show <id>`` -> read the manifest + ``answer.md`` and print them.
         **No runtime.** A missing run returns 2 with a clear message.
      3. ``--export <id> --destination <dir>`` -> ``runstore.export_run``; print the
         bundle path, blob count and size. **No runtime.**
      4. ``--import <bundle>`` -> ``runstore.verify_bundle`` then
         ``runstore.import_run``; print the installed run id or the verification
         errors. **No runtime.**
      5. Otherwise a query is required (exit 2 if absent). ``--resume`` forces
         ``run_id`` and ``resume=True`` and takes the query from the stored
         manifest (a resumed run is not a new question); ``--fork`` creates the
         child manifest first and runs with ``parent_run_id``. Then
         ``build_runtime(...)`` -> ``runtime.run()`` -> print the answer and one
         summary line (``run_id``, cost, ``partial``, ``stop_reason``, artifact
         paths).

    Import the store **module** (``from lens.agent import runstore``) inside this
    function rather than its functions, so tests can patch it
    (``tests/test_run_wiring.py``), and keep the heavy imports inside
    ``build_runtime``.

    ``server.stop()`` goes in a ``finally``; the sandbox is stopped by
    ``Runtime._finalize``. The printed cost is raw ``usage.cost`` — a *reported*
    ``$/query`` adds the 5.5 % credit-load fee, which is a reporting step, not
    something this CLI should bake in.
    """
    from lens.agent import runstore  # module, so tests can patch it

    # The answer is model prose and contains whatever Unicode it likes (an arrow,
    # an em dash, a Greek letter). The Windows console is cp1252, so the final
    # `print` raised UnicodeEncodeError **after** the run had finished and been
    # finalized — a completed run reported as a traceback. Keep the
    # console's encoding, stop crashing on characters it cannot show.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)

    # ---------------------------------------------------------- read-only paths
    if args.list:
        runstore.sweep_interrupted()  # running -> interrupted, once
        for manifest in runstore.list_runs():
            cost = (manifest.totals or {}).get("cost_usd") or 0.0
            print(
                f"{manifest.run_id}  {manifest.status:11s}  "
                f"{manifest.updated_at or '-':32s}  ${cost:8.4f}  {manifest.query[:70]}"
            )
        return 0

    if args.show:
        manifest = runstore.read_manifest(runstore.runs_root() / args.show)

        if manifest is None:
            print(f"no run {args.show!r} under {runstore.runs_root()}")
            return 2

        print(f"run_id       {manifest.run_id}")
        print(f"status       {manifest.status}")
        print(f"query        {manifest.query}")
        print(f"parent       {manifest.parent_run_id or '-'}")
        print(f"created_at   {manifest.created_at}")
        print(f"updated_at   {manifest.updated_at}")
        print(f"iterations   {manifest.iterations}")
        print(f"models       {manifest.models}")

        answer_path = runstore.runs_root() / args.show / "answer.md"
        print("--- answer.md ---")
        print(
            answer_path.read_text(encoding="utf-8")
            if answer_path.is_file()
            else "(no answer.md in this run)"
        )
        return 0

    if args.export:
        bundle = runstore.export_run(args.export, args.destination)
        print(f"exported {bundle}")
        return 0

    if args.import_bundle:
        report = runstore.verify_bundle(args.import_bundle)

        if not report["ok"]:
            # Verify first: a tampered bundle must install nothing.
            print(f"{args.import_bundle}: verification failed")
            for error in report["errors"]:
                print(f"  - {error}")
            return 2

        manifest = runstore.import_run(args.import_bundle)
        print(f"imported {manifest.run_id} into {runstore.runs_root()}")
        return 0

    # ------------------------------------------------------------------ a run
    resume = args.resume is not None
    fork_source = args.fork

    if resume:
        run_id = args.resume
        stored = runstore.read_manifest(runstore.runs_root() / run_id)

        if stored is None:
            print(f"no run {run_id!r} to resume under {runstore.runs_root()}")
            return 2

        # A resumed run is not a new question: the query comes from the manifest,
        # not from argv, or the log and the prompt would disagree.
        query = stored.query
    else:
        if not args.query:
            print(
                "a query is required (or use --list / --show / --resume / "
                "--fork / --export / --import)"
            )
            return 2

        query = args.query
        run_id = args.run_id or _timestamp_run_id()

        if fork_source:
            # Provenance, not continuation: the child gets its own log.
            runstore.fork_run(fork_source, run_id, query=query)

    runtime, server, sandbox = build_runtime(
        query,
        run_id,
        recorded=args.recorded,
        allow_sql=args.allow_sql,
        resume=resume,
        parent_run_id=fork_source,
        replay=not args.no_replay,
    )

    try:
        result = runtime.run()
    finally:
        # The port and the DB connection are per-process resources; the sandbox is
        # stopped by ``Runtime._finalize``. Exceptions propagate — the caller sees
        # the traceback, not an exit code that hides it.
        server.stop()
        _close_quietly(getattr(runtime, "conn", None))

    print(result.answer)
    print(
        f"run {run_id}: ${result.cost_usd:.4f}, {result.iterations} iterations, "
        f"partial={result.partial}, stop_reason={result.stop_reason}"
    )
    print(f"artifacts: {result.answer_path} | {result.trajectory_path} | {result.tasks_path}")

    return 0


def _timestamp_run_id() -> str:
    """Return a sortable default run id (``run-YYYYmmdd-HHMMSS``, UTC)."""
    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")


def _close_quietly(connection: Any) -> None:
    """Close a psycopg connection when one was opened; never raise from cleanup."""
    if connection is None:
        return

    try:
        connection.close()
    except Exception:  # noqa: BLE001 - cleanup must not mask the run's outcome
        pass


if __name__ == "__main__":
    raise SystemExit(main())
