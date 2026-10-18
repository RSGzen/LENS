"""CLI entry point for one LENS query, plus run listing/resume (M8/M8.5).

Spec: ``MVP Build Roadmap.md`` M8/M9/M8.5.

Usage::

    python -m lens.run "How do RLM systems aggregate long contexts?"
    python -m lens.run --list                     # saved runs, newest first
    python -m lens.run --show <run_id>            # read-only: manifest + answer
    python -m lens.run --resume <run_id>          # continue an interrupted run
    python -m lens.run --fork <run_id> --run-id new "same query, other arm"

``--list`` / ``--show`` are **read-only**: they never build a runtime, call a
model, or start a container. That is what makes the M10 UI cheap.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from lens import config


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m lens.run",
        description="Run one LENS query end-to-end (M8 harness), or inspect saved runs.",
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
    # --- M8.5 session management (no runtime for --list / --show) -------------
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
      3. ``Embedder`` / ``Reranker`` / ``KGExtractor`` (all lazy).
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
         ``server.start()`` — **then** set ``sandbox.host_rpc_url =
         f"http://host.docker.internal:{server.port}"``, because the server may
         have bound port 0. The sandbox is started by ``Runtime.run``.
      6. Pass ``resume`` / ``parent_run_id`` / ``replay`` straight through to
         ``Runtime`` (M8.5). On resume, the trajectory and manifest already exist;
         do not truncate or recreate them.
    """
    raise NotImplementedError("M8 scaffold: implement run.build_runtime")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args, dispatch, and return an exit code.

    Dispatch order matters:
      1. ``--list`` -> :func:`lens.agent.runstore.list_runs` (sweep ``running`` ->
         ``interrupted`` first), print one line per run (id, status, date, query,
         cost), return 0. **No runtime.**
      2. ``--show <id>`` -> read the manifest + ``answer.md`` and print them.
         **No runtime.** A missing run returns 2 with a clear message.
      3. Otherwise a query is required (exit 2 if absent). ``--resume`` forces
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
    ``$/query`` adds the 5.5 % credit-load fee (R4), which is an M11 reporting
    step, not something this CLI should bake in.
    """
    raise NotImplementedError("M8 scaffold: implement run.main")


if __name__ == "__main__":
    raise SystemExit(main())
