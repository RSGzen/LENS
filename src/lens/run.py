"""CLI entry point for one LENS query.

Spec: ``MVP Build Roadmap.md`` M8/M9.

Usage::

    python -m lens.run "How do RLM systems aggregate long contexts?"

The module only wires dependencies; the loop lives in
:class:`lens.agent.runtime.Runtime`.
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
        description="Run one LENS query end-to-end (M8 harness).",
    )
    parser.add_argument("query", help="natural-language research query")
    parser.add_argument(
        "--run-id",
        default=None,
        help="run id / artifact directory name (default: generated timestamp)",
    )
    # BooleanOptionalAction, not store_true: the env may already set these, and
    # the CLI must be able to *turn them off* for a recorded run.
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
    return parser


def build_runtime(
    query: str,
    run_id: str,
    *,
    recorded: bool = config.LENS_RECORDED,
    allow_sql: bool = config.ALLOW_SQL,
) -> tuple[Any, Any, Any]:
    """Build the real dependency graph; return ``(runtime, server, sandbox)``.

    Kept as one function so the heavy imports stay out of module import time
    (``--help`` must not load torch) and so tests can patch this single seam
    (``tests/test_run_wiring.py``).

    Outline:
      1. ``workspace = Trajectory(run_id).run_dir / "work"`` — the sandbox mounts
         only the workspace; ``trajectory.jsonl`` / ``tasks.json`` / ``answer.md``
         stay in ``run_dir`` (the RLM must not be able to rewrite them).
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
    """
    raise NotImplementedError("M8 scaffold: implement run.build_runtime")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args, build the runtime, run one query, print the artifacts.

    Order: parse -> ``run_id`` (default ``run-%Y%m%d-%H%M%S``) ->
    ``build_runtime(...)`` -> ``runtime.run()`` -> print the answer and one
    summary line (``run_id``, cost, ``partial``, ``stop_reason``, artifact paths).

    ``server.stop()`` goes in a ``finally``; the sandbox is stopped by
    ``Runtime._finalize``. The printed cost is raw ``usage.cost`` — a *reported*
    ``$/query`` adds the 5.5 % credit-load fee (R4), which is an M11 reporting
    step, not something this CLI should bake in.
    """
    raise NotImplementedError("M8 scaffold: implement run.main")


if __name__ == "__main__":
    raise SystemExit(main())
