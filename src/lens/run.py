"""CLI entry point for one LENS query.

Spec: ``MVP Build Roadmap.md`` M8/M9.

Usage::

    python -m lens.run "How do RLM systems aggregate long contexts?"

The module only wires dependencies; the loop lives in :class:`lens.agent.runtime.Runtime`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

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
    parser.add_argument(
        "--recorded",
        action="store_true",
        default=config.LENS_RECORDED,
        help="disable dev-only escape hatches (T3 SQL) for a recorded run",
    )
    parser.add_argument(
        "--allow-sql",
        action="store_true",
        default=config.ALLOW_SQL,
        help="allow the gated T3 query(sql) tool (dev only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args, build real dependencies, run the harness, return an exit code.

    Wiring outline:
      1. ``run_id = args.run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")``.
      2. ``Trajectory(run_id)``; ``LensClient(trajectory)``.
      3. ``db.connect()``; ``Embedder()``; ``Reranker()``; ``KGExtractor()``.
      4. Build a :class:`ToolRegistry` with the M5-M7 tools + ledger tools.
      5. ``token = secrets.token_urlsafe(32)``; ``ToolServer(runtime, token=token)``.
      6. ``SandboxSession(run_id, trajectory.run_dir, token=token)``.
      7. ``Runtime(...).run()``; print the answer path + cost; stop the RPC server.
    """
    raise NotImplementedError("M8 scaffold: implement run.main")


if __name__ == "__main__":
    raise SystemExit(main())
