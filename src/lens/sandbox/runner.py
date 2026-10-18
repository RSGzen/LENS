"""Passive, persistent Python executor for the RLM sandbox.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a.

This process is **not** the agent. It never calls an LLM and never decides the
next step. It keeps one Python namespace alive so variables survive across
host-driven steps, and exposes only two HTTP endpoints::

    GET  /health -> {"ok": true}
    POST /exec   -> {"ok", "stdout", "stderr", "result", "error", "duration_ms"}

The runner is single-threaded on purpose: one block executes at a time, and the
Linux ``SIGALRM`` timeout can interrupt it.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

try:  # container: runner.py and rlm_api.py are siblings at /app
    import rlm_api
except ImportError:  # host tests: package import
    from lens.sandbox import rlm_api  # type: ignore[no-redef]

# Hard cap on the code block itself (a guardrail, not a scientific value).
MAX_CODE_CHARS = 200_000
DEFAULT_PORT = 8080
DEFAULT_STDOUT_MAX_CHARS = int(os.environ.get("LENS_SANDBOX_STDOUT_MAX_CHARS", "20000"))


class Executor:
    """One persistent namespace + one ``exec`` entry point."""

    def __init__(self, *, stdout_max_chars: int = DEFAULT_STDOUT_MAX_CHARS) -> None:
        self.stdout_max_chars = stdout_max_chars
        self.namespace: dict[str, Any] = {}

    def run(self, code: str) -> dict[str, Any]:
        """Execute ``code`` in the persistent namespace and return the envelope.

        Contract:
          - Capture stdout/stderr separately.
          - On success, ``ok=True`` and ``result=None`` (use ``print``).
          - On an exception, ``ok=False`` with ``{"type", "message"}`` and the
            traceback in ``stderr``.
          - Enforce ``config.MAX_EXEC_BLOCK_S`` with ``signal.setitimer`` on
            Linux; return a timeout error rather than killing the process.
          - Truncate captured output to ``stdout_max_chars`` and set
            ``truncated=True`` when it happened.
        """
        raise NotImplementedError("M8 scaffold: implement Executor.run")

    def _truncate(self, text: str) -> tuple[str, bool]:
        """Return ``(text, truncated)`` for one captured stream."""
        raise NotImplementedError("M8 scaffold: implement Executor._truncate")


class ExecHandler(BaseHTTPRequestHandler):
    """HTTP handler bound to one :class:`Executor` (set as ``server.executor``)."""

    server: Any

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        """Serve ``/health``."""
        raise NotImplementedError("M8 scaffold: implement ExecHandler.do_GET")

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        """Serve ``/exec``."""
        raise NotImplementedError("M8 scaffold: implement ExecHandler.do_POST")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep container logs quiet; the host owns observability."""
        return


def make_server(host: str, port: int, executor: Executor) -> HTTPServer:
    """Build the single-threaded HTTP server with ``executor`` attached."""
    raise NotImplementedError("M8 scaffold: implement make_server")


def main() -> None:
    """Read ``HOST``/``PORT`` from the environment and serve forever."""
    raise NotImplementedError("M8 scaffold: implement runner.main")


if __name__ == "__main__":
    main()
