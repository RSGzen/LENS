"""Passive, persistent Python executor for the RLM sandbox.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a.

This process is **not** the agent. It never calls an LLM and never decides the
next step. It keeps one Python namespace alive so variables survive across
host-driven steps, and exposes only two HTTP endpoints::

    GET  /health -> {"ok": true}
    POST /exec   -> {"ok", "stdout", "stderr", "result", "error", "duration_ms"}

The runner is single-threaded on purpose: one block executes at a time, and the
Linux ``SIGALRM`` timeout can interrupt it. ``Executor.run`` must therefore be
called from the **main thread** (``signal.signal`` refuses any other thread),
which is exactly what :func:`main` does — never serve this from a worker thread.
"""

from __future__ import annotations

import json
import os
import signal
import time
import traceback
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
from typing import Any

try:  # container: runner.py and rlm_api.py are siblings at /app
    import rlm_api  # noqa: F401 - imported so the RLM's blocks can `import rlm_api`
except ImportError:  # host tests: package import
    from lens.sandbox import rlm_api  # type: ignore[no-redef]  # noqa: F401

# Hard cap on the code block itself (a guardrail, not a scientific value).
MAX_CODE_CHARS = 200_000
DEFAULT_PORT = 8080
# First cut on observation size: the host applies a second, tighter cut
# (config.TOOL_RESULT_SUMMARY_CHARS) before the text becomes a message. 6,000
# chars is ~1.5k tokens — enough for a digest, small enough that 8 turns do not
# fill the root context on their own.
DEFAULT_STDOUT_MAX_CHARS = int(os.environ.get("LENS_SANDBOX_STDOUT_MAX_CHARS", "6000"))
# Local copy of ``config.MAX_EXEC_BLOCK_S``: ``lens.config`` is **not** in the
# image (the runner ships with rlm_api.py and nothing else), so the value has to
# arrive as the environment variable ``SandboxSession`` sets. The host's cap and
# this one must be the same number, or the host gives up on a block the runner
# would still finish.
MAX_EXEC_BLOCK_S = float(os.environ.get("LENS_MAX_EXEC_BLOCK_S", "120"))


class _ExecTimeout(Exception):
    """Raised from the ``SIGALRM`` handler; caught inside :meth:`Executor.run`."""


def _on_alarm(signum: int, frame: Any) -> None:
    """Turn the alarm into an exception the interpreter can unwind through."""
    raise _ExecTimeout(f"execution exceeded {MAX_EXEC_BLOCK_S:.0f}s")


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
          - Enforce ``MAX_EXEC_BLOCK_S`` with ``signal.setitimer`` on Linux;
            return a timeout error rather than killing the process. A timer left
            armed would fire during the *next* block, so it is disarmed in
            ``finally``.
          - Truncate captured output to ``stdout_max_chars`` and set
            ``truncated=True`` when it happened.

        The namespace is ``self.namespace`` **and it is the globals dict**: one
        dict, never a separate ``locals``. With split globals/locals a function
        defined in the block cannot see names assigned in the same block
        (``x = 1`` lands in locals, the function's ``__globals__`` is globals).
        """
        if len(code) > MAX_CODE_CHARS:
            return {
                "ok": False,
                "stdout": "",
                "stderr": "",
                "result": None,
                "error": {
                    "type": "CodeTooLarge",
                    "message": f"{len(code)} chars exceeds the {MAX_CODE_CHARS} cap",
                },
                "duration_ms": 0,
                "truncated": False,
            }

        start_timer = time.perf_counter()

        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        error = None

        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.setitimer(signal.ITIMER_REAL, MAX_EXEC_BLOCK_S)

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, self.namespace)

        except _ExecTimeout as exc:
            error = {"type": "TimeoutError", "message": str(exc)}
            # stderr, not stdout: the contract puts the diagnosis there, and the
            # model reads stdout as its own output.
            stderr_buffer.write(f"TimeoutError: {exc}\n")

        except SystemExit as exc:
            # Not an ``Exception``: without this a block calling exit() takes the
            # whole runner down and the host sees a dead container.
            error = {"type": "SystemExit", "message": str(exc)}

        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:300]}
            traceback.print_exc(file=stderr_buffer)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, 0)

            duration_ms = int((time.perf_counter() - start_timer) * 1000)

        stdout_text, cut_out = self._truncate(stdout_buffer.getvalue())
        stderr_text, cut_err = self._truncate(stderr_buffer.getvalue())

        return {
            "ok": error is None,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "result": None,
            "error": error,
            "duration_ms": duration_ms,
            "truncated": cut_out or cut_err,
        }

    def _truncate(self, text: str) -> tuple[str, bool]:
        """Return ``(text, truncated)`` for one captured stream."""
        if len(text) > self.stdout_max_chars:
            return (text[: self.stdout_max_chars], True)

        return (text, False)


class ExecHandler(BaseHTTPRequestHandler):
    """HTTP handler bound to one :class:`Executor` (set as ``server.executor``)."""

    server: Any

    protocol_version = "HTTP/1.1"
    timeout = 300

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep container logs quiet; the host owns observability."""
        return

    def _send(self, status: int, body: Mapping[str, Any]) -> None:
        """Write one JSON response with an exact ``Content-Length``."""
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> tuple[bool, Any]:
        """Return ``(ok, payload)``.

        ``ok=False`` means the bytes were not valid JSON, which is a **400**
        answered here: :meth:`handle_request` takes an already-parsed
        payload, so it can never see a parse failure.
        ``Content-Length: 0`` (no body) is ``(True, None)``.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return True, None

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False, None

        return True, payload

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        """Serve ``/health``; anything else is a 404."""
        path = self.path.split("?", 1)[0]

        if path != "/health":
            self._send(
                404,
                {
                    "ok": False,
                    "error": {"type": "NotFound", "message": f"unknown endpoint {path!r}"},
                },
            )
            return

        self._send(200, {"ok": True})

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        """Serve ``/exec``; the body is ``{"code": str}`` and the reply is the envelope."""
        path = self.path.split("?", 1)[0]

        if path != "/exec":
            self._send(
                404,
                {
                    "ok": False,
                    "error": {"type": "NotFound", "message": f"unknown endpoint {path!r}"},
                },
            )
            return

        ok, payload = self._read_json()
        code = payload.get("code") if ok and isinstance(payload, Mapping) else None

        if not isinstance(code, str):
            self._send(
                400,
                {
                    "ok": False,
                    "error": {"type": "BadRequest", "message": 'body must be {"code": str}'},
                },
            )
            return

        # The runner is single-threaded, so nothing else runs while this blocks.
        self._send(200, self.server.executor.run(code))


def make_server(host: str, port: int, executor: Executor) -> HTTPServer:
    """Build the single-threaded HTTP server with ``executor`` attached.

    ``HTTPServer`` (not ``ThreadingHTTPServer``) is deliberate: the ``SIGALRM``
    timeout only fires in the main thread, and one block must never run while
    another is still executing. The handler reaches the executor through
    ``self.server.executor``.
    """
    server = HTTPServer((host, port), ExecHandler)
    server.executor = executor

    return server


def main() -> None:
    """Read ``HOST``/``PORT`` from the environment and serve forever."""
    # Bind all interfaces: the host reaches the runner through `docker compose exec`
    # (a process inside this container dials 127.0.0.1), and nothing is published.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", str(DEFAULT_PORT)))

    # The only log line: log_message() is silenced, so without this a healthy
    # container and a hung one look identical in ``docker compose logs``.
    print(f"runner listening on {host}:{port}", flush=True)

    make_server(host, port, Executor()).serve_forever()


if __name__ == "__main__":
    main()
