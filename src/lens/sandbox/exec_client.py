"""In-container client for the runner's ``/exec`` endpoint (option C).

The host cannot reach the runner over TCP: ``sandbox-net`` is ``internal: true``
and Docker installs no DNAT rule for an internal network, so the published port
is silently dropped. ``docker compose exec`` runs *inside* the
container instead, where ``127.0.0.1:8080`` is the runner.

Usage (from ``SandboxSession``)::

    docker compose exec -T lens-repl python /app/exec_client.py health
    docker compose exec -T lens-repl python /app/exec_client.py exec   # code on stdin

Both print one JSON object on stdout and exit 0 whenever a body could be built —
including a failed block, which is ``{"ok": false, ...}``. A non-zero exit means
the transport itself broke, and the host maps that to an ``ExecResult`` error.

stdlib only: the image deliberately has no ``requests`` and no ``lens`` package.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

RUNNER_URL = os.environ.get("LENS_RUNNER_URL", "http://127.0.0.1:8080")


def _call(path: str, payload: dict | None = None) -> dict:
    """POST/GET one JSON object and return the parsed response."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        RUNNER_URL + path,
        data=data,
        method="GET" if data is None else "POST",
    )
    if data is not None:
        request.add_header("Content-Type", "application/json")

    # The runner's own SIGALRM fires first; this only bounds a hung socket.
    timeout = float(os.environ.get("LENS_MAX_EXEC_BLOCK_S", "120")) + 10

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "exec"

    try:
        if mode == "health":
            body = _call("/health")
        else:
            body = _call("/exec", {"code": sys.stdin.read()})
    except (urllib.error.URLError, OSError, ValueError) as exc:
        body = {
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)[:300]},
        }

    json.dump(body, sys.stdout)
    sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
