"""Host-side lifecycle + exec client for the ``lens-repl`` container.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a/§4b; MVP Roadmap M8.

The runtime starts **one** container per query session, mounts only the run
workspace (``<run_dir>/work``, never the artifact directory — see
:mod:`lens.agent.runtime`), passes the per-run RPC token, and talks to the
passive runner over HTTP. The sandbox never receives the OpenRouter key or DB
credentials.
"""

from __future__ import annotations

import requests
import subprocess
import time
import json
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lens import config


@dataclass
class ExecResult:
    """One ``/exec`` response from ``sandbox/runner.py``."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    result: Any = None
    error: dict[str, Any] | None = None
    duration_ms: int | None = None
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the RPC-shaped dict the orchestrator appends as an observation."""
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
        }


class SandboxSession:
    """Docker Compose lifecycle plus the host-side ``/exec`` client."""

    def __init__(
        self,
        run_id: str,
        workspace: str | Path,
        *,
        image: str = config.SANDBOX_IMAGE,
        compose_file: Path = config.SANDBOX_COMPOSE_FILE,
        service: str = config.SANDBOX_SERVICE,
        port: int = config.SANDBOX_PORT,
        token: str,
        exec_timeout_s: float = config.MAX_EXEC_BLOCK_S,
        session: Any | None = None,
        host_rpc_url: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.workspace = Path(workspace)
        self.image = image
        self.compose_file = Path(compose_file)
        self.service = service
        self.port = port
        self.token = token
        self.exec_timeout_s = exec_timeout_s
        # The injected session is the test seam and the production client at
        # once: ``requests`` (the module) has ``get``/``post`` with the same
        # signature, so no wrapper is needed.
        self.session = session or requests
        # The URL the *container* uses to reach the host RPC server. Built by
        # run.py from the server's real bound port, because ``ToolServer.start()``
        # may bind port 0 and pick a free one. Default keeps the documented
        # single-container setup working.
        self.host_rpc_url = host_rpc_url or (
            f"http://host.docker.internal:{config.HOST_RPC_PORT}"
        )

    @property
    def url(self) -> str:
        """Host-side runner URL (compose publishes the port on 127.0.0.1)."""
        return f"http://127.0.0.1:{self.port}"

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Create/recreate the container and wait for ``/health``.

        Steps:
          1. Build the compose environment: ``LENS_RUN_WORKSPACE``,
             ``LENS_RUN_ID``, ``LENS_RPC_TOKEN``, ``LENS_HOST_RPC_URL``,
             ``LENS_SANDBOX_IMAGE``, ``LENS_SANDBOX_PORT``.
          2. Run ``docker compose -f <file> up -d --no-deps --force-recreate
             <service>`` from the repo root.
          3. Poll :meth:`wait_until_healthy`; on failure, call :meth:`stop` and
             raise a clear error.

        ``--force-recreate`` is the load-bearing flag: a stopped container is
        reused with its **previous** environment, so the new token and RPC URL
        would silently not apply (symptom: every call is a 401).
        ``--no-deps`` keeps ``lens-pg`` untouched.
        """
        result = self._run_compose(
            ["up", "-d", "--no-deps", "--force-recreate", self.service]
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"docker compose up failed for {self.service!r}: "
                f"{(result.stderr or result.stdout).strip()[:500]}"
            )

        if not self.wait_until_healthy():
            # Leave nothing running: a half-started container would hold the
            # published port and the next run would fail on "port is allocated".
            self.stop()
            raise RuntimeError(
                f"sandbox {self.service!r} started but /health never answered"
            )

    def stop(self) -> None:
        """Remove the container (``docker compose ... rm -sf <service>``).

        ``rm -sf`` and **never** ``down``: ``down`` removes the networks, and
        ``down -v`` would delete the ``lens-pgdata`` volume — the loaded corpus.
        Best-effort by design (it runs from a ``finally``): removing a container
        that is not there is a no-op, and a failure must not mask the original
        error.
        """
        self._run_compose(["rm", "-sf", self.service])

    def wait_until_healthy(self, timeout_s: float = 30.0) -> bool:
        """Poll ``/health`` until ready or the timeout expires."""
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            try:
                if self.health():
                    return True

            # Not listening yet: normal for the first second
            except requests.RequestException:
                pass                      
            time.sleep(0.25)

        return False

    # -------------------------------------------------------------- transport
    def health(self) -> bool:
        """Return ``True`` when ``GET /health`` answers 200.

        The per-request timeout keeps a half-open socket from blocking the
        caller; the *retry budget* lives in :meth:`wait_until_healthy`. They are
        different clocks and the second one is not a substitute for the first.
        """
        response = self.session.get(f"{self.url}/health", timeout=2)

        return response.status_code == 200

    def exec(self, code: str) -> ExecResult:
        """POST one Python block to ``/exec`` and parse the response.

        Request body: ``data=json.dumps({"code": code})`` with a JSON content
        type. Client-side timeout is ``exec_timeout_s + 10`` so the runner's own
        timeout fires first. Transport errors become ``ExecResult(ok=False, ...)``.
        """
        try:
            response = self.session.post(
                f"{self.url}/exec",
                data=json.dumps({"code": code}),
                timeout=self.exec_timeout_s + 10,
            )
            data = response.json()
        except requests.RequestException as exc:
            # Refused / timed out / reset: the container is unreachable or dead.
            return ExecResult(
                ok=False,
                error={"type": type(exc).__name__, "message": str(exc)[:300]},
            )

        if response.status_code != 200:
            return ExecResult(
                ok=False,
                error={
                    "type": "HTTPError",
                    "message": f"runner answered {response.status_code}",
                },
            )

        return ExecResult(
            # The runner reports a failed block as 200 + ``ok: false``; an HTTP
            # error above means the transport broke, not the model's code.
            ok=data.get("error") is None,
            stdout=data.get("stdout") or "",
            stderr=data.get("stderr") or "",
            result=data.get("result"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms"),
            truncated=bool(data.get("truncated")),
        )

    # ------------------------------------------------------------------ utils
    def _compose_env(self) -> dict[str, str]:
        """Return the environment for the compose subprocess.

        Must **not** contain ``OPENROUTER_API_KEY`` or any ``LENS_DB_*`` secret —
        the compose file interpolates only what is listed here.

        Two traps:
          - ``LENS_RUN_WORKSPACE`` goes into a volume spec, so it must be a
            **forward-slash** path (``Path.as_posix()``); backslashes break
            ``"<host>:/workspace"`` on Windows.
          - ``LENS_HOST_RPC_URL`` is ``self.host_rpc_url`` (built from the port
            the RPC server actually bound), not a constant.

        ``LENS_MAX_EXEC_BLOCK_S`` must be passed as **the same value** as
        ``exec_timeout_s`` (i.e. ``config.MAX_EXEC_BLOCK_S``): ``lens.config`` is
        not in the image, so the runner's timeout comes from this variable alone.
        Let the two drift and the host gives up on a block the runner would still
        finish (or waits on one it already killed).

        ``LENS_SANDBOX_STDOUT_MAX_CHARS`` is passed too: the compose default is
        20,000 while ``config.SANDBOX_STDOUT_MAX_CHARS`` is 6,000, so leaving it
        to the file silently widens the sandbox's cut.

        Every value must be a **string** — ``env=`` is passed straight to
        ``subprocess``, and an ``int`` there is a ``TypeError`` on Windows.
        """
        return {
            "LENS_RUN_WORKSPACE": self.workspace.as_posix(),
            "LENS_RUN_ID": self.run_id,
            "LENS_RPC_TOKEN": self.token,
            # The *container's* view of the host, not this host-side URL.
            "LENS_HOST_RPC_URL": self.host_rpc_url,
            "LENS_SANDBOX_IMAGE": self.image,
            "LENS_SANDBOX_PORT": str(self.port),
            "LENS_MAX_EXEC_BLOCK_S": str(self.exec_timeout_s),
            "LENS_SANDBOX_STDOUT_MAX_CHARS": str(config.SANDBOX_STDOUT_MAX_CHARS),
        }

    def _run_compose(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Run ``docker compose -f <file> <args>`` and return the completed process.

        ``env`` is the ambient environment **plus** :meth:`_compose_env`, never
        the latter alone: without ``PATH``/``DOCKER_HOST`` the ``docker`` lookup
        itself fails. Failures are returned, not raised — the caller decides
        (``start`` raises, ``stop`` ignores).
        """
        return subprocess.run(
            ["docker", "compose", "-f", str(self.compose_file), *args],
            cwd=config.REPO_ROOT,
            env={**os.environ, **self._compose_env()},
            capture_output=True,
            text=True,
            timeout=120,
        )
