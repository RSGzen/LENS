"""Host-side lifecycle + exec client for the ``lens-repl`` container.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a/§4b; MVP Roadmap M8.

The runtime starts **one** container per query session, mounts only the run
workspace (``<run_dir>/work``, never the artifact directory — see
:mod:`lens.agent.runtime`), passes the per-run RPC token, and talks to the
passive runner over HTTP. The sandbox never receives the OpenRouter key or DB
credentials.
"""

from __future__ import annotations

import subprocess
import time
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
        raise NotImplementedError("M8 scaffold: implement ExecResult.to_dict")


class SandboxSession:
    """Docker Compose lifecycle plus the host-side ``/exec`` client."""

    def __init__(
        self,
        run_id: str,
        workspace: str | Path,
        *,
        image: str = config.SANDBOX_IMAGE,
        compose_file: str = config.SANDBOX_COMPOSE_FILE,
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
        self.session = session
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
        """
        raise NotImplementedError("M8 scaffold: implement SandboxSession.start")

    def stop(self) -> None:
        """Remove the container (``docker compose ... rm -sf <service>``)."""
        raise NotImplementedError("M8 scaffold: implement SandboxSession.stop")

    def wait_until_healthy(self, timeout_s: float = 30.0) -> bool:
        """Poll ``/health`` until ready or the timeout expires."""
        raise NotImplementedError("M8 scaffold: implement SandboxSession.wait_until_healthy")

    # -------------------------------------------------------------- transport
    def health(self) -> bool:
        """Return ``True`` when ``GET /health`` answers ``ok``."""
        raise NotImplementedError("M8 scaffold: implement SandboxSession.health")

    def exec(self, code: str) -> ExecResult:
        """POST one Python block to ``/exec`` and parse the response.

        Request body: ``data=json.dumps({"code": code})`` with a JSON content
        type. Client-side timeout is ``exec_timeout_s + 10`` so the runner's own
        timeout fires first. Transport errors become ``ExecResult(ok=False, ...)``.
        """
        raise NotImplementedError("M8 scaffold: implement SandboxSession.exec")

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
        """
        raise NotImplementedError("M8 scaffold: implement SandboxSession._compose_env")

    def _run_compose(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Run ``docker compose -f <file> <args>`` and return the completed process."""
        raise NotImplementedError("M8 scaffold: implement SandboxSession._run_compose")
