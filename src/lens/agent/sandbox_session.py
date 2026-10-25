"""Host-side lifecycle + exec client for the ``lens-repl`` container.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a/§4b; MVP Roadmap M8.

The runtime starts **one** container per query session, mounts only the run
workspace (``<run_dir>/work``, never the artifact directory — see
:mod:`lens.agent.runtime`), passes the per-run RPC token, and talks to the
passive runner over HTTP. The sandbox never receives the OpenRouter key or DB
credentials.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lens import config

#: Runs **inside** the container and posts to the runner there (option C). The
#: runner is not reachable over TCP while ``sandbox-net`` is ``internal: true``
#: (LOG-107), so the host reaches it with ``docker compose exec``.
EXEC_CLIENT = "/app/exec_client.py"


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
        host_rpc_url: str | None = None,
        host_rpc_port: int = config.HOST_RPC_PORT,
        relay_service: str = config.SANDBOX_RELAY_SERVICE,
    ) -> None:
        self.run_id = run_id
        self.workspace = Path(workspace)
        self.image = image
        self.compose_file = Path(compose_file)
        self.service = service
        self.port = port
        self.token = token
        self.exec_timeout_s = exec_timeout_s
        self.relay_service = relay_service
        # The port the RELAY forwards to (`run.py` sets it from the server's real
        # bound port: ``ToolServer.start()`` may bind port 0).
        self.host_rpc_port = host_rpc_port
        # The URL the *container* uses. It is the relay's address on the internal
        # network, never the host's: from an internal bridge there is no route to
        # the host at all (LOG-114), so the relay is the only peer.
        self.host_rpc_url = host_rpc_url or config.SANDBOX_RELAY_URL

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

        The **relay** starts first (and is recreated for the same reason): it is
        the sandbox's only route to the host RPC server, so a stale relay target
        port would send every ``tool()`` to a closed port.
        """
        result = self._run_compose(
            [
                "up",
                "-d",
                "--no-deps",
                "--force-recreate",
                self.relay_service,
                self.service,
            ]
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
        error. The relay goes with it - one run, one sandbox, one relay.
        """
        self._run_compose(["rm", "-sf", self.service, self.relay_service])

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
        """Return ``True`` when the in-container ``GET /health`` answers ``ok``.

        Runs over ``docker exec`` (option C): the runner is not reachable from the
        host as long as ``sandbox-net`` stays ``internal: true``. The 30 s
        deadline here is the compose call's; the retry budget lives in
        :meth:`wait_until_healthy` — different clocks, both needed.
        """
        result = self._run_compose(
            ["exec", "-T", self.service, "python", EXEC_CLIENT, "health"], timeout=30
        )

        if result.returncode != 0:
            return False

        try:
            return bool(json.loads(result.stdout).get("ok"))
        except (json.JSONDecodeError, AttributeError):
            return False

    def exec(self, code: str) -> ExecResult:
        """Run one Python block in the container and parse the runner's envelope.

        The code travels on **stdin** (no quoting, no shell) to
        ``/app/exec_client.py``, which posts it to the runner inside the
        container. Client-side timeout is ``exec_timeout_s + 20`` (docker exec
        overhead on top of the runner's own cap), so a runaway block comes back
        as a structured ``TimeoutError`` instead of a client abort.
        """
        try:
            result = self._run_compose(
                ["exec", "-T", self.service, "python", EXEC_CLIENT, "exec"],
                input=code,
                timeout=self.exec_timeout_s + 20,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                ok=False,
                error={"type": "TimeoutExpired", "message": str(exc)[:300]},
            )

        if result.returncode != 0:
            # The transport broke (container gone, docker down) — distinct from a
            # block that failed, which is a 0 exit with ``ok: false``.
            return ExecResult(
                ok=False,
                error={
                    "type": "ContainerExecError",
                    "message": (result.stderr or result.stdout).strip()[:300],
                },
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return ExecResult(
                ok=False,
                error={"type": "BadEnvelope", "message": result.stdout[:300]},
            )

        return ExecResult(
            # The runner reports a failed block as 200 + ``ok: false``; a
            # non-zero exit above means the transport broke, not the model's code.
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
          - ``LENS_HOST_RPC_URL`` is the **relay's** address on the internal
            network (``self.host_rpc_url``), not the host's; the relay's target
            port is passed separately (``LENS_RELAY_TARGET_PORT``) because the RPC
            server may bind a port the compose file cannot know.

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
            # The *container's* view: the relay on the internal network.
            "LENS_HOST_RPC_URL": self.host_rpc_url,
            # What the relay forwards to (the RPC server's real bound port).
            "LENS_RELAY_TARGET_PORT": str(self.host_rpc_port),
            "LENS_SANDBOX_IMAGE": self.image,
            "LENS_SANDBOX_PORT": str(self.port),
            "LENS_MAX_EXEC_BLOCK_S": str(self.exec_timeout_s),
            "LENS_SANDBOX_STDOUT_MAX_CHARS": str(config.SANDBOX_STDOUT_MAX_CHARS),
        }

    def _run_compose(
        self,
        args: list[str],
        *,
        input: str | None = None,
        timeout: float = 120,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``docker compose -f <file> <args>`` and return the completed process.

        ``env`` is the ambient environment **plus** :meth:`_compose_env`, never
        the latter alone: without ``PATH``/``DOCKER_HOST`` the ``docker`` lookup
        itself fails. Failures are returned, not raised — the caller decides
        (``start`` raises, ``stop`` ignores).

        ``input`` feeds stdin, which is how code reaches
        ``exec_client.py``: no shell, no quoting, no argv length limit.

        ``encoding="utf-8"`` is load-bearing, not tidiness: with ``text=True``
        alone Python encodes stdin with the **locale** encoding (cp1252 here), so
        an em dash the model wrote arrives as byte ``0x97``, the container decodes
        it as a lone surrogate, and ``exec()`` dies with ``UnicodeEncodeError`` —
        a whole turn lost to one non-ASCII character (found by the M9 smoke,
        LOG-115). UTF-8 on both directions matches what the image uses.
        """
        return subprocess.run(
            ["docker", "compose", "-f", str(self.compose_file), *args],
            cwd=config.REPO_ROOT,
            env={**os.environ, **self._compose_env()},
            input=input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
