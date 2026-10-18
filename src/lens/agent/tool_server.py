"""Host RPC server the sandbox calls back into.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a/§4b.

Only these endpoints cross the boundary::

    GET  /health       liveness (no token)
    POST /tool         {"name": str, "args": {...}}
    POST /sub_call     {"prompt": str, "context": str | null}
    GET  /budget       read-only budget snapshot
    POST /emit         {"value": any, "label": str | null}
    POST /note         {"text": str}

The sandbox holds no API key and no DB driver. The per-run token is the auth
boundary; the server never logs it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lens import config


class ToolServer:
    """Thin HTTP wrapper around ``runtime.handle_rpc(endpoint, payload)``.

    ``runtime`` is any object exposing::

        handle_rpc(endpoint: str, payload: Mapping[str, Any]) -> Mapping[str, Any]

    Keeping the transport here and the logic in the orchestrator lets the same
    handler be unit-tested without opening a socket.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        host: str = config.HOST_RPC_BIND,
        port: int = config.HOST_RPC_PORT,
        token: str,
    ) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self.token = token
        self._httpd: Any | None = None
        self._thread: Any | None = None

    @property
    def url(self) -> str:
        """Host-side URL (the sandbox uses ``host.docker.internal`` instead)."""
        return f"http://127.0.0.1:{self.port}"

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Bind and serve in a background thread. Port ``0`` selects a free port."""
        raise NotImplementedError("M8 scaffold: implement ToolServer.start")

    def stop(self) -> None:
        """Shut the server down and join the thread (idempotent)."""
        raise NotImplementedError("M8 scaffold: implement ToolServer.stop")

    # ------------------------------------------------------------ transport
    def handle_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        *,
        token: str | None,
    ) -> tuple[int, dict[str, Any]]:
        """Return ``(status_code, body)`` for one request.

        - ``/health`` is public.
        - Every other path requires ``Authorization: Bearer <token>``.
        - Unknown paths -> 404; wrong method -> 405; bad token -> 401.
        - Handler exceptions become ``{"ok": False, "error": {...}}`` with 500.
        """
        raise NotImplementedError("M8 scaffold: implement ToolServer.handle_request")

    def _build_handler(self) -> type:
        """Build the ``BaseHTTPRequestHandler`` subclass bound to this server."""
        raise NotImplementedError("M8 scaffold: implement ToolServer._build_handler")
