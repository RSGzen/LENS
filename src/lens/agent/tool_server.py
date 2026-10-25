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

Transport
---------
JSON over HTTP/1.1, one bearer header, one request at a time per connection.
:meth:`ToolServer.handle_request` is a **pure function of (method, path, payload,
token)** returning ``(status, dict)`` — every response is a JSON object, never an
HTML error page, because ``rlm_api._post`` parses the body unconditionally.

``http.server`` is used rather than a raw socket on purpose: request framing
(``Content-Length``, keep-alive, a malformed request line) is the stdlib's job,
and re-implementing it would add a few hundred lines of bug surface for one
loopback client. ``ThreadingHTTPServer`` (not ``HTTPServer``) keeps a long tool
call from blocking the ``/health`` probe.
"""

from __future__ import annotations

import hmac
import json
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from lens import config

#: path -> the one method it accepts. A POST-only endpoint answering GET is a
#: **405**, which is why this is a table and not an ``if``/``elif`` chain: the
#: method check falls out of membership instead of being restated per branch.
_ROUTES: dict[str, frozenset[str]] = {
    "/tool": frozenset({"POST"}),
    "/sub_call": frozenset({"POST"}),
    "/budget": frozenset({"GET"}),
    "/emit": frozenset({"POST"}),
    "/note": frozenset({"POST"}),
}

#: Endpoints whose body must be a JSON object. ``GET /budget`` has no body.
_BODYLESS = frozenset({"/health", "/budget"})


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

        # Fail closed: an empty token would make a missing header "match".
        if not token:
            raise ValueError("ToolServer requires a non-empty RPC token")

    @property
    def url(self) -> str:
        """Host-side URL. The container never uses it — it dials the relay."""
        return f"http://127.0.0.1:{self.port}"

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Bind and serve in a background thread. Port ``0`` selects a free port.

        After binding, ``self.port`` holds the **actual** port. The relay forwards to
        it: ``run.py`` sets ``sandbox.host_rpc_port = server.port`` and the compose
        environment carries it as ``LENS_RELAY_TARGET_PORT``. The relay's own
        address is the constant ``config.SANDBOX_RELAY_URL``, which is what the
        container's ``LENS_HOST_RPC_URL`` holds — a random port never has to be
        written into the compose file.
        """
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        self.port = self._httpd.server_address[1]

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Shut the server down and join the thread (idempotent).

        Three steps, in order: ``shutdown()`` wakes ``serve_forever`` so it stops
        accepting, ``server_close()`` releases the socket (skip it and the port
        leaks until the process exits — painful when ``run.py`` starts several
        runs in one session), then ``join`` makes sure the thread is really gone.
        Never started -> every branch is skipped, no exception.
        """
        if isinstance(self._httpd, ThreadingHTTPServer):
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

        if isinstance(self._thread, threading.Thread):
            self._thread.join(timeout=5)
            self._thread = None

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

        The route is resolved **here**, not by calling back into this server over
        HTTP: this *is* the server. The payload goes to
        ``runtime.handle_rpc(path, payload)``, which owns every endpoint's
        semantics; the transport only decides status codes.

        Order matters: public route, then auth (**401 before knowing whether the
        path exists** — an unauthenticated caller learns nothing about the
        surface), then 404, then 405, then 400 for a missing body.
        """
        if method == "GET" and path == "/health":
            return 200, {"ok": True}

        # compare_digest on bytes: a non-ASCII token must be a 401, not a
        # TypeError from the str-only overload.
        if not token or not hmac.compare_digest(
            token.encode("utf-8"), self.token.encode("utf-8")
        ):
            return 401, {
                "ok": False,
                "error": {"type": "Unauthorized", "message": "bad or missing token"},
            }

        allowed = _ROUTES.get(path)
        if allowed is None:
            return 404, {
                "ok": False,
                "error": {"type": "NotFound", "message": f"unknown endpoint {path!r}"},
            }

        if method not in allowed:
            return 405, {
                "ok": False,
                "error": {
                    "type": "MethodNotAllowed",
                    "message": f"{path} accepts {', '.join(sorted(allowed))}",
                },
            }

        if path not in _BODYLESS and not isinstance(payload, Mapping):
            return 400, {
                "ok": False,
                "error": {"type": "BadRequest", "message": f"{path} needs a JSON object"},
            }

        try:
            return 200, dict(self.runtime.handle_rpc(path, payload or {}))
        except Exception as exc:  # noqa: BLE001 - a handler failure is a 500, not a crash
            return 500, {
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)[:300]},
            }

    def _build_handler(self) -> type:
        """Build the ``BaseHTTPRequestHandler`` subclass bound to this server.

        Called once per :meth:`start`. It returns a **class** (not an instance):
        ``ThreadingHTTPServer`` instantiates it once per accepted connection, so
        all per-request state lives on the handler instance and everything shared
        (the runtime, the token, the route table) is captured by the closure.

        Each ``do_<VERB>`` does the same three things: parse the body, ask
        :meth:`handle_request` for ``(status, body)``, write it as JSON. No route
        logic here — that is the point of the split, and it is why this class can
        stay identical for every endpoint.
        """
        server = self  # the handler needs the *server*; its own ``self`` is the request

        class _RPCHandler(BaseHTTPRequestHandler):
            # HTTP/1.1 -> keep-alive, so an exact Content-Length is mandatory on
            # every response (HTTP/1.0 semantics would drop the connection each
            # call, adding a TCP handshake to every tool dispatch).
            protocol_version = "HTTP/1.1"
            # A client that connects and stalls must not pin a worker thread.
            timeout = 300

            def log_message(self, format: str, *args: Any) -> None:
                # Default writes every request line to stderr, which would
                # interleave with the agent loop's CLI output. Silent on purpose:
                # the trajectory is the record, not stderr.
                return

            def _send(self, status: int, body: Mapping[str, Any]) -> None:
                """Write one JSON response with an exact ``Content-Length``.

                ``default=str`` because a tool result may carry a DB type
                (``date`` / ``Decimal`` / ``UUID``). The registry already narrows
                the blob with ``str``; the envelope must not be the weaker link, or
                a serialisation error becomes a 500 the model reads as a transport
                failure (M9 smoke, LOG-115).
                """
                data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _bearer(self) -> str | None:
                """Return the token from ``Authorization: Bearer <token>``, or None."""
                scheme, _, value = (self.headers.get("Authorization") or "").partition(" ")
                return value if scheme.lower() == "bearer" and value else None

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

            def _dispatch(self, method: str) -> None:
                ok, payload = (True, None) if method == "GET" else self._read_json()

                if not ok:
                    self._send(
                        400,
                        {
                            "ok": False,
                            "error": {"type": "BadRequest", "message": "body is not valid JSON"},
                        },
                    )
                    return

                # Strip the query string: an endpoint is a path, and tool
                # arguments travel in the body, never in the URL.
                status, body = server.handle_request(
                    method,
                    self.path.split("?", 1)[0],
                    payload,
                    token=self._bearer(),
                )
                self._send(status, body)

            def do_GET(self) -> None:
                self._dispatch("GET")

            def do_POST(self) -> None:
                self._dispatch("POST")

            def do_PUT(self) -> None:
                self._dispatch("PUT")

            def do_DELETE(self) -> None:
                self._dispatch("DELETE")

            def do_PATCH(self) -> None:
                self._dispatch("PATCH")

        return _RPCHandler
