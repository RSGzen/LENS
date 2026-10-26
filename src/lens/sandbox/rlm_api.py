"""Façade imported by the RLM's generated Python code.

Only these methods exist at depth 1::

    tool(name, **kwargs) -> Any
    sub_call(prompt, context=None) -> str
    budget() -> dict
    emit(value, label=None) -> dict
    note(text) -> None

There is **no** ``self_call`` / recursion here. Every method is
an RPC to the host; this module contains no tool logic and no credentials.

Also importable in the namespace: ``networkx``, ``numpy``, ``pandas``, and
``kg_query`` (the pure KG-traversal module — ``kg_neighbors`` / ``kg_triples`` /
``kg_subgraph``) for the graph returned by ``tool("build_kg", ...)``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

# Environment set by ``docker compose`` at container start. The URL points at the
# host; the token is a per-run RPC token, not the OpenRouter key.
HOST_URL_ENV = "LENS_HOST_RPC_URL"
TOKEN_ENV = "LENS_RPC_TOKEN"

DEFAULT_TIMEOUT_S = 180.0

#: endpoint -> HTTP method. Mirrors ``tool_server._ROUTES``: ``/budget`` is
#: GET-only (a POST is a 405), everything else takes a JSON body.
#:
#: Duplicated **on purpose**: the image ships this module on its own — there is no
#: ``lens`` package in ``lens-repl``, only ``runner.py``, ``rlm_api.py``
#: and ``kg_query.py`` — so importing the host's route table is a
#: ``ModuleNotFoundError`` inside the sandbox, i.e. the façade never loads.
_METHODS: dict[str, str] = {"/budget": "GET"}


class RLMError(RuntimeError):
    """A host RPC failed. ``status`` is the HTTP status when one was received."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error = dict(error or {})


def _host_url() -> str:
    """Return the host RPC base URL as the **container** sees it.

    ``LENS_HOST_RPC_URL`` is set by the compose environment from the relay's
    address (``config.SANDBOX_RELAY_URL``). The fallback is that same relay
    address: ``host.docker.internal`` is unreachable from an ``internal: true``
    bridge, so defaulting to it would turn a missing variable into a
    confusing "network unreachable" instead of a connection to the one peer the
    sandbox is allowed to have.
    """
    return os.environ.get(HOST_URL_ENV, "http://lens-relay:8090").rstrip("/")


def _token() -> str:
    """Return the per-run RPC token from the container environment."""
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise RLMError(f"{TOKEN_ENV} is not set in the sandbox environment")
    return token


def _post(
    endpoint: str,
    payload: Mapping[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """POST one JSON payload to the host and return the **parsed envelope**.

    Contract:
      - Return the host's JSON body unchanged: ``{"ok": true, "result": ...}``
        or ``{"ok": false, "error": {...}}``. Do **not** unwrap here — the
        callers pick out ``result`` and there is exactly one place that decides
        what a failure looks like.
      - Send ``Authorization: Bearer <token>`` from :func:`_token`.
      - On a transport/HTTP error raise :class:`RLMError`; on ``ok: false`` raise
        :class:`RLMError` carrying the structured error body.
      - Use ``httpx`` (baked into the image) **lazily** so the host can import
        this module for contract tests without the sandbox dependency installed.
    """
    import httpx

    method = _METHODS.get(endpoint, "POST")
    headers = {"Authorization": f"Bearer {_token()}"}

    try:
        response = httpx.request(
            method,
            _host_url() + endpoint,
            # No body on GET: the server does not read one for GET, so its bytes
            # would sit in the keep-alive socket and corrupt the next request.
            json=payload if method == "POST" else None,
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise RLMError(f"host RPC {endpoint} failed: {exc}") from exc

    try:
        body = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RLMError(
            f"host RPC {endpoint} returned non-JSON (HTTP {response.status_code})",
            status=response.status_code,
        ) from exc

    if not body.get("ok"):
        error = body.get("error") or {
            "type": "HostRPCError",
            "message": f"host RPC {endpoint} failed with HTTP {response.status_code}",
        }
        raise RLMError(
            error.get("message", "host RPC failed"),
            status=response.status_code,
            error=error,
        )

    return body


def tool(name: str, **kwargs: Any) -> Any:
    """Call a registered host tool by name; return its ``result``."""
    payload = {"name": name, "args": kwargs}

    return _post("/tool", payload)["result"]


def sub_call(prompt: str, context: str | None = None) -> str:
    """Run a plain depth-1 GPT-5-mini call and return its text.

    No tools, no REPL, no recursion. The host performs the call and logs it.
    """
    payload = {"prompt": prompt, "context": context}

    return _post("/sub_call", payload)["result"]


def budget() -> dict[str, Any]:
    """Return the host's read-only budget snapshot (the ``result`` field)."""
    payload = {}

    return _post("/budget", payload)["result"]


def emit(value: Any, label: str | None = None) -> dict[str, Any]:
    """Offload ``value`` to the run workspace.

    Returns the host's ``result`` dict::

        {"ref": "sha256:<hex>",   # content address (also the trajectory ref)
         "path": "/workspace/emits/<name>",   # what *you* can open()
         "host_path": "D:\\...",  # host-side copy; not visible here
         "bytes": 1234}

    Use ``path`` — ``host_path`` is meaningless inside the container.
    """
    payload = {"value": value, "label": label}

    return _post("/emit", payload)["result"]


def note(text: str) -> None:
    """Append one free-form note to the trajectory."""
    payload = {"text": text}

    return _post("/note", payload)["result"]


__all__ = ["RLMError", "budget", "emit", "note", "sub_call", "tool"]
