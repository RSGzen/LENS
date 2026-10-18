"""Façade imported by the RLM's generated Python code.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a (façade API) and §4b.

Only these methods exist at depth 1::

    tool(name, **kwargs) -> Any
    sub_call(prompt, context=None) -> str
    budget() -> dict
    emit(value, label=None) -> dict
    note(text) -> None

There is **no** ``self_call`` / recursion here (LOG-13, A-01). Every method is
an RPC to the host; this module contains no tool logic and no credentials.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

# Environment set by ``docker compose`` at container start. The URL points at the
# host; the token is a per-run RPC token, not the OpenRouter key.
HOST_URL_ENV = "LENS_HOST_RPC_URL"
TOKEN_ENV = "LENS_RPC_TOKEN"

DEFAULT_TIMEOUT_S = 180.0


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
    """Return the host RPC base URL from the container environment."""
    return os.environ.get(HOST_URL_ENV, "http://host.docker.internal:8765").rstrip("/")


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
    """POST one JSON payload to the host and unwrap ``{"ok": true, "result": ...}``.

    Use ``httpx`` (baked into the image) lazily so the host can import this
    module for contract tests without the sandbox dependency installed. On a
    transport/HTTP error raise :class:`RLMError`; on ``ok: false`` raise
    :class:`RLMError` with the structured error body.
    """
    raise NotImplementedError("M8 scaffold: implement rlm_api._post")


def tool(name: str, **kwargs: Any) -> Any:
    """Call a registered host tool by name; return its result."""
    raise NotImplementedError("M8 scaffold: implement rlm_api.tool")


def sub_call(prompt: str, context: str | None = None) -> str:
    """Run a plain depth-1 GPT-5-mini call and return its text.

    No tools, no REPL, no recursion. The host performs the call and logs it.
    """
    raise NotImplementedError("M8 scaffold: implement rlm_api.sub_call")


def budget() -> dict[str, Any]:
    """Return the host's read-only budget snapshot."""
    raise NotImplementedError("M8 scaffold: implement rlm_api.budget")


def emit(value: Any, label: str | None = None) -> dict[str, Any]:
    """Offload ``value`` to the run workspace; return ``{"ref", "path", ...}``."""
    raise NotImplementedError("M8 scaffold: implement rlm_api.emit")


def note(text: str) -> None:
    """Append one free-form note to the trajectory."""
    raise NotImplementedError("M8 scaffold: implement rlm_api.note")


__all__ = ["RLMError", "budget", "emit", "note", "sub_call", "tool"]
