r"""Contract tests for the host RPC server (no network, no key).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_tool_server -v
"""

from __future__ import annotations

import unittest
from typing import Any, Mapping

from lens.agent.tool_server import ToolServer


class FakeRuntime:
    def handle_rpc(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        # A dispatch failure (not a route failure): unknown paths are 404, and a
        # handler exception is a 500.
        if payload.get("name") == "boom":
            raise RuntimeError("boom")
        return {"ok": True, "endpoint": endpoint, "payload": dict(payload)}


class ToolServerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ToolServer(
            FakeRuntime(), host="127.0.0.1", port=0, token="secret-token"
        )

    def test_health_is_public(self) -> None:
        status, body = self.server.handle_request("GET", "/health", None, token=None)
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_tool_requires_token(self) -> None:
        status, body = self.server.handle_request(
            "POST", "/tool", {"name": "x", "args": {}}, token=None
        )
        self.assertEqual(status, 401)
        self.assertFalse(body["ok"])

    def test_wrong_token_is_rejected(self) -> None:
        status, _ = self.server.handle_request(
            "POST", "/tool", {"name": "x", "args": {}}, token="wrong"
        )
        self.assertEqual(status, 401)

    def test_tool_dispatches_to_runtime(self) -> None:
        status, body = self.server.handle_request(
            "POST",
            "/tool",
            {"name": "echo", "args": {"text": "hi"}},
            token="secret-token",
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["endpoint"], "/tool")
        self.assertEqual(body["payload"]["name"], "echo")

    def test_budget_is_get_only(self) -> None:
        status, _ = self.server.handle_request("GET", "/budget", None, token="secret-token")
        self.assertEqual(status, 200)
        status, _ = self.server.handle_request("POST", "/budget", {}, token="secret-token")
        self.assertEqual(status, 405)

    def test_unknown_path_is_404(self) -> None:
        status, _ = self.server.handle_request("GET", "/nope", None, token="secret-token")
        self.assertEqual(status, 404)

    def test_runtime_exception_becomes_500(self) -> None:
        status, body = self.server.handle_request(
            "POST", "/tool", {"name": "boom", "args": {}}, token="secret-token"
        )
        self.assertEqual(status, 500)
        self.assertFalse(body["ok"])
        self.assertNotIn("secret-token", str(body))


if __name__ == "__main__":
    unittest.main()
