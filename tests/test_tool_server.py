r"""Contract tests for the host RPC server (no network, no key).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_tool_server -v
"""

from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
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
        self.server = ToolServer(FakeRuntime(), host="127.0.0.1", port=0, token="secret-token")

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


class ToolServerTransportTestCase(unittest.TestCase):
    """One real loopback socket: start(), a JSON round-trip, stop().

    The contracts above call ``handle_request`` directly, so they say nothing
    about ``_build_handler`` — the only code that can break between the socket and
    the route table. This is that coverage.
    """

    def setUp(self) -> None:
        self.server = ToolServer(FakeRuntime(), host="127.0.0.1", port=0, token="secret-token")
        self.server.start()
        self.addCleanup(self.server.stop)
        self.assertTrue(self.server.port, "port 0 must be replaced by the bound port")

    def _call(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        token: str | None = "secret-token",
        raw: bytes | None = None,
    ) -> tuple[int, dict]:
        data = (
            raw
            if raw is not None
            else (None if payload is None else json.dumps(payload).encode("utf-8"))
        )
        request = urllib.request.Request(self.server.url + path, data=data, method=method)
        if token is not None:
            request.add_header("Authorization", f"Bearer {token}")
        if data is not None:
            request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:  # 4xx/5xx raise by default
            return exc.code, json.loads(exc.read())

    def test_health_over_the_wire(self) -> None:
        status, body = self._call("GET", "/health", token=None)
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_bearer_header_is_required(self) -> None:
        status, body = self._call("GET", "/budget", token=None)
        self.assertEqual(status, 401)
        self.assertFalse(body["ok"])

    def test_malformed_json_is_400(self) -> None:
        status, _ = self._call("POST", "/tool", token="secret-token", raw=b"{not json")
        self.assertEqual(status, 400)

    def test_post_round_trip_dispatches(self) -> None:
        status, body = self._call("POST", "/tool", {"name": "echo", "args": {}})
        self.assertEqual(status, 200)
        self.assertEqual(body["endpoint"], "/tool")

    def test_query_string_is_not_part_of_the_route(self) -> None:
        status, _ = self._call("GET", "/health?probe=1", token=None)
        self.assertEqual(status, 200)

    def test_wrong_method_is_405(self) -> None:
        status, _ = self._call("PUT", "/tool", {"name": "echo", "args": {}})
        self.assertEqual(status, 405)

    def test_stop_is_idempotent(self) -> None:
        self.server.stop()
        self.server.stop()  # cleanup calls it a third time


if __name__ == "__main__":
    unittest.main()
