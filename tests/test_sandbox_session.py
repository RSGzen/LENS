r"""Contract tests for the host-side sandbox session (no Docker, no key).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_sandbox_session -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lens.agent.sandbox_session import ExecResult, SandboxSession


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {"ok": True, "stdout": "2\n"}
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append(("GET", {"url": url, **kwargs}))
        return FakeResponse({"ok": True})

    def post(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append(("POST", {"url": url, **kwargs}))
        return FakeResponse(self.payload)


class SandboxSessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.fake = FakeSession()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_exec_result_to_dict(self) -> None:
        result = ExecResult(ok=True, stdout="hi", duration_ms=3)
        self.assertEqual(result.to_dict()["stdout"], "hi")

    def test_compose_env_contains_no_secrets(self) -> None:
        session = SandboxSession(
            "run-1", self.workspace, token="rpc-token", session=self.fake
        )
        env = session._compose_env()
        self.assertNotIn("OPENROUTER_API_KEY", env)
        self.assertNotIn("LENS_DB_PASSWORD", env)
        self.assertEqual(env["LENS_RPC_TOKEN"], "rpc-token")
        self.assertEqual(env["LENS_RUN_ID"], "run-1")
        self.assertIn(str(self.workspace), env["LENS_RUN_WORKSPACE"])

    def test_exec_posts_code_and_parses_result(self) -> None:
        session = SandboxSession(
            "run-1", self.workspace, token="rpc-token", session=self.fake
        )
        result = session.exec("print(1 + 1)")
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout, "2\n")
        payload = json.loads(self.fake.calls[-1][1]["data"])
        self.assertEqual(payload["code"], "print(1 + 1)")

    def test_health(self) -> None:
        session = SandboxSession(
            "run-1", self.workspace, token="rpc-token", session=self.fake
        )
        self.assertTrue(session.health())


if __name__ == "__main__":
    unittest.main()
