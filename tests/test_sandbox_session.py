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
        # Forward slashes: the value goes into a "<host>:/workspace" volume spec.
        self.assertEqual(Path(env["LENS_RUN_WORKSPACE"]).as_posix(), self.workspace.as_posix())

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


class SandboxSessionLifecycleTestCase(unittest.TestCase):
    """Compose wiring, without Docker: ``_run_compose`` is the seam."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.calls: list[list[str]] = []

        class Recorder:
            returncode = 0
            stdout = ""
            stderr = ""

        self.recorded = Recorder()
        self.session = SandboxSession(
            "run-1", self.workspace, token="rpc-token", session=FakeSession()
        )
        self.session._run_compose = lambda args: (self.calls.append(args) or self.recorded)
        self.addCleanup(self._tmp.cleanup)

    def test_compose_env_is_all_strings_and_uses_the_container_url(self) -> None:
        env = self.session._compose_env()
        self.assertTrue(all(isinstance(v, str) for v in env.values()))
        # Not the host-side 127.0.0.1 URL: the container cannot reach that.
        self.assertEqual(env["LENS_HOST_RPC_URL"], self.session.host_rpc_url)
        self.assertNotIn("127.0.0.1", env["LENS_HOST_RPC_URL"])
        self.assertEqual(env["LENS_MAX_EXEC_BLOCK_S"], str(self.session.exec_timeout_s))
        self.assertIn("LENS_SANDBOX_STDOUT_MAX_CHARS", env)

    def test_start_ups_force_recreate_then_waits(self) -> None:
        self.session.wait_until_healthy = lambda timeout_s=30.0: True
        self.session.start()
        self.assertEqual(
            self.calls,
            [["up", "-d", "--no-deps", "--force-recreate", "lens-repl"]],
        )

    def test_start_raises_and_stops_when_never_healthy(self) -> None:
        self.session.wait_until_healthy = lambda timeout_s=30.0: False
        with self.assertRaises(RuntimeError):
            self.session.start()
        self.assertEqual(self.calls[-1], ["rm", "-sf", "lens-repl"])

    def test_start_raises_on_a_compose_failure(self) -> None:
        self.recorded.returncode = 1
        self.recorded.stderr = "port is already allocated"
        with self.assertRaises(RuntimeError) as caught:
            self.session.start()
        self.assertIn("port is already allocated", str(caught.exception))

    def test_stop_removes_only_the_service(self) -> None:
        self.session.stop()
        self.assertEqual(self.calls, [["rm", "-sf", "lens-repl"]])

    def test_exec_http_error_is_structured(self) -> None:
        class FailingSession:
            def post(self, url, **kwargs):
                return FakeResponse({}, status_code=500)

        session = SandboxSession(
            "run-1", self.workspace, token="t", session=FailingSession()
        )
        result = session.exec("print(1)")
        self.assertFalse(result.ok)
        self.assertEqual(result.error["type"], "HTTPError")

    def test_exec_transport_error_is_structured(self) -> None:
        import requests

        class DeadSession:
            def post(self, url, **kwargs):
                raise requests.ConnectionError("refused")

        session = SandboxSession("run-1", self.workspace, token="t", session=DeadSession())
        result = session.exec("print(1)")
        self.assertFalse(result.ok)
        self.assertEqual(result.error["type"], "ConnectionError")


if __name__ == "__main__":
    unittest.main()
