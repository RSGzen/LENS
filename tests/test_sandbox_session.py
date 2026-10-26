r"""Contract tests for the host-side sandbox session (no Docker, no key).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_sandbox_session -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lens.agent.sandbox_session import ExecResult, SandboxSession


class FakeCompose:
    """Stand-in for ``_run_compose``: records the call, returns a canned result."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, args: list[str], **kwargs) -> "FakeCompose":
        self.calls.append((args, kwargs))
        return self

    @property
    def last_args(self) -> list[str]:
        return self.calls[-1][0]


class SandboxSessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_exec_result_to_dict(self) -> None:
        result = ExecResult(ok=True, stdout="hi", duration_ms=3)
        self.assertEqual(result.to_dict()["stdout"], "hi")
        self.assertIn("stderr", result.to_dict())

    def test_compose_env_contains_no_secrets(self) -> None:
        session = SandboxSession("run-1", self.workspace, token="rpc-token")
        env = session._compose_env()
        self.assertNotIn("OPENROUTER_API_KEY", env)
        self.assertNotIn("LENS_DB_PASSWORD", env)
        self.assertEqual(env["LENS_RPC_TOKEN"], "rpc-token")
        self.assertEqual(env["LENS_RUN_ID"], "run-1")
        # Forward slashes: the value goes into a "<host>:/workspace" volume spec.
        self.assertEqual(Path(env["LENS_RUN_WORKSPACE"]).as_posix(), self.workspace.as_posix())

    def test_exec_sends_code_on_stdin_and_parses_result(self) -> None:
        envelope = {"ok": True, "stdout": "2\n", "stderr": "", "error": None}
        fake = FakeCompose(stdout=json.dumps(envelope))
        session = SandboxSession("run-1", self.workspace, token="t")
        session._run_compose = fake

        result = session.exec("print(1 + 1)")

        self.assertTrue(result.ok)
        self.assertEqual(result.stdout, "2\n")
        self.assertEqual(fake.calls[-1][1]["input"], "print(1 + 1)")
        self.assertEqual(
            fake.last_args,
            ["exec", "-T", "lens-repl", "python", "/app/exec_client.py", "exec"],
        )

    def test_exec_block_failure_is_not_a_transport_error(self) -> None:
        envelope = {"ok": False, "error": {"type": "ValueError", "message": "bad"}}
        session = SandboxSession("run-1", self.workspace, token="t")
        session._run_compose = FakeCompose(stdout=json.dumps(envelope))

        result = session.exec("raise ValueError('bad')")

        self.assertFalse(result.ok)
        self.assertEqual(result.error["type"], "ValueError")

    def test_exec_transport_failure_is_structured(self) -> None:
        session = SandboxSession("run-1", self.workspace, token="t")
        session._run_compose = FakeCompose(returncode=1, stderr="no such container")

        result = session.exec("print(1)")

        self.assertFalse(result.ok)
        self.assertEqual(result.error["type"], "ContainerExecError")
        self.assertIn("no such container", result.error["message"])

    def test_health(self) -> None:
        session = SandboxSession("run-1", self.workspace, token="t")
        session._run_compose = FakeCompose(stdout='{"ok": true}')
        self.assertTrue(session.health())

        session._run_compose = FakeCompose(returncode=1)
        self.assertFalse(session.health())


class SandboxSessionLifecycleTestCase(unittest.TestCase):
    """Compose wiring, without Docker: ``_run_compose`` is the seam."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.fake = FakeCompose()
        self.session = SandboxSession("run-1", self.workspace, token="rpc-token")
        self.session._run_compose = self.fake
        self.addCleanup(self._tmp.cleanup)

    def test_compose_env_is_all_strings_and_uses_the_container_url(self) -> None:
        env = self.session._compose_env()
        self.assertTrue(all(isinstance(v, str) for v in env.values()))
        # Not the host-side 127.0.0.1 URL: the container cannot reach that.
        self.assertEqual(env["LENS_HOST_RPC_URL"], self.session.host_rpc_url)
        self.assertNotIn("127.0.0.1", env["LENS_HOST_RPC_URL"])
        self.assertEqual(env["LENS_MAX_EXEC_BLOCK_S"], str(self.session.exec_timeout_s))
        self.assertIn("LENS_SANDBOX_STDOUT_MAX_CHARS", env)
        # The relay target is the port the RPC server actually bound.
        self.assertEqual(env["LENS_RELAY_TARGET_PORT"], str(self.session.host_rpc_port))

    def test_start_ups_force_recreate_then_waits(self) -> None:
        self.session.wait_until_healthy = lambda timeout_s=30.0: True
        self.session.start()
        # The relay comes with the sandbox: it is the only route to the host RPC
        # server, and a stale relay target port would break every tool call.
        self.assertEqual(
            [args for args, _ in self.fake.calls],
            [
                [
                    "up",
                    "-d",
                    "--no-deps",
                    "--force-recreate",
                    "lens-relay",
                    "lens-repl",
                ]
            ],
        )

    def test_start_raises_and_stops_when_never_healthy(self) -> None:
        self.session.wait_until_healthy = lambda timeout_s=30.0: False
        with self.assertRaises(RuntimeError):
            self.session.start()
        self.assertEqual(self.fake.last_args, ["rm", "-sf", "lens-repl", "lens-relay"])

    def test_start_raises_on_a_compose_failure(self) -> None:
        self.fake.returncode = 1
        self.fake.stderr = "port is already allocated"
        with self.assertRaises(RuntimeError) as caught:
            self.session.start()
        self.assertIn("port is already allocated", str(caught.exception))

    def test_stop_removes_both_containers(self) -> None:
        self.session.stop()
        self.assertEqual(self.fake.last_args, ["rm", "-sf", "lens-repl", "lens-relay"])


class ExecClientTestCase(unittest.TestCase):
    """The in-container client: stdin in, one JSON object out (option C)."""

    def test_exec_uses_utf8_on_the_transport(self) -> None:
        # With text=True alone, Python encodes stdin with the locale (cp1252 on
        # Windows): an em dash the model wrote arrives as 0x97, the container
        # decodes a lone surrogate, and exec() dies. One character, one lost turn.
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        captured: dict = {}

        def fake_run(command, **kwargs):
            captured.update(kwargs)
            captured["command"] = command

            class Done:
                returncode = 0
                stdout = '{"ok": true, "stdout": "2\\n"}'
                stderr = ""

            return Done()

        session = SandboxSession("run-1", Path(tempfile.mkdtemp()), token="t")

        with patch("lens.agent.sandbox_session.subprocess.run", fake_run):
            session.exec("print('\u2014')")

        self.assertEqual(captured["encoding"], "utf-8")
        self.assertTrue(captured["text"])

    def test_exec_posts_the_code_from_stdin(self) -> None:
        import io
        from unittest.mock import patch

        from lens.sandbox import exec_client

        captured: dict = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self) -> bytes:
                return b'{"ok": true, "stdout": "2\\n"}'

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["body"] = request.data
            captured["method"] = request.get_method()
            return Response()

        with (
            patch("urllib.request.urlopen", fake_urlopen),
            patch("sys.stdin", io.StringIO("print(1 + 1)")),
            patch("sys.stdout", new_callable=io.StringIO) as out,
        ):
            self.assertEqual(exec_client.main(["exec_client.py", "exec"]), 0)

        self.assertEqual(captured["method"], "POST")
        self.assertTrue(captured["url"].endswith("/exec"))
        self.assertEqual(json.loads(captured["body"]), {"code": "print(1 + 1)"})
        self.assertTrue(json.loads(out.getvalue())["ok"])

    def test_transport_failure_still_prints_an_envelope(self) -> None:
        import io
        from unittest.mock import patch

        from lens.sandbox import exec_client

        with (
            patch("urllib.request.urlopen", side_effect=OSError("refused")),
            patch("sys.stdin", io.StringIO("print(1)")),
            patch("sys.stdout", new_callable=io.StringIO) as out,
        ):
            self.assertEqual(exec_client.main(["exec_client.py", "exec"]), 0)

        body = json.loads(out.getvalue())
        self.assertFalse(body["ok"])
        self.assertIn("refused", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
