r"""Contract tests for the passive executor (no Docker, no key).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_runner -v
"""

from __future__ import annotations

import json
import socket
import threading
import unittest
import urllib.error
import urllib.request

from lens.sandbox import runner
from lens.sandbox.runner import Executor


class ExecutorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = Executor(stdout_max_chars=50)

    def test_namespace_persists_across_blocks(self) -> None:
        first = self.executor.run("x = 41")
        self.assertTrue(first["ok"])
        second = self.executor.run("print(x + 1)")
        self.assertTrue(second["ok"])
        self.assertEqual(second["stdout"].strip(), "42")

    def test_stdout_and_stderr_are_captured(self) -> None:
        result = self.executor.run("import sys\nprint('out')\nprint('err', file=sys.stderr)")
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"].strip(), "out")
        self.assertEqual(result["stderr"].strip(), "err")

    def test_exception_is_structured(self) -> None:
        result = self.executor.run("raise ValueError('bad')")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "ValueError")
        self.assertIn("bad", result["error"]["message"])

    def test_output_is_truncated(self) -> None:
        result = self.executor.run("print('y' * 200)")
        self.assertTrue(result["ok"])
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["stdout"]), 50)


class ExecutorGuardTestCase(unittest.TestCase):
    """Regressions: the paths the four contracts cannot see."""

    def setUp(self) -> None:
        self.executor = Executor()

    def test_oversized_code_is_a_structured_error(self) -> None:
        result = self.executor.run("x = 1\n" * runner.MAX_CODE_CHARS)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "CodeTooLarge")

    def test_system_exit_does_not_escape(self) -> None:
        # Not an Exception: without the explicit clause the runner dies and the
        # host sees a dead container.
        result = self.executor.run("raise SystemExit(3)")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "SystemExit")

    def test_untruncated_output_reports_false(self) -> None:
        self.assertFalse(self.executor.run("print('ok')")["truncated"])

    def test_timeout_is_structured(self) -> None:
        import signal

        if not hasattr(signal, "SIGALRM"):     # Windows host: no alarm
            self.skipTest("SIGALRM is Linux-only")

        from lens.sandbox import runner as runner_module

        original = runner_module.MAX_EXEC_BLOCK_S
        runner_module.MAX_EXEC_BLOCK_S = 0.2
        self.addCleanup(setattr, runner_module, "MAX_EXEC_BLOCK_S", original)

        result = self.executor.run("while True: pass")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "TimeoutError")
        self.assertIn("TimeoutError", result["stderr"])


class ExecHandlerTransportTestCase(unittest.TestCase):
    """The handler the contracts never touch: one real loopback socket."""

    def setUp(self) -> None:
        self.executor = Executor(stdout_max_chars=50)
        self.server = runner.make_server("127.0.0.1", 0, self.executor)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)

    def _stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _call(
        self, method: str, path: str, body: dict | None = None, raw: bytes | None = None
    ) -> tuple[int, dict]:
        data = raw if raw is not None else (
            None if body is None else json.dumps(body).encode("utf-8")
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method
        )
        if data is not None:
            request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health(self) -> None:
        status, body = self._call("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_exec_round_trip(self) -> None:
        status, body = self._call("POST", "/exec", {"code": "print(6 * 7)"})
        self.assertEqual(status, 200)
        self.assertEqual(body["stdout"].strip(), "42")
        self.assertFalse(body["truncated"])

    def test_namespace_persists_across_requests(self) -> None:
        self._call("POST", "/exec", {"code": "y = 5"})
        self.assertEqual(self._call("POST", "/exec", {"code": "print(y * 2)"})[1]["stdout"].strip(), "10")

    def test_exec_error_envelope(self) -> None:
        status, body = self._call("POST", "/exec", {"code": "raise ValueError('bad')"})
        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"]["type"], "ValueError")

    def test_bad_body_is_400(self) -> None:
        self.assertEqual(self._call("POST", "/exec", raw=b"{not json")[0], 400)
        self.assertEqual(self._call("POST", "/exec", {})[0], 400)

    def test_unknown_path_answers_once(self) -> None:
        # The draft matched on the path, then fell through and wrote a second
        # response: two responses on one connection corrupt the next request.
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
            sock.sendall(b"GET /nope HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            chunks = []
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                chunks.append(data)

        wire = b"".join(chunks)
        self.assertIn(b"404", wire.split(b"\r\n", 1)[0])
        self.assertNotIn(b"200 OK", wire)          # exactly one response


if __name__ == "__main__":
    unittest.main()
