r"""Offline contract tests for the three sandbox-side modules that had none.

``sandbox/relay.py``, ``sandbox/exec_client.py`` and ``observability/run_logging.py``
were exercised only by ``scripts/verify_sandbox.py`` and live runs, so a refactor
could pass the whole suite and break every run. These tests use real loopback
sockets and a real temp log file: slower than a pure unit test, and the only way to
pin a transport.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_sandbox_offline -v
"""

from __future__ import annotations

import io
import json
import logging
import socket
import socketserver
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from lens import config
from lens.observability import run_logging
from lens.sandbox import exec_client, relay


class _EchoHandler(socketserver.BaseRequestHandler):
    """Upstream stand-in: sends every byte back."""

    def handle(self) -> None:
        while True:
            data = self.request.recv(65536)
            if not data:
                return
            self.request.sendall(data)


def _serve(server: socketserver.BaseServer) -> threading.Thread:
    # A 50 ms poll makes shutdown() fast: the default 0.5 s per server dominated
    # this file's runtime.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.daemon = True
    thread.start()
    return thread


class RelayTestCase(unittest.TestCase):
    """The relay forwards bytes to one target and does not accept a destination."""

    def setUp(self) -> None:
        self._saved = (relay.TARGET_HOST, relay.TARGET_PORT)
        self.upstream = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _EchoHandler)
        _serve(self.upstream)
        relay.TARGET_HOST = "127.0.0.1"
        relay.TARGET_PORT = self.upstream.server_address[1]

        self.relay = relay._Server(("127.0.0.1", 0), relay._Handler)
        _serve(self.relay)
        self.port = self.relay.server_address[1]

        self.addCleanup(self._stop)

    def _stop(self) -> None:
        for server in (self.relay, self.upstream):
            server.shutdown()
            server.server_close()
        relay.TARGET_HOST, relay.TARGET_PORT = self._saved

    def test_bytes_reach_the_target_and_come_back(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
            client.sendall(b"ping")
            self.assertEqual(client.recv(64), b"ping")

    def test_a_dead_target_closes_the_connection_instead_of_hanging(self) -> None:
        # The relay exists so a sandbox call fails fast; a wedged thread per dead
        # attempt would leak threads for the length of a run.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed_port = probe.getsockname()[1]
        relay.TARGET_PORT = closed_port

        started = time.monotonic()
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as client:
            client.sendall(b"ping")
            self.assertEqual(client.recv(64), b"")
        self.assertLess(time.monotonic() - started, 5)


class _RunnerHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for the in-container runner."""

    def log_message(self, *_args) -> None:  # silence the test output
        return

    def _send(self, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # /health
        self._send({"ok": True})

    def do_POST(self) -> None:  # /exec
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self._send({"ok": True, "stdout": f"ran: {body.get('code')}", "stderr": ""})


class ExecClientTestCase(unittest.TestCase):
    """``exec_client`` prints one JSON object and exits 0 whenever a body exists."""

    def setUp(self) -> None:
        self._saved = exec_client.RUNNER_URL
        self.server = HTTPServer(("127.0.0.1", 0), _RunnerHandler)
        _serve(self.server)
        exec_client.RUNNER_URL = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self._stop)

    def _stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        exec_client.RUNNER_URL = self._saved

    def _run(self, argv: list[str], stdin: str = "") -> tuple[int, dict]:
        out = io.StringIO()
        with (
            mock.patch.object(sys, "stdin", io.StringIO(stdin)),
            mock.patch.object(sys, "stdout", out),
        ):
            code = exec_client.main(argv)
        return code, json.loads(out.getvalue())

    def test_health(self) -> None:
        code, body = self._run(["exec_client.py", "health"])
        self.assertEqual(code, 0)
        self.assertTrue(body["ok"])

    def test_code_travels_on_stdin(self) -> None:
        code, body = self._run(["exec_client.py", "exec"], stdin="print(1)")
        self.assertEqual(code, 0)
        self.assertEqual(body["stdout"], "ran: print(1)")

    def test_a_transport_failure_is_a_body_not_an_exit_code(self) -> None:
        # The host maps a non-zero exit to a transport error; an unreachable runner
        # is a body with ok=false, so the model sees a diagnosis instead of a dead
        # container.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed_port = probe.getsockname()[1]
        exec_client.RUNNER_URL = f"http://127.0.0.1:{closed_port}"

        code, body = self._run(["exec_client.py", "health"])

        self.assertEqual(code, 0)
        self.assertFalse(body["ok"])
        self.assertIn("error", body)


class StreamLoggerTestCase(unittest.TestCase):
    """``StreamLogger`` forwards whole lines and buffers the rest."""

    def test_a_complete_line_is_logged_and_a_partial_one_buffered(self) -> None:
        logger = logging.getLogger("lens.test.stream")
        logger.setLevel(logging.INFO)
        records: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record.getMessage())
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)

        stream = run_logging.StreamLogger(logger)

        stream.write("first line\nhalf a li")
        self.assertEqual(records, ["first line"])

        stream.write("ne\n")
        self.assertEqual(records, ["first line", "half a line"])

    def test_flush_emits_the_remainder_without_a_newline(self) -> None:
        logger = logging.getLogger("lens.test.flush")
        logger.setLevel(logging.INFO)
        records: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record.getMessage())
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)

        stream = run_logging.StreamLogger(logger)
        stream.write("no newline here")
        self.assertEqual(records, [])

        stream.flush()
        self.assertEqual(records, ["no newline here"])

    def test_isatty_is_false_for_third_party_libraries(self) -> None:
        stream = run_logging.StreamLogger(logging.getLogger("lens.test.tty"))
        self.assertFalse(stream.isatty())
        self.assertTrue(stream.writable())


class SetupFileLoggingTestCase(unittest.TestCase):
    """``setup_file_logging`` mirrors ``print()`` into a timestamped file."""

    def test_prints_reach_the_log_file(self) -> None:
        saved = (sys.stdout, sys.stderr)
        self.addCleanup(setattr, sys, "stdout", saved[0])
        self.addCleanup(setattr, sys, "stderr", saved[1])

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(config, "LENS_LOGS_ROOT_PATH", Path(tmp)):
                logger = run_logging.setup_file_logging("lens.test.file", "unit")

                print("batch 1 of 2")
                sys.stderr.write("nothing to see\n")
                for handler in logger.handlers:
                    handler.flush()

                logs = list(Path(tmp).glob("unit_*.log"))
                self.assertEqual(len(logs), 1)

                text = logs[0].read_text(encoding="utf-8")
                self.assertIn("batch 1 of 2", text)
                self.assertIn("nothing to see", text)

                # Windows will not delete a file a handler still holds open.
                for handler in logger.handlers:
                    handler.close()
                logger.handlers.clear()


if __name__ == "__main__":
    unittest.main()
