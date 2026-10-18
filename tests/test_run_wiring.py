r"""Wiring tests for the CLI entry point (no key, no Docker, zero cost).

``run.py`` is wiring only, which is exactly where the port/token/workspace bugs
live, so it gets a seam instead of an integration test: ``main`` must build
everything through ``build_runtime`` and clean up unconditionally.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_run_wiring -v
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import lens.run as run_module
from lens import config
from lens.agent.runtime import RunResult


class FakeRuntime:
    """Stands in for :class:`lens.agent.runtime.Runtime`."""

    def __init__(self, *, exc: Exception | None = None) -> None:
        self.exc = exc
        self.ran = False
        self.run_dir = Path(tempfile.gettempdir()) / "lens-wiring-run"

    def run(self) -> RunResult:
        self.ran = True
        if self.exc is not None:
            raise self.exc
        return RunResult(
            answer="The grounded answer [c1].",
            partial=False,
            stop_reason=None,
            iterations=3,
            cost_usd=0.1234,
            trajectory_path=self.run_dir / "trajectory.jsonl",
            tasks_path=self.run_dir / "tasks.json",
            answer_path=self.run_dir / "answer.md",
        )


class FakeServer:
    def __init__(self) -> None:
        self.stopped = False
        self.port = 0

    def stop(self) -> None:
        self.stopped = True


class FakeSandbox:
    port = 0


class ParserTestCase(unittest.TestCase):
    def test_defaults_follow_config(self) -> None:
        args = run_module.build_parser().parse_args(["a query"])
        self.assertEqual(args.recorded, config.LENS_RECORDED)
        self.assertEqual(args.allow_sql, config.ALLOW_SQL)
        self.assertEqual(args.query, "a query")

    def test_flags_can_be_turned_off_as_well_as_on(self) -> None:
        # BooleanOptionalAction: an env-set flag must be overridable per run.
        parser = run_module.build_parser()
        self.assertFalse(parser.parse_args(["q", "--no-recorded"]).recorded)
        self.assertTrue(parser.parse_args(["q", "--recorded"]).recorded)
        self.assertFalse(parser.parse_args(["q", "--no-allow-sql"]).allow_sql)
        self.assertTrue(parser.parse_args(["q", "--allow-sql"]).allow_sql)


class MainWiringTestCase(unittest.TestCase):
    def _invoke(self, runtime: FakeRuntime, argv: list[str]) -> int:
        self.server = FakeServer()
        sandbox = FakeSandbox()
        self.out = io.StringIO()
        with patch.object(
            run_module, "build_runtime", return_value=(runtime, self.server, sandbox)
        ) as build:
            self.build_call = build
            with redirect_stdout(self.out):
                self.code = run_module.main(argv)
        return self.code

    def test_main_runs_the_loop_and_prints_the_answer(self) -> None:
        runtime = FakeRuntime()
        code = self._invoke(runtime, ["how do RLMs aggregate?", "--run-id", "run-x"])
        self.assertEqual(code, 0)
        self.assertTrue(runtime.ran)
        self.assertIn("The grounded answer", self.out.getvalue())
        self.assertIn("0.1234", self.out.getvalue())
        self.assertTrue(self.server.stopped)

    def test_main_passes_the_parsed_flags_and_run_id_to_the_wiring(self) -> None:
        self._invoke(FakeRuntime(), ["the query", "--run-id", "run-x", "--recorded", "--allow-sql"])
        args, kwargs = self.build_call.call_args
        self.assertEqual(args[1], "run-x")
        self.assertIs(kwargs["recorded"], True)
        self.assertIs(kwargs["allow_sql"], True)

    def test_server_is_stopped_even_when_the_run_raises(self) -> None:
        runtime = FakeRuntime(exc=RuntimeError("loop blew up"))
        with self.assertRaises(RuntimeError):
            self._invoke(runtime, ["q"])
        self.assertTrue(runtime.ran)
        self.assertTrue(self.server.stopped)

    def test_run_id_defaults_to_a_timestamp(self) -> None:
        self._invoke(FakeRuntime(), ["q"])
        run_id = self.build_call.call_args.args[1]
        self.assertTrue(run_id.startswith("run-"), run_id)


if __name__ == "__main__":
    unittest.main()
