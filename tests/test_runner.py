r"""Contract tests for the passive executor (no Docker, no key).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_runner -v
"""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
