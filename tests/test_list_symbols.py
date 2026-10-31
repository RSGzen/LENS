r"""Contract tests for the symbol-listing script (no imports of the package).

``scripts/list_symbols.py`` exists so the hand-maintained module index can be
diffed against the code in one command. These tests pin what "the surface" means:
top-level classes and functions, public names only, and no import side effects.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_list_symbols -v
"""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "list_symbols.py"


def _load_script():
    """Import the script as a module without adding ``scripts`` to ``sys.path``."""
    spec = importlib.util.spec_from_file_location("list_symbols", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ListSymbolsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.script = _load_script()

    def _run(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            code = self.script.main(list(argv))
        return code, out.getvalue()

    def test_it_lists_top_level_public_symbols(self) -> None:
        code, text = self._run("--package", "tools")

        self.assertEqual(code, 0)
        self.assertIn("lens.tools.retrieval:rerank_chunks", text)
        self.assertIn("lens.tools.evaluate:evaluate_stage", text)
        # Private helpers are not part of the surface the index names.
        self.assertNotIn("_format_evidence", text)

    def test_the_package_filter_narrows_the_output(self) -> None:
        _, everything = self._run()
        _, tools_only = self._run("--package", "tools")

        self.assertGreater(len(everything.splitlines()), len(tools_only.splitlines()))
        self.assertTrue(all("lens.tools" in line for line in tools_only.splitlines()))

    def test_an_unknown_package_is_an_error_not_empty_output(self) -> None:
        code, text = self._run("--package", "not_a_package")
        self.assertEqual(code, 2)
        self.assertEqual(text, "")

    def test_reading_the_package_imports_nothing(self) -> None:
        # The index is documentation; generating it must not load torch or open a DB.
        before = set(sys.modules)
        self.script.surface(
            self.script.PACKAGE_ROOT / "tools" / "evaluate.py", self.script.PACKAGE_ROOT
        )
        self.assertEqual({m for m in set(sys.modules) - before if m.startswith("lens")}, set())


if __name__ == "__main__":
    unittest.main()
