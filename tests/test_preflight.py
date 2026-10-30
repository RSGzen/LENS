r"""Contract tests for the environment preflight (no key, no database, no corpus).

The preflight exists so a missing setting fails once, with the variable named,
instead of surfacing as a confusing error inside a run. These tests pin what it
accepts, what it refuses, and the rule that a secret is never printed.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_preflight -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lens import preflight
from lens.dataset import paths

GOOD_KEY = "sk-or-v1-" + "a" * 40


class PreflightTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()

        for name in (
            "OPENROUTER_API_KEY",
            "LENS_DB_PASSWORD",
            "LENS_DB_RO_PASSWORD",
            "LENS_CORPUS_DIR",
            "LENS_GROBID_XML_DIR",
            "LENS_ARXIV_METADATA",
        ):
            os.environ.pop(name, None)

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._data_root = mock.patch.object(paths.config, "DATA_ROOT", self.root)
        self._data_root.start()
        self._db = mock.patch.multiple(
            paths.config,
            LENS_DB_PASSWORD=None,
            LENS_DB_RO_PASSWORD=None,
            LENS_DB_HOST="localhost",
            LENS_DB_PORT=5433,
            LENS_DB_NAME="lens",
            LENS_DB_USER="lens",
        )
        self._db.start()

    def tearDown(self) -> None:
        self._db.stop()
        self._data_root.stop()
        self._tmp.cleanup()
        self._env.stop()

    def _corpus(self, count: int = 1) -> Path:
        directory = self.root / "xml"
        directory.mkdir(parents=True, exist_ok=True)

        for index in range(count):
            (directory / f"1501.0000{index}v1.tei.xml").write_text("<TEI/>", encoding="utf-8")

        os.environ["LENS_GROBID_XML_DIR"] = str(directory)

        return directory

    def _metadata(self) -> Path:
        path = self.root / "metadata.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        os.environ["LENS_ARXIV_METADATA"] = str(path)

        return path

    # ------------------------------------------------------------- the key
    def test_missing_key_is_reported_with_the_variable_name(self) -> None:
        check = preflight.check_environment(need=(preflight.API_KEY,))[0]

        self.assertFalse(check.ok)
        self.assertIn("OPENROUTER_API_KEY", check.detail)
        self.assertIn(".env", check.hint)

    def test_placeholder_key_is_refused(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-..."

        check = preflight.check_environment(need=(preflight.API_KEY,))[0]

        self.assertFalse(check.ok)
        self.assertIn("placeholder", check.detail)

    def test_a_real_looking_key_passes_and_is_masked(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = GOOD_KEY

        check = preflight.check_environment(need=(preflight.API_KEY,))[0]

        self.assertTrue(check.ok)
        self.assertNotIn(GOOD_KEY, check.line())
        self.assertNotIn(GOOD_KEY[10:30], check.line())

    # ------------------------------------------------------------ database
    def test_both_db_passwords_are_required(self) -> None:
        os.environ["LENS_DB_PASSWORD"] = "admin"
        check = preflight.check_environment(need=(preflight.DB,))[0]

        self.assertFalse(check.ok)
        self.assertIn("LENS_DB_RO_PASSWORD", check.detail)

    def test_db_check_names_the_target(self) -> None:
        with mock.patch.multiple(
            paths.config, LENS_DB_PASSWORD="admin", LENS_DB_RO_PASSWORD="readonly"
        ):
            check = preflight.check_environment(need=(preflight.DB,))[0]

        self.assertTrue(check.ok)
        self.assertIn("lens@localhost:5433/lens", check.detail)
        self.assertNotIn("admin", check.line())

    # -------------------------------------------------------------- corpus
    def test_empty_corpus_fails_the_check(self) -> None:
        directory = self.root / "xml"
        directory.mkdir()
        os.environ["LENS_GROBID_XML_DIR"] = str(directory)

        check = preflight.check_environment(need=(preflight.CORPUS,))[0]

        self.assertFalse(check.ok)
        self.assertIn("fetch_corpus", check.hint)

    def test_corpus_count_is_reported(self) -> None:
        self._corpus(count=3)

        check = preflight.check_environment(need=(preflight.CORPUS,))[0]

        self.assertTrue(check.ok)
        self.assertIn("3 TEI", check.detail)

    # ------------------------------------------------------------ metadata
    def test_missing_metadata_points_at_kaggle_and_the_variable(self) -> None:
        check = preflight.check_environment(need=(preflight.METADATA,))[0]

        self.assertFalse(check.ok)
        self.assertIn("kaggle.com/datasets/Cornell-University/arxiv", check.hint)
        self.assertIn("LENS_ARXIV_METADATA", check.hint)

    def test_metadata_file_passes_with_its_size(self) -> None:
        self._metadata()

        check = preflight.check_environment(need=(preflight.METADATA,))[0]

        self.assertTrue(check.ok)
        self.assertIn("MB", check.detail)

    # ------------------------------------------------------- reporting shape
    def test_report_lists_env_file_and_every_requested_check(self) -> None:
        checks = preflight.check_environment(need=(preflight.API_KEY, preflight.DB))
        text = preflight.report(checks)

        self.assertIn(".env", text)
        self.assertIn("OPENROUTER_API_KEY", text)
        self.assertIn("LENS_DB_PASSWORD", text)
        self.assertIn("2 requirement(s) missing", text)

    def test_require_environment_exits_two_without_everything(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            preflight.require_environment(need=(preflight.API_KEY,))

        self.assertEqual(caught.exception.code, 2)

    def test_require_environment_is_silent_when_satisfied(self) -> None:
        os.environ["OPENROUTER_API_KEY"] = GOOD_KEY

        preflight.require_environment(need=(preflight.API_KEY,))  # must not raise

    def test_asking_for_nothing_checks_nothing(self) -> None:
        self.assertEqual(preflight.check_environment(need=()), [])
        self.assertTrue(set(preflight.ALL) >= {preflight.API_KEY, preflight.DB, preflight.CORPUS})


if __name__ == "__main__":
    unittest.main()
