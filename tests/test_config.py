r"""Contract tests for path resolution (M8.0) — no key, no DB, no Docker.

The point is portability: nothing in `config.py` may be an absolute literal, and
a machine without the corpus must still be able to import the harness and run
`--help`. These tests pin the anchors, not a particular machine's layout.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_config -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lens import config


class StageThresholdsTestCase(unittest.TestCase):
    """``LENS_STAGE_THRESHOLDS`` parsing (LOG-113).

    Empty is the honest default (tier 1 skipped); a malformed entry must raise,
    because a typo that silently drops a threshold makes the gate *look* calibrated.
    """

    def test_default_is_empty(self) -> None:
        self.assertEqual(config._parse_stage_thresholds(""), {})

    def test_pairs_parse_with_whitespace(self) -> None:
        self.assertEqual(
            config._parse_stage_thresholds(" retrieval=0.72, kg=0.65 "),
            {"retrieval": 0.72, "kg": 0.65},
        )

    def test_malformed_entry_raises(self) -> None:
        with self.assertRaises(ValueError):
            config._parse_stage_thresholds("retrieval:0.7")
        with self.assertRaises(ValueError):
            config._parse_stage_thresholds("retrieval=high")


class PathAnchorTestCase(unittest.TestCase):
    def test_no_absolute_path_literals(self) -> None:
        # The portability bug this closes: five hard-coded ``r"D:\..."`` defaults.
        # A comment may *mention* a path as an example (that is documentation);
        # what must never exist is a raw-string default.
        source = Path(config.__file__).read_text(encoding="utf-8")
        self.assertNotIn('r"D:', source)

    def test_repo_root_resolves_to_the_checkout(self) -> None:
        # Derived from __file__, not from the cwd: launching from anywhere works.
        self.assertTrue((config.REPO_ROOT / "pyproject.toml").is_file())
        self.assertEqual(config.SANDBOX_COMPOSE_FILE, config.REPO_ROOT / "docker-compose.yaml")

    def test_data_root_override_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LENS_DATA_ROOT": tmp}):
                self.assertEqual(config._data_root(), Path(tmp))

    def test_derived_roots_live_under_the_data_root(self) -> None:
        # Every writable root must be relative to DATA_ROOT, never a literal.
        for path in (
            config.LENS_RUNS_ROOT_PATH,
            config.LENS_LOGS_ROOT_PATH,
            config.LENS_CACHE_ROOT_PATH,
            config.LENS_MANIFESTS_ROOT_PATH,
        ):
            self.assertIsInstance(path, Path)
            self.assertIn(config.DATA_ROOT, [path, *path.parents])

    def test_corpus_paths_have_no_default(self) -> None:
        # They are machine-specific and huge: absent unless configured, and their
        # absence must not break importing the harness.
        self.assertIsNone(config.ARXIV_METADATA_JSON_PATH)
        self.assertIsNone(config.GROBID_XML_DIR_PATH)


class RequirePathTestCase(unittest.TestCase):
    def test_missing_value_exits_with_the_env_var_name(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            config.require_path("LENS_GROBID_XML_DIR", None)
        self.assertIn("LENS_GROBID_XML_DIR", str(ctx.exception))

    def test_missing_directory_exits_with_the_path(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            config.require_path(
                "LENS_GROBID_XML_DIR", str(Path(tempfile.gettempdir()) / "nope-xyz")
            )
        self.assertIn("nope-xyz", str(ctx.exception))

    def test_valid_path_returns_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(config.require_path("X", tmp), Path(tmp))


if __name__ == "__main__":
    unittest.main()
