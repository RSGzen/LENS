r"""Contract tests for the corpus path resolver (no corpus, no database).

The resolver decides which directory the pipeline reads, and it is the one place
where "where did that value come from" is answerable. These tests pin the
precedence chain and the preflight that stops a load when the directory is empty.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_paths -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lens.dataset import paths


class ResolveCorpusTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for name in (
            "LENS_CORPUS_DIR",
            "LENS_GROBID_XML_DIR",
            "LENS_PDF_DIR",
            "LENS_ARXIV_METADATA",
        ):
            os.environ.pop(name, None)
        self._root = mock.patch.object(paths.config, "DATA_ROOT", Path("D:/data"))
        self._root.start()

    def tearDown(self) -> None:
        self._root.stop()
        self._env.stop()

    def test_defaults_live_under_the_data_root(self) -> None:
        resolved = paths.resolve_corpus()

        self.assertEqual(resolved.xml_dir, Path("D:/data/corpus/xml"))
        self.assertEqual(resolved.pdf_dir, Path("D:/data/corpus/pdf"))
        self.assertEqual(resolved.metadata, Path("D:/data/corpus/metadata.jsonl"))
        self.assertEqual(set(resolved.source.values()), {"default"})

    def test_command_line_beats_environment_beats_default(self) -> None:
        os.environ["LENS_GROBID_XML_DIR"] = "E:/env/xml"
        os.environ["LENS_CORPUS_DIR"] = "E:/root"

        from_env = paths.resolve_corpus()
        from_cli = paths.resolve_corpus(xml_dir="E:/cli/xml")

        self.assertEqual(from_env.xml_dir, Path("E:/env/xml"))
        self.assertEqual(from_env.source["xml_dir"], "env:LENS_GROBID_XML_DIR")
        self.assertEqual(from_cli.xml_dir, Path("E:/cli/xml"))
        self.assertEqual(from_cli.source["xml_dir"], "command line")
        # A role without a specific override still follows the shared root.
        self.assertEqual(from_env.pdf_dir, Path("E:/root/pdf"))
        self.assertEqual(from_env.source["pdf_dir"], "default")

    def test_corpus_root_moves_every_default(self) -> None:
        os.environ["LENS_CORPUS_DIR"] = "E:/root"

        resolved = paths.resolve_corpus()

        self.assertEqual(resolved.xml_dir, Path("E:/root/xml"))
        self.assertEqual(resolved.metadata, Path("E:/root/metadata.jsonl"))

    def test_describe_names_every_role_and_source(self) -> None:
        lines = paths.resolve_corpus(xml_dir="E:/xml").describe()

        self.assertEqual(len(lines), 3)
        self.assertIn("xml_dir = E:\\xml", lines[0])
        self.assertIn("command line", lines[0])


class RequireXmlCorpusTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _corpus(self, *names: str) -> paths.CorpusPaths:
        directory = self.root / "xml"
        directory.mkdir(parents=True, exist_ok=True)

        for name in names:
            (directory / name).write_text("<TEI/>", encoding="utf-8")

        return paths.resolve_corpus(xml_dir=directory, pdf_dir=self.root / "pdf")

    def test_missing_directory_is_reported_with_the_next_command(self) -> None:
        resolved = paths.resolve_corpus(xml_dir=self.root / "gone", pdf_dir=self.root / "pdf")

        with self.assertRaises(paths.CorpusError) as caught:
            paths.require_xml_corpus(resolved)

        message = " ".join([str(caught.exception), *caught.exception.details])
        self.assertIn("does not exist", message)
        self.assertIn("fetch_corpus.py fetch", message)

    def test_empty_directory_fails_the_preflight(self) -> None:
        resolved = self._corpus()

        with self.assertRaises(paths.CorpusError) as caught:
            paths.require_xml_corpus(resolved)

        self.assertIn("no .tei.xml files", str(caught.exception))

    def test_one_tei_file_is_enough_and_the_count_is_returned(self) -> None:
        resolved = self._corpus("1501.00601v3.tei.xml")

        directory, count = paths.require_xml_corpus(resolved)

        self.assertEqual(directory, self.root / "xml")
        self.assertEqual(count, 1)

    def test_nested_layout_is_found_when_the_flat_listing_is_empty(self) -> None:
        nested = self.root / "xml" / "1501"
        nested.mkdir(parents=True)
        (nested / "1501.00601v3.tei.xml").write_text("<TEI/>", encoding="utf-8")
        resolved = paths.resolve_corpus(xml_dir=self.root / "xml", pdf_dir=self.root / "pdf")

        _, count = paths.require_xml_corpus(resolved)

        self.assertEqual(count, 1)

    def test_other_files_do_not_count_as_corpus(self) -> None:
        resolved = self._corpus("notes.txt", "1501.00601v3.pdf")

        with self.assertRaises(paths.CorpusError):
            paths.require_xml_corpus(resolved)


if __name__ == "__main__":
    unittest.main()
