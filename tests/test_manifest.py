r"""Contract tests for the M1 manifest builder (no key, no network, zero cost).

Fixtures are tiny fabricated JSONL + ``.tei.xml`` files in a temp dir, so the
real ~4.9 GB snapshot is never touched. They fail with ``NotImplementedError``
until the fill points are implemented.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lens import config
from lens.dataset.manifest import (
    build_manifest,
    checkpoint_logging,
    extract_manifest_entry,
    is_candidate,
    load_manifest,
)

# --- fixture metadata: cs.AI? / id-year 15-26? / has a TEI file? -----------------
RECORDS = [
    {"id": "1401.00004", "title": "too old", "authors": "A", "categories": "cs.AI",
     "update_date": "2014-12-31", "abstract": "a", "versions": [{"version": "v1"}]},
    {"id": "1501.00007", "title": "start edge", "authors": "B", "categories": "cs.AI cs.LG",
     "update_date": "2015-01-01", "abstract": "a", "versions": [{"version": "v1"}]},
    {"id": "1501.00601", "title": "one", "authors": "C", "categories": "cs.AI",
     "doi": "10.1/x", "journal-ref": "J.1", "update_date": "2016-05-05", "abstract": "a",
     "versions": [{"version": "v1"}, {"version": "v3"}]},
    {"id": "1601.00002", "title": "two", "authors": "D", "categories": "cs.AI",
     "update_date": "2015-06-06", "abstract": "a", "versions": [{"version": "v2"}]},
    {"id": "1701.00003", "title": "not ai", "authors": "E", "categories": "cs.LG cs.CV",
     "update_date": "2017-03-03", "abstract": "a", "versions": [{"version": "v1"}]},
    {"id": "1801.00005", "title": "missing xml", "authors": "F", "categories": "cs.AI",
     "update_date": "2018-01-01", "abstract": "a", "versions": [{"version": "v1"}]},
    {"id": "1901.00008", "title": "no versions", "authors": "G", "categories": "cs.AI",
     "update_date": "2019-01-01", "abstract": "a"},
    {"id": "2601.00006", "title": "end edge", "authors": "H", "categories": "cs.AI",
     "update_date": "2026-12-31", "abstract": "a", "versions": [{"version": "v1"}]},
    {"id": "2701.00009", "title": "too new", "authors": "I", "categories": "cs.AI",
     "update_date": "2027-01-01", "abstract": "a", "versions": [{"version": "v1"}]},
    {"id": "acc-phys/9411001", "title": "old style", "authors": "J", "categories": "cs.AI",
     "update_date": "1994-11-01", "abstract": "a", "versions": [{"version": "v1"}]},
]
# Only these ids have a TEI file on disk (plus one orphan not in metadata).
XML_NAMES = ["1501.00007v1.tei.xml", "1501.00601v3.tei.xml", "1601.00002v2.tei.xml",
             "2601.00006v1.tei.xml", "9999.99999v1.tei.xml"]
# The 4 entries build_manifest should keep (order = source order).
EXPECTED_IDS = ["1501.00007", "1501.00601", "1601.00002", "2601.00006"]


class ManifestTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.metadata = self.root / "metadata.jsonl"
        self.xml_dir = self.root / "xml"
        self.xml_dir.mkdir()
        self.manifest_path = self.root / "manifest.json"
        self.checkpoint_path = self.root / "manifest.checkpoint.json"

        self.metadata.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
        for name in XML_NAMES:
            (self.xml_dir / name).write_text("<TEI/>", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ------------------------------------------------------------- filtering
    def _cand(self, record: dict) -> bool:
        return is_candidate(record, config.TARGET_CATEGORY, config.PAPER_YEAR_RANGE)

    def test_is_candidate_category(self) -> None:
        self.assertTrue(self._cand(RECORDS[1]))  # "cs.AI cs.LG"
        self.assertFalse(self._cand(RECORDS[4]))  # "cs.LG cs.CV"

    def test_is_candidate_year_bounds_inclusive(self) -> None:
        self.assertTrue(self._cand(RECORDS[1]))  # year 15 (start)
        self.assertTrue(self._cand(RECORDS[7]))  # year 26 (end)
        self.assertFalse(self._cand(RECORDS[0]))  # year 14
        self.assertFalse(self._cand(RECORDS[8]))  # year 27

    def test_is_candidate_old_style_id_does_not_crash(self) -> None:
        self.assertFalse(self._cand(RECORDS[9]))  # "acc-phys/9411001"

    # ------------------------------------------------------------- extracting
    def test_extract_manifest_entry_fields(self) -> None:
        entry = extract_manifest_entry(RECORDS[2], "1501.00601v3.tei.xml")
        self.assertEqual(entry["arxiv_id"], "1501.00601")
        self.assertEqual(entry["version"], "v3")  # latest of versions
        self.assertEqual(entry["tei_xml"], "1501.00601v3.tei.xml")
        self.assertEqual(entry["categories"], ["cs.AI"])
        self.assertEqual(entry["authors"], "C")
        self.assertEqual(entry["arxiv_doi"], "10.1/x")
        self.assertEqual(entry["journal_ref"], "J.1")
        self.assertEqual(entry["title"], "one")
        self.assertEqual(entry["update_date"], "2016-05-05")

    # ---------------------------------------------------------------- build
    def test_build_manifest_end_to_end(self) -> None:
        manifest = build_manifest(self.metadata, self.xml_dir, self.manifest_path,
                                  checkpoint_path=self.checkpoint_path)

        self.assertEqual(manifest["count"], len(EXPECTED_IDS))
        self.assertEqual([e["arxiv_id"] for e in manifest["papers"]], EXPECTED_IDS)
        for key in ("generated_at", "target_category", "year_range", "metadata_source", "xml_dir"):
            self.assertIn(key, manifest)

        # Every kept entry resolves to a real TEI file on disk.
        for entry in manifest["papers"]:
            self.assertTrue((self.xml_dir / entry["tei_xml"]).exists())

        # Checkpoint is JSONL (history retained); the last line is the final state.
        lines = self.checkpoint_path.read_text(encoding="utf-8").strip().splitlines()
        state = json.loads(lines[-1])
        self.assertTrue(state["done"])

        # Round-trips through disk and reloads identically.
        self.assertEqual(load_manifest(self.manifest_path), manifest)

    def test_checkpoint_logging_retains_history(self) -> None:
        checkpoint_logging(self.checkpoint_path, 10, 1)
        checkpoint_logging(self.checkpoint_path, 20, 2)
        checkpoint_logging(self.checkpoint_path, 30, 3, done_flag=True)
        lines = self.checkpoint_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)  # append-only history, one JSON per line
        self.assertFalse(json.loads(lines[0])["done"])
        self.assertTrue(json.loads(lines[-1])["done"])

    def test_build_manifest_skips_missing_xml_and_versionless(self) -> None:
        manifest = build_manifest(self.metadata, self.xml_dir, self.manifest_path,
                                  checkpoint_path=self.checkpoint_path)
        kept = {e["arxiv_id"] for e in manifest["papers"]}
        self.assertNotIn("1801.00005", kept)  # cs.AI in range but no TEI file
        self.assertNotIn("1901.00008", kept)  # no "versions" -> cannot form a filename


if __name__ == "__main__":
    unittest.main()
