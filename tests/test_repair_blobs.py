r"""Tests for the one-off blob repair - no key, no DB, no Docker.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_repair_blobs -v
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_blobs.py"
_spec = importlib.util.spec_from_file_location("repair_blobs", _SCRIPT)
repair_blobs = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(repair_blobs)


class RepairBlobsTestCase(unittest.TestCase):
    def test_repairs_crlf_blobs_and_leaves_good_ones_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blobs = Path(tmp)

            good_text = "line one\nline two"
            good = hashlib.sha256(good_text.encode("utf-8")).hexdigest()
            (blobs / good).write_bytes(good_text.encode("utf-8"))

            legacy_text = "alpha\nbeta"
            legacy = hashlib.sha256(legacy_text.encode("utf-8")).hexdigest()
            (blobs / legacy).write_bytes(b"alpha\r\nbeta")  # what write_text produced

            counts = repair_blobs.repair(blobs, dry_run=False)

            self.assertEqual(counts, {"ok": 1, "repaired": 1, "corrupt": 0})
            self.assertEqual((blobs / legacy).read_bytes(), legacy_text.encode("utf-8"))
            # Idempotent: a second pass finds nothing to do.
            self.assertEqual(
                repair_blobs.repair(blobs, dry_run=False), {"ok": 2, "repaired": 0, "corrupt": 0}
            )

    def test_a_genuinely_corrupt_blob_is_reported_not_guessed(self) -> None:
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            blobs = Path(tmp)
            name = hashlib.sha256(b"the real payload").hexdigest()
            (blobs / name).write_bytes(b"something else entirely")

            # The reporter prints the corrupt name; keep the suite output clean.
            with contextlib.redirect_stdout(io.StringIO()):
                counts = repair_blobs.repair(blobs, dry_run=False)

            self.assertEqual(counts["corrupt"], 1)
            self.assertEqual((blobs / name).read_bytes(), b"something else entirely")

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blobs = Path(tmp)
            legacy = hashlib.sha256(b"a\nb").hexdigest()
            (blobs / legacy).write_bytes(b"a\r\nb")

            counts = repair_blobs.repair(blobs, dry_run=True)

            self.assertEqual(counts["repaired"], 1)
            self.assertEqual((blobs / legacy).read_bytes(), b"a\r\nb")


if __name__ == "__main__":
    unittest.main()
