r"""Contract tests for the corpus builder (no network, no GROBID, no Docker).

The download and the conversion are injected, so every branch of the pipeline is
reachable offline: the resume decision, the download failure that must not abort a
batch, the responses GROBID actually returns, and the failures file that a retry
rewrites rather than grows.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_fetch -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import requests

from lens.dataset import fetch


class TargetTestCase(unittest.TestCase):
    def test_names_and_object_key(self) -> None:
        target = fetch.Target(arxiv_id="1501.00601", version="v3")

        self.assertEqual(target.pdf_name, "1501.00601v3.pdf")
        self.assertEqual(target.tei_name, "1501.00601v3.tei.xml")
        self.assertEqual(target.object_key, "arxiv/arxiv/pdf/1501/1501.00601v3.pdf")
        self.assertTrue(target.url.startswith("https://storage.googleapis.com/arxiv-dataset/"))

    def test_parse_accepts_both_shapes_and_rejects_junk(self) -> None:
        self.assertEqual(fetch.parse_target("1501/1501.00601v3.pdf").stem, "1501.00601v3")
        self.assertEqual(fetch.parse_target(" 1501.00601v3 \n").stem, "1501.00601v3")
        self.assertIsNone(fetch.parse_target("notes.txt"))
        self.assertIsNone(fetch.parse_target("1501.00601.pdf"))
        self.assertIsNone(fetch.parse_target(""))


class PlanActionTestCase(unittest.TestCase):
    def test_three_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "a.pdf"
            xml = Path(tmp) / "a.tei.xml"

            self.assertEqual(fetch.plan_action(pdf, xml), "download")
            pdf.write_text("pdf", encoding="utf-8")
            self.assertEqual(fetch.plan_action(pdf, xml), "process")
            xml.write_text("<TEI/>", encoding="utf-8")
            self.assertEqual(fetch.plan_action(pdf, xml), "skip")


class GrobidConvertTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.pdf = Path(self._tmp.name) / "1501.00601v3.pdf"
        self.pdf.write_bytes(b"%PDF-1.4")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _convert(self, responses, *, retries=3):
        queue = list(responses)

        def post(_path: Path) -> tuple[int, str]:
            item = queue.pop(0)

            if isinstance(item, Exception):
                raise item

            return item

        return fetch.grobid_convert(
            self.pdf,
            url="http://grobid:8070",
            retries=retries,
            post=post,
            sleep=lambda _seconds: None,
        )

    def test_200_returns_the_tei(self) -> None:
        tei, failure = self._convert([(200, "<TEI/>")])

        self.assertEqual(tei, "<TEI/>")
        self.assertIsNone(failure)

    def test_busy_then_success(self) -> None:
        tei, failure = self._convert([(503, ""), (200, "<TEI/>")])

        self.assertEqual(tei, "<TEI/>")
        self.assertIsNone(failure)

    def test_timeout_and_connection_error_are_retried(self) -> None:
        tei, failure = self._convert(
            [requests.exceptions.Timeout(), requests.exceptions.ConnectionError(), (200, "<TEI/>")]
        )

        self.assertEqual(tei, "<TEI/>")

    def test_hard_status_is_not_retried(self) -> None:
        calls: list[int] = []

        def post(_path: Path) -> tuple[int, str]:
            calls.append(1)
            return 400, "bad request"

        tei, failure = fetch.grobid_convert(
            self.pdf, url="http://grobid:8070", retries=3, post=post, sleep=lambda _s: None
        )

        self.assertIsNone(tei)
        self.assertEqual(failure, "status 400")
        self.assertEqual(len(calls), 1)

    def test_exhausted_retries_report_the_count(self) -> None:
        tei, failure = self._convert([(503, ""), (503, ""), (503, "")], retries=3)

        self.assertIsNone(tei)
        self.assertEqual(failure, "exhausted 3 attempts")

    def test_unexpected_exception_is_fatal_not_retried(self) -> None:
        tei, failure = self._convert([RuntimeError("boom")])

        self.assertIsNone(tei)
        self.assertTrue(failure.startswith(fetch.FATAL_EXCEPTION))


class BuildOneTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.xml_dir = self.root / "xml"
        self.pdf_dir = self.root / "pdf"
        self.target = fetch.Target(arxiv_id="1501.00601", version="v3")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self, *, fetch_bytes=None, post=None, keep_pdf=False):
        return fetch.build_one(
            self.target,
            xml_dir=self.xml_dir,
            pdf_dir=self.pdf_dir,
            grobid_url="http://grobid:8070",
            retries=3,
            keep_pdf=keep_pdf,
            fetch=fetch_bytes or (lambda _url: b"%PDF-1.4"),
            post=post or (lambda _path: (200, "<TEI/>")),
            sleep=lambda _seconds: None,
        )

    def test_success_writes_tei_and_burns_the_pdf(self) -> None:
        status, retry = self._build()

        self.assertEqual(status, fetch.SUCCESS_FIRST_TRY)
        self.assertIsNone(retry)
        self.assertEqual(
            (self.xml_dir / self.target.tei_name).read_text(encoding="utf-8"), "<TEI/>"
        )
        self.assertFalse((self.pdf_dir / self.target.pdf_name).exists())

    def test_keep_pdf_leaves_the_pdf_in_place(self) -> None:
        self._build(keep_pdf=True)

        self.assertTrue((self.pdf_dir / self.target.pdf_name).exists())

    def test_second_run_skips_a_finished_paper(self) -> None:
        self._build()

        status, _ = self._build(fetch_bytes=lambda _url: self.fail("must not download again"))

        self.assertEqual(status, fetch.ALREADY_PROCESSED)

    def test_leftover_pdf_is_converted_without_downloading(self) -> None:
        self.pdf_dir.mkdir(parents=True)
        (self.pdf_dir / self.target.pdf_name).write_bytes(b"%PDF-1.4")

        status, _ = self._build(fetch_bytes=lambda _url: self.fail("must not download again"))

        self.assertEqual(status, fetch.SUCCESS_AFTER_RETRY)

    def test_missing_upstream_is_not_a_failure_to_retry(self) -> None:
        def missing(_url: str) -> bytes:
            raise fetch.UpstreamMissing("gone")

        status, retry = self._build(fetch_bytes=missing)

        self.assertEqual(status, fetch.MISSING_UPSTREAM)
        self.assertIsNone(retry)

    def test_download_error_is_recorded_for_retry(self) -> None:
        def broken(_url: str) -> bytes:
            raise OSError("disk full")

        status, retry = self._build(fetch_bytes=broken)

        self.assertEqual(status, fetch.PERMANENT_FAILURE)
        self.assertEqual(retry, self.target)

    def test_conversion_failure_keeps_no_tei(self) -> None:
        status, retry = self._build(post=lambda _path: (400, ""))

        self.assertEqual(status, fetch.PERMANENT_FAILURE)
        self.assertEqual(retry, self.target)
        self.assertFalse((self.xml_dir / self.target.tei_name).exists())


class RunBatchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_tally_and_retry_list(self) -> None:
        targets = [fetch.Target(arxiv_id=f"1501.0060{index}", version="v1") for index in range(3)]

        def post(path: Path) -> tuple[int, str]:
            return (400, "") if path.stem.startswith("1501.00601") else (200, "<TEI/>")

        report = fetch.run_batch(
            targets,
            xml_dir=self.root / "xml",
            pdf_dir=self.root / "pdf",
            workers=2,
            fetch=lambda _url: b"%PDF-1.4",
            post=post,
            sleep=lambda _seconds: None,
        )

        self.assertEqual(report.tally[fetch.SUCCESS_FIRST_TRY], 2)
        self.assertEqual(report.tally[fetch.PERMANENT_FAILURE], 1)
        self.assertEqual([target.stem for target in report.failed], ["1501.00601v1"])
        self.assertIn("SUCCESS_FIRST_TRY", report.summary())

    def test_empty_batch_is_not_an_error(self) -> None:
        report = fetch.run_batch([], xml_dir=self.root / "xml", pdf_dir=self.root / "pdf")

        self.assertEqual(report.summary(), "nothing to do")


class FailuresFileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "failed_papers.txt"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_read_dedupes_and_keeps_order(self) -> None:
        self.path.write_text(
            "1501/1501.00601v3.pdf\n1501/1501.01179v1.pdf\n1501/1501.00601v3.pdf\n\n",
            encoding="utf-8",
        )

        stems = [target.stem for target in fetch.read_targets(self.path)]

        self.assertEqual(stems, ["1501.00601v3", "1501.01179v1"])

    def test_write_replaces_instead_of_appending(self) -> None:
        targets = [fetch.Target(arxiv_id="1501.00601", version="v3")]

        written = fetch.write_targets(self.path, targets)
        again = fetch.write_targets(self.path, targets)

        self.assertEqual((written, again), (1, 1))
        self.assertEqual(self.path.read_text(encoding="utf-8").count("\n"), 1)
        self.assertIn("1501/1501.00601v3.pdf", self.path.read_text(encoding="utf-8"))

    def test_empty_selection_removes_the_file(self) -> None:
        fetch.write_targets(self.path, [fetch.Target(arxiv_id="1501.00601", version="v3")])

        written = fetch.write_targets(self.path, [])

        self.assertEqual(written, 0)
        self.assertFalse(self.path.exists())


class ProbeMetadataTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "metadata.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_real_looking_dump_passes(self) -> None:
        self.path.write_text(
            json.dumps({"id": "1501.00601", "versions": [{"version": "v3"}]}) + "\n",
            encoding="utf-8",
        )

        fetch.probe_metadata(self.path)  # must not raise

    def test_json_array_is_refused_with_the_reason(self) -> None:
        self.path.write_text('[{"id": "1501.00601"}]', encoding="utf-8")

        with self.assertRaises(ValueError) as caught:
            fetch.probe_metadata(self.path)

        self.assertIn("does not look like the arXiv metadata dump", str(caught.exception))

    def test_empty_file_is_refused(self) -> None:
        self.path.write_text("\n\n", encoding="utf-8")

        with self.assertRaises(ValueError) as caught:
            fetch.probe_metadata(self.path)

        self.assertIn("no records", str(caught.exception))

    def test_unparseable_line_points_at_the_format(self) -> None:
        self.path.write_text("id,authors\n1501.00601,A\n", encoding="utf-8")

        with self.assertRaises(ValueError) as caught:
            fetch.probe_metadata(self.path)

        self.assertIn("not JSONL", str(caught.exception))


class IterTargetsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "metadata.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _record(
        self, arxiv_id: str, *, categories: str = "cs.AI cs.LG", version: str = "v2"
    ) -> str:
        return json.dumps(
            {
                "id": arxiv_id,
                "categories": categories,
                "versions": [{"version": "v1"}, {"version": version}],
            }
        )

    def test_filters_by_category_and_year_and_uses_the_latest_version(self) -> None:
        self.path.write_text(
            "\n".join(
                [
                    self._record("1501.00601"),
                    self._record("1501.01179", categories="cs.CL"),
                    self._record("1401.00001"),
                    self._record("2601.00002"),
                    json.dumps({"id": "1601.00003", "categories": "cs.AI", "versions": []}),
                    "not json",
                ]
            ),
            encoding="utf-8",
        )

        targets = list(fetch.iter_targets(self.path, category="cs.AI", years={"15", "16", "26"}))

        self.assertEqual([target.stem for target in targets], ["1501.00601v2", "2601.00002v2"])

    def test_limit_stops_early(self) -> None:
        self.path.write_text(
            "\n".join(self._record(f"1501.0000{index}") for index in range(5)), encoding="utf-8"
        )

        targets = list(fetch.iter_targets(self.path, category="cs.AI", years={"15"}, limit=2))

        self.assertEqual(len(targets), 2)


if __name__ == "__main__":
    unittest.main()
