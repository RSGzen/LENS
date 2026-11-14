r"""Contract tests for the OOLONG bridge converter (no network, no key).

The conversion is the one place where the release's text could be altered, so the
tests pin what must survive: every non-empty line, no label fragment, and a
document under the flat arm's budget.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lens.experiments.oolong import (
    LABEL_FRAGMENT,
    build_instance,
    check_context,
    document_name,
    split_context,
    write_document,
)

CONTEXT = (
    "The following lines contain 2 general-knowledge questions, one per line.\n"
    "\n"
    "Date: 2023-01-01 || User: 1 || Instance: What is a dog?\n"
    "Date: 2023-01-02 || User: 2 || Instance: Name a colour.\n"
)
ROW = {
    "id": 17000206,
    "context_len": 131072,
    "context_window_id": 6,
    "dataset": "trec_coarse",
    "task": "TASK_TYPE.MOST_FREQ",
    "answer": "['numeric value']",
    "answer_type": "ANSWER_TYPE.LABEL",
    "num_labels": 6,
    "question": "In the above data, which label is the most common?",
}


class SplitContextTest(unittest.TestCase):
    def test_one_record_per_non_empty_line(self) -> None:
        records = split_context(CONTEXT)

        self.assertEqual(len(records), 3)
        self.assertEqual([record["id"] for record in records], ["rec-0000", "rec-0001", "rec-0002"])
        self.assertTrue(records[1]["text"].startswith("Date: "))

    def test_the_blank_line_is_not_a_record(self) -> None:
        records = split_context(CONTEXT)

        self.assertEqual(sum(1 for record in records if record["text"] == ""), 0)

    def test_a_label_fragment_is_refused(self) -> None:
        leaked = CONTEXT.replace(
            "Instance: What is a dog?", "Instance: What is a dog? || Label: description"
        )

        with self.assertRaises(ValueError):
            check_context(leaked, split_context(leaked))

    def test_the_records_reproduce_the_context(self) -> None:
        records = split_context(CONTEXT)

        check_context(CONTEXT, records)

        self.assertEqual(
            "\n".join(record["text"] for record in records),
            "\n".join(line for line in CONTEXT.splitlines() if line.strip()),
        )

    def test_a_missing_record_is_refused(self) -> None:
        records = split_context(CONTEXT)[:-1]

        with self.assertRaises(ValueError):
            check_context(CONTEXT, records)


class BuildInstanceTest(unittest.TestCase):
    def test_fields_come_from_the_row_verbatim(self) -> None:
        instance = build_instance(ROW, document_file="oolong-ctx-6.jsonl", subset=False, seed=7)

        self.assertEqual(instance["instance_id"], "oolong-tc-17000206")
        self.assertEqual(instance["query"], ROW["question"])
        self.assertEqual(instance["gold"]["answer"], ROW["answer"])
        self.assertEqual(instance["answer_type"], "ANSWER_TYPE.LABEL")
        self.assertEqual(instance["evidence"]["file"], "oolong-ctx-6.jsonl")
        self.assertEqual(instance["meta"]["context_window_id"], 6)
        self.assertFalse(instance["meta"]["subset"])

    def test_the_subset_flag_is_carried(self) -> None:
        instance = build_instance(ROW, document_file="oolong-ctx-6.jsonl", subset=True, seed=7)

        self.assertTrue(instance["meta"]["subset"])

    def test_document_names_are_per_window(self) -> None:
        self.assertEqual(document_name(6), "oolong-ctx-6.jsonl")
        self.assertNotEqual(document_name(6), document_name(8))


class WriteDocumentTest(unittest.TestCase):
    def test_the_file_round_trips_and_carries_no_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_document(split_context(CONTEXT), Path(tmp) / "doc.jsonl")
            raw = path.read_bytes()
            text = path.read_text(encoding="utf-8")

        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(len(text.splitlines()), 3)
        self.assertEqual(set(json.loads(text.splitlines()[0])), {"id", "text", "label"})
        self.assertNotIn(LABEL_FRAGMENT, text)


if __name__ == "__main__":
    unittest.main()
