"""Contract tests for the typed sub-tasks (no model call, no database)."""

from __future__ import annotations

import json
import unittest

from lens.agent import sub_tasks
from lens.agent.sub_tasks import (
    EXTRACT_FIELDS,
    MAX_IDS_PER_CALL,
    SUB_TASKS,
    UNSURE_LABEL,
    get_task,
    parse_result,
    render_instruction,
    render_items,
    render_prompt_block,
    response_format,
    split_batches,
    validate_params,
)


def sample_params(name: str) -> dict:
    """Return one valid parameter set per task."""
    if name == "extract":
        return {"chunk_ids": ["c-1"], "fields": ["claims", "method"]}
    if name == "summarise":
        return {"text": "some text", "focus": "cost", "max_words": 60}

    return {"chunk_ids": ["c-1"], "question": "does it report a latency?", "labels": ["yes", "no"]}


class RegistryTestCase(unittest.TestCase):
    def test_the_three_tasks_are_registered(self) -> None:
        self.assertEqual(sorted(SUB_TASKS), ["classify", "extract", "summarise"])
        self.assertEqual(
            sorted(task.function for task in SUB_TASKS.values()),
            ["sub_classify", "sub_extract", "sub_summarise"],
        )

    def test_unknown_name_returns_none(self) -> None:
        self.assertIsNone(get_task("invent"))

    def test_every_signature_names_the_parameters_it_accepts(self) -> None:
        # The prompt advertises the signature; the host rejects anything else. A
        # parameter documented but not accepted (or the reverse) is a turn lost to
        # a 400 the model cannot explain.
        for task in SUB_TASKS.values():
            for param in task.params:
                self.assertIn(param, task.signature, f"{task.name} does not document {param}")

    def test_extract_fields_are_a_closed_whitelist(self) -> None:
        self.assertEqual(
            EXTRACT_FIELDS,
            ("claims", "method", "dataset", "metric", "numbers", "limitations"),
        )


class SplitBatchesTestCase(unittest.TestCase):
    """The fan-out unit: one RPC becomes ceil(n / max_items) model calls."""

    def test_twelve_ids_become_three_batches_of_four(self) -> None:
        items = [(f"c-{index}", "text") for index in range(12)]

        batches = split_batches(items, 4)

        self.assertEqual([len(batch) for batch in batches], [4, 4, 4])
        # Order is preserved, so a merged result reads in the order the caller asked.
        self.assertEqual([batch[0][0] for batch in batches], ["c-0", "c-4", "c-8"])

    def test_a_remainder_is_its_own_batch(self) -> None:
        items = [(f"c-{index}", "text") for index in range(7)]

        self.assertEqual([len(batch) for batch in split_batches(items, 4)], [4, 3])

    def test_an_empty_list_is_no_batches(self) -> None:
        self.assertEqual(split_batches([], 4), [])

    def test_a_batch_size_below_one_is_a_bug_not_a_division(self) -> None:
        with self.assertRaises(ValueError):
            split_batches([("c-1", "text")], 0)


class ValidateParamsTestCase(unittest.TestCase):
    def test_unknown_parameter_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_params(get_task("extract"), {"chunk_ids": ["c-1"], "depth": 3})

    def test_item_count_is_capped_per_call_not_per_batch(self) -> None:
        # One call may carry more ids than a single batch: the host fans them out into
        # batches of max_items and runs those concurrently. The ceiling is the call.
        task = get_task("extract")

        accepted = validate_params(
            task, {"chunk_ids": [f"c-{index}" for index in range(MAX_IDS_PER_CALL)]}
        )
        self.assertEqual(len(accepted["chunk_ids"]), MAX_IDS_PER_CALL)

        with self.assertRaises(ValueError):
            validate_params(
                task, {"chunk_ids": [f"c-{index}" for index in range(MAX_IDS_PER_CALL + 1)]}
            )

    def test_at_least_one_id_is_required(self) -> None:
        with self.assertRaises(ValueError):
            validate_params(get_task("extract"), {"chunk_ids": []})

    def test_unknown_extract_field_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_params(get_task("extract"), {"chunk_ids": ["c-1"], "fields": ["samples"]})

    def test_summarise_needs_text_or_ids(self) -> None:
        with self.assertRaises(ValueError):
            validate_params(get_task("summarise"), {"focus": "cost"})

    def test_summarise_defaults_focus_and_words(self) -> None:
        selected = validate_params(get_task("summarise"), {"text": "abc"})

        self.assertEqual(selected["max_words"], 120)
        self.assertTrue(selected["focus"])

    def test_summarise_bounds_max_words(self) -> None:
        task = get_task("summarise")

        with self.assertRaises(ValueError):
            validate_params(task, {"text": "abc", "max_words": 5})

        self.assertEqual(
            validate_params(task, {"text": "abc", "max_words": 10_000})["max_words"],
            sub_tasks.MAX_WORDS_CEILING,
        )

    def test_classify_bounds_the_label_set(self) -> None:
        task = get_task("classify")

        with self.assertRaises(ValueError):
            validate_params(task, {"chunk_ids": ["c-1"], "question": "q", "labels": ["only"]})

        with self.assertRaises(ValueError):
            validate_params(
                task,
                {
                    "chunk_ids": ["c-1"],
                    "question": "q",
                    "labels": [f"label-{index}" for index in range(sub_tasks.MAX_LABELS + 1)],
                },
            )

    def test_classify_rejects_duplicate_labels(self) -> None:
        with self.assertRaises(ValueError):
            validate_params(
                get_task("classify"),
                {"chunk_ids": ["c-1"], "question": "q", "labels": ["yes", "yes"]},
            )


class SchemaTestCase(unittest.TestCase):
    def test_every_schema_is_strict(self) -> None:
        # Strict mode rejects a schema whose properties are not all required, so a
        # "leave the key out" answer would be an API error rather than a result.
        for task in SUB_TASKS.values():
            payload = response_format(task, sample_params(task.name))
            schema = payload["json_schema"]["schema"]

            self.assertTrue(payload["json_schema"]["strict"])
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(schema["required"], list(schema["properties"]))

    def test_extract_schema_only_carries_the_requested_fields(self) -> None:
        params = {"chunk_ids": ["c-1"], "fields": ["claims", "numbers"]}
        schema = response_format(get_task("extract"), params)["json_schema"]["schema"]
        record = schema["properties"]["records"]["items"]

        # ``quotes`` is always there: it is the only part of a record an answer may
        # cite, so it cannot be something the caller forgets to ask for.
        self.assertEqual(record["required"], ["id", "claims", "numbers", "quotes"])

    def test_quotes_are_last_whatever_is_requested(self) -> None:
        # The request bytes are the prompt's cache key, so the field order may not
        # depend on set iteration.
        for fields in (None, ["numbers"], ["claims", "method", "quotes"]):
            params = {"chunk_ids": ["c-1"], "fields": fields}
            schema = response_format(get_task("extract"), params)["json_schema"]["schema"]
            record = schema["properties"]["records"]["items"]

            self.assertEqual(record["required"][-1], "quotes")
            self.assertEqual(record["required"].count("quotes"), 1)

    def test_classify_schema_offers_an_unsure_label(self) -> None:
        schema = response_format(get_task("classify"), sample_params("classify"))["json_schema"][
            "schema"
        ]
        label = schema["properties"]["labels"]["items"]["properties"]["label"]

        self.assertIn(UNSURE_LABEL, label["enum"])
        self.assertIn("yes", label["enum"])


class RenderTestCase(unittest.TestCase):
    def test_instruction_has_no_unfilled_placeholder(self) -> None:
        for task in SUB_TASKS.values():
            params = validate_params(task, sample_params(task.name))
            text = render_instruction(task, params, "[c-1]\nsome text")

            self.assertNotIn("<<", text)
            self.assertIn("some text", text)

    def test_items_are_truncated_per_item(self) -> None:
        rendered = render_items([("c-1", "x" * (sub_tasks.ITEM_CHARS + 500))])

        self.assertIn("[c-1]", rendered)
        self.assertEqual(len(rendered.splitlines()[1]), sub_tasks.ITEM_CHARS)

    def test_prompt_block_names_every_registered_function(self) -> None:
        block = render_prompt_block()

        for task in SUB_TASKS.values():
            self.assertIn(task.function, block)

    def test_every_signature_carries_its_types(self) -> None:
        # Five code blocks in three recorded runs died guessing the argument and
        # return shapes: KeyError on the record key, AttributeError treating a quote
        # as a dict. The host rejected bad parameters already; the result had no
        # contract at all.
        for task in SUB_TASKS.values():
            self.assertIn(": ", task.signature, f"{task.name} signature has no types")
            self.assertIn("-> dict", task.signature)
            self.assertIn('"id"', task.result_shape, f"{task.name} does not name its key")
            self.assertIn(task.result_key, task.result_shape)

    def test_the_prompt_block_shows_the_result_shape_and_the_id_key(self) -> None:
        block = render_prompt_block()

        self.assertIn('"records": [{"id"', block)
        self.assertIn("not ``chunk_id``", block)
        self.assertIn("not of dicts", block)

    def test_prompt_block_states_the_call_limit(self) -> None:
        self.assertIn("sub-task calls per query", render_prompt_block())


class QuoteVerificationTestCase(unittest.TestCase):
    """A quote the caller cannot find in the chunk must not survive the call."""

    TEXT = (
        "We propose a context compression model structure based on an autoencoder, "
        "which we call the Recurrent Context Compression model."
    )

    def test_a_verbatim_quote_is_kept(self) -> None:
        records, dropped = sub_tasks.verify_quotes(
            [{"id": "c-1", "quotes": ["based on an autoencoder, which we call the"]}],
            [("c-1", self.TEXT)],
        )

        self.assertEqual(records[0]["quotes"], ["based on an autoencoder, which we call the"])
        self.assertEqual(dropped, 0)

    def test_a_rewritten_quote_is_dropped(self) -> None:
        records, dropped = sub_tasks.verify_quotes(
            [{"id": "c-1", "quotes": ["based on a deep autoencoder that they call RCC"]}],
            [("c-1", self.TEXT)],
        )

        self.assertEqual(records[0]["quotes"], [])
        self.assertEqual(dropped, 1)

    def test_a_quote_from_another_chunk_is_dropped(self) -> None:
        # The id is what the answer cites, so the span has to be in *that* text.
        records, dropped = sub_tasks.verify_quotes(
            [{"id": "c-2", "quotes": ["based on an autoencoder, which we call the"]}],
            [
                ("c-1", self.TEXT),
                ("c-2", "An unrelated sentence about caching, long enough to say."),
            ],
        )

        self.assertEqual(records[0]["quotes"], [])
        self.assertEqual(dropped, 1)

    def test_line_wrapping_does_not_break_a_quote(self) -> None:
        records, dropped = sub_tasks.verify_quotes(
            [{"id": "c-1", "quotes": ["based on an\n  autoencoder,   which we call the"]}],
            [("c-1", self.TEXT.replace(" ", "\n", 3))],
        )

        self.assertEqual(len(records[0]["quotes"]), 1)
        self.assertEqual(dropped, 0)

    def test_a_short_or_missing_field_is_tolerated(self) -> None:
        records, dropped = sub_tasks.verify_quotes(
            [{"id": "c-1", "quotes": ["too short"]}, {"id": "c-1", "claims": []}],
            [("c-1", self.TEXT)],
        )

        self.assertEqual(records[0]["quotes"], [])
        # A record that came back without the field does not gain one: the check
        # removes quotes, it never invents them.
        self.assertNotIn("quotes", records[1])
        self.assertEqual(dropped, 1)

    def test_quotes_are_capped_in_count_and_length(self) -> None:
        # An unbounded quote list is what truncated the first recorded call.
        long_quote = "based on an autoencoder, which we call the " + "x" * 500
        text = self.TEXT + " " + long_quote
        records, dropped = sub_tasks.verify_quotes(
            [
                {
                    "id": "c-1",
                    "quotes": [
                        "which we call the Recurrent Context Compression model.",
                        "based on an autoencoder, which we call the",
                        "We propose a context compression model structure",
                        "model structure based on an autoencoder",
                        long_quote,
                    ],
                }
            ],
            [("c-1", text)],
        )

        self.assertEqual(len(records[0]["quotes"]), sub_tasks.MAX_QUOTES_PER_RECORD)
        self.assertEqual(dropped, 2)


class ParseResultTestCase(unittest.TestCase):
    def test_a_record_with_an_unrequested_id_is_dropped(self) -> None:
        records, missing, errors = parse_result(
            get_task("extract"),
            json.dumps({"records": [{"id": "c-1", "claims": []}, {"id": "c-999", "claims": []}]}),
            ["c-1"],
        )

        self.assertEqual([record["id"] for record in records], ["c-1"])
        self.assertEqual(missing, [])
        self.assertTrue(any("c-999" in error for error in errors))

    def test_a_missing_id_is_reported_not_invented(self) -> None:
        records, missing, _errors = parse_result(
            get_task("extract"), json.dumps({"records": [{"id": "c-1"}]}), ["c-1", "c-2"]
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(missing, ["c-2"])

    def test_a_duplicate_record_is_dropped(self) -> None:
        records, _missing, errors = parse_result(
            get_task("extract"),
            json.dumps({"records": [{"id": "c-1"}, {"id": "c-1"}]}),
            ["c-1"],
        )

        self.assertEqual(len(records), 1)
        self.assertTrue(any("duplicate" in error for error in errors))

    def test_non_json_reports_every_id_as_missing(self) -> None:
        records, missing, errors = parse_result(get_task("extract"), "not json", ["c-1"])

        self.assertEqual(records, [])
        self.assertEqual(missing, ["c-1"])
        self.assertEqual(len(errors), 1)

    def test_a_fenced_reply_is_accepted(self) -> None:
        text = '```json\n{"records": [{"id": "c-1", "claims": ["a"]}]}\n```'
        records, missing, errors = parse_result(get_task("extract"), text, ["c-1"])

        self.assertEqual(len(records), 1)
        self.assertEqual((missing, errors), ([], []))

    def test_extract_fields_are_normalised(self) -> None:
        records, _missing, _errors = parse_result(
            get_task("extract"),
            json.dumps(
                {
                    "records": [
                        {
                            "id": "c-1",
                            "claims": "one claim",
                            "method": None,
                            "numbers": [1, 2],
                        }
                    ]
                }
            ),
            ["c-1"],
        )

        self.assertEqual(records[0]["claims"], ["one claim"])
        self.assertIsNone(records[0]["method"])
        self.assertEqual(records[0]["numbers"], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
