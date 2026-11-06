"""Contract tests for the host-side claim extractor.

The model is faked: every test asserts what the host does with a reply, not
what a provider returns.
"""

from __future__ import annotations

import json
import unittest

from lens.models import CallResult
from lens.observability.claim_extract import _schema, extract_frames, verify_answer

CHUNK_A = "aaaaaaaa-0000-0000-0000-000000000001"
CHUNK_B = "bbbbbbbb-0000-0000-0000-000000000002"


def _frame(unit_index: int, chunk_id: str | None = CHUNK_A, **overrides: object) -> dict:
    base = {
        "unit_index": unit_index,
        "predicate": "measures",
        "subject": "RLM",
        "object": None,
        "benchmark": "OOLONG",
        "metric": "accuracy",
        "value": "56.0",
        "unit": "%",
        "direction": None,
        "quote": "reaches 56.0 accuracy on OOLONG",
        "chunk_id": chunk_id,
    }
    base.update(overrides)

    return base


def _reply(*frames: dict) -> str:
    return json.dumps({"frames": list(frames)})


class FakeClient:
    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple] = []

    def chat(self, role: str, messages: list, **kwargs: object) -> CallResult:
        self.calls.append((role, messages, kwargs))
        reply = self.replies.pop(0)

        if isinstance(reply, Exception):
            raise reply

        return CallResult(
            text=str(reply),
            message={"role": "assistant", "content": str(reply)},
            model="fake",
            provider="fake",
            model_snapshot=None,
            tokens_in=10,
            tokens_out=5,
            latency_ms=1,
            cost_usd=0.0,
        )


class FakeBudget:
    def __init__(self) -> None:
        self.chats = 0
        self.sub_tasks = 0

    def record_chat(self, _result: CallResult) -> None:
        self.chats += 1

    def record_sub_task(self) -> None:
        self.sub_tasks += 1


class ExtractFramesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.units = [
            ("RLM reaches 56.0 accuracy on OOLONG.", [CHUNK_A]),
            ("The flat baseline reaches 44.0 accuracy on OOLONG.", [CHUNK_B]),
        ]

    def test_one_batch_returns_frames_and_records_the_call(self) -> None:
        budget = FakeBudget()
        client = FakeClient([_reply(_frame(0), _frame(1, CHUNK_B, value="44.0"))])

        frames, errors = extract_frames(client, self.units, None, budget=budget)

        self.assertEqual(errors, [])
        self.assertEqual([frame.unit_index for frame in frames], [0, 1])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(budget.chats, 1)
        self.assertEqual(budget.sub_tasks, 1)

    def test_more_units_than_the_batch_are_split(self) -> None:
        units = [(f"claim {index}", [CHUNK_A]) for index in range(7)]
        client = FakeClient([_reply(_frame(0)), _reply(_frame(6))])

        frames, errors = extract_frames(client, units, None, batch_size=6)

        self.assertEqual(errors, [])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual([frame.unit_index for frame in frames], [0, 6])

    def test_a_frame_from_another_batch_is_dropped(self) -> None:
        units = [(f"claim {index}", [CHUNK_A]) for index in range(7)]
        client = FakeClient([_reply(_frame(6)), _reply(_frame(6))])

        frames, errors = extract_frames(client, units, None, batch_size=6)

        self.assertEqual(len(frames), 1)
        self.assertTrue(any("not shown" in error for error in errors))

    def test_a_non_json_reply_is_an_error(self) -> None:
        client = FakeClient(["not json"])

        frames, errors = extract_frames(client, self.units, None)

        self.assertEqual(frames, [])
        self.assertTrue(any("not JSON" in error for error in errors))

    def test_a_client_failure_is_an_error_not_an_exception(self) -> None:
        client = FakeClient([RuntimeError("provider down")])

        frames, errors = extract_frames(client, self.units, None)

        self.assertEqual(frames, [])
        self.assertTrue(any("claim extraction failed" in error for error in errors))

    def test_the_schema_lists_every_frame_field(self) -> None:
        schema = _schema()
        item = schema["properties"]["frames"]["items"]

        self.assertTrue(item["additionalProperties"] is False)
        self.assertIn("unit_index", item["required"])
        self.assertIn("quote", item["required"])

    def test_verify_answer_marks_a_supported_frame(self) -> None:
        answer = (
            "The RLM method reaches 56.0 accuracy on the OOLONG trec_coarse "
            f"benchmark in this reported run [{CHUNK_A}]."
        )
        texts = {
            CHUNK_A: ("The RLM method reaches 56.0 accuracy on the OOLONG trec_coarse benchmark.")
        }
        client = FakeClient(
            [
                _reply(
                    _frame(
                        0,
                        CHUNK_A,
                        quote="reaches 56.0 accuracy on the OOLONG trec_coarse benchmark",
                    )
                )
            ]
        )

        report = verify_answer(answer, texts, client, None, run_id="run-1")

        self.assertEqual(report.units, 1)
        self.assertEqual(report.counts["grounded"], 1)
        self.assertEqual(report.coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
