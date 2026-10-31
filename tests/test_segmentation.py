r"""Contract tests for segmentation (no API key, no network, zero cost).

A ``FakeClient`` replaces ``LensClient``, so these tests never leave the machine.
The segmenter is implemented; these are contract tests.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lens.agent.segmentation import (
    BUCKET_ORDER,
    SEGMENTATION_RESPONSE_FORMAT,
    SEGMENTATION_SYSTEM_PROMPT,
    build_segmentation_messages,
    normalize_buckets,
    parse_segmentation_json,
    segment_query,
    write_query_json,
)
from lens.models import CallResult
from lens.observability.trajectory import TraceCtx
from lens.tools.retrieval import format_query


def chat_result(text: str) -> CallResult:
    """A canned ``CallResult`` whose ``text`` is the model's reply."""
    return CallResult(
        text=text,
        message={"role": "assistant", "content": text},
        model="openai/gpt-5-mini",
        provider="openai",
        model_snapshot="openai/gpt-5-mini-20260101",
        tokens_in=10,
        tokens_out=5,
        latency_ms=1,
        cost_usd=0.0001,
    )


class FakeClient:
    """Minimal stand-in for ``LensClient``; records the ``chat`` kwargs."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    def chat(self, role, messages, *, trace_ctx, response_format=None, **kw) -> CallResult:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "trace_ctx": trace_ctx,
                "response_format": response_format,
                "extra": kw.get("extra"),
            }
        )
        return chat_result(self._text)


def good_json() -> str:
    return json.dumps(
        {
            "objectives": ["improve long-context aggregation"],
            "problems": ["naive RAG loses evidence"],
            "constraints": ["bounded token budget"],
            "requirements": ["citation-traceable"],
        }
    )


class BuildMessagesTest(unittest.TestCase):
    def test_system_then_user(self) -> None:
        messages = build_segmentation_messages("why is X slow?")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], SEGMENTATION_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "why is X slow?")

    def test_system_prompt_names_all_buckets(self) -> None:
        for name in BUCKET_ORDER:
            self.assertIn(name, SEGMENTATION_SYSTEM_PROMPT)


class ParseSegmentationJsonTest(unittest.TestCase):
    def test_bare_json(self) -> None:
        self.assertEqual(
            parse_segmentation_json(good_json())["objectives"], ["improve long-context aggregation"]
        )

    def test_fenced_json(self) -> None:
        fenced = "```json\n" + good_json() + "\n```"
        self.assertIn("problems", parse_segmentation_json(fenced))

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_segmentation_json("not json at all")


class NormalizeBucketsTest(unittest.TestCase):
    def test_always_returns_fixed_order(self) -> None:
        out = normalize_buckets({})
        self.assertEqual(tuple(out), BUCKET_ORDER)
        self.assertEqual(out, {name: [] for name in BUCKET_ORDER})

    def test_scalar_becomes_single_item_list(self) -> None:
        self.assertEqual(
            normalize_buckets({"objectives": "improve Y"})["objectives"], ["improve Y"]
        )

    def test_strips_and_drops_empty_clauses(self) -> None:
        out = normalize_buckets({"problems": ["  p1 ", "", "   ", "p2"]})
        self.assertEqual(out["problems"], ["p1", "p2"])

    def test_ignores_unknown_keys(self) -> None:
        self.assertNotIn("note", normalize_buckets({"note": ["x"]}))

    def test_drops_duplicate_clauses_across_buckets(self) -> None:
        # Same clause (case-/space-insensitive) in two buckets -> kept once, in
        # the first bucket by BUCKET_ORDER (objectives before problems).
        out = normalize_buckets(
            {
                "objectives": ["Improve Y"],
                "problems": ["improve   y", "why X"],
                "constraints": ["Improve Y"],
            }
        )
        self.assertEqual(out["objectives"], ["Improve Y"])
        self.assertEqual(out["problems"], ["why X"])
        self.assertEqual(out["constraints"], [])

    def test_drops_duplicate_clauses_within_a_bucket(self) -> None:
        out = normalize_buckets({"requirements": ["budget", "Budget", "budget "]})
        self.assertEqual(out["requirements"], ["budget"])


class SegmentQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = TraceCtx(trace_id="t1", step_idx=0, actor="root")

    def test_calls_sub_role_with_response_format(self) -> None:
        client = FakeClient(good_json())

        buckets = segment_query(client, "why is X slow?", trace_ctx=self.ctx)

        call = client.calls[0]
        self.assertEqual(call["role"], "sub")
        self.assertEqual(call["response_format"], SEGMENTATION_RESPONSE_FORMAT)
        self.assertEqual(call["trace_ctx"], self.ctx)
        self.assertEqual(call["messages"][1]["content"], "why is X slow?")
        self.assertEqual(buckets["objectives"], ["improve long-context aggregation"])

    def test_the_call_is_tagged_as_segmentation(self) -> None:
        # It logs at step_idx = 0 like a root turn; without the tag a reader cannot
        # tell the one pre-loop call from the turns the model made.
        client = FakeClient(good_json())

        segment_query(client, "q", trace_ctx=self.ctx)

        self.assertEqual(client.calls[0]["extra"], {"source": "segmentation"})

    def test_blank_query_rejected_without_calling(self) -> None:
        client = FakeClient(good_json())

        with self.assertRaises(ValueError):
            segment_query(client, "   ", trace_ctx=self.ctx)

        self.assertEqual(client.calls, [])

    def test_output_feeds_retrieval_format_query(self) -> None:
        client = FakeClient(good_json())

        buckets = segment_query(client, "q", trace_ctx=self.ctx)

        # format_query flattens in BUCKET_ORDER insertion order (plain clause join).
        self.assertEqual(
            format_query(buckets),
            "improve long-context aggregation naive RAG loses evidence "
            "bounded token budget citation-traceable",
        )


class WriteQueryJsonTest(unittest.TestCase):
    def test_writes_utf8_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "query.json"
            buckets = {"objectives": ["caf\u00e9 aggregation"], "problems": []}

            returned = write_query_json(buckets, path)

            self.assertEqual(returned, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), buckets)


if __name__ == "__main__":
    unittest.main()
