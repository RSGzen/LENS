r"""Contract tests for the experimental arms: prompt variant, routes, registration.

An arm is a process environment, so the things that define it are checked here:
which instruction block the root sees, which RPC routes exist, and what the
manifest records. No key, no database, no Docker, zero cost.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_arms -v
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lens import config
from lens.agent import context
from lens.agent.tool_server import ToolServer
from lens.experiments.arm import arm_fingerprint

MANIFEST = "1. shallow_match(buckets)\n2. rerank_chunks(chunk_ids)"


def prompt(**kwargs: object) -> str:
    """Build the system prompt against a fixed manifest, buckets and caps."""
    return context.build_system_prompt(MANIFEST, {}, {"iterations": 8}, **kwargs)  # type: ignore[arg-type]


class PromptVariantTests(unittest.TestCase):
    def test_full_names_the_kg_and_the_sub_tasks(self) -> None:
        text = prompt(variant="full", kg=True, sub_calls=True)

        self.assertIn("kg_query", text)

        for function in ("sub_extract", "sub_summarise", "sub_classify"):
            self.assertIn(function, text)

    def test_kg_off_drops_every_kg_mention(self) -> None:
        # The prompt must not advertise a tool the run cannot call: the model
        # spends turns on it and the arm measures the wrong thing.
        text = prompt(variant="full", kg=False, sub_calls=True)

        self.assertNotIn("kg_query", text)
        self.assertNotIn("when relations matter", text)
        self.assertNotIn("knowledge graph", text)

    def test_sub_calls_off_drops_the_whole_section(self) -> None:
        # The off arm must not mention a capability the sandbox route removes.
        text = prompt(variant="full", kg=True, sub_calls=False)

        self.assertNotIn("sub_extract", text)
        self.assertNotIn("Sub-tasks", text)
        self.assertIn("kg_query", text)

    def test_protected_rules_survive_every_full_family_variant(self) -> None:
        for keep in ({"kg": True, "sub_calls": True}, {"kg": False, "sub_calls": False}):
            text = prompt(variant="full", **keep)

            self.assertIn("FINAL_ANSWER:", text)
            self.assertIn("Search the variable", text)
            self.assertIn("negative finding", text)

    def test_host_segmentation_shows_the_bucket_block(self) -> None:
        text = prompt(variant="full", segmentation="host", raw_query=False)

        self.assertIn("Query buckets (query.json)", text)
        self.assertNotIn("The host did not segment", text)

    def test_root_segmentation_asks_the_root_to_write_the_dict(self) -> None:
        text = prompt(variant="full", segmentation="root", question="the raw question")

        self.assertNotIn("Query buckets (query.json)", text)
        self.assertIn("The host did not segment", text)
        self.assertIn("write ``buckets``", text)
        self.assertIn("the raw question", text)

    def test_raw_query_adds_the_wording_without_removing_the_buckets(self) -> None:
        text = prompt(
            variant="full", segmentation="host", raw_query=True, question="the raw wording"
        )

        self.assertIn("Query buckets (query.json)", text)
        self.assertIn("the raw wording", text)

    def test_no_marker_leaks(self) -> None:
        for variant in config.PROMPT_VARIANTS:
            self.assertNotIn("<<", prompt(variant=variant))

    def test_flat_variant_has_no_tools(self) -> None:
        text = prompt(variant="naive")

        self.assertNotIn("shallow_match", text)
        self.assertNotIn("FINAL_ANSWER", text)
        self.assertIn("Cite the id", text)

    def test_document_variant_points_at_the_evidence_file(self) -> None:
        text = prompt(variant="document", kg=False, sub_calls=False)

        self.assertIn("documents.jsonl", text)
        self.assertIn("shallow_match", text)  # the manifest is still injected
        self.assertNotIn("kg_query", text)

    def test_unknown_variant_fails_loud(self) -> None:
        with self.assertRaises(ValueError):
            prompt(variant="vibes")


class SubCallRouteTests(unittest.TestCase):
    def test_route_absent_when_sub_calls_are_off(self) -> None:
        server = ToolServer(runtime=object(), token="t", sub_calls=False)

        self.assertNotIn("/sub_task", server.routes)

        status, body = server.handle_request("POST", "/sub_task", {}, token="t")

        self.assertEqual(status, 404)
        self.assertFalse(body["ok"])

    def test_route_present_when_sub_calls_are_on(self) -> None:
        server = ToolServer(runtime=object(), token="t", sub_calls=True)

        self.assertIn("/sub_task", server.routes)


class ArmFingerprintTests(unittest.TestCase):
    def test_records_the_harness_that_ran(self) -> None:
        arm = arm_fingerprint(recorded=True, replay=True)

        self.assertEqual(arm["mode"]["run"], config.RUN_MODE)
        self.assertEqual(arm["mode"]["evidence"], config.EVIDENCE_SOURCE)
        self.assertEqual(arm["mode"]["treatment"], config.TREATMENT)
        self.assertEqual(arm["mode"]["segmentation"], config.SEGMENTATION)
        self.assertEqual(arm["mode"]["raw_query"], config.RAW_QUERY)
        self.assertEqual(arm["mode"]["prompt"], config.PROMPT_VARIANT)
        self.assertEqual(arm["mode"]["sub_calls"], config.SUB_CALLS_ENABLED)
        self.assertEqual(arm["mode"]["document_evidence"], config.DOCUMENT_EVIDENCE)
        self.assertIn("documents", arm["mode"])

    def test_a_document_run_records_the_file_and_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "evidence.jsonl"
            source.write_text('{"id": "a", "text": "alpha"}\n', encoding="utf-8")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()

            with patch.object(config, "DOCUMENTS_PATH", str(source)):
                arm = arm_fingerprint(recorded=True, replay=True)

            documents = arm["mode"]["documents"]
            self.assertEqual(documents["path"], str(source))
            self.assertEqual(documents["sha256"], expected)

    def test_a_corpus_run_records_no_document(self) -> None:
        with patch.object(config, "DOCUMENTS_PATH", None):
            arm = arm_fingerprint(recorded=True, replay=True)

        self.assertIsNone(arm["mode"]["documents"])


class ModeValidationTests(unittest.TestCase):
    def test_document_mode_requires_a_documents_path(self) -> None:
        with (
            patch.object(config, "RUN_MODE", "document"),
            patch.object(config, "PROMPT_VARIANT", "document"),
            patch.object(config, "DOCUMENTS_PATH", None),
        ):
            with self.assertRaises(ValueError):
                config._check_run_mode()

    def test_mode_and_variant_must_agree(self) -> None:
        # A document run with the full prompt is a silent misconfiguration: the
        # model would be told to call tools that were never registered.
        with (
            patch.object(config, "RUN_MODE", "naive"),
            patch.object(config, "PROMPT_VARIANT", "full"),
        ):
            with self.assertRaises(ValueError):
                config._check_run_mode()

    def test_defaults_pass(self) -> None:
        with (
            patch.object(config, "RUN_MODE", "rlm"),
            patch.object(config, "PROMPT_VARIANT", "full"),
        ):
            config._check_run_mode()


if __name__ == "__main__":
    unittest.main()
