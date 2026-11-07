r"""Contract tests for citation resolution - no key, no DB, no Docker.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_citations -v
"""

from __future__ import annotations

import json
import tempfile
import unittest

import lens.observability.trajectory as trajectory_module
from lens.observability.citations import (
    CitationReport,
    cited_ids,
    claim_units,
    expand_short_ids,
    recorded_chunk_texts,
    resolve_citations,
    resolve_merged_ids,
    verify_quoted_spans,
    write_sources_md,
)
from lens.observability.trajectory import TraceCtx, Trajectory

CHUNK = "11111111-1111-5111-8111-111111111111"
OTHER = "22222222-2222-5222-8222-222222222222"
GHOST = "99999999-9999-5999-8999-999999999999"


class UncitedUnitTestCase(unittest.TestCase):
    """The claims the aggregate sees and the per-claim questions could not."""

    ONE = "11111111-1111-5111-8111-111111111111"

    def test_a_scope_sentence_without_ids_becomes_a_unit(self) -> None:
        answer = (
            "From the 15 cited chunks we inspected, drawn from 100 reranked chunks "
            "across 19 papers, the text supports the following methods.\n"
            f"- A method with evidence [{self.ONE}]\n"
        )

        units = claim_units(answer, include_uncited=True)

        self.assertEqual(len(units), 2)
        self.assertEqual(units[0][1], [self.ONE])
        self.assertEqual(units[1][1], [])
        self.assertIn("15 cited chunks", units[1][0])

    def test_the_same_text_without_the_flag_is_cited_only(self) -> None:
        answer = (
            "From the 15 cited chunks we inspected, drawn from 100 reranked chunks "
            "across 19 papers, the text supports the following methods.\n"
            f"- A method with evidence [{self.ONE}]\n"
        )

        self.assertEqual(len(claim_units(answer)), 1)

    def test_structure_lines_are_not_uncited_claims(self) -> None:
        answer = (
            "# A heading long enough to look like a sentence about the evidence\n"
            "- A bullet that names a method but carries no citation at all\n"
        )

        self.assertEqual(claim_units(answer, include_uncited=True), [])

    def test_a_short_line_is_not_a_claim(self) -> None:
        self.assertEqual(claim_units("Scope inspected:", include_uncited=True), [])

    def test_a_lead_in_line_is_not_a_claim(self) -> None:
        # Measured: these scored highest of every unit because they cite nothing, so
        # the gate's feedback kept naming headings.
        answer = (
            "How they differ in where aggregation happens (stated via the quotes above):\n"
            f"- A method with evidence that is long enough to be a claim [{self.ONE}]\n"
        )

        units = claim_units(answer, include_uncited=True)

        self.assertEqual([ids for _, ids in units], [[self.ONE]])

    def test_a_heading_without_a_full_stop_is_not_a_claim(self) -> None:
        # Both lines are verbatim from a recorded answer. They have no `:`, `#`, `-`
        # or `*`, and each is at least twelve words, so they were asked about as
        # uncited claims; an excerpt cannot support a heading, so both came back at
        # exactly 0.50 — the aggregate threshold — and were named as the offenders.
        answer = (
            "Q2 — How these methods differ in where aggregation happens (grounded by quotes)\n"
            "Strictly supported findings (each claim is followed by a verbatim sentence "
            "from a retrieved chunk)\n"
            f"- A method with evidence that is long enough to be a claim [{self.ONE}]\n"
        )

        units = claim_units(answer, include_uncited=True)

        self.assertEqual([ids for _, ids in units], [[self.ONE]])

    def test_a_scope_sentence_is_still_a_unit(self) -> None:
        # The punctuation test must not remove the sentences it was introduced for.
        sentence = (
            "From the 12 chunks inspected, none of them states the figure that the "
            "question asks about."
        )

        self.assertEqual(len(claim_units(sentence, include_uncited=True)), 1)


class QuotedSpanTests(unittest.TestCase):
    """Fidelity of the answer's own quotes: the model retypes them, and it drifts."""

    A = "11111111-1111-5111-8111-111111111111"
    B = "22222222-2222-5222-8222-222222222222"

    TEXTS = {
        A: "We propose a context compression model structure based on an autoencoder, "
        "which we call the Recurrent Context Compression model.",
        B: "Retrieval-Augmented Generation (RAG) represents another line of research, "
        "designed to retrieve answers from long contexts.",
    }

    def test_a_quote_inside_its_cited_chunk_counts(self) -> None:
        answer = (
            "\u201cbased on an autoencoder, which we call the Recurrent Context Compression\u201d "
            f"[{self.A}]"
        )

        self.assertEqual(verify_quoted_spans(answer, self.TEXTS), (1, 0, 1))

    def test_a_paraphrase_in_quotation_marks_is_found_nowhere(self) -> None:
        # Measured: one answer put a sub-task summary in quotation marks, so the
        # reader could not find the sentence in the chunk it cited.
        answer = f"\u201ca paraphrase nobody wrote about compressing the context\u201d [{self.A}]"

        self.assertEqual(verify_quoted_spans(answer, self.TEXTS), (0, 0, 1))

    def test_a_real_quote_under_the_wrong_id_is_a_mis_citation(self) -> None:
        # The words exist but the reader who follows the id finds different text.
        answer = f"\u201cdesigned to retrieve answers from long contexts\u201d [{self.A}]"

        self.assertEqual(verify_quoted_spans(answer, self.TEXTS), (0, 1, 1))

    def test_an_elided_quote_verifies_when_every_part_is_present(self) -> None:
        answer = (
            "\u201cWe propose a context compression model ... "
            "Recurrent Context Compression model\u201d "
            f"[{self.A}]"
        )

        self.assertEqual(verify_quoted_spans(answer, self.TEXTS), (1, 0, 1))

    def test_straight_quotes_count_too(self) -> None:
        # One recorded answer used straight quotes throughout; a check that knew only
        # typographic quotes would have scored it as having no quotes at all.
        answer = f'"designed to retrieve answers from long contexts" [{self.B}]'

        self.assertEqual(verify_quoted_spans(answer, self.TEXTS), (1, 0, 1))

    def test_an_abbreviated_cited_id_is_expanded_before_the_check(self) -> None:
        chunk = "abc12345-1111-5111-8111-111111111111"
        quote = "designed to retrieve answers from long contexts"
        answer = f'"designed to retrieve answers from long contexts" [{chunk[:8]}]'
        texts = {chunk: f"Retrieval-Augmented Generation {quote}."}

        self.assertEqual(verify_quoted_spans(answer, texts), (1, 0, 1))

    def test_a_quoted_task_name_with_no_citation_is_not_counted(self) -> None:
        answer = (
            'Ledger: task "identify aggregation methods and where aggregation happens" was done.'
        )

        self.assertEqual(verify_quoted_spans(answer, self.TEXTS), (0, 0, 0))

    def test_an_answer_without_quotes_is_zero_of_zero(self) -> None:
        self.assertEqual(verify_quoted_spans("A claim with no quotation.", self.TEXTS), (0, 0, 0))

    def test_the_texts_come_from_the_recorded_payloads(self) -> None:
        # The check is only meaningful against text the run actually held.
        case = CitationTestCase("test_resolved_and_unresolved_chunks")
        case.setUp()
        case._tool(
            1,
            "retrieval_funnel",
            {
                "chunks": [{"chunk_uuid": CHUNK, "chunk_text": "the retrieved sentence"}],
                "papers": {},
            },
        )

        try:
            texts = recorded_chunk_texts(case.traj)
        finally:
            case.tearDown()

        self.assertEqual(texts, {CHUNK: "the retrieved sentence"})


class ExpandShortIdsTestCase(unittest.TestCase):
    """A model that prints 8-character prefixes cites the prefix it saw."""

    KNOWN = [
        "bc4e8dcb-1111-5111-8111-111111111111",
        "92287ee9-2222-5222-8222-222222222222",
        "92287ee0-3333-5333-8333-333333333333",
    ]

    def test_every_written_form_expands(self) -> None:
        text = "x [chunk_uuid=bc4e8dcb, arxiv_id=2604.22294]; y [bc4e8dcb]; z bc4e8dcb…"

        out = expand_short_ids(text, self.KNOWN)

        self.assertEqual(out.count(self.KNOWN[0]), 3)
        self.assertEqual(cited_ids(out), [self.KNOWN[0], "2604.22294"])

    def test_ambiguous_unknown_and_word_tokens_stay(self) -> None:
        text = "92287ee [deadbeef] decade 2604.22294 " + self.KNOWN[1]

        # 92287ee prefixes two ids; deadbeef matches none; "decade" has no digit;
        # a full uuid is never rewritten.
        self.assertEqual(expand_short_ids(text, self.KNOWN), text)


class MergedIdTestCase(unittest.TestCase):
    """A full uuid with an invented suffix is a typo, not a missing chunk."""

    REAL = "c3be54f5-88d8-5111-8111-111111111111"
    WRITTEN = "c3be54f5-8d39-5905-b23a-f911f0437cd2"
    AMBIGUOUS_A = "c3be54f5-1111-5111-8111-111111111111"
    AMBIGUOUS_B = "c3be54f5-2222-5222-8222-222222222222"

    def test_a_unique_prefix_names_the_chunk(self) -> None:
        text, repairs = resolve_merged_ids(f"See [{self.WRITTEN}].", [self.REAL])

        self.assertEqual(text, f"See [{self.REAL}].")
        self.assertEqual(repairs, [(self.WRITTEN, self.REAL)])

    def test_an_ambiguous_prefix_stays_as_written(self) -> None:
        text = f"See [{self.WRITTEN}]."

        out, repairs = resolve_merged_ids(text, [self.AMBIGUOUS_A, self.AMBIGUOUS_B])

        self.assertEqual(out, text)
        self.assertEqual(repairs, [])

    def test_a_known_id_is_left_alone(self) -> None:
        text, repairs = resolve_merged_ids(f"See [{self.REAL}].", [self.REAL])

        self.assertEqual(text, f"See [{self.REAL}].")
        self.assertEqual(repairs, [])

    def test_the_report_counts_merged_separately(self) -> None:
        report = CitationReport(
            run_id="r",
            merged=[(self.WRITTEN, self.REAL)],
            unresolved=[self.AMBIGUOUS_A],
        )

        self.assertEqual(report.counts["merged"], 1)
        self.assertEqual(report.counts["unresolved"], 1)
        self.assertEqual(report.to_dict()["merged"][0]["chunk_uuid"], self.REAL)

    def test_a_quote_near_a_merged_id_is_in_cited(self) -> None:
        quote = "designed to retrieve answers from long contexts"
        answer = f'"{quote}" [{self.WRITTEN}]'
        texts = {self.REAL: f"Retrieval-Augmented Generation {quote}."}

        self.assertEqual(verify_quoted_spans(answer, texts), (1, 0, 1))


class ClaimUnitShapeTestCase(unittest.TestCase):
    """The answer shapes the model actually writes."""

    def test_a_bullet_with_a_separate_citations_line_is_one_claim(self) -> None:
        # Recorded shape: the claim, a "Where:" line, then a "Citations:" line. The
        # sentence splitter dropped both halves -- the claim had no ids, the citation
        # line had no words -- and a nine-claim answer produced one unit.
        answer = (
            "- Retrieval-augmented aggregation across calls\n"
            "  - Where: outside the model, in the retrieval/synthesis pipeline\n"
            "  - Citations: 9db55e2a-afdf-5842-bbbb-80e36dc75bce / 2406.12331\n"
            "- Hierarchical summarization over chunks\n"
            "  - Where: between model calls, chunk summaries then combined\n"
            "  - Citations: 8db3b3d8-e812-5675-966e-e000d199be37 / 2508.10391\n"
        )

        units = claim_units(answer)

        self.assertEqual(len(units), 2)
        self.assertTrue(all("Citations:" in text for text, _ids in units))
        self.assertTrue(all(len(ids) >= 2 for _text, ids in units))

    def test_a_bullet_without_ids_is_not_a_claim(self) -> None:
        answer = "- A heading with no evidence\n- Another heading, also bare\n"

        units = claim_units(answer)

        # Nothing cited: the answer collapses to one unit so it is still judged once.
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0][1], [])

    def test_prose_without_bullets_still_splits_by_sentence(self) -> None:
        one = "11111111-1111-5111-8111-111111111111"

        units = claim_units(f"ReCAP keeps a sliding window [{one}]. The cost stays bounded.")

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0][1], [one])


class CitationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = self._tmp.name
        self.traj = Trajectory("run-cite")

    def tearDown(self) -> None:
        trajectory_module.LENS_RUNS_ROOT_PATH = self._orig
        self._tmp.cleanup()

    # ---------------------------------------------------------------- helpers
    def _tool(self, step: int, name: str, result: object) -> None:
        self.traj.log_event(
            trace_ctx=TraceCtx("run-cite", step, "tool"),
            call_type="tool",
            action="tool.dispatch",
            tool=name,
            args_hash="hash-" + name,
            args_ref=self.traj.store_blob("{}"),
            result_ref=self.traj.store_blob(json.dumps(result)),
            result_summary=f"{name}: ok",
            latency_ms=10,
        )

    def _answer(self, text: str) -> None:
        (self.traj.run_dir / "answer.md").write_text(text, encoding="utf-8")

    def _bundle(self, extra_uuid: str | None = None) -> None:
        """The shape ``retrieval_funnel`` returns: {chunks, papers: {arxiv_id: {...}}}.

        The ``papers`` map is keyed by id; it used to be read as a list of rows, so
        titles and paper ids from this payload never resolved.
        """
        extra = (
            [{"chunk_uuid": extra_uuid, "arxiv_id": "2512.24601", "section_type": "Methodology"}]
            if extra_uuid
            else []
        )
        self._tool(
            1,
            "retrieval_funnel",
            {
                "chunks": [
                    {
                        "chunk_uuid": CHUNK,
                        "arxiv_id": "2512.24601",
                        "section_type": "Methodology",
                    },
                    {
                        "chunk_uuid": OTHER,
                        "arxiv_id": "2511.02817",
                        "section_type": "Result/Discussion",
                    },
                    *extra,
                ],
                "papers": {
                    "2512.24601": {"title": "Recursive Language Models"},
                    "2511.02817": {"title": "OOLONG"},
                },
            },
        )

    # ------------------------------------------------------------------ tests
    def test_resolved_and_unresolved_chunks(self) -> None:
        self._bundle()
        self._answer(f"RLMs recurse on chunks ({CHUNK}). A ghost is cited ({GHOST}).")

        report = resolve_citations("run-cite")

        self.assertEqual(report.counts, {"cited": 2, "resolved": 1, "merged": 0, "unresolved": 1})
        self.assertEqual(report.unresolved, [GHOST])

        resolved = [c for c in report.citations if c.resolved][0]
        self.assertEqual(resolved.arxiv_id, "2512.24601")
        self.assertEqual(resolved.section_type, "Methodology")
        self.assertEqual(resolved.title, "Recursive Language Models")
        self.assertEqual(resolved.turn, 1)

    def test_arxiv_ids_resolve_against_retrieved_papers(self) -> None:
        self._bundle()
        self._answer("See 2512.24601 and 2601.99999 for the comparison.")

        report = resolve_citations("run-cite")

        by_token = {c.token: c for c in report.citations}
        self.assertTrue(by_token["2512.24601"].resolved)
        self.assertFalse(by_token["2601.99999"].resolved)
        self.assertEqual(report.unresolved, ["2601.99999"])

    def test_an_abbreviated_id_resolves_to_the_one_chunk_it_matches(self) -> None:
        # A model that writes the first eight characters is citing the chunk. A
        # strict matcher counted those answers as uncited, which under-reported
        # the flat baseline's grounding.
        self._bundle()
        self._answer(f"RLMs recurse on chunks [{CHUNK[:8]}].")

        report = resolve_citations("run-cite")

        self.assertEqual(report.counts, {"cited": 1, "resolved": 1, "merged": 0, "unresolved": 0})
        self.assertEqual(report.citations[0].token, CHUNK)

    def test_an_ambiguous_prefix_stays_unresolved(self) -> None:
        self._bundle()
        self._bundle(extra_uuid="11111111-aaaa-5aaa-8aaa-aaaaaaaaaaaa")
        self._answer("Two chunks share a prefix [11111111].")

        report = resolve_citations("run-cite")

        self.assertEqual(report.counts["resolved"], 0)
        self.assertEqual(report.unresolved, ["11111111"])

    def test_cited_ids_dedupes_and_keeps_order(self) -> None:
        # The final gate needs the ids the answer actually names.
        text = f"First {CHUNK}, then 2512.24601, then {CHUNK} and {GHOST}."

        self.assertEqual(cited_ids(text), [CHUNK, "2512.24601", GHOST])
        self.assertEqual(cited_ids("no ids here"), [])

    def test_duplicate_tokens_count_once(self) -> None:
        self._bundle()
        self._answer(f"{CHUNK} again {CHUNK} and {CHUNK}")

        self.assertEqual(resolve_citations("run-cite").counts["cited"], 1)

    def test_non_hydrating_tools_are_not_evidence(self) -> None:
        # deep_match names candidates; the model could not have read them.
        self._tool(1, "deep_match", [CHUNK])
        self._answer(f"Quoting {CHUNK}")

        self.assertFalse(resolve_citations("run-cite").citations[0].resolved)

    def test_missing_answer_is_an_empty_report(self) -> None:
        # A failed run must still open: no answer.md is not an exception.
        report = resolve_citations("run-cite")

        self.assertEqual(report.counts, {"cited": 0, "resolved": 0, "merged": 0, "unresolved": 0})
        self.assertEqual(report.answer_chars, 0)

    def test_damaged_payload_does_not_raise(self) -> None:
        self.traj.log_event(
            trace_ctx=TraceCtx("run-cite", 1, "tool"),
            call_type="tool",
            action="tool.dispatch",
            tool="rerank_chunks",
            args_hash="h",
            result_ref="sha256:deadbeef",  # never stored
            result_summary="boom",
        )
        self._answer(f"Quoting {CHUNK}")

        self.assertEqual(resolve_citations("run-cite").counts["resolved"], 0)

    def test_sources_md_lists_resolved_then_unresolved(self) -> None:
        self._bundle()
        self._answer(f"{CHUNK} and {GHOST}")

        report = resolve_citations("run-cite")
        path = write_sources_md(report, self.traj.run_dir / "sources.md")
        text = path.read_text(encoding="utf-8")

        self.assertIn("2512.24601", text)
        self.assertIn("## Unresolved", text)
        self.assertIn(GHOST, text)
        # answer.md stays the model's words; the trace lives beside it.
        self.assertNotIn("## Unresolved", (self.traj.run_dir / "answer.md").read_text())


class ClaimUnitTests(unittest.TestCase):
    def test_a_sentence_with_words_and_an_id_is_a_claim(self) -> None:
        unit = (
            "ReCAP keeps the prompt within a sliding window of K rounds "
            "[11111111-1111-5111-8111-111111111111]."
        )

        units = claim_units(unit)

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0][1], ["11111111-1111-5111-8111-111111111111"])

    def test_an_id_dump_is_not_a_claim(self) -> None:
        # A bullet that is only ids asserts nothing, so a verdict on it can only
        # name a non-claim as the failing one.
        text = (
            "- 85849030-2267-5a7a-8da5-2557ffd56876@2510.04532, "
            "11111111-1111-5111-8111-111111111111"
        )

        units = claim_units(text)

        # Falls back to one whole-answer unit, because nothing qualified.
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0][1], [])

    def test_an_answer_with_no_citation_is_judged_once(self) -> None:
        units = claim_units("No evidence was found for this question.")

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0][1], [])

    def test_prose_findings_survive_a_bulleted_gap_list(self) -> None:
        # Measured on a rig-v10 answer: prose findings plus a bulleted gaps section.
        # The bullet path used to replace the sentence path, so the findings vanished
        # and coverage read 0 %.
        text = (
            f"The RLM externalizes the prompt and recurses over it [{CHUNK}].\n\n"
            f"ReCAP keeps the plan in a sliding window [{OTHER}].\n\n"
            "Claims I could not support:\n"
            f"- I did not read the SLIDERS chunks in detail [{CHUNK}].\n"
            f"- I did not extract the CORG aggregation step [{OTHER}].\n"
        )

        units = claim_units(text)

        self.assertEqual(len(units), 4)
        joined = " ".join(unit for unit, _ids in units)
        self.assertIn("externalizes the prompt", joined)
        self.assertIn("sliding window", joined)

    def test_each_sentence_is_scored_separately(self) -> None:
        text = (
            "The window is K rounds [11111111-1111-5111-8111-111111111111]. "
            "The cost is bounded [22222222-2222-5222-8222-222222222222]."
        )

        self.assertEqual(len(claim_units(text)), 2)


if __name__ == "__main__":
    unittest.main()
