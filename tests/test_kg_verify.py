r"""Contract tests for M7.3 / Phase 1.5 (span filter, coref, direction verifier)."""

from __future__ import annotations

import unittest

from lens.kg_verify import (
    DirectionVerifier,
    ambiguity,
    cue_check,
    filter_spans,
    is_generic_span,
    resolve_coref,
)
from lens.models import JevResult


def triple(head, head_type, relation, tail, tail_type, **extra):
    base = {
        "head": head,
        "head_type": head_type,
        "relation": relation,
        "tail": tail,
        "tail_type": tail_type,
        "score": 0.8,
        "chunk_uuid": "u1",
        "arxiv_id": "2101.1",
        "section_type": "Experiment",
    }
    base.update(extra)
    return base


def fake_jev(choice, confidence=0.9):
    def call(state, questions):
        return JevResult(
            answers={"direction": {"type": "choice", "choice": choice, "confidence": confidence}},
            model="fake",
            provider="fake",
            tokens_in=10,
            cost_usd=0.0,
            latency_ms=1,
        )

    return call


class SpanFilterTest(unittest.TestCase):
    def test_generic_spans(self) -> None:
        for text in ("novels", "long-text scenarios", "target model", "academic terms", "it"):
            self.assertTrue(is_generic_span(text), text)

    def test_named_spans_survive(self) -> None:
        for text in ("BERT", "spaCy", "Full Context", "GPT-3.5", "SQuAD"):
            self.assertFalse(is_generic_span(text), text)

    def test_filter_drops_bad_endpoint(self) -> None:
        kept = triple("BERT", "Method", "uses", "spaCy", "Method")
        dropped = triple("novels", "Method", "outperforms", "GTE", "Method")
        self.assertEqual(filter_spans([kept, dropped]), [kept])


class CorefTest(unittest.TestCase):
    def test_resolves_pronoun_to_nearest_preceding(self) -> None:
        text = "BERT is evaluated on SQuAD. It outperforms GPT."
        triples = [
            triple("BERT", "Method", "evaluated_on", "SQuAD", "Dataset"),
            triple("It", "Method", "outperforms", "GPT", "Method"),
        ]
        out = resolve_coref(triples, {"u1": text})
        self.assertEqual(out[1]["head"], "BERT")

    def test_unresolved_pronoun_is_left_for_filter(self) -> None:
        out = resolve_coref([triple("It", "Method", "uses", "BERT", "Method")], {"u1": "It uses BERT."})
        self.assertEqual(out[0]["head"], "It")


class CueTest(unittest.TestCase):
    def test_agree_and_disagree(self) -> None:
        sentence = "BERT outperforms GPT on the benchmark."
        self.assertEqual(cue_check(sentence, "BERT", "GPT", "outperforms"), "agree")
        self.assertEqual(cue_check(sentence, "GPT", "BERT", "outperforms"), "disagree")

    def test_unknown_without_cue(self) -> None:
        self.assertEqual(
            cue_check("The result is interesting.", "A", "B", "outperforms"),
            "unknown",
        )


class AmbiguityTest(unittest.TestCase):
    def test_cue_agreement_is_not_ambiguous(self) -> None:
        t = triple("BERT", "Method", "outperforms", "GPT", "Method", direction_margin=0.5)
        self.assertFalse(ambiguity(t, "BERT outperforms GPT.")[0])

    def test_cue_disagreement_is_ambiguous(self) -> None:
        t = triple("GPT", "Method", "outperforms", "BERT", "Method", direction_margin=0.5)
        self.assertTrue(ambiguity(t, "BERT outperforms GPT.")[0])

    def test_small_margin_without_cue_is_ambiguous(self) -> None:
        t = triple("A", "Method", "outperforms", "B", "Method", direction_margin=0.01)
        self.assertTrue(ambiguity(t, "The result is interesting.")[0])

    def test_negation_is_ambiguous(self) -> None:
        t = triple("BERT", "Method", "outperforms", "GPT", "Method", direction_margin=0.5)
        self.assertTrue(ambiguity(t, "BERT does not outperform GPT.")[0])


class VerifierTest(unittest.TestCase):
    def test_flips_on_ba(self) -> None:
        triples = [triple("A", "Method", "outperforms", "B", "Method", direction_margin=0.01)]
        verifier = DirectionVerifier(fake_jev("ba"), budget=5)
        out, stats = verifier.verify(triples, {"u1": "The result is interesting."})
        self.assertEqual(out[0]["head"], "B")
        self.assertEqual(out[0]["tail"], "A")
        self.assertEqual(out[0]["direction_source"], "jev")
        self.assertEqual(stats.flipped, 1)

    def test_drops_on_neither(self) -> None:
        triples = [triple("A", "Method", "outperforms", "B", "Method", direction_margin=0.01)]
        verifier = DirectionVerifier(fake_jev("neither"), budget=5)
        out, stats = verifier.verify(triples, {"u1": "The result is interesting."})
        self.assertEqual(out, [])
        self.assertEqual(stats.dropped, 1)

    def test_budget_marks_remaining_unverified(self) -> None:
        triples = [
            triple("A", "Method", "outperforms", "B", "Method", direction_margin=0.01),
            triple("C", "Method", "outperforms", "D", "Method", direction_margin=0.02),
        ]
        verifier = DirectionVerifier(fake_jev("ab"), budget=1)
        out, stats = verifier.verify(triples, {"u1": "The result is interesting."})
        self.assertEqual(stats.verified, 1)
        self.assertEqual(stats.ambiguous, 2)
        self.assertTrue(any(t["direction_source"] == "unverified" for t in out))


if __name__ == "__main__":
    unittest.main()
