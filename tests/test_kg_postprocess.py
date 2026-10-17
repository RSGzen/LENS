r"""Contract tests for the M7.1 KG post-process (no DB, no key, zero cost).

Pure functions only. Run:
  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from lens.kg_postprocess import (
    apply_gazetteer,
    apply_role_filter,
    infer_types_from_relations,
    is_benchmark_span,
    postprocess,
    resolve_directionality,
    suffix_type_hint,
)


def triple(head, head_type, relation, tail, tail_type, score=0.9, ctx="u1"):
    return {
        "head": head,
        "head_type": head_type,
        "relation": relation,
        "tail": tail,
        "tail_type": tail_type,
        "score": score,
        "chunk_uuid": ctx,
        "arxiv_id": "2101.1",
        "section_type": "Experiment",
    }


class RelationTypingTest(unittest.TestCase):
    def test_evaluated_on_tail_is_dataset(self) -> None:
        out = infer_types_from_relations(
            [triple("BERT", "Method", "evaluated_on", "SQuAD", "Method")]
        )[0]
        self.assertEqual(out["tail_type"], "Dataset")
        self.assertEqual(out["tail_type_source"], "relation")

    def test_measures_tail_is_metric(self) -> None:
        out = infer_types_from_relations(
            [triple("BERT", "Method", "measures", "F1", "Concept")]
        )[0]
        self.assertEqual(out["tail_type"], "Metric")

    def test_outperforms_forces_method_both_sides(self) -> None:
        out = infer_types_from_relations(
            [triple("BERT", "Concept", "outperforms", "GPT", "Concept")]
        )[0]
        self.assertEqual(out["head_type"], "Method")
        self.assertEqual(out["tail_type"], "Method")

    def test_ambiguous_uses_tail_keeps_type(self) -> None:
        out = infer_types_from_relations(
            [triple("BERT", "Method", "uses", "SQuAD", "Dataset")]
        )[0]
        self.assertEqual(out["tail_type"], "Dataset")


class GazetteerTest(unittest.TestCase):
    def test_override(self) -> None:
        gazetteer = {"squad": {"name": "SQuAD", "type": "Dataset"}}
        out = apply_gazetteer(
            [triple("BERT", "Method", "uses", "SQuAD", "Method")], gazetteer
        )[0]
        self.assertEqual(out["tail_type"], "Dataset")
        self.assertEqual(out["tail_type_source"], "gazetteer")

    def test_low_confidence_is_ignored(self) -> None:
        gazetteer = {"squad": {"name": "SQuAD", "type": "Dataset", "confidence": 0.7}}
        out = apply_gazetteer(
            [triple("BERT", "Method", "uses", "SQuAD", "Method")], gazetteer
        )[0]
        self.assertEqual(out["tail_type"], "Method")
        self.assertEqual(out.get("tail_type_source", "model"), "model")


class SuffixHintTest(unittest.TestCase):
    def test_suffix_types(self) -> None:
        out = suffix_type_hint(
            [triple("five open-domain QA datasets", "Method", "uses", "ROUGE score", "Method")]
        )[0]
        self.assertEqual(out["head_type"], "Dataset")
        self.assertEqual(out["tail_type"], "Metric")
        self.assertEqual(out["head_type_source"], "suffix")

    def test_no_suffix_unchanged(self) -> None:
        out = suffix_type_hint([triple("BERT", "Method", "uses", "spaCy", "Method")])[0]
        self.assertEqual(out["tail_type"], "Method")


class DirectionalityTest(unittest.TestCase):
    def test_drop_self_loop(self) -> None:
        out = resolve_directionality(
            [triple("BERT", "Method", "outperforms", "BERT", "Method")]
        )
        self.assertEqual(out, [])

    def test_same_context_keeps_higher_direction(self) -> None:
        out = resolve_directionality(
            [
                triple("A", "Method", "outperforms", "B", "Method", score=0.6),
                triple("B", "Method", "outperforms", "A", "Method", score=0.8),
            ]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["head"], "B")
        self.assertAlmostEqual(out[0]["score"], 0.8)

    def test_different_contexts_keep_both(self) -> None:
        out = resolve_directionality(
            [
                triple("A", "Method", "outperforms", "B", "Method", score=0.6, ctx="u1"),
                triple("B", "Method", "outperforms", "A", "Method", score=0.8, ctx="u2"),
            ]
        )
        self.assertEqual(len(out), 2)

    def test_symmetric_relation_single_edge(self) -> None:
        out = resolve_directionality(
            [
                triple("A", "Method", "compares_to", "B", "Method", score=0.6),
                triple("B", "Method", "compares_to", "A", "Method", score=0.8),
            ]
        )
        self.assertEqual(len(out), 1)

    def test_cycle_break_drops_lowest_score(self) -> None:
        out = resolve_directionality(
            [
                triple("A", "Method", "outperforms", "B", "Method", score=0.9),
                triple("B", "Method", "outperforms", "C", "Method", score=0.8),
                triple("C", "Method", "outperforms", "A", "Method", score=0.7),
            ],
            cycle_break=True,
        )
        self.assertEqual(len(out), 2)
        self.assertNotIn(
            ("C", "A"), {(t["head"], t["tail"]) for t in out}
        )


class BenchmarkSpanTest(unittest.TestCase):
    def test_strict_positives(self) -> None:
        for name in (
            "WCEP", "BookSum", "QM-Sum", "AcademicEval", "2WikiMultihopQA",
            "MultiFieldQA", "Long-Bench", "EN.MC", "En.QA", "biomedical QA",
        ):
            self.assertTrue(is_benchmark_span(name), name)

    def test_strict_negatives(self) -> None:
        for name in (
            "quality-aware retrieval", "retrieval", "checksum", "BERT", "GPT-4o",
            "ARGUE F1", "Mixtral", "GoR",
        ):
            self.assertFalse(is_benchmark_span(name), name)

    def test_broad_keeps_substring_behaviour(self) -> None:
        # Scoring leniency may still read "quality-aware" as QuALITY.
        self.assertTrue(is_benchmark_span("quality-aware retrieval", broad=True))


class RoleFilterTest(unittest.TestCase):
    def test_drop_benchmark_head_of_evaluated_on(self) -> None:
        out, dropped = apply_role_filter(
            [triple("WCEP", "Method", "evaluated_on", "BookSum", "Dataset")]
        )
        self.assertEqual(out, [])
        self.assertEqual(dropped, 1)

    def test_drop_benchmark_head_of_outperforms(self) -> None:
        out, dropped = apply_role_filter(
            [triple("biomedical QA", "Method", "outperforms", "GPT-4o", "Method")]
        )
        self.assertEqual(out, [])
        self.assertEqual(dropped, 1)

    def test_keep_benchmark_head_of_measures(self) -> None:
        out, dropped = apply_role_filter(
            [triple("2WikiMultihopQA", "Method", "measures", "F1", "Metric")]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)

    def test_drop_benchmark_tail_of_measures(self) -> None:
        out, dropped = apply_role_filter(
            [triple("GoR", "Method", "measures", "QASPER", "Dataset")]
        )
        self.assertEqual(out, [])
        self.assertEqual(dropped, 1)

    def test_keep_benchmark_tail_of_evaluated_on(self) -> None:
        out, dropped = apply_role_filter(
            [triple("GoR", "Method", "evaluated_on", "WCEP", "Dataset")]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)

    def test_keep_method_outperforms(self) -> None:
        out, dropped = apply_role_filter(
            [triple("Mixtral", "Method", "outperforms", "GPT-4o", "Method")]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)

    def test_keep_task_tail_of_addresses(self) -> None:
        out, dropped = apply_role_filter(
            [triple("BERT", "Method", "addresses", "question answering", "Task")]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)

    def test_drop_benchmark_tail_of_addresses(self) -> None:
        out, dropped = apply_role_filter(
            [triple("BERT", "Method", "addresses", "QASPER", "Dataset")]
        )
        self.assertEqual(out, [])
        self.assertEqual(dropped, 1)

    def test_metric_head_is_kept_without_metric_hints(self) -> None:
        # ARGUE F1 is a Metric typed Method; metric hints are held for now.
        out, dropped = apply_role_filter(
            [triple("ARGUE F1", "Method", "evaluated_on", "NeuCLIR", "Dataset")]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(dropped, 0)

    def test_role_filter_runs_before_directionality(self) -> None:
        # The invalid direction has the higher score. If directionality ran first
        # it would keep QASPER -> BM25, then the role filter would drop it and the
        # valid BM25 -> QASPER edge would be lost.
        out = postprocess(
            [
                triple("QASPER", "Dataset", "uses", "BM25", "Method", score=0.9),
                triple("BM25", "Method", "uses", "QASPER", "Dataset", score=0.5),
            ],
            role_filter=True,
            directionality=True,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["head"], "BM25")


class PipelineTest(unittest.TestCase):
    def test_soft_fields_default_to_model(self) -> None:
        out = postprocess(
            [triple("BERT", "Method", "evaluated_on", "SQuAD", "Dataset")],
            type_inference=False,
            directionality=False,
        )[0]
        self.assertEqual(out["head_type_source"], "model")
        self.assertEqual(out["head_type_confidence"], 0.5)

    def test_pipeline_runs_all_stages(self) -> None:
        out = postprocess(
            [
                triple("BERT", "Method", "evaluated_on", "SQuAD", "Method"),
                triple("BERT", "Method", "outperforms", "BERT", "Method"),
            ],
            gazetteer={},
            type_inference=True,
            directionality=True,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tail_type"], "Dataset")

    def test_role_filter_stats(self) -> None:
        stats: dict = {}
        out = postprocess(
            [triple("WCEP", "Method", "evaluated_on", "BookSum", "Dataset")],
            stats=stats,
        )
        self.assertEqual(out, [])
        self.assertEqual(stats["role_dropped"], 1)

    def test_role_filter_can_be_disabled(self) -> None:
        out = postprocess(
            [triple("WCEP", "Method", "evaluated_on", "BookSum", "Dataset")],
            role_filter=False,
        )
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
