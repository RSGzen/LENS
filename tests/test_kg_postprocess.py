r"""Contract tests for the M7.1 KG post-process (no DB, no key, zero cost).

Pure functions only. Run:
  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from lens.kg_postprocess import (
    apply_gazetteer,
    infer_types_from_relations,
    postprocess,
    resolve_directionality,
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


if __name__ == "__main__":
    unittest.main()
