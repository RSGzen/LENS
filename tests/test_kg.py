r"""Contract tests for M7 KG (no DB, no key, zero cost).

The pure graph helpers are tested directly; ``build_kg`` is exercised with a
``FakeExtractor`` and ``fetch_chunks`` patched (the DB row shape is not the point).
No DB, no torch — the real GLiNER/GLiREL are never loaded. The KG pipeline is
implemented; these are contract tests.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from lens import config
from lens import kg_extract
from lens.kg_extract import (
    ENTITY_LABEL_SETS,
    ENTITY_TYPES,
    RELATION_SCHEMA,
    GlinerExtractor,
)
from lens.tools.kg import (
    build_graph,
    build_kg,
    cache_key,
    dedup_triples,
    filter_ontology,
    graph_from_nodelink,
    graph_to_nodelink,
    kg_index_summary,
    kg_neighbors,
    kg_subgraph,
    kg_triples,
    load_ontology,
    normalize_entity,
    normalize_relation,
)


def triple(head, head_type, relation, tail, tail_type, score=0.9, **prov):
    base = {
        "head": head,
        "head_type": head_type,
        "relation": relation,
        "tail": tail,
        "tail_type": tail_type,
        "score": score,
        "chunk_uuid": "u1",
        "arxiv_id": "2101.1",
        "section_type": "Methodology",
    }
    base.update(prov)
    return base


class FakeExtractor:
    """Returns canned triples per chunk; records the texts it was asked to extract."""

    def __init__(self, triples_by_text=None, default=None) -> None:
        self._by_text = triples_by_text or {}
        self._default = default if default is not None else []
        self.texts: list[str] = []

    def extract(self, text):
        self.texts.append(text)
        return [dict(t) for t in self._by_text.get(text, self._default)]


class OntologyRegistryTest(unittest.TestCase):
    """Registry sanity — passes without any student logic (guards the constants)."""

    def test_entity_types_present(self) -> None:
        self.assertEqual(
            set(ENTITY_TYPES),
            {"Method", "Dataset", "Metric", "Task", "Problem", "Concept"},
        )

    def test_relation_schema_wellformed(self) -> None:
        for name, types in RELATION_SCHEMA.items():
            self.assertTrue(types["head"], name)
            self.assertTrue(types["tail"], name)
            self.assertLessEqual(set(types["head"]), set(ENTITY_TYPES))
            self.assertLessEqual(set(types["tail"]), set(ENTITY_TYPES))

    def test_glirel_labels_derived_from_schema(self) -> None:
        labels = kg_extract.relation_label_config()["glirel_labels"]
        for name, types in RELATION_SCHEMA.items():
            self.assertEqual(set(labels[name]["allowed_head"]), set(types["head"]))
            self.assertEqual(set(labels[name]["allowed_tail"]), set(types["tail"]))

    def test_load_ontology_shape(self) -> None:
        onto = load_ontology()
        self.assertEqual(onto["version"], config.KG_ONTOLOGY_VERSION)
        self.assertEqual(set(onto["entities"]), set(ENTITY_TYPES))
        self.assertEqual(set(onto["relations"]), set(RELATION_SCHEMA))


class NormalizeTest(unittest.TestCase):
    def test_entity_nfc_and_whitespace(self) -> None:
        self.assertEqual(normalize_entity("  BERT\u00a0 base  "), "BERT base")

    def test_entity_keeps_case_and_symbols(self) -> None:
        self.assertEqual(normalize_entity("λ-RLM ∈ X"), "λ-RLM ∈ X")

    def test_relation_synonym_mapping(self) -> None:
        self.assertEqual(normalize_relation("Evaluated On"), "evaluated_on")
        self.assertEqual(normalize_relation("used-for"), "uses")

    def test_relation_unknown_passes_through(self) -> None:
        self.assertEqual(normalize_relation("frobnicates"), "frobnicates")


class DedupTest(unittest.TestCase):
    def test_exact_dedup_case_insensitive(self) -> None:
        triples = [
            triple("BERT", "Method", "uses", "SQuAD", "Dataset"),
            triple("bert", "Method", "uses", "squad", "Dataset"),
        ]
        out = dedup_triples(triples)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["head"], "BERT")  # first occurrence kept

    def test_distinct_relation_not_merged(self) -> None:
        triples = [
            triple("A", "Method", "extends", "B", "Method"),
            triple("A", "Method", "outperforms", "B", "Method"),
        ]
        self.assertEqual(len(dedup_triples(triples)), 2)


class FilterOntologyTest(unittest.TestCase):
    def test_keeps_on_ontology(self) -> None:
        good = triple("BERT", "Method", "evaluated_on", "SQuAD", "Dataset")
        self.assertEqual(filter_ontology([good]), [good])

    def test_drops_unknown_relation(self) -> None:
        bad = triple("A", "Method", "frobnicates", "B", "Method")
        self.assertEqual(filter_ontology([bad]), [])

    def test_drops_type_violation(self) -> None:
        # evaluated_on requires head=Method, tail=Dataset.
        bad = triple("SQuAD", "Dataset", "evaluated_on", "BERT", "Method")
        self.assertEqual(filter_ontology([bad]), [])


class GraphTest(unittest.TestCase):
    def _graph(self):
        triples = [
            triple("BERT", "Method", "evaluated_on", "SQuAD", "Dataset",
                   chunk_uuid="u1", arxiv_id="2101.1", section_type="Experiment"),
            triple("BERT", "Method", "outperforms", "GPT", "Method",
                   chunk_uuid="u1", arxiv_id="2101.1", section_type="Experiment"),
            triple("GPT", "Method", "uses", "SQuAD", "Dataset",
                   chunk_uuid="u2", arxiv_id="2102.2", section_type="Methodology"),
        ]
        return build_graph(triples)

    def test_nodes_and_edges(self) -> None:
        g = self._graph()
        self.assertEqual(g.number_of_nodes(), 3)
        self.assertEqual(g.number_of_edges(), 3)

    def test_casefold_merge(self) -> None:
        g = build_graph([
            triple("BERT", "Method", "extends", "GPT", "Method"),
            triple("bert", "Method", "extends", "gpt", "Method"),
        ])
        self.assertEqual(g.number_of_nodes(), 2)

    def test_nodelink_roundtrip(self) -> None:
        g = self._graph()
        payload = graph_to_nodelink(g)
        g2 = graph_from_nodelink(payload)
        self.assertEqual(g.number_of_nodes(), g2.number_of_nodes())
        self.assertEqual(g.number_of_edges(), g2.number_of_edges())

    def test_edges_carry_provenance(self) -> None:
        g = self._graph()
        for _, _, _, data in g.edges(keys=True, data=True):
            self.assertIn("chunk_uuid", data)
            self.assertIn("arxiv_id", data)

    def test_index_summary(self) -> None:
        summary = kg_index_summary(self._graph())
        self.assertEqual(summary["n_nodes"], 3)
        self.assertEqual(summary["n_edges"], 3)
        self.assertEqual(summary["entity_types"]["Method"], 2)
        # Relations are counted off the edge attribute ``relation`` (not a typo'd
        # ``relations``); this guards the zero-count regression seen in the probe.
        self.assertEqual(summary["relations"]["evaluated_on"], 1)
        self.assertEqual(summary["relations"]["outperforms"], 1)
        self.assertEqual(summary["relations"]["uses"], 1)
        self.assertIn("top_entities", summary)

    def test_neighbors_out_and_both(self) -> None:
        g = self._graph()
        out = kg_neighbors(g, "bert", direction="out")   # case-insensitive
        self.assertEqual(len(out), 2)
        both = kg_neighbors(g, "SQuAD", direction="both")
        self.assertEqual(len(both), 2)

    def test_triples_filter(self) -> None:
        g = self._graph()
        got = kg_triples(g, relation="outperforms")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["head"], "BERT")

    def test_subgraph(self) -> None:
        g = self._graph()
        sub = kg_subgraph(g, ["bert", "gpt"])
        self.assertEqual(sub.number_of_nodes(), 2)
        self.assertEqual(sub.number_of_edges(), 1)


class BuildKgTest(unittest.TestCase):
    def _rows(self):
        return [
            {"chunk_uuid": "u1", "arxiv_id": "2101.1", "section_type": "Experiment",
             "section_order": 0, "chunk_text": "text-one", "token_count": 5},
            {"chunk_uuid": "u2", "arxiv_id": "2102.2", "section_type": "Methodology",
             "section_order": 1, "chunk_text": "text-two", "token_count": 5},
        ]

    def test_empty_chunks_is_empty(self) -> None:
        out = build_kg(None, [], FakeExtractor())
        self.assertEqual(out["summary"]["n_nodes"], 0)
        self.assertEqual(out["stats"]["triples_raw"], 0)

    def test_kg_disabled_is_empty(self) -> None:
        out = build_kg(None, ["u1"], FakeExtractor(), kg_enabled=False)
        self.assertEqual(out["summary"]["n_nodes"], 0)
        self.assertFalse(out["stats"]["kg_enabled"])

    def test_builds_with_provenance(self) -> None:
        by_text = {
            "text-one": [triple("BERT", "Method", "evaluated_on", "SQuAD", "Dataset")],
            "text-two": [triple("BERT", "Method", "outperforms", "GPT", "Method")],
        }
        with mock.patch("lens.tools.kg.fetch_chunks", return_value=self._rows()):
            out = build_kg(
                None, ["u1", "u2"], FakeExtractor(by_text),
                use_cache=False, postprocess=False,
            )

        self.assertEqual(out["stats"]["triples_raw"], 2)
        self.assertEqual(out["stats"]["triples_kept"], 2)
        self.assertEqual(out["summary"]["n_nodes"], 3)
        self.assertEqual(out["summary"]["n_edges"], 2)
        g = graph_from_nodelink(out["graph"])
        for _, _, _, data in g.edges(keys=True, data=True):
            self.assertIn(data["chunk_uuid"], {"u1", "u2"})

    def test_role_filter_drops_benchmark_head(self) -> None:
        by_text = {
            "text-one": [
                triple("WCEP", "Method", "evaluated_on", "BookSum", "Dataset"),
                triple("GoR", "Method", "evaluated_on", "WCEP", "Dataset"),
            ],
        }
        with mock.patch("lens.tools.kg.fetch_chunks", return_value=self._rows()[:1]):
            out = build_kg(
                None, ["u1"], FakeExtractor(by_text),
                use_cache=False, role_filter=True,
            )

        self.assertTrue(out["stats"]["role_filter"])
        self.assertEqual(out["stats"]["role_dropped"], 1)
        self.assertEqual(out["stats"]["triples_kept"], 1)

    def test_role_filter_can_be_disabled(self) -> None:
        by_text = {
            "text-one": [
                triple("WCEP", "Method", "evaluated_on", "BookSum", "Dataset"),
            ],
        }
        with mock.patch("lens.tools.kg.fetch_chunks", return_value=self._rows()[:1]):
            out = build_kg(
                None, ["u1"], FakeExtractor(by_text),
                use_cache=False, role_filter=False,
            )

        self.assertEqual(out["stats"]["role_dropped"], 0)
        self.assertEqual(out["stats"]["triples_kept"], 1)


class BuildKgCacheTest(unittest.TestCase):
    def _rows(self):
        return [
            {"chunk_uuid": "u1", "arxiv_id": "2101.1", "section_type": "Experiment",
             "section_order": 0, "chunk_text": "text-one", "token_count": 5},
            {"chunk_uuid": "u2", "arxiv_id": "2102.2", "section_type": "Methodology",
             "section_order": 1, "chunk_text": "text-two", "token_count": 5},
        ]

    def test_cache_hit_replays_full_payload(self) -> None:
        by_text = {
            "text-one": [triple("BERT", "Method", "evaluated_on", "SQuAD", "Dataset")],
            "text-two": [triple("BERT", "Method", "outperforms", "GPT", "Method")],
        }
        with tempfile.TemporaryDirectory() as cache_dir, mock.patch(
            "lens.tools.kg.fetch_chunks", return_value=self._rows()
        ):
            first = build_kg(None, ["u1", "u2"], FakeExtractor(by_text), cache_dir=cache_dir)
            second = build_kg(None, ["u1", "u2"], FakeExtractor(by_text), cache_dir=cache_dir)

        self.assertFalse(first["stats"]["cached"])
        self.assertTrue(second["stats"]["cached"])
        # A hit replays the exact build metrics, it does not fabricate them.
        self.assertEqual(second["stats"]["triples_raw"], first["stats"]["triples_raw"])
        self.assertEqual(second["stats"]["triples_kept"], first["stats"]["triples_kept"])
        self.assertEqual(second["summary"], first["summary"])
        self.assertEqual(second["graph"], first["graph"])


class CacheKeyTest(unittest.TestCase):
    def test_order_independent_and_version_sensitive(self) -> None:
        k1 = cache_key(["b", "a", "a"], "v1")
        k2 = cache_key(["a", "b"], "v1")
        k3 = cache_key(["a", "b"], "v2")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k2, k3)

    def test_extractor_fingerprint_is_part_of_the_key(self) -> None:
        k1 = cache_key(["a", "b"], "v1", extractor_fingerprint="abstract")
        k2 = cache_key(["a", "b"], "v1", extractor_fingerprint="descriptive")
        self.assertNotEqual(k1, k2)


class FakeNerModel:
    """Stand-in for GLiNER: returns canned entities (labels lower-cased, like GLiNER)."""

    def __init__(self, entities):
        self._entities = entities

    def predict_entities(self, text, labels, threshold=0.5):
        return [dict(e) for e in self._entities]


class FakeRelModel:
    """Stand-in for GLiREL: returns canned relations; records the call args."""

    def __init__(self, relations):
        self._relations = relations
        self.calls = []

    def predict_relations(self, text, labels, threshold=0.5, ner=None, top_k=-1):
        self.calls.append({"tokens": list(text), "labels": list(labels), "ner": ner})
        return [dict(r) for r in self._relations]


class ExtractPipelineTest(unittest.TestCase):
    def _extractor(self, entities, relations):
        return GlinerExtractor(
            ner_model=FakeNerModel(entities),
            rel_model=FakeRelModel(relations),
        )

    def test_blank_returns_empty_without_calling_models(self) -> None:
        ex = self._extractor([], [])
        self.assertEqual(ex.extract("   "), [])

    def test_fewer_than_two_entities_returns_empty(self) -> None:
        ex = self._extractor(
            [{"start": 0, "end": 4, "text": "BERT", "label": "method", "score": 0.9}],
            [],
        )
        self.assertEqual(ex.extract("BERT is a model."), [])

    def test_char_to_token_mapping_multi_token(self) -> None:
        # "The BERT base model uses SQuAD." under GLiREL's regex.
        spans = [(0, 3), (4, 8), (9, 13), (14, 19), (20, 24), (25, 30), (30, 31)]
        ents = [{"start": 4, "end": 13, "text": "BERT base", "label": "Method", "score": 0.9}]
        ex = GlinerExtractor(ner_model=FakeNerModel(ents), rel_model=FakeRelModel([]))
        self.assertEqual(
            ex.map_entity_token_indices(spans, ents),
            [[1, 2, "Method", "BERT base"]],
        )

    def test_extract_assembles_triple_and_canonicalises_type(self) -> None:
        text = "BERT is evaluated on SQuAD."
        ents = [
            {"start": 0, "end": 4, "text": "BERT", "label": "method", "score": 0.9},
            {"start": 21, "end": 26, "text": "SQuAD", "label": "dataset", "score": 0.9},
        ]
        # head_pos/tail_pos are end-EXCLUSIVE (GLiREL adds +1).
        rels = [{
            "head_pos": [0, 1], "tail_pos": [4, 5],
            "head_text": ["BERT"], "tail_text": ["SQuAD"],
            "label": "evaluated_on", "score": 0.88,
        }]
        ex = self._extractor(ents, rels)
        out = ex.extract(text)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["head"], "BERT")
        self.assertEqual(out[0]["head_type"], "Method")   # canonicalised from "method"
        self.assertEqual(out[0]["relation"], "evaluated_on")
        self.assertEqual(out[0]["tail"], "SQuAD")
        self.assertEqual(out[0]["tail_type"], "Dataset")
        self.assertAlmostEqual(out[0]["score"], 0.88)

    def test_descriptive_labels_map_to_canonical(self) -> None:
        text = "BERT is evaluated on SQuAD."
        ents = [
            {"start": 0, "end": 4, "text": "BERT",
             "label": "machine learning method, model, or algorithm", "score": 0.9},
            {"start": 21, "end": 26, "text": "SQuAD",
             "label": "dataset, corpus, or benchmark", "score": 0.9},
        ]
        rels = [{
            "head_pos": [0, 1], "tail_pos": [4, 5],
            "head_text": ["BERT"], "tail_text": ["SQuAD"],
            "label": "evaluated_on", "score": 0.88,
        }]
        ex = GlinerExtractor(
            ner_model=FakeNerModel(ents),
            rel_model=FakeRelModel(rels),
            entity_labels=ENTITY_LABEL_SETS["descriptive"],
        )
        out = ex.extract(text)
        self.assertEqual(out[0]["head_type"], "Method")
        self.assertEqual(out[0]["tail_type"], "Dataset")

    def test_text_windows_respect_max_tokens_with_overlap(self) -> None:
        ex = GlinerExtractor(
            ner_model=FakeNerModel([]),
            rel_model=FakeRelModel([]),
            max_tokens=5,
            overlap=2,
        )
        windows = ex._text_windows(
            "one two three four five six seven eight nine ten"
        )
        self.assertGreaterEqual(len(windows), 2)
        for window in windows:
            self.assertLessEqual(len(window.split()), 5)
        self.assertTrue(windows[0].endswith("five"))
        self.assertTrue(windows[1].startswith("four five"))  # overlap carried

    def test_short_text_is_one_window(self) -> None:
        ex = self._extractor([], [])
        self.assertEqual(len(ex._text_windows("short text here.")), 1)

    def test_fingerprint_differs_by_label_set(self) -> None:
        a = GlinerExtractor(
            ner_model=FakeNerModel([]),
            rel_model=FakeRelModel([]),
            entity_labels=ENTITY_LABEL_SETS["abstract"],
        )
        b = GlinerExtractor(
            ner_model=FakeNerModel([]),
            rel_model=FakeRelModel([]),
            entity_labels=ENTITY_LABEL_SETS["descriptive"],
        )
        self.assertNotEqual(a.fingerprint, b.fingerprint)

    def test_dotted_entity_uses_surface_text(self) -> None:
        # GLiREL tokenises "GPT-3.5" as ["GPT-3", ".", "5"]; the triple must keep
        # GLiNER's surface span ("GPT-3.5"), not the space-joined token list.
        text = "GPT-3.5 is evaluated on SQuAD."
        ents = [
            {"start": 0, "end": 7, "text": "GPT-3.5", "label": "method", "score": 0.9},
            {"start": 24, "end": 29, "text": "SQuAD", "label": "dataset", "score": 0.9},
        ]
        rels = [{
            "head_pos": [0, 3], "tail_pos": [6, 7],
            "head_text": ["GPT-3", ".", "5"], "tail_text": ["SQuAD"],
            "label": "evaluated_on", "score": 0.9,
        }]
        out = self._extractor(ents, rels).extract(text)
        self.assertEqual(out[0]["head"], "GPT-3.5")
        self.assertEqual(out[0]["tail"], "SQuAD")


if __name__ == "__main__":
    unittest.main()
