r"""Contract tests for the sandbox-side KG helpers (no DB, no key, no Docker).

``lens.kg_query`` is the *pure* module (stdlib + networkx) that is copied into the
``lens-repl`` image as ``kg_query.py``, so the RLM traverses the graph with the
same code the host re-exports from ``lens.tools.kg``. These tests pin both halves
of that promise: the module must import with no project dependencies, and the
helpers must behave the same through either import path.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_kg_query -v
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import networkx as nx

import lens.kg_query as kg_query
from lens.tools import kg as kg_tools


def sample_graph() -> nx.MultiDiGraph:
    """Two papers, one shared benchmark — enough for direction and filtering."""
    graph = nx.MultiDiGraph()
    graph.add_node("rlm", display="RLM", type="Method", type_source="model", type_confidence=0.9)
    graph.add_node("rag", display="RAG", type="Method", type_source="model", type_confidence=0.8)
    graph.add_node("hotpotqa", display="HotpotQA", type="Dataset", type_source="gazetteer", type_confidence=0.9)
    graph.add_edge(
        "rlm", "hotpotqa", key="e1",
        relation="evaluated_on", score=0.7,
        head_type="Method", tail_type="Dataset",
        chunk_uuid="c1", arxiv_id="2512.24601", section_type="Results",
    )
    graph.add_edge("rlm", "rag", key="e2", relation="outperforms", score=0.6, chunk_uuid="c2")
    return graph


class KgQueryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = sample_graph()

    def test_helpers_are_the_same_objects_as_tools_kg(self) -> None:
        # One source of truth: no duplicated copy that can drift.
        for name in ("kg_neighbors", "kg_triples", "kg_subgraph", "normalize_entity"):
            self.assertIs(getattr(kg_query, name), getattr(kg_tools, name), name)

    def test_module_has_no_project_imports(self) -> None:
        source = Path(kg_query.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from lens", source)
        self.assertNotIn("import lens", source)

    def test_module_imports_in_a_clean_interpreter(self) -> None:
        # The container has no `lens` package: import the file by path.
        code = (
            "import importlib.util,sys;"
            "spec=importlib.util.spec_from_file_location('kg_query', sys.argv[1]);"
            "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
            "print(sorted(n for n in m.__all__))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code, str(Path(kg_query.__file__))],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("kg_neighbors", out.stdout)

    def test_neighbors_follow_direction(self) -> None:
        out = kg_query.kg_neighbors(self.graph, "rlm", direction="out")
        self.assertEqual({t["relation"] for t in out}, {"evaluated_on", "outperforms"})
        in_edges = kg_query.kg_neighbors(self.graph, "RLM", direction="in")
        self.assertEqual(in_edges, [])
        both = kg_query.kg_neighbors(self.graph, "HotpotQA", direction="both")
        self.assertEqual(len(both), 1)
        self.assertEqual(both[0]["head"], "RLM")
        self.assertIn("chunk_uuid", both[0])

    def test_neighbors_filter_by_relation_and_unknown_entity(self) -> None:
        out = kg_query.kg_neighbors(self.graph, "rlm", relation="outperforms")
        self.assertEqual([t["relation"] for t in out], ["outperforms"])
        self.assertEqual(kg_query.kg_neighbors(self.graph, "unknown"), [])
        with self.assertRaises(ValueError):
            kg_query.kg_neighbors(self.graph, "rlm", direction="sideways")

    def test_triples_filters_are_and(self) -> None:
        fn = kg_query.kg_triples
        self.assertEqual(len(fn(self.graph)), 2)
        self.assertEqual(len(fn(self.graph, relation="evaluated_on")), 1)
        self.assertEqual(len(fn(self.graph, head="rlm", tail="rag")), 1)
        self.assertEqual(fn(self.graph, head="rlm", tail="HotpotQA", relation="outperforms"), [])

    def test_subgraph_is_a_copy(self) -> None:
        sub = kg_query.kg_subgraph(self.graph, ["rlm", "HotpotQA", "missing"])
        self.assertEqual(set(sub.nodes), {"rlm", "hotpotqa"})
        sub.remove_node("rlm")
        self.assertIn("rlm", self.graph)


if __name__ == "__main__":
    unittest.main()
