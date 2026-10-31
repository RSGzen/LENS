r"""Contract tests for the D-11 changes: KG discovery, use signal, trust counts.

The graph was buildable and untraversable: the helpers existed in the image but
were not bound, not listed in ``tools.md``, and nothing recorded whether a run ever
walked the graph — so the KG on/off ablation measured availability, not use.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_kg_discovery -v
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest

import networkx as nx

from lens.agent import context
from lens.tools.kg import kg_index_summary


def _sandbox_namespace() -> tuple[dict, list[str]]:
    """Exec the generated binding block against a fake ``kg_query`` and ``rlm_api``."""
    notes: list[str] = []

    def kg_neighbors(graph, entity, relation=None, direction="both"):
        return [("B", "uses")]

    def kg_triples(graph, head=None, relation=None, tail=None):
        return [("A", "uses", "B")]

    def kg_subgraph(graph, entities):
        return "sub"

    fake_kg_query = types.SimpleNamespace(
        kg_neighbors=kg_neighbors, kg_triples=kg_triples, kg_subgraph=kg_subgraph
    )
    fake_rlm_api = types.SimpleNamespace(note=notes.append)

    saved = {name: sys.modules.get(name) for name in ("kg_query", "rlm_api")}
    sys.modules["kg_query"] = fake_kg_query
    sys.modules["rlm_api"] = fake_rlm_api

    try:
        namespace: dict = {}
        exec(context.kg_bindings(), namespace)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    return namespace, notes


class KgBindingTestCase(unittest.TestCase):
    def test_the_helpers_are_bound_as_plain_functions(self) -> None:
        namespace, _notes = _sandbox_namespace()

        for name in context.KG_HELPER_SIGNATURES:
            self.assertIn(name, namespace, name)
            self.assertTrue(namespace[name].__doc__.startswith("KG helper"), name)

    def test_a_call_reaches_the_module_without_an_import(self) -> None:
        namespace, _notes = _sandbox_namespace()

        self.assertEqual(namespace["kg_neighbors"](object(), "A"), [("B", "uses")])

    def test_only_the_first_call_is_noted(self) -> None:
        # One note per run is the use signal; one per call would be noise the
        # trajectory already carries as tool events.
        namespace, notes = _sandbox_namespace()

        namespace["kg_neighbors"](object(), "A")
        namespace["kg_triples"](object())
        namespace["kg_subgraph"](object(), ["A"])

        self.assertEqual(notes, ["kg used: kg_neighbors"])

    def test_a_missing_kg_query_still_seeds(self) -> None:
        # An image without the module must not stop a run that never uses the graph.
        saved = sys.modules.get("kg_query")
        sys.modules.pop("kg_query", None)
        try:
            namespace: dict = {}
            exec(context.kg_bindings(), namespace)
        finally:
            if saved is not None:
                sys.modules["kg_query"] = saved

        self.assertNotIn("kg_neighbors", namespace)
        self.assertIn("KG_HELPERS", namespace)


class ContextFilesTestCase(unittest.TestCase):
    def test_tools_md_lists_the_helpers_when_the_kg_is_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = context.write_context_files(
                tmp,
                tool_manifest="1. build_kg(chunk_ids)",
                budget_snapshot={},
                kg_helpers=True,
            )
            text = paths["tools.md"].read_text(encoding="utf-8")

        self.assertIn("build_kg", text)
        self.assertIn("kg_neighbors(graph, entity", text)
        self.assertIn("direction_verified", text)

    def test_tools_md_is_untouched_when_the_kg_is_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = context.write_context_files(
                tmp, tool_manifest="1. shallow_match(buckets)", budget_snapshot={}
            )
            text = paths["tools.md"].read_text(encoding="utf-8")

        self.assertNotIn("kg_neighbors", text)

    def test_the_prompt_names_the_bound_helpers(self) -> None:
        # It used to say `kg_query` with no import and no binding; the names the
        # model can actually call are what belong in the instruction block.
        prompt = context.build_system_prompt(
            "1. build_kg(chunk_ids)",
            {},
            {},
            kg=True,
        )

        self.assertIn("kg_neighbors", prompt)
        self.assertIn("direction_verified", prompt)


class SummaryDirectionTestCase(unittest.TestCase):
    def test_the_summary_splits_verified_from_unverified_edges(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_node("A", type="Method", type_source="model")
        graph.add_node("B", type="Method", type_source="model")
        graph.add_edge("A", "B", relation="uses", direction_verified=True)
        graph.add_edge("B", "A", relation="uses", direction_verified=False)
        graph.add_edge("A", "B", relation="extends")

        summary = kg_index_summary(graph)

        self.assertEqual(summary["direction"], {"verified": 1, "unverified": 1, "unknown": 1})


if __name__ == "__main__":
    unittest.main()
