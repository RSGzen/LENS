"""Pure knowledge-graph query helpers shared by the host and the RLM sandbox.

Why this module exists
----------------------
The traversal helpers run **inside the container**, on the graph variable the RLM
built from ``rlm_api.tool("build_kg", ...)["graph"]``. The container holds only
``runner.py``, ``rlm_api.py`` and this file (see ``sandbox/Dockerfile``), so this
module must have **no project imports** — stdlib and ``networkx`` only. That is
also why it is not part of ``tools/kg.py``: that module imports the DB stack,
``config`` and the extractors, none of which exist in the sandbox.

``lens.tools.kg`` re-exports everything here, so host callers and
``tests/test_kg.py`` keep **one** source of truth. There is no second copy to
drift.

Use in the sandbox::

    import networkx as nx, kg_query
    G = nx.node_link_graph(kg["graph"])
    kg_query.kg_neighbors(G, "GPT-4", direction="in")
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any

import networkx as nx

# A triple is a plain dict (same shape end to end so it round-trips through JSON):
#   {"head", "head_type", "relation", "tail", "tail_type", "score",
#    "chunk_uuid", "arxiv_id", "section_type", ...}
Triple = dict[str, Any]


def normalize_entity(text: str) -> str:
    """Canonical **display** form of an entity surface string.

    NFC, collapse Unicode whitespace runs to a single ASCII space, strip ends,
    strip control characters. This is the only lossless normaliser in the tree: the
    ingestion path collapses whitespace per paragraph and leaves the rest alone, and
    the query path normalises nothing, so these rules apply to **entity names only**.
    **Do not** lowercase or strip symbols (that would destroy meaningful math/Greek
    tokens); case-insensitive merging is handled by ``dedup_triples`` /
    ``build_graph`` via a separate fold key.
    """
    normalized = unicodedata.normalize("NFC", text)
    normalized = " ".join(normalized.split())
    normalized = "".join(ch for ch in normalized if not unicodedata.category(ch).startswith("C"))

    return normalized


def create_triple_instance(graph: nx.MultiDiGraph, u: str, v: str, data: dict[str, Any]) -> Triple:
    """Rebuild one triple dict from an edge, using **display** names (not keys)."""
    triple: Triple = {
        "head": graph.nodes[u].get("display", u),
        "head_type": data.get("head_type") or graph.nodes[u].get("type", "Concept"),
        "head_type_source": data.get("head_type_source")
        or graph.nodes[u].get("type_source", "model"),
        "head_type_confidence": (
            data["head_type_confidence"]
            if data.get("head_type_confidence") is not None
            else graph.nodes[u].get("type_confidence", 0.5)
        ),
        "relation": data.get("relation"),
        "tail": graph.nodes[v].get("display", v),
        "tail_type": data.get("tail_type") or graph.nodes[v].get("type", "Concept"),
        "tail_type_source": data.get("tail_type_source")
        or graph.nodes[v].get("type_source", "model"),
        "tail_type_confidence": (
            data["tail_type_confidence"]
            if data.get("tail_type_confidence") is not None
            else graph.nodes[v].get("type_confidence", 0.5)
        ),
        "score": data.get("score"),
        "chunk_uuid": data.get("chunk_uuid"),
        "arxiv_id": data.get("arxiv_id"),
        "section_type": data.get("section_type"),
        "direction_verified": data.get("direction_verified"),
        "direction_confidence": data.get("direction_confidence"),
        "direction_source": data.get("direction_source"),
        "direction_margin": data.get("direction_margin"),
        "direction_reasons": data.get("direction_reasons"),
    }

    return triple


# ------------------------------------------------------------------ traversal
def kg_neighbors(
    graph: nx.MultiDiGraph,
    entity: str,
    *,
    relation: str | None = None,
    direction: str = "out",
) -> list[Triple]:
    """Triples incident to ``entity`` (case-insensitive match).

    ``direction``: ``"out"`` (entity is head), ``"in"`` (entity is tail), or
    ``"both"`` (union of both). Optional ``relation`` filters by relation name.
    Returns triple dicts (with provenance). Unknown entity -> ``[]``.
    """
    if direction not in {"out", "in", "both"}:
        raise ValueError(f"direction must be 'out', 'in' or 'both', got {direction!r}")

    key = normalize_entity(entity).casefold()
    if key not in graph:
        return []

    triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()

    def collect(edges) -> None:
        for u, v, edge_key, data in edges:
            if relation is not None and data.get("relation") != relation:
                continue
            marker = (u, v, edge_key)
            if marker in seen:  # a self-loop is both an in- and an out-edge
                continue
            seen.add(marker)
            triples.append(create_triple_instance(graph, u, v, data))

    if direction in {"out", "both"}:
        collect(graph.out_edges(key, keys=True, data=True))
    if direction in {"in", "both"}:
        collect(graph.in_edges(key, keys=True, data=True))

    return triples


def kg_triples(
    graph: nx.MultiDiGraph,
    *,
    head: str | None = None,
    relation: str | None = None,
    tail: str | None = None,
) -> list[Triple]:
    """Return edges matching any subset of ``head`` / ``relation`` / ``tail``.

    All supplied filters must match (AND); name filters are case-insensitive. All
    ``None`` -> every edge. Returns triple dicts with provenance.
    """
    head_key = normalize_entity(head).casefold() if head is not None else None
    tail_key = normalize_entity(tail).casefold() if tail is not None else None

    matched_edges: list[Triple] = []
    for u, v, _edge_key, data in graph.edges(keys=True, data=True):
        if head_key is not None and u != head_key:
            continue
        if tail_key is not None and v != tail_key:
            continue
        if relation is not None and data.get("relation") != relation:
            continue
        matched_edges.append(create_triple_instance(graph, u, v, data))

    return matched_edges


def kg_subgraph(graph: nx.MultiDiGraph, entities: Sequence[str]) -> nx.MultiDiGraph:
    """Induced subgraph over ``entities`` (case-insensitive) — a focused view.

    Returns a **copy**, so the caller cannot mutate the full graph through a view.
    Unknown entities are ignored.
    """
    keys = [normalize_entity(entity).casefold() for entity in entities]
    keys = [key for key in keys if key in graph]

    return graph.subgraph(keys).copy()


__all__ = [
    "Triple",
    "create_triple_instance",
    "kg_neighbors",
    "kg_subgraph",
    "kg_triples",
    "normalize_entity",
]
