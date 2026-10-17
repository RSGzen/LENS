r"""M7 — knowledge-graph tool: ``build_kg`` + graph assembly + RLM helpers.

Spec: ``MVP Build Roadmap.md`` §4 (M7); ``Thesis Drafts/Tech Stack (Draft).md``
§2 (stage 12), §4i. Decisions: D-06/A-02 (query-time, deterministic), K-01
(ontology), K-02 (dedup), K-06 (cache + latency), K-07 (KG <-> RLM contract).

Pipeline::

    chunk_ids -> fetch_chunks (T2) -> per-chunk text
              -> extractor.extract(text)            (lens.kg_extract, host-side)
              -> filter_ontology -> dedup_triples   (deterministic)
              -> build_graph -> graph_to_nodelink    (networkx)
              -> cache by (paper set + ontology ver + extractor id)  (K-06)
    -> {"kg_id", "graph": <node-link>, "summary", "stats"}

The RLM contract (K-07)
-----------------------
The full graph travels as **node-link JSON** and is loaded in the sandbox into a
``networkx`` namespace variable (``networkx`` is baked into the ``lens-repl``
image), so the root model navigates it with Python rather than dumping it into
context. Only :func:`kg_index_summary` — a compact index — is small enough to sit
in root context, and it is what tells the RLM what is in the graph. Traversal
helpers (:func:`kg_neighbors`, :func:`kg_triples`, :func:`kg_subgraph`) run
sandbox-side on that variable.

Traceability (the M7 verify clause)
-----------------------------------
Every triple carries its source chunk (``chunk_uuid``, ``arxiv_id``,
``section_type``), so any edge in the graph resolves back to the exact chunk it
came from.

Scaffold-and-fill (LOG-33)
--------------------------
Signatures, type hints, docstrings, and tests land here; the **student fills the
logic bodies** (``raise NotImplementedError``). ``tests/test_kg.py`` is the
contract (no DB, no key, zero cost).
"""

from __future__ import annotations

import hashlib
import json
import unicodedata

from collections.abc import Sequence
from collections import Counter
from typing import Any
from heapq import nlargest
from pathlib import Path

import networkx as nx

from lens import config
from lens.kg_extract import (
    ENTITY_TYPES,
    ONTOLOGY_VERSION,
    RELATION_SCHEMA,
    KGExtractor,
)
from lens.tools.retrieval import fetch_chunks

# A triple is a plain dict (same shape end to end so it round-trips through JSON):
#   {"head", "head_type", "relation", "tail", "tail_type", "score",
#    "chunk_uuid", "arxiv_id", "section_type"}
Triple = dict[str, Any]


# ------------------------------------------------------------------ ontology
def load_ontology() -> dict[str, Any]:
    """Return the active ontology as registry data (K-01).

    Shape::

        {"version": <str>,
         "entities": [<str>, ...],
         "relations": {<name>: {"head": [...], "tail": [...]}, ...}}

    This is the in-memory mirror of ``docs/ontology.md``; the harness will also
    copy that file into the sandbox workspace as a context file (Tech Stack §4).
    """
    ontology_dict = {
        "version": ONTOLOGY_VERSION,
        "entities": ENTITY_TYPES,
        "relations": RELATION_SCHEMA
    }

    return ontology_dict


# ------------------------------------------------------------ text normalisation
def normalize_entity(text: str) -> str:
    """Canonical **display** form of an entity surface string.

    NFC, collapse Unicode whitespace runs to a single ASCII space, strip ends,
    strip control characters (same lossless rules as ``chunking.normalize_text``).
    **Do not** lowercase or strip symbols (that would destroy meaningful math/Greek
    tokens); case-insensitive merging is handled by :func:`dedup_triples` /
    :func:`build_graph` via a separate fold key.
    """
    # Unicode canonical composition first, then collapse whitespace and strip
    # control characters. No lowercasing / NFKC / symbol stripping (math survives).
    normalized = unicodedata.normalize("NFC", text)
    normalized = " ".join(normalized.split())
    normalized = "".join(
        ch for ch in normalized if not unicodedata.category(ch).startswith("C")
    )

    return normalized


# Conservative surface-synonym backstop. GLiREL returns the canonical relation
# names we pass it, so this only matters for a joint model or a renamed label.
# Do NOT add inverse relations (e.g. ``extended_by``) — they would flip edge
# direction. Unknown labels fall through unchanged so ``filter_ontology`` drops them.
_RELATION_SYNONYMS = {
    "used_for": "uses",
    "utilizes": "uses",
    "utilises": "uses",
    "evaluate_for": "evaluated_on",
    "evaluated_for": "evaluated_on",
    "compared_to": "compares_to",
    "beats": "outperforms",
    "solves": "addresses",
    "targets": "addresses",
    "investigates": "explores",
}


def normalize_relation(label: str) -> str:
    """Map a raw relation label to a canonical ontology relation name.

    Lowercase, trim, spaces/hyphens -> underscores, then map common surface
    synonyms onto :data:`RELATION_SCHEMA` keys via :data:`_RELATION_SYNONYMS`
    (e.g. ``"used for"`` -> ``"uses"``, ``"evaluate for"`` -> ``"evaluated_on"``).
    Return the input unchanged if it is already canonical; return it as-is (so
    :func:`filter_ontology` can drop it) if it maps to nothing.
    """
    if label in RELATION_SCHEMA:
        return label

    normalized = label.lower().strip().replace(" ", "_").replace("-", "_")

    return _RELATION_SYNONYMS.get(normalized, normalized)


# ------------------------------------------------------------------ dedup / filter
def dedup_triples(triples: Sequence[Triple]) -> list[Triple]:
    """Exact dedup (K-02, first half) — case-/whitespace-insensitive, order-stable.

    Key = ``(fold(head), relation, fold(tail))`` where ``fold`` is
    ``normalize_entity`` + casefold. The **first** occurrence is kept (its display
    casing and score survive). No fuzzy/entity-resolution merging here — that is a
    deferred sensitivity item (K-02 remainder), not part of M7.
    """
    normalized_triples = []
    key_check = []

    for triple in triples:
        normalized_head = normalize_entity(triple["head"]).casefold()
        normalized_tail = normalize_entity(triple["tail"]).casefold()

        key = (normalized_head, triple["relation"], normalized_tail)

        if key not in key_check:
            key_check.append(key)

            temp_insert = Triple(
                {
                    "head": triple["head"],
                    "head_type": triple["head_type"],
                    "relation": triple["relation"],
                    "tail": triple["tail"],
                    "tail_type": triple["tail_type"],
                    "score": triple["score"],
                    "chunk_uuid": triple["chunk_uuid"],
                    "arxiv_id": triple["arxiv_id"],
                    "section_type": triple["section_type"]
                }
            )

            normalized_triples.append(temp_insert)

    return normalized_triples


def filter_ontology(triples: Sequence[Triple]) -> list[Triple]:
    """Drop off-ontology triples (the M7 "entities/relations in ontology" check).

    Keep a triple only when:
      * ``relation`` is a key of :data:`RELATION_SCHEMA`,
      * ``head_type`` is one of that relation's allowed ``head`` types, and
      * ``tail_type`` is one of its allowed ``tail`` types.
    Also drop malformed triples (missing/empty head or tail). Return the kept list
    in input order.
    """
    allowed_triples = []
    
    for triple in triples:
        t_relation = triple["relation"]

        if t_relation not in list(RELATION_SCHEMA.keys()):
            continue

        t_r_dict = RELATION_SCHEMA[t_relation]

        if triple["head_type"] not in list(t_r_dict["head"]):
            continue

        if triple["tail_type"] not in list(t_r_dict["tail"]):
            continue

        allowed_triples.append(triple)

    return allowed_triples


# ------------------------------------------------------------------ graph build
def build_graph(triples: Sequence[Triple]) -> nx.MultiDiGraph:
    """Assemble a :class:`networkx.MultiDiGraph` from triples.

    Nodes are entities keyed by a **case-folded** canonical key (so ``"BERT"`` and
    ``"bert"`` merge); each node carries ``type`` (first seen) and ``display`` (the
    first-seen display string). Edges are directed head -> tail, ``key`` = relation
    name, with edge attrs ``relation``, ``score``, ``chunk_uuid``, ``arxiv_id``,
    ``section_type``. ``MultiDiGraph`` because the same pair can hold several
    relations / sources.
    """
    graph = nx.MultiDiGraph()

    for triple in triples:
        key_head = normalize_entity(triple["head"]).casefold()

        if key_head not in graph:
            graph.add_node(key_head, 
                           type=triple["head_type"],
                           display=triple["head"])

        key_tail = normalize_entity(triple["tail"]).casefold()
        
        if key_tail not in graph:
            graph.add_node(key_tail, 
                           type=triple["tail_type"],
                           display=triple["tail"])

        graph.add_edge(
            u_for_edge=key_head, 
            v_for_edge=key_tail, 
            key=triple["relation"],
            relation=triple["relation"],
            score=triple["score"],
            chunk_uuid=triple["chunk_uuid"],
            arxiv_id=triple["arxiv_id"],
            section_type=triple["section_type"]
        )

    return graph


def graph_to_nodelink(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Serialise ``graph`` to node-link JSON (``networkx.node_link_data``).

    This is the payload handed to the sandbox; it must round-trip through
    :func:`graph_from_nodelink` and through ``json.dumps`` (plain Python types only).
    """
    graph_data = nx.node_link_data(graph)

    return graph_data


def graph_from_nodelink(payload: dict[str, Any]) -> nx.MultiDiGraph:
    """Rebuild a graph from a :func:`graph_to_nodelink` payload (sandbox side)."""
    graph = nx.node_link_graph(payload)

    return graph


def kg_index_summary(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Compact index of the graph — the only KG artefact that enters root context.

    Shape::

        {"ontology_version": <str>,
         "n_nodes": <int>, "n_edges": <int>,
         "entity_types": {<type>: <count>, ...},
         "relations": {<relation>: <count>, ...},
         "top_entities": [{"entity": <str>, "type": <str>, "degree": <int>}, ...]}

    ``top_entities`` is the highest-degree entities, capped at
    :data:`config.KG_SUMMARY_TOP_ENTITIES` so the RLM sees the graph's hubs without
    the graph itself. Graph topology lives **here**, not in ``build_kg``'s stats.
    """
    relations_dict = {}
    target_attr = "relations"

    for relation in list(RELATION_SCHEMA.keys()):
        count = sum(
            1
            for u, v, k, data in graph.edges(data=True, keys=True)
            if data.get(target_attr) == relation
        )

        relations_dict[relation] = count

    top = nlargest(config.KG_SUMMARY_TOP_ENTITIES, graph.degree(), key=lambda nd: nd[1])

    top_entities = [
        {
            "entity": graph.nodes[n].get("display", n),
            "type": graph.nodes[n].get("type", "Concept"),
            "degree": d,
        }
        for n, d in top
    ]
    entity_types = Counter(graph.nodes[n].get("type", "Concept") for n in graph)

    summary_dict = {
        "ontology_version": ONTOLOGY_VERSION,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "entity_types": dict(entity_types),
        "relations": relations_dict,
        "top_entities": top_entities
    }

    return summary_dict

# ------------------------------------------------------------------ helper functions

def create_triple_instance(graph: nx.MultiDiGraph, u: str, v: str, data: dict[str, Any]) -> Triple:
    """Rebuild one triple dict from an edge, using **display** names (not keys)."""
    triple: Triple = {
        "head": graph.nodes[u].get("display", u),
        "head_type": graph.nodes[u].get("type", "Concept"),
        "relation": data.get("relation"),
        "tail": graph.nodes[v].get("display", v),
        "tail_type": graph.nodes[v].get("type", "Concept"),
        "score": data.get("score"),
        "chunk_uuid": data.get("chunk_uuid"),
        "arxiv_id": data.get("arxiv_id"),
        "section_type": data.get("section_type"),
    }

    return triple

# ------------------------------------------------------------------ cache (K-06)
def cache_key(
    arxiv_ids: Sequence[str],
    ontology_version: str,
    *,
    ner_model_id: str = config.KG_NER_MODEL_ID,
    rel_model_id: str = config.KG_REL_MODEL_ID,
    min_score: float = config.KG_MIN_SCORE,
) -> str:
    """Deterministic cache key for a subgraph (K-06).

    The key covers **everything that changes the extracted graph**: the paper set,
    the ontology version, the NER/RE checkpoints, and the minimum score. It is a
    canonical JSON payload (sorted keys; sorted + de-duplicated ``arxiv_ids``)
    hashed with SHA-256, so the key is order-independent and delimiter-safe.
    Return a hex digest.

    **Caveat — extractor identity is part of the key.** Without the model ids and
    the threshold, swapping ``KG_NER_MODEL_ID`` / ``KG_REL_MODEL_ID`` or tuning
    ``KG_MIN_SCORE`` would silently serve a graph built by a *different* extractor.
    Extraction *code* changes (prompts, post-processing) are not hashed, so bump
    ``KG_ONTOLOGY_VERSION`` or clear ``<cache_root>/kg`` when you change those.
    """
    payload = json.dumps(
        {
            "arxiv_ids": sorted(set(arxiv_ids)),
            "ontology_version": ontology_version,
            "ner_model_id": ner_model_id,
            "rel_model_id": rel_model_id,
            "min_score": str(float(min_score)),
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ tool (host)
def build_kg(
    conn: Any,
    chunk_ids: Sequence[str],
    extractor: KGExtractor,
    *,
    ontology_version: str = ONTOLOGY_VERSION,
    cache_dir: Any | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Build the query-time KG over ``chunk_ids`` (M7 entry point).

    Steps:
      1. Empty ``chunk_ids`` -> an empty graph + zeroed stats (no DB read).
      2. ``rows = fetch_chunks(conn, chunk_ids)`` (T2); re-order to the input order.
      3. Cache lookup (K-06): ``key = cache_key(<distinct arxiv_ids>,
         ontology_version)``; on a hit, load the **full cached payload** (graph +
         summary + stats), set ``stats["cached"] = True`` and return it.
      4. For each chunk, ``extractor.extract(row["chunk_text"])`` and attach
         provenance (``chunk_uuid``, ``arxiv_id``, ``section_type``) to every triple;
         normalize head/tail via :func:`normalize_entity` and the relation via
         :func:`normalize_relation`.
      5. ``filter_ontology`` -> ``dedup_triples`` -> ``build_graph``.
      6. Serialise (``graph_to_nodelink``) + :func:`kg_index_summary`; cache the
         **whole payload** under ``<cache_dir>/<key>.json`` when ``use_cache``.

    Returns::

        {"kg_id": <cache_key>, "graph": <node-link dict>,
         "summary": {"ontology_version", "n_nodes", "n_edges", "entity_types",
                     "relations", "top_entities"},
         "stats": {"chunks": int, "triples_raw": int, "triples_kept": int,
                   "cached": bool}}

    ``summary`` owns the graph topology; ``stats`` owns build bookkeeping only, so
    ``n_nodes`` / ``n_edges`` are **not** duplicated here. ``triples_kept`` is the
    extraction yield after ontology-filter + dedup (it currently equals
    ``summary["n_edges"]`` because each deduped triple becomes one edge).

    The full payload is cached, not just the graph, so a cache hit can report the
    exact ``triples_raw`` / ``triples_kept`` instead of fabricating them.

    ``kg_id`` is the cache key, so two calls over the same paper set + ontology
    version + extractor identity share one id and one cache entry.
    """
    if not chunk_ids:
        empty = nx.MultiDiGraph()
        return {
            "kg_id": cache_key([], ontology_version),
            "graph": graph_to_nodelink(empty),
            "summary": kg_index_summary(empty),
            "stats": {"chunks": 0, "triples_raw": 0, "triples_kept": 0, "cached": False},
        }

    chunk_rows = fetch_chunks(conn, chunk_ids)

    # Re-order to the input (ANN) order and collect the paper set for the key.
    by_id = {row["chunk_uuid"]: row for row in chunk_rows}
    reordered_chunk_rows = [by_id[cid] for cid in chunk_ids if cid in by_id]
    arxiv_ids = [row["arxiv_id"] for row in reordered_chunk_rows]

    key = cache_key(arxiv_ids=arxiv_ids, ontology_version=ontology_version)

    cache_path: Path | None = None
    if use_cache:
        cache_root = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(config.LENS_CACHE_ROOT_PATH) / config.KG_CACHE_DIR_NAME
        )
        cache_path = cache_root / f"{key}.json"
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload["stats"]["cached"] = True
            return payload

    triples_kept: list[Triple] = []
    triples_raw = 0

    for chunk in reordered_chunk_rows:
        raw_triples = extractor.extract(chunk["chunk_text"])
        triples_raw += len(raw_triples)

        for raw_triple in raw_triples:
            triples_kept.append(
                {
                    "head": normalize_entity(raw_triple["head"]),
                    "head_type": raw_triple["head_type"],
                    "relation": normalize_relation(raw_triple["relation"]),
                    "tail": normalize_entity(raw_triple["tail"]),
                    "tail_type": raw_triple["tail_type"],
                    "score": raw_triple["score"],
                    "chunk_uuid": chunk["chunk_uuid"],
                    "arxiv_id": chunk["arxiv_id"],
                    "section_type": chunk["section_type"],
                }
            )

    triples_kept = dedup_triples(filter_ontology(triples_kept))
    graph = build_graph(triples_kept)

    payload = {
        "kg_id": key,
        "graph": graph_to_nodelink(graph),
        "summary": kg_index_summary(graph),
        "stats": {
            "chunks": len(reordered_chunk_rows),
            "triples_raw": triples_raw,
            "triples_kept": len(triples_kept),
            "cached": False,
        },
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    return payload

# ------------------------------------------------------------ RLM helpers (sandbox)
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
