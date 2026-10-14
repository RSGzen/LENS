r"""M5 — retrieval tools: the T1 funnel + T2 fetch-by-id.

Spec: ``MVP Build Roadmap.md`` §4 (M5); ``Thesis Drafts/Tech Stack (Draft).md``
§2 (stages 10-11), §3 (HNSW tuning), §4i (the retrieval / hydration / asset
contract).

Funnel — the graded path (R-01)::

    shallow_match   global HNSW over papers.abstract_embedding   -> k arxiv_ids
    rerank_papers   bge-reranker-v2-m3 (Stage A) over abstracts  -> n papers
    deep_match      filtered HNSW over chunks within the n       -> C chunk_uuids
    rerank_chunks   bge-reranker-v2-m3 (Stage B) over chunk_text -> c hydrated

Design rules (do not deviate)
-----------------------------
- **The index buys candidate recall; the cross-encoder buys top-precision.** A
  reranker cannot recover a candidate the HNSW scan never returned, so end-to-end
  quality is bounded by candidate recall — hence the generous ``k=100 -> n=20`` /
  ``C=100 -> c=5`` funnel and why ``k`` / ``C`` are the recall knobs.
- **Hydration is host-side, at the deep stage only** (§4i): ``shallow_match``,
  ``rerank_papers`` and ``deep_match`` return ids only; ``rerank_chunks`` returns
  the ``{chunks, papers, assets}`` payload the orchestrator ``emit()``s.
- **The RLM never issues SQL here.** Every function runs on a host connection
  (``lens.db.connect``); in recorded runs the host uses the read-only ``lens_ro``
  role. The sandbox holds no DB driver (LOG-32).

Query input
-----------
The four funnel tools take ``buckets`` — the M6 segmentation output (a mapping of
bucket name -> clause text), or a plain query string. :func:`format_query`
normalises both into the single string that is embedded (with the nomic
``search_query:`` prefix) at the shallow/deep stages and passed to the
cross-encoder at the rerank stages. Document and query paths **must** use the same
prefix, or the vector space skews.

Embeddings — text-literal cast
------------------------------
The query vector is sent as a pgvector **text literal** with an explicit
``::<config.EMBEDDING_PGVECTOR_TYPE>`` cast (:func:`vector_literal`), so the read
connection needs **no** psycopg vector adapter registered and the tools stay easy
to unit-test with a fake connection.

Assets
------
Figures/tables (``assets``) are a follow-up module (``tools/assets.py``); until it
lands, ``rerank_chunks`` returns an empty ``assets`` map and ``asset_refs: []`` so
the §4i payload shape is already stable.

``tests/test_retrieval.py`` is the contract (no DB, no key, zero cost). The bodies
raise :class:`NotImplementedError` until the student fills them (scaffold-and-fill,
LOG-33).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

from lens import config
from lens.embeddings import Embedder
from lens.reranking import Reranker

# Generic item type for :func:`top_by_score`.
T = TypeVar("T")

# SELECT column lists mirror the §3 table schemas. The order here is the order the
# query must select them in, so :func:`row_to_dict` can zip row -> dict.
PAPER_COLUMNS = (
    "arxiv_id",
    "title",
    "authors",
    "categories",
    "arxiv_doi",
    "journal_ref",
    "current_ver",
    "update_date",
    "abstract",
)
CHUNK_COLUMNS = (
    "chunk_uuid",
    "arxiv_id",
    "section_type",
    "section_order",
    "chunk_text",
    "token_count",
)

# Amount the embedder/reranker score in per forward pass when ranking one query
# against a candidate list (Stage A: k abstracts; Stage B: C chunks).
_QUERY_ENCODE_BATCH = 1


# ------------------------------------------------------------------ helpers
def format_query(buckets: Mapping[str, str] | str) -> str:
    """Normalise the query input into the single string to embed.

    - ``str`` -> returned as-is (already formatted; e.g. a raw query).
    - ``Mapping`` -> the clause texts joined with a single space, in
      :data:`config.QUERY_BUCKET_ORDER` for known keys, then any remaining keys in
      their insertion order. Empty / whitespace-only values are skipped.

    Example: ``{"objective": "improve Y", "problem": "why X"}`` -> ``"why X improve Y"``.
    """
    raise NotImplementedError(
        "M5 scaffold: implement format_query — str passes through; a mapping is "
        "joined in config.QUERY_BUCKET_ORDER then in insertion order, skipping "
        "empty values."
    )


def year_from_arxiv_id(arxiv_id: str) -> int | None:
    """Publication year from the arXiv id prefix (YYMM).

    ``"2103.12345"`` -> ``2021``. Returns ``None`` when the first two characters
    are not a two-digit year (old-style ids like ``"acc-phys/9411001"``, or
    malformed input). Mirrors the M1 manifest's id-prefix year rule (Tech Stack §4f).
    """
    raise NotImplementedError(
        "M5 scaffold: implement year_from_arxiv_id — 2000 + int(arxiv_id[:2]) when "
        "arxiv_id[:2] is two digits, else None."
    )


def row_to_dict(columns: Sequence[str], row: Sequence[Any]) -> dict[str, Any]:
    """Zip a result row to ``{column: value}`` using the SELECT column order."""
    raise NotImplementedError(
        "M5 scaffold: implement row_to_dict — dict(zip(columns, row))."
    )


def vector_literal(vec: np.ndarray) -> str:
    """pgvector text literal for a 1-D vector, e.g. ``"[0.1,-0.2,...]"``.

    Used with an explicit ``::<config.EMBEDDING_PGVECTOR_TYPE>`` cast in the ANN
    queries, so no psycopg vector adapter is needed on the read connection.
    """
    raise NotImplementedError(
        "M5 scaffold: implement vector_literal — flatten to 1-D and join "
        "'[' + ','.join(repr(float(x)) for x in vec) + ']'."
    )


def hnsw_config_sql(ef_search: int, iterative_scan: bool = False) -> list[str]:
    """The ``SET`` statements to run before an HNSW query (E-04).

    Always sets ``hnsw.ef_search`` from ``ef_search``; when ``iterative_scan`` is
    true also sets ``hnsw.iterative_scan = relaxed_order``. Iterative scan is
    required for the **filtered** Stage-B (deep) scan, because HNSW applies the
    SQL ``WHERE`` *after* the graph scan and can otherwise under-return
    (pgvector >= 0.8). Return the statements in execution order.
    """
    raise NotImplementedError(
        "M5 scaffold: implement hnsw_config_sql — always 'SET hnsw.ef_search = <n>', "
        "and 'SET hnsw.iterative_scan = relaxed_order' when iterative_scan is true."
    )


def top_by_score(
    items: Sequence[T], scores: Sequence[float], top: int
) -> list[tuple[T, float]]:
    """Return the ``top`` ``(item, score)`` pairs, highest score first.

    Stable on ties (original order preserved). ``top <= 0`` -> ``[]``;
    ``top >= len(items)`` -> all, still score-sorted. A length mismatch between
    ``items`` and ``scores`` is a programming error -> :class:`ValueError`.
    """
    raise NotImplementedError(
        "M5 scaffold: implement top_by_score — validate equal lengths, sort the "
        "(item, score) pairs by score descending (stable), truncate to top."
    )


# ------------------------------------------------------------ T1 funnel tools
def shallow_match(
    conn: Any,
    buckets: Mapping[str, str] | str,
    embedder: Embedder,
    k: int = config.RETRIEVAL_K,
) -> list[str]:
    """T1 Stage A candidate search — global HNSW cosine over ``papers``.

    Steps:
      1. ``query = format_query(buckets)``; skip (return ``[]``) if blank.
      2. Embed with :data:`config.EMBEDDING_QUERY_PREFIX` (the *same* nomic query
         prefix used at index time) — one text, batch size 1.
      3. Run :func:`hnsw_config_sql` (:data:`config.HNSW_EF_SEARCH`, no iterative
         scan — this scan is unfiltered).
      4. ``SELECT arxiv_id FROM papers WHERE abstract_embedding IS NOT NULL
         ORDER BY abstract_embedding <=> %s::<type> LIMIT %s`` with the query
         vector as :func:`vector_literal` and ``k`` as the limit.

    Returns the ``arxiv_id`` list in ANN order (nearest first).
    """
    raise NotImplementedError(
        "M5 scaffold: implement shallow_match — embed the formatted query with the "
        "query prefix, SET ef_search, SELECT arxiv_id ordered by "
        "abstract_embedding <=> %s::<type> LIMIT k."
    )


def fetch_papers(conn: Any, arxiv_ids: Sequence[str]) -> list[dict[str, Any]]:
    """T2 — fetch full paper rows by ``arxiv_id``.

    ``SELECT <PAPER_COLUMNS> FROM papers WHERE arxiv_id = ANY(%s)``; map each row
    with :func:`row_to_dict`. Rows are returned in the DB's order (unordered) —
    callers that need a specific order re-index by ``arxiv_id``. Empty input ->
    ``[]`` (do **not** run the query). Uses a host connection; ``categories`` comes
    back as a Python ``list[str]``.
    """
    raise NotImplementedError(
        "M5 scaffold: implement fetch_papers — SELECT PAPER_COLUMNS ... WHERE "
        "arxiv_id = ANY(%s), map rows via row_to_dict; short-circuit empty input."
    )


def rerank_papers(
    conn: Any,
    arxiv_ids: Sequence[str],
    buckets: Mapping[str, str] | str,
    reranker: Reranker,
    n: int = config.RETRIEVAL_N,
    batch_size: int = config.RERANK_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Stage A — cross-encoder over abstracts -> top-``n`` papers.

    Steps:
      1. ``rows = fetch_papers(conn, arxiv_ids)``; **re-order rows to the input
         ``arxiv_ids`` order** (skip ids with no row) so ties are deterministic.
      2. ``query = format_query(buckets)``;
         ``scores = reranker.score(query, [row["abstract"] or "" for row in rows],
         batch_size)``.
      3. Keep the top-``n`` via :func:`top_by_score`.

    Returns, best first, ``{"arxiv_id", "title", "abstract", "score"}`` dicts.
    """
    raise NotImplementedError(
        "M5 scaffold: implement rerank_papers — fetch, re-order to input ids, score "
        "the abstracts, keep top-n via top_by_score, return dicts+score."
    )


def deep_match(
    conn: Any,
    arxiv_ids: Sequence[str],
    buckets: Mapping[str, str] | str,
    embedder: Embedder,
    candidate_chunks: int = config.RETRIEVAL_C,
) -> list[str]:
    """T1 Stage B candidate search — filtered HNSW over ``chunks`` within ``n`` papers.

    Steps:
      1. Return ``[]`` for empty ``arxiv_ids`` or a blank formatted query.
      2. Embed the query with :data:`config.EMBEDDING_QUERY_PREFIX` (as in
         :func:`shallow_match`).
      3. Run :func:`hnsw_config_sql` with ``iterative_scan=True`` —
         ``hnsw.ef_search`` **and** ``hnsw.iterative_scan = relaxed_order`` — the
         WHERE filter is applied after the graph scan.
      4. ``SELECT chunk_uuid::text FROM chunks
         WHERE arxiv_id = ANY(%s) AND chunk_embedding IS NOT NULL
         ORDER BY chunk_embedding <=> %s::<type> LIMIT %s`` (``candidate_chunks``).

    Returns ``chunk_uuid`` strings (cast to text) in ANN order.
    """
    raise NotImplementedError(
        "M5 scaffold: implement deep_match — embed query, SET ef_search + iterative "
        "scan, SELECT chunk_uuid::text WHERE arxiv_id = ANY(%s) ordered by "
        "chunk_embedding <=> %s::<type> LIMIT C."
    )


def fetch_chunks(conn: Any, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
    """T2 — fetch chunk rows by ``chunk_uuid``.

    ``SELECT chunk_uuid::text, arxiv_id, section_type, section_order, chunk_text,
    token_count FROM chunks WHERE chunk_uuid = ANY(%s::uuid[])``; map with
    :func:`row_to_dict` (``chunk_uuid`` returned as ``str``). Empty input -> ``[]``.
    """
    raise NotImplementedError(
        "M5 scaffold: implement fetch_chunks — SELECT CHUNK_COLUMNS (chunk_uuid::text) "
        "WHERE chunk_uuid = ANY(%s::uuid[]), map rows via row_to_dict."
    )


def rerank_chunks(
    conn: Any,
    chunk_ids: Sequence[str],
    buckets: Mapping[str, str] | str,
    reranker: Reranker,
    top_chunks: int = config.RETRIEVAL_C_TOP,
    batch_size: int = config.RERANK_BATCH_SIZE,
) -> dict[str, Any]:
    """Stage B — cross-encoder over ``chunk_text`` -> top-``c`` hydrated chunks.

    Steps:
      1. ``rows = fetch_chunks(conn, chunk_ids)``; re-order to the input order.
      2. ``scores = reranker.score(format_query(buckets),
         [row["chunk_text"] or "" for row in rows], batch_size)``.
      3. Keep the top-``c`` via :func:`top_by_score`.
      4. **Hydrate host-side (§4i):** build the chunk dicts
         ``{"chunk_uuid", "arxiv_id", "section_type", "section_order",
         "chunk_text", "token_count", "asset_refs": []}`` (``asset_refs`` is filled
         by the assets follow-up), and a ``papers`` map ``{arxiv_id: {"title",
         "authors", "year", "abstract"}}`` via :func:`fetch_papers` +
         :func:`year_from_arxiv_id`.

    Returns ``{"chunks": [...], "papers": {...}, "assets": {}}`` — the §4i payload
    the orchestrator ``emit()``s. ``assets`` stays ``{}`` until ``tools/assets.py``
    lands.
    """
    raise NotImplementedError(
        "M5 scaffold: implement rerank_chunks — fetch, re-order, score, keep top-c, "
        "then hydrate chunks (+asset_refs=[]) and the papers map "
        "(year via year_from_arxiv_id); return {chunks, papers, assets:{}}."
    )
