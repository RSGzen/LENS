r"""— retrieval tools: the T1 funnel + T2 fetch-by-id.

Funnel — the graded path ::

    shallow_match   global HNSW over papers.abstract_embedding   -> k arxiv_ids
    rerank_papers   bge-reranker-v2-m3 (Stage A) over abstracts  -> n papers
    deep_match      filtered HNSW over chunks within the n       -> C chunk_uuids
    rerank_chunks   bge-reranker-v2-m3 (Stage B) over chunk_text -> c hydrated

Design rules (do not deviate)
-----------------------------
- **The index buys candidate recall; the cross-encoder buys top-precision.** A
  reranker cannot recover a candidate the HNSW scan never returned, so end-to-end
  quality is bounded by candidate recall — hence the generous ``k=100 -> n=20`` /
  ``C=100`` funnel and why ``k`` / ``C`` are the recall knobs. ``c`` is
  ``RETRIEVAL_C_TOP`` (100 by default), so Stage B reorders the candidates rather
  than truncating them.
- **Hydration is host-side, at the deep stage only** : ``shallow_match``,
  ``rerank_papers`` and ``deep_match`` return ids only; ``rerank_chunks`` returns
  the ``{chunks, papers, assets}`` payload the orchestrator ``emit()``s.
- **The RLM never issues SQL here.** Every function runs on a host connection
  (``lens.db.connect``); in recorded runs the host uses the read-only ``lens_ro``
  role. The sandbox holds no DB driver.

Query input
-----------
The four funnel tools take ``buckets`` — the segmentation output (a mapping of
bucket name -> **list** of clause texts, e.g.
``{"objectives": ["improve Y", "test Y"], "problems": ["Y is costly"]}``), or a
plain query string. :func:`format_query`
joins both into the single string that is embedded (with the nomic
``search_query:`` prefix) at the shallow/deep stages and passed to the
cross-encoder at the rerank stages. Document and query paths **must** use the same
prefix, or the vector space skews. Nothing here normalises the text: the visible
clauses are stripped and joined, which is all the query path does.

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
the payload shape is already stable.

``tests/test_retrieval.py`` is the contract (no DB, no key, zero cost). The funnel
tools are implemented; ``tools/assets.py`` remains the only deferred follow-up.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

from lens import config
from lens.embeddings import Embedder
from lens.reranking import Reranker

# Generic item type for :func:`top_by_score`.
T = TypeVar("T")

# Query input for the funnel: the segmentation output (bucket name -> clause
# texts, e.g. ``{"objectives": ["...", "..."], "problems": [...]}``) or a
# pre-formatted query string. owns the dict build order; :func:`format_query`
# joins in insertion order.
QueryBuckets = Mapping[str, Sequence[str]] | str

# SELECT column lists mirror the table schemas. The order here is the order the
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
def format_query(buckets: QueryBuckets) -> str:
    """Normalise the query input into the single string to embed.

    - ``str`` -> returned as-is (already formatted; e.g. a raw query).
    - ``Mapping`` -> every clause in every bucket, flattened and joined with a
      single space, in the mapping's **insertion order** ( owns building the
      dict in a fixed order). Empty / whitespace-only values are skipped.

    Flattening is shape-tolerant on purpose. The shape is
    ``{"objectives": ["clause", ...], ...}``, but the RLM may hand over
    ``{"query": "one clause"}`` or a nested list. Iterating a *string* value as if
    it were a list of clauses splits it into **characters** — the embedded query
    becomes ``"s o m e   c l a u s e"`` and the funnel returns unrelated papers.
    So a string is one clause, a mapping or a sequence is flattened recursively,
    and anything else is ignored.

    Example: ``{"objective": ["improve Y"], "problem": ["why X"]}`` ->
    ``"improve Y why X"``.
    """
    if isinstance(buckets, str):
        return buckets

    parts: list[str] = []

    for clause in _flatten_clauses(buckets):
        if clause.strip():
            parts.append(clause.strip())

    return " ".join(parts)


def _flatten_clauses(value: Any) -> Iterator[str]:
    """Yield every clause string under ``value``, whatever the nesting.

    A plain string is yielded whole (never character by character); mappings and
    sequences are walked; numbers/None/booleans carry no clause text and are
    skipped, so a stray field cannot poison the embedded query.
    """
    if isinstance(value, str):
        yield value

    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _flatten_clauses(nested)

    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for nested in value:
            yield from _flatten_clauses(nested)


def year_from_arxiv_id(arxiv_id: str) -> int | None:
    """Publication year from the arXiv id prefix (YYMM).

    ``"2103.12345"`` -> ``2021``. Returns ``None`` when the first two characters
    are not a two-digit year (old-style ids like ``"acc-phys/9411001"``, or
    malformed input). Mirrors the manifest's id-prefix year rule.
    """

    # Check if first two characters are all digits
    if arxiv_id[:2].isdigit():
        yr = 2000 + int(arxiv_id[:2])
        return yr

    else:
        return None


def row_to_dict(columns: Sequence[str], row: Sequence[Any]) -> dict[str, Any]:
    """Zip a result row to ``{column: value}`` using the SELECT column order."""

    return dict(zip(columns, row))


def vector_literal(vec: np.ndarray) -> str:
    """Format a vector as a pgvector **text literal**, e.g. ``"[0.1,-0.2,...]"``.

    This only formats the *value*; the ``::<config.EMBEDDING_PGVECTOR_TYPE>`` cast
    (e.g. ``%s::halfvec``) is written into the ANN SQL, where the vector stays a
    plain ``%s`` bind parameter — only the cast target is composed from config.
    Text + cast is used instead of registering a psycopg vector adapter on the
    read connection (``register_vector``), which keeps the tools testable with a
    fake connection. Flatten to 1-D because the embedder returns shape
    ``(1, dim)`` for a single query while the literal is a flat list.
    """
    # Accept anything array-like and normalise to nomic dtype (float32) and then flatten it
    flat = np.asarray(vec, dtype=np.float32).reshape(-1)

    # pgvector's text format, e.g. "[0.1,-0.02,...]"
    return "[" + ",".join(repr(float(x)) for x in flat) + "]"


def hnsw_config_sql(ef_search: int, iterative_scan: bool = False) -> list[str]:
    """The ``SET`` statements to run before an HNSW query.

    Always sets ``hnsw.ef_search`` from ``ef_search``; when ``iterative_scan`` is
    true also sets ``hnsw.iterative_scan = relaxed_order``. Iterative scan is
    required for the **filtered** Stage-B (deep) scan, because HNSW applies the
    SQL ``WHERE`` *after* the graph scan and can otherwise under-return
    (pgvector >= 0.8). Return the statements in execution order.
    """
    # Set the initial candidate graph pool
    set_statement_list = [f"SET hnsw.ef_search = {ef_search};"]

    # Enable iterative scanning (Scans until reaches enough matching rows / hit safety limit)
    if iterative_scan:
        set_statement_list.append("SET hnsw.iterative_scan = 'relaxed_order';")

    return set_statement_list


def top_by_score(items: Sequence[T], scores: Sequence[float], top: int) -> list[tuple[T, float]]:
    """Return the ``top`` ``(item, score)`` pairs, highest score first.

    Stable on ties (original order preserved). ``top <= 0`` -> ``[]``;
    ``top >= len(items)`` -> all, still score-sorted. A length mismatch between
    ``items`` and ``scores`` is a programming error -> :class:`ValueError`.
    """
    if len(items) != len(scores):
        raise ValueError("items and scores must have the same length")

    if top <= 0:
        return []

    # Highest score first; Python's sort is stable, so ties keep input order.
    ranked = sorted(zip(items, scores), key=lambda pair: pair[1], reverse=True)

    return ranked[:top]


# ------------------------------------------------------------------ T1 funnel tools
def shallow_match(
    conn: Any,
    buckets: QueryBuckets,
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
    query = format_query(buckets=buckets)
    if not query.strip():
        return []

    vec = embedder.encode_documents(
        texts=[query], task_prefix=config.EMBEDDING_QUERY_PREFIX, batch_size=_QUERY_ENCODE_BATCH
    )

    with conn.cursor() as cur:
        # Set configurations for HNSW. `ef_search` bounds the rows an unfiltered
        # HNSW scan returns (pgvector default 40), so it must be >= the LIMIT.
        for stmt in hnsw_config_sql(max(config.HNSW_EF_SEARCH, k)):
            cur.execute(stmt)

        cur.execute(
            "SELECT arxiv_id FROM papers "
            "WHERE abstract_embedding IS NOT NULL "
            f"ORDER BY abstract_embedding <=> %s::{config.EMBEDDING_PGVECTOR_TYPE} "
            "LIMIT %s",
            (vector_literal(vec), k),
        )

        rows = cur.fetchall()

        # Select first col (arxiv_ID) that are already in order
        return [r[0] for r in rows]


def fetch_papers(conn: Any, arxiv_ids: Sequence[str]) -> list[dict[str, Any]]:
    """T2 — fetch full paper rows by ``arxiv_id``.

    ``SELECT <PAPER_COLUMNS> FROM papers WHERE arxiv_id = ANY(%s)``; map each row
    with :func:`row_to_dict`. Rows are returned in the DB's order (unordered) —
    callers that need a specific order re-index by ``arxiv_id``. Empty input ->
    ``[]`` (do **not** run the query). Uses a host connection; ``categories`` comes
    back as a Python ``list[str]``.
    """
    if not arxiv_ids:
        return []

    # Select PAPER_COLUMNS in order so row_to_dict can zip them; the vector
    # columns (abstract_embedding / embedding_version) are intentionally excluded.
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(PAPER_COLUMNS)} FROM papers WHERE arxiv_id = ANY(%s)",
            (list(arxiv_ids),),
        )
        return [row_to_dict(PAPER_COLUMNS, row) for row in cur.fetchall()]


def rerank_papers(
    conn: Any,
    arxiv_ids: Sequence[str],
    buckets: QueryBuckets,
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
    rows = fetch_papers(conn, arxiv_ids)

    order_map = {name: index for index, name in enumerate(arxiv_ids)}
    sorted_rows = sorted(rows, key=lambda x: order_map.get(x["arxiv_id"], float("inf")))

    query = format_query(buckets)

    texts = [row["abstract"] or "" for row in sorted_rows]
    scores = reranker.score(query, texts, batch_size)

    return [
        {
            "arxiv_id": row["arxiv_id"],
            "title": row["title"],
            "abstract": row["abstract"],
            "score": score,
        }
        for row, score in top_by_score(sorted_rows, scores, n)
    ]


def deep_match(
    conn: Any,
    arxiv_ids: Sequence[str],
    buckets: QueryBuckets,
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
    if not arxiv_ids:
        return []

    query = format_query(buckets)
    if not query.strip():
        return []

    vec = embedder.encode_documents(
        texts=[query], task_prefix=config.EMBEDDING_QUERY_PREFIX, batch_size=_QUERY_ENCODE_BATCH
    )

    with conn.cursor() as cur:
        # Filtered HNSW applies the WHERE after the graph scan -> iterative scan.
        # `ef_search` must also be >= the LIMIT (see shallow_match).
        for stmt in hnsw_config_sql(
            ef_search=max(config.HNSW_EF_SEARCH, candidate_chunks), iterative_scan=True
        ):
            cur.execute(stmt)

        cur.execute(
            "SELECT chunk_uuid::text FROM chunks "
            "WHERE arxiv_id = ANY(%s) AND chunk_embedding IS NOT NULL "
            f"ORDER BY chunk_embedding <=> %s::{config.EMBEDDING_PGVECTOR_TYPE} "
            "LIMIT %s",
            (arxiv_ids, vector_literal(vec), candidate_chunks),
        )

        # chunk_uuids are already in ANN order (nearest first).
        return [r[0] for r in cur.fetchall()]


def fetch_chunks(conn: Any, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
    """T2 — fetch chunk rows by ``chunk_uuid``.

    ``SELECT chunk_uuid::text, arxiv_id, section_type, section_order, chunk_text,
    token_count FROM chunks WHERE chunk_uuid = ANY(%s::uuid[])``; map with
    :func:`row_to_dict` (``chunk_uuid`` returned as ``str``). Empty input -> ``[]``.
    """
    if not chunk_ids:
        return []

    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_uuid::text, arxiv_id, section_type, section_order, "
            "chunk_text, token_count "
            "FROM chunks WHERE chunk_uuid = ANY(%s::uuid[])",
            (list(chunk_ids),),
        )
        return [row_to_dict(CHUNK_COLUMNS, row) for row in cur.fetchall()]


def rerank_chunks(
    conn: Any,
    chunk_ids: Sequence[str],
    buckets: QueryBuckets,
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
      4. **Hydrate host-side :** build the chunk dicts
         ``{"chunk_uuid", "arxiv_id", "section_type", "section_order",
         "chunk_text", "token_count", "asset_refs": []}`` (``asset_refs`` is filled
         by the assets follow-up), and a ``papers`` map ``{arxiv_id: {"title",
         "authors", "year", "abstract"}}`` via :func:`fetch_papers` +
         :func:`year_from_arxiv_id`.

    Returns ``{"chunks": [...], "papers": {...}, "assets": {}}`` — the payload
    the orchestrator ``emit()``s. ``assets`` stays ``{}`` until ``tools/assets.py``
    lands.
    """
    rows = fetch_chunks(conn, chunk_ids)

    # Re-order to the input (ANN) order so a tie keeps the index's ranking.
    order_map = {chunk_uuid: index for index, chunk_uuid in enumerate(chunk_ids)}
    sorted_rows = sorted(rows, key=lambda x: order_map.get(x["chunk_uuid"], float("inf")))

    query = format_query(buckets)
    texts = [row["chunk_text"] or "" for row in sorted_rows]

    scores = reranker.score(query, texts, batch_size)

    top_rows = [row for row, _ in top_by_score(sorted_rows, scores, top_chunks)]

    # payload: chunk_text keeps the visible labels; asset_refs is filled by
    # the assets follow-up (tools/assets.py), so it stays [] for now.
    chunks = [
        {
            "chunk_uuid": row["chunk_uuid"],
            "arxiv_id": row["arxiv_id"],
            "section_type": row["section_type"],
            "section_order": row["section_order"],
            "chunk_text": row["chunk_text"],
            "token_count": row["token_count"],
            "asset_refs": [],
        }
        for row in top_rows
    ]

    # Papers map: metadata for every paper the kept chunks came from, so the RLM
    # can attribute chunks without another fetch. year comes from the id prefix.
    arxiv_ids = list(dict.fromkeys(row["arxiv_id"] for row in top_rows))
    papers = {
        paper["arxiv_id"]: {
            "title": paper["title"],
            "authors": paper["authors"],
            "year": year_from_arxiv_id(paper["arxiv_id"]),
            "abstract": paper["abstract"],
        }
        for paper in fetch_papers(conn, arxiv_ids)
    }

    return {"chunks": chunks, "papers": papers, "assets": {}}
