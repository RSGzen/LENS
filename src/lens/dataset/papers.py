"""M3 — load the manifest papers with abstract embeddings.

Spec: ``MVP Build Roadmap.md`` §4 (M3); ``Thesis Drafts/Tech Stack (Draft).md``
§3 (``papers`` columns) and §4f (the M1 manifest this consumes).

Pipeline
--------
``manifest.json`` (166,704 papers, M1)
  -> :func:`load_papers` — embed each abstract via :class:`lens.embeddings.Embedder`
     and upsert one ``papers`` row (metadata + ``abstract_embedding`` +
     ``embedding_version``).

The **full** manifest is loaded — there is no paper subset (advisor decision,
6 Oct 2026). M3 verify clause: ``rows == manifest["count"]`` and no NULL
``abstract_embedding``. ``tests/DESTRUCTIVE_db_tests/test_papers.py`` is the
contract — it drops the schema, so run it only against a throwaway ``lens_test``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector


from lens import config
from lens.embeddings import Embedder

def chunked_iterable(iterable, size):
    """Yield successive chunks from the data source."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def load_papers(
    conn: psycopg.Connection,
    entries: Sequence[Mapping[str, Any]],
    process_chunk_size: int,
    encode_batch_size: int,
    embedder: Embedder,
    *,
    embedding_version: str = config.EMBEDDING_VERSION,
) -> int:
    """Embed each abstract and upsert one ``papers`` row per entry; return the count.

    ``conn`` is an admin connection (this writes). The load is **chunked** to bound
    memory and expose per-chunk timing:

      1. Split ``entries`` into ``process_chunk_size`` batches.
      2. Per batch: ``encode_documents(abstracts, task_prefix=EMBEDDING_DOC_PREFIX,
         batch_size=encode_batch_size)``; each 512-d vector is wrapped in
         :class:`pgvector.HalfVector` (the ``halfvec`` dumper accepts only that).
      3. ``register_vector(conn)`` once — pgvector's psycopg3 adapter — so the
         ``halfvec(512)`` column accepts a ``HalfVector``.
      4. ``executemany`` an ``INSERT … ON CONFLICT (arxiv_id) DO UPDATE`` keyed on
         the PK (idempotent re-run). Manifest ``version`` -> ``current_ver``;
         ``categories`` is a list (``TEXT[]``); ``arxiv_doi`` / ``journal_ref`` may
         be ``None``; ``update_date`` is an ISO ``YYYY-MM-DD`` string.
      5. Print one line per chunk (``encode`` / ``insert`` / ``chunk`` seconds and
         cumulative progress); the runner mirrors these lines to the run log.
      6. Do **not** commit here — the caller owns the transaction.

    The caller passes ``manifest["papers"]`` (all 166,704). Build the HNSW indexes
    **after** this load (``scripts/build_indexes.py``) — pgvector recommends it, and
    inserts are much faster without graph maintenance (Roadmap §10).
    """
    num_processed_entries = 0

    task_prefix = config.EMBEDDING_DOC_PREFIX

    # pgvector's psycopg3 adapter: required so a HalfVector can be sent to the
    # halfvec(512) column (the halfvec dumper accepts only HalfVector).
    register_vector(conn)

    total_entries = len(entries)
    run_start = time.perf_counter()

    with conn.cursor() as cur:
        for chunk_index, chunk in enumerate(
            chunked_iterable(entries, process_chunk_size), start=1
        ):
            chunk_start = time.perf_counter()

            # 1. Extract abstract just for this chunk
            abstract_lists = [e["abstract"] for e in chunk]

            # 2. Encode the abstract embeddings
            encode_start = time.perf_counter()
            embeddings = embedder.encode_documents(abstract_lists,
                                                task_prefix=task_prefix,
                                                batch_size=encode_batch_size)
            encode_s = time.perf_counter() - encode_start

            # 3. Prepare list of parameter tuples
            records = [
                (m["arxiv_id"], m["title"], m["authors"], m["categories"], m["arxiv_doi"], m["journal_ref"], m["version"], m["update_date"], m["abstract"], HalfVector(emb), embedding_version)
                for m, emb in zip(chunk, embeddings)
            ]

            # 4. Define the SQL query with placeholders
            query = """
                INSERT INTO papers (arxiv_id, title, authors, categories, arxiv_doi,
                                   journal_ref, current_ver, update_date, abstract,
                                   abstract_embedding, embedding_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    categories = EXCLUDED.categories,
                    arxiv_doi = EXCLUDED.arxiv_doi,
                    journal_ref = EXCLUDED.journal_ref,
                    current_ver = EXCLUDED.current_ver,
                    update_date = EXCLUDED.update_date,
                    abstract = EXCLUDED.abstract,
                    abstract_embedding = EXCLUDED.abstract_embedding,
                    embedding_version = EXCLUDED.embedding_version
            """

            # 5. Execute the batch
            insert_start = time.perf_counter()
            cur.executemany(query, records)
            insert_s = time.perf_counter() - insert_start

            # 6. Record number of inserted entries
            num_processed_entries += len(abstract_lists)

            # 7. Per-chunk progress (mirrored to the run log by the runner)
            chunk_s = time.perf_counter() - chunk_start
            elapsed_s = time.perf_counter() - run_start
            print(
                f"[chunk {chunk_index}] rows={len(abstract_lists)} "
                f"encode={encode_s:.2f}s insert={insert_s:.2f}s chunk={chunk_s:.2f}s "
                f"| processed={num_processed_entries}/{total_entries} elapsed={elapsed_s:.1f}s"
            )

    return num_processed_entries
