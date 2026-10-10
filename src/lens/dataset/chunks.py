r"""M4 — load the manifest papers' chunks with embeddings into ``chunks``.

Spec: ``MVP Build Roadmap.md`` §4 (M4); ``Thesis Drafts/Tech Stack (Draft).md``
§3 (``chunks`` columns) and §4h (the M3 loader this mirrors).

Pipeline
--------
``manifest.json`` (166,704 papers, M1) + ``papers`` (loaded by M3)
  -> :func:`chunk_paper`  parse ``.tei.xml`` -> :class:`lens.chunking.Chunk` list
  -> :func:`load_chunks`  batch -> embed -> idempotent upsert into ``chunks``.

The **full** manifest is chunked — no paper subset (advisor decision, 6 Oct
2026). M4 verify clause: **0 FK orphans, sane chunk counts** (Roadmap §4). FK
safety relies on M3 having loaded every manifest ``arxiv_id`` into ``papers``.

Idempotency / resume
--------------------
``chunk_uuid`` is **deterministic** (:func:`chunk_uuid`, ``uuid5`` over
``(arxiv_id, section_type, section_order)``) so a multi-day load is resumable:
re-running a completed batch hits ``ON CONFLICT (chunk_uuid) DO UPDATE`` and
replaces rather than duplicates. Commit **per batch** so partial progress is
durable (the loop prints one timing line per batch for the run log).

Insert speed
------------
The ``chunks_chunk_embedding_hnsw`` index already exists (M2). Per-row HNSW graph
maintenance makes bulk inserts slow, so — exactly as M3 did for ``papers`` —
**drop the chunks HNSW index before the load and rebuild with**
``scripts/build_indexes.py`` after (Roadmap §4 M3/M4, §10).

``tests/DESTRUCTIVE_db_tests/test_chunks.py`` is the contract — it drops the
schema, so run it only against a throwaway ``lens_test`` (no API key, zero cost).
"""

from __future__ import annotations

import datetime
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg
from lxml import etree
from pgvector import HalfVector
from pgvector.psycopg import register_vector

from lens import config
from lens.chunking import CHUNK_UUID_NAMESPACE, Chunk, TokenCounter, chunk_tei
from lens.embeddings import Embedder


def eta_calculation(elapsed_s, total_processed, total_papers):
    """Return ``(eta_timestamp, remaining)`` from the run's cumulative rate.

    ``remaining = (total_papers - total_processed) * (elapsed_s / total_processed)``
    — i.e. the average seconds/paper so far, so the ETA does **not** jump around
    with a single slow/fast batch. ``remaining`` is a ``datetime.timedelta``
    string, e.g. ``"18:58:29"`` or ``"1 day, 14:32:00"``.
    """
    processed = max(total_processed, 1)
    remaining_secs = (total_papers - total_processed) * (elapsed_s / processed)

    eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_secs)

    return (
        eta.strftime("%Y-%m-%d %H:%M:%S"),
        str(datetime.timedelta(seconds=int(remaining_secs))),
    )


def chunked_iterable(iterable, size):
    """Yield successive chunks from the data source."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def chunk_paper(
    entry: Mapping[str, Any],
    xml_dir: str | Path,
    count_tokens: TokenCounter,
    threshold: int,
    overlap: int = 0,
) -> list[Chunk]:
    """Chunk one manifest ``entry`` into an ordered :class:`Chunk` list.

    Resolve ``Path(xml_dir) / entry["tei_xml"]`` and delegate to
    :func:`lens.chunking.chunk_tei`. A missing/unparseable file returns ``[]``
    (the caller counts it as a skipped paper, not a crash).
    """
    tei_xml_filepath = Path(xml_dir) / Path(entry["tei_xml"])
    if not tei_xml_filepath.exists():
        return []

    try:
        return chunk_tei(tei_xml_filepath, count_tokens, threshold, overlap)
    except etree.XMLSyntaxError:
        return []


def chunk_uuid(arxiv_id: str, section_type: str, section_order: int) -> uuid.UUID:
    """Deterministic, stable PK for a chunk (idempotent upsert).

    ``uuid.uuid5(CHUNK_UUID_NAMESPACE, f"{arxiv_id}:{section_type}:{section_order}")``.
    Uses :data:`lens.chunking.CHUNK_UUID_NAMESPACE`; never rebase once rows exist.
    """
    c_uuid = uuid.uuid5(namespace=CHUNK_UUID_NAMESPACE,
                        name=f"{arxiv_id}:{section_type}:{section_order}")

    return c_uuid


def load_chunks(
    conn: psycopg.Connection,
    entries: Sequence[Mapping[str, Any]],
    *,
    xml_dir: str | Path,
    count_tokens: TokenCounter,
    embedder: Embedder,
    papers_per_batch: int,
    encode_batch_size: int,
    threshold: int,
    overlap: int = 0,
    embedding_version: str = config.EMBEDDING_VERSION,
) -> tuple[int, int]:
    """Chunk + embed + upsert every paper in ``entries``; return counts.

    ``conn`` is an admin connection (this writes). ``entries`` is the manifest's
    ``papers`` list (or a slice). Steps:

      1. ``register_vector(conn)`` **once** — pgvector's psycopg3 adapter, so the
         ``halfvec(512)`` column accepts a :class:`pgvector.HalfVector`.
      2. Split ``entries`` into ``papers_per_batch`` batches. For each batch:
         a. For every entry, :func:`chunk_paper`; collect all chunks across the
            batch (skip papers that yielded none; count them).
         b. ``encode_documents([c.chunk_text for c in batch_chunks],
            task_prefix=config.EMBEDDING_DOC_PREFIX, batch_size=encode_batch_size)``
            -> one 512-d vector per chunk. Wrap each in :class:`HalfVector`.
         c. ``executemany`` an idempotent upsert keyed on the PK. Columns:
            ``chunk_uuid, arxiv_id, section_type, chunk_text, chunk_embedding,
            token_count, section_order, embedding_version`` with
            ``ON CONFLICT (chunk_uuid) DO UPDATE SET`` the non-key columns.
            ``chunk_uuid`` comes from :func:`chunk_uuid` (not the column default)
            so re-runs are idempotent.
         d. Commit **per batch** — the runner passes ``autocommit=True`` — so
            progress survives a crash.
         e. Print **one line per batch**: batch number, ``papers=``/``chunks=``,
            ``encode=``/``insert=``/``batch=`` seconds, cumulative
            ``papers=x/N chunks=y/M``, ``elapsed=`` and a simple ETA.
            (The runner mirrors these lines to the timestamped run log.)
      3. Return ``(papers_processed, chunks_processed)``.

    Do **not** build the HNSW index here — drop it before the load and rebuild
    afterwards via ``scripts/build_indexes.py``.
    """
    register_vector(conn)

    task_prefix = config.EMBEDDING_DOC_PREFIX
    num_processed_entries = 0
    num_processed_chunks = 0

    total_entries = len(entries)

    run_start = time.perf_counter()

    with conn.cursor() as cur:
        for batch_index, batch in enumerate(
            chunked_iterable(entries, papers_per_batch)
        ):
            batch_start = time.perf_counter()

            uuid_list = []
            arxivID_list = []
            section_type_list = []
            chunk_txt_list = []
            token_count_list = []
            section_order_list = []

            for paper_entry in batch:
                paper_chunk_list = chunk_paper(entry=paper_entry,
                                               xml_dir=xml_dir,
                                               count_tokens=count_tokens,
                                               threshold=threshold)

                # Collect the per-chunk fields for this paper, and derive the
                # deterministic PK from *each chunk's own* (arxiv_id, type, order).
                for chunk in paper_chunk_list:
                    arxivID_list.append(paper_entry["arxiv_id"])
                    section_type_list.append(chunk.section_type)
                    chunk_txt_list.append(chunk.chunk_text)
                    token_count_list.append(chunk.token_count)
                    section_order_list.append(chunk.section_order)
                    uuid_list.append(
                        chunk_uuid(paper_entry["arxiv_id"], chunk.section_type, chunk.section_order)
                    )

            # 2. Encode chunk texts (skip the model call for an empty batch)
            encode_start = time.perf_counter()

            if chunk_txt_list:
                embeddings = embedder.encode_documents(texts=chunk_txt_list,
                                                       task_prefix=task_prefix,
                                                       batch_size=encode_batch_size)
            else:
                embeddings = []

            encode_s = time.perf_counter() - encode_start

            # 3. Prepare list of parameter tuples
            records = [
                (uuid, arxiv_id, section_type, text, HalfVector(emb), t_count, section_order, embedding_version)
                for uuid, arxiv_id, section_type, text, emb, t_count, section_order in zip(uuid_list, arxivID_list, section_type_list, chunk_txt_list, embeddings, token_count_list, section_order_list)
            ]

            # 4. Define the SQL query with placeholders
            query = """
                INSERT INTO chunks (chunk_uuid, arxiv_id, section_type, chunk_text, 
                                    chunk_embedding, token_count, section_order,
                                    embedding_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_uuid) DO UPDATE SET
                    arxiv_id = EXCLUDED.arxiv_id,
                    section_type = EXCLUDED.section_type,
                    chunk_text = EXCLUDED.chunk_text,
                    chunk_embedding = EXCLUDED.chunk_embedding,
                    token_count = EXCLUDED.token_count,
                    section_order = EXCLUDED.section_order,
                    embedding_version = EXCLUDED.embedding_version
            """

            # 5. Execute the batch
            insert_start = time.perf_counter()
            cur.executemany(query, records)
            insert_s = time.perf_counter() - insert_start

            # 6. Record number of inserted entries and chunks
            num_processed_entries += len(set(arxivID_list))
            num_processed_chunks += len(uuid_list)

            # 7. Per-chunk progress (mirrored to the run log by the runner)
            batch_s = time.perf_counter() - batch_start
            elapsed_s = time.perf_counter() - run_start
            eta_str, remaining_str = eta_calculation(elapsed_s=elapsed_s,
                                                     total_processed=num_processed_entries,
                                                     total_papers=total_entries)

            print(
                f"[batch {batch_index}] papers={len(set(arxivID_list))} chunks={len(uuid_list)} "
                f"encode={encode_s:.2f}s insert={insert_s:.2f}s batch={batch_s:.2f}s "
                f"| processed={num_processed_entries}/{total_entries} elapsed={elapsed_s:.1f}s "
                f"| ETA {eta_str} | Remaining {remaining_str}"
            )

    return num_processed_entries, num_processed_chunks
