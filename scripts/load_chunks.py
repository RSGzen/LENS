r"""M4 runner — chunk + embed the full manifest into ``chunks``.

Wiring only: read the M1 manifest, build the nomic token counter + embedder, and
call :func:`lens.dataset.chunks.load_chunks`. All stdout — including the
per-batch timing lines printed by ``dataset/chunks.py`` — is mirrored to a
timestamped log file under ``config.LENS_LOGS_ROOT_PATH`` for thesis evidence
(Roadmap §4 M4, §8).

Prereqs:
  * ``lens-pg`` running with the schema applied (``python scripts/setup_db.py``);
  * M3 complete (every manifest ``arxiv_id`` present in ``papers`` -> FK safety);
  * the ``chunks_chunk_embedding_hnsw`` index dropped for the load — this script
    drops it, then **you rebuild it afterwards** with
    ``python scripts/build_indexes.py`` (Roadmap §10; per-row HNSW maintenance
    would otherwise dominate insert time).

Run:  .\\.venv\\Scripts\\python.exe scripts\\load_chunks.py
"""

from __future__ import annotations

import time
from pathlib import Path

from lens import config
from lens.dataset.chunks import load_chunks
from lens.dataset.manifest import load_manifest
from lens.db.connect import connect
from lens.embeddings import Embedder
from lens.observability.run_logging import setup_file_logging


def _build_token_counter(embedder: Embedder):
    """Return ``text -> nomic token count`` from the embedder's tokenizer."""
    tokenizer = embedder.model.tokenizer

    def count_tokens(text: str) -> int:
        # truncation=False: count the whole piece (some sections exceed 8192).
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))

    return count_tokens


def _drop_chunks_index(conn) -> None:
    """Drop the chunks HNSW index so bulk inserts stay fast (rebuild after)."""
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS chunks_chunk_embedding_hnsw")
    print("dropped chunks_chunk_embedding_hnsw (rebuild after with build_indexes.py)")


def _filter_loaded(conn, entries):
    """Drop manifest entries whose ``arxiv_id`` is already in ``chunks`` (resume).

    Batches commit per batch, so a re-run only needs the papers that are not yet
    present. ``SELECT DISTINCT arxiv_id`` is a one-off scan (~2 min at 4.5 M rows).
    """
    print("resume: scanning chunks for already-loaded papers (can take ~2 min)...")
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT arxiv_id FROM chunks")
        loaded = {row[0] for row in cur.fetchall()}
    if not loaded:
        print("resume: chunks is empty -> nothing to skip")
        return entries
    remaining = [entry for entry in entries if entry["arxiv_id"] not in loaded]
    print(f"resume: {len(entries) - len(remaining)} papers already loaded, {len(remaining)} remaining")
    return remaining


def main() -> None:
    setup_file_logging("lens.load_chunks", "load_chunks")

    manifest_path = Path(config.LENS_MANIFESTS_ROOT_PATH) / config.MANIFEST_FILENAME
    manifest = load_manifest(manifest_path)

    entries = manifest["papers"]
    if config.PAPERS_LIMIT:
        entries = entries[: config.PAPERS_LIMIT]

    print(
        f"config: limit={config.PAPERS_LIMIT or 'all'} papers={len(entries)} "
        f"papers_per_batch={config.PAPERS_PER_BATCH} "
        f"encode_batch={config.EMBEDDING_CHUNK_BATCH_SIZE} "
        f"threshold={config.CHUNK_TOKEN_THRESHOLD} overlap={config.CHUNK_OVERLAP_TOKENS}"
    )

    embedder = Embedder()
    count_tokens = _build_token_counter(embedder)

    start = time.perf_counter()
    with connect(autocommit=True) as conn:
        if config.RESUME:
            entries = _filter_loaded(conn, entries)
        _drop_chunks_index(conn)
        papers, chunks = load_chunks(
            conn=conn,
            entries=entries,
            xml_dir=config.GROBID_XML_DIR_PATH,
            count_tokens=count_tokens,
            embedder=embedder,
            papers_per_batch=config.PAPERS_PER_BATCH,
            encode_batch_size=config.EMBEDDING_CHUNK_BATCH_SIZE,
            threshold=config.CHUNK_TOKEN_THRESHOLD,
            overlap=config.CHUNK_OVERLAP_TOKENS,
        )
    elapsed_s = time.perf_counter() - start

    rate = chunks / elapsed_s if elapsed_s > 0 else 0.0
    print(
        f"loaded {chunks} chunks from {papers} papers in {elapsed_s:.1f}s "
        f"({rate:.1f} chunks/s; embedding_version={config.EMBEDDING_VERSION})"
    )
    print("next: rebuild the HNSW indexes -> python scripts/build_indexes.py")


if __name__ == "__main__":
    main()
