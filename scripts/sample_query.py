r"""M5 runner — one sample query through the full funnel (``k -> n -> C -> c``).

Wiring only: connect **read-only** (``lens_ro``), build the nomic ``Embedder`` +
bge ``Reranker``, then run
``shallow_match -> rerank_papers -> deep_match -> rerank_chunks`` and print each
stage's latency plus the hydrated result. Stdout is mirrored to a timestamped log
under ``config.LENS_LOGS_ROOT_PATH`` (Roadmap §4 M5 verify clause).

E-04: HNSW ``m`` / ``ef_construction`` are build-time (``scripts/build_indexes.py``);
``ef_search`` is in :data:`config.HNSW_EF_SEARCH` and recorded per run.

Prereqs: ``lens-pg`` running with M3+M4 loaded and the HNSW indexes built.

Run:  .\\.venv\\Scripts\\python.exe scripts\\sample_query.py "your query here"
"""

from __future__ import annotations

import sys
import time

from lens import config
from lens.db.connect import connect
from lens.embeddings import Embedder
from lens.observability.run_logging import setup_file_logging
from lens.reranking import Reranker
from lens.tools.retrieval import (
    deep_match,
    rerank_chunks,
    rerank_papers,
    shallow_match,
)

DEFAULT_QUERY = (
    "How do retrieval-augmented language models aggregate evidence "
    "across a long context?"
)


def main() -> None:
    setup_file_logging("lens.sample_query", "sample_query")

    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    print(f"query: {query!r}")
    print(
        f"funnel: k={config.RETRIEVAL_K} n={config.RETRIEVAL_N} "
        f"C={config.RETRIEVAL_C} c={config.RETRIEVAL_C_TOP} | "
        f"hnsw.ef_search={config.HNSW_EF_SEARCH} | reranker={config.RERANKER_MODEL_ID}"
    )

    embedder = Embedder()
    reranker = Reranker()

    with connect(readonly=True) as conn:
        t0 = time.perf_counter()
        paper_ids = shallow_match(conn, query, embedder)
        t_shallow = time.perf_counter() - t0

        t0 = time.perf_counter()
        papers = rerank_papers(conn, paper_ids, query, reranker)
        t_rerank_papers = time.perf_counter() - t0
        n_ids = [p["arxiv_id"] for p in papers]

        t0 = time.perf_counter()
        chunk_ids = deep_match(conn, n_ids, query, embedder)
        t_deep = time.perf_counter() - t0

        t0 = time.perf_counter()
        hydrated = rerank_chunks(conn, chunk_ids, query, reranker)
        t_rerank_chunks = time.perf_counter() - t0

    print(
        f"stage latencies (s): shallow={t_shallow:.3f} rerank_papers={t_rerank_papers:.3f} "
        f"deep={t_deep:.3f} rerank_chunks={t_rerank_chunks:.3f}"
    )
    print(f"funnel counts: k={len(paper_ids)} n={len(n_ids)} "
          f"C={len(chunk_ids)} c={len(hydrated['chunks'])}")
    print("top papers:")
    for p in papers[:5]:
        print(f"  {p['score']:.4f}  {p['arxiv_id']}  {p.get('title', '')}")
    print("top chunks:")
    for c in hydrated["chunks"]:
        preview = (c["chunk_text"] or "")[:120].replace("\n", " ")
        print(f"  {c['arxiv_id']}  {c['section_type']}#{c['section_order']}  {preview}")


if __name__ == "__main__":
    main()
