r"""M7 K-05 probe — GLiNER-family extraction over a controlled sample.

Wiring only: read-only connect (``lens_ro``), pick a sample, run ``build_kg`` with a
fresh extraction (cache bypassed so timing is real), and print per-type/relation
counts + timings. Writes the full result JSON under ``config.LENS_LOGS_ROOT_PATH``.

Sample arms
-----------
- ``head``   — the first ``--papers`` ``arxiv_id``s, all their chunks (legacy default).
- ``random`` — ``--papers`` papers ordered by ``md5(arxiv_id || seed)`` (reproducible
  random), all their chunks.
- ``query``  — run the M5 funnel for ``--query`` and take the KG **scope**:
  ``c`` (reranked chunks, production answer-context), ``C`` (deep-match candidates),
  or ``papers`` (all chunks of the top-``n`` papers).

Extractor arms
--------------
``--ner-model`` / ``--rel-model`` / ``--min-score`` / ``--labels``
(``abstract`` | ``descriptive`` | ``scierc`` — see ``kg_extract.ENTITY_LABEL_SETS``).

Prereqs: ``lens-pg`` running with M3+M4 loaded.

Run:
  .\.venv\Scripts\python.exe scripts\kg_probe.py --papers 20
  .\.venv\Scripts\python.exe scripts\kg_probe.py --sample random --seed 7
  .\.venv\Scripts\python.exe scripts\kg_probe.py --sample query --scope C --labels descriptive
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lens import config
from lens.db.connect import connect
from lens.kg_extract import ENTITY_LABEL_SETS, GlinerExtractor
from lens.observability.run_logging import setup_file_logging
from lens.tools.kg import build_kg

DEFAULT_PAPERS = 20
DEFAULT_QUERY = (
    "How do retrieval-augmented language models aggregate evidence "
    "across a long context?"
)


# ------------------------------------------------------------------ sampling
def _chunks_for_papers(conn, arxiv_ids: list[str]) -> list[str]:
    """All chunks of ``arxiv_ids`` (deterministic order)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_uuid::text FROM chunks "
            "WHERE arxiv_id = ANY(%s) ORDER BY arxiv_id, section_order",
            (arxiv_ids,),
        )
        return [row[0] for row in cur.fetchall()]


def _head_chunk_ids(conn, papers: int) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT arxiv_id FROM papers ORDER BY arxiv_id LIMIT %s", (papers,)
        )
        arxiv_ids = [row[0] for row in cur.fetchall()]
    return _chunks_for_papers(conn, arxiv_ids)


def _random_chunk_ids(conn, papers: int, seed: int) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT arxiv_id FROM papers "
            "ORDER BY md5(arxiv_id || %s) LIMIT %s",
            (str(seed), papers),
        )
        arxiv_ids = [row[0] for row in cur.fetchall()]
    return _chunks_for_papers(conn, arxiv_ids)


def _query_chunk_ids(conn, query: str, scope: str) -> list[str]:
    """Run the M5 funnel and return the chunk set for the requested KG scope."""
    from lens.embeddings import Embedder
    from lens.reranking import Reranker
    from lens.tools.retrieval import (
        deep_match,
        rerank_chunks,
        rerank_papers,
        shallow_match,
    )

    embedder = Embedder()
    reranker = Reranker()

    paper_ids = shallow_match(conn, query, embedder)
    papers = rerank_papers(conn, paper_ids, query, reranker)
    n_ids = [p["arxiv_id"] for p in papers]

    if scope == "papers":
        return _chunks_for_papers(conn, n_ids)

    candidate_chunks = deep_match(conn, n_ids, query, embedder)  # C
    if scope == "C":
        return candidate_chunks

    # scope == "c": the reranked answer-context chunks (production default).
    hydrated = rerank_chunks(conn, candidate_chunks, query, reranker)
    return [chunk["chunk_uuid"] for chunk in hydrated["chunks"]]


def _sample_chunk_ids(conn, args: argparse.Namespace) -> list[str]:
    if args.sample == "head":
        return _head_chunk_ids(conn, args.papers)
    if args.sample == "random":
        return _random_chunk_ids(conn, args.papers, args.seed)
    return _query_chunk_ids(conn, args.query, args.scope)


# ------------------------------------------------------------------ main
def main() -> None:
    setup_file_logging("lens.kg_probe", "kg_probe")

    parser = argparse.ArgumentParser(description="M7 K-05 KG extraction probe")
    parser.add_argument("--papers", type=int, default=DEFAULT_PAPERS)
    parser.add_argument(
        "--sample", choices=("head", "random", "query"), default="head"
    )
    parser.add_argument("--scope", choices=("c", "C", "papers"), default="C",
                        help="query-arm KG scope (production default: c=5 chunks)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--labels", choices=tuple(ENTITY_LABEL_SETS), default="abstract")
    parser.add_argument("--ner-model", default=config.KG_NER_MODEL_ID)
    parser.add_argument("--rel-model", default=config.KG_REL_MODEL_ID)
    parser.add_argument("--min-score", type=float, default=config.KG_MIN_SCORE)
    args = parser.parse_args()

    print(
        f"probe: sample={args.sample} scope={args.scope} papers={args.papers} "
        f"seed={args.seed} labels={args.labels}\n"
        f"       ner={args.ner_model} rel={args.rel_model} "
        f"min_score={args.min_score} ontology={config.KG_ONTOLOGY_VERSION}"
    )
    if args.sample == "query":
        print(f"query: {args.query!r}")

    extractor = GlinerExtractor(
        ner_model_id=args.ner_model,
        rel_model_id=args.rel_model,
        entity_labels=ENTITY_LABEL_SETS[args.labels],
        threshold=args.min_score,
    )

    with connect(readonly=True) as conn:
        chunk_ids = _sample_chunk_ids(conn, args)
        print(f"sample: {len(chunk_ids)} chunks")

        t0 = time.perf_counter()
        result = build_kg(conn, chunk_ids, extractor, use_cache=False)
        elapsed = time.perf_counter() - t0

    stats = result["stats"]
    summary = result["summary"]
    print(f"build_kg: {elapsed:.1f}s  kg_id={result['kg_id']}  stats={stats}")
    print(f"entities by type: {summary['entity_types']}")
    print(f"relations: {summary['relations']}")
    print("top entities:")
    for node in summary["top_entities"]:
        print(f"  {node['degree']:>4}  {node['type']:<8}  {node['entity']}")

    triples = result["graph"].get("links", result["graph"].get("edges", []))
    print(f"triples: {len(triples)}")
    for edge in triples[:20]:
        print(f"  {edge}")

    out_path = (
        Path(config.LENS_LOGS_ROOT_PATH)
        / f"kg_probe_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
