r"""probe — GLiNER-family extraction over a controlled sample.

Wiring only: read-only connect (``lens_ro``), pick a sample, run ``build_kg`` with a
fresh extraction (cache bypassed so timing is real), and print per-type/relation
counts + timings. Writes the full result JSON under ``config.LENS_LOGS_ROOT_PATH``.

Sample arms
-----------
- ``head``   — the first ``--papers`` ``arxiv_id``s, all their chunks (legacy default).
- ``random`` — ``--papers`` papers ordered by ``md5(arxiv_id || seed)`` (reproducible
  random), all their chunks.
- ``query`` — run the funnel for ``--query`` and take the KG **scope**:
  ``c`` (reranked chunks, production answer-context), ``C`` (deep-match candidates),
  or ``papers`` (all chunks of the top-``n`` papers).

Extractor arms
--------------
``--ner-model`` / ``--rel-model`` / ``--min-score`` / ``--labels``
(``abstract`` | ``descriptive`` | ``scierc`` — see ``kg_extract.ENTITY_LABEL_SETS``).

Prereqs: ``lens-pg`` running with + loaded.

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
    "How do retrieval-augmented language models aggregate evidence across a long context?"
)


def _gpu_stats() -> dict:
    """GPU device + VRAM figures for the run log (empty on CPU)."""
    try:
        import torch
    except Exception:
        return {"device": "unavailable"}
    if torch.cuda.is_available():
        return {
            "device": torch.cuda.get_device_name(0),
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
            "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        }
    return {"device": "cpu"}


def _reset_peak_vram() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


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
        cur.execute("SELECT arxiv_id FROM papers ORDER BY arxiv_id LIMIT %s", (papers,))
        arxiv_ids = [row[0] for row in cur.fetchall()]
    return _chunks_for_papers(conn, arxiv_ids)


def _random_chunk_ids(conn, papers: int, seed: int) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT arxiv_id FROM papers ORDER BY md5(arxiv_id || %s) LIMIT %s",
            (str(seed), papers),
        )
        arxiv_ids = [row[0] for row in cur.fetchall()]
    return _chunks_for_papers(conn, arxiv_ids)


def _query_chunk_ids(
    conn, query: str, scope: str, retrieval_c: int, retrieval_c_top: int
) -> list[str]:
    """Run the funnel and return the chunk set for the requested KG scope."""
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

    candidate_chunks = deep_match(conn, n_ids, query, embedder, candidate_chunks=retrieval_c)  # C
    if scope == "C":
        return candidate_chunks

    # scope == "c": the reranked answer-context chunks.
    hydrated = rerank_chunks(conn, candidate_chunks, query, reranker, top_chunks=retrieval_c_top)
    return [chunk["chunk_uuid"] for chunk in hydrated["chunks"]]


def _sample_chunk_ids(conn, args: argparse.Namespace) -> list[str]:
    if args.sample == "head":
        return _head_chunk_ids(conn, args.papers)
    if args.sample == "random":
        return _random_chunk_ids(conn, args.papers, args.seed)
    return _query_chunk_ids(conn, args.query, args.scope, args.retrieval_c, args.retrieval_c_top)


# ------------------------------------------------------------------ main
def main() -> None:
    setup_file_logging("lens.kg_probe", "kg_probe")

    parser = argparse.ArgumentParser
    parser.add_argument("--papers", type=int, default=DEFAULT_PAPERS)
    parser.add_argument("--sample", choices=("head", "random", "query"), default="head")
    parser.add_argument(
        "--scope",
        choices=("c", "C", "papers"),
        default="C",
        help="query-arm KG scope (production KG scope: C=100; c=5 is the answer context)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--retrieval-c", type=int, default=config.RETRIEVAL_C)
    parser.add_argument("--retrieval-c-top", type=int, default=config.RETRIEVAL_C_TOP)
    parser.add_argument(
        "--chunks-file", default=None, help="reuse a saved chunk-id list (skips sampling)"
    )
    parser.add_argument(
        "--save-chunks", default=None, help="write the sampled chunk-id list to this JSON path"
    )
    parser.add_argument("--labels", choices=tuple(ENTITY_LABEL_SETS), default="abstract")
    parser.add_argument("--ner-model", default=config.KG_NER_MODEL_ID)
    parser.add_argument("--rel-model", default=config.KG_REL_MODEL_ID)
    parser.add_argument("--min-score", type=float, default=config.KG_MIN_SCORE)
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="disable the post-process (pre/post comparison)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="enable budgeted JEV direction verification (live API)",
    )
    parser.add_argument("--verify-budget", type=int, default=config.KG_VERIFY_BUDGET)
    args = parser.parse_args()
    postprocess = not args.no_postprocess

    print(
        f"probe: sample={args.sample} scope={args.scope} papers={args.papers} "
        f"seed={args.seed} labels={args.labels} postprocess={postprocess}\n"
        f"       C={args.retrieval_c} c={args.retrieval_c_top} "
        f"ner={args.ner_model} rel={args.rel_model} "
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

    verifier = None
    if args.verify:
        from lens.clients import LensClient
        from lens.kg_verify import DirectionVerifier
        from lens.observability.trajectory import TraceCtx, Trajectory

        run_id = time.strftime("kgprobe_%Y%m%d_%H%M%S")
        trajectory = Trajectory(run_id)
        client = LensClient(trajectory)
        trace = TraceCtx(trace_id=run_id, step_idx=0, actor="tool")
        verifier = DirectionVerifier(
            lambda state, questions: client.jev(state, questions, trace_ctx=trace),
            budget=args.verify_budget,
        )
        print(f"verify: JEV budget={args.verify_budget} run={run_id}")

    with connect(readonly=True) as conn:
        chunks_path = Path(args.chunks_file) if args.chunks_file else None
        if chunks_path is not None and chunks_path.is_file():
            chunk_ids = json.loads(chunks_path.read_text(encoding="utf-8"))
            print(f"reused {len(chunk_ids)} chunks from {chunks_path}")
        else:
            chunk_ids = _sample_chunk_ids(conn, args)
            print(f"sample: {len(chunk_ids)} chunks")
            if args.save_chunks:
                saved = Path(args.save_chunks)
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_text(json.dumps(chunk_ids), encoding="utf-8")
                print(f"saved chunks -> {saved}")

        _reset_peak_vram()
        t0 = time.perf_counter()
        result = build_kg(
            conn,
            chunk_ids,
            extractor,
            use_cache=False,
            postprocess=postprocess,
            verifier=verifier,
        )
        elapsed = time.perf_counter() - t0

        result["stats"]["timing"] = {
            "overall_s": round(elapsed, 3),
            "chunks_per_s": round(len(chunk_ids) / elapsed, 2) if elapsed else 0.0,
        }
        result["stats"]["gpu"] = _gpu_stats()

    stats = result["stats"]
    summary = result["summary"]
    print(f"build_kg: {elapsed:.1f}s  kg_id={result['kg_id']}  stats={stats}")
    print(f"timing: {stats.get('timing')} gpu={stats.get('gpu')}")
    print(f"verify: {stats.get('verify')}")
    print(f"entities by type: {summary['entity_types']}")
    print(f"type sources: {summary.get('type_sources')}")
    print(f"relations: {summary['relations']}")
    print("top entities:")
    for node in summary["top_entities"]:
        print(f"  {node['degree']:>4}  {node['type']:<8}  {node['entity']}")

    triples = result["graph"].get("links", result["graph"].get("edges", []))
    print(f"triples: {len(triples)}")
    for edge in triples[:20]:
        print(f"  {edge}")

    out_path = Path(config.LENS_LOGS_ROOT_PATH) / f"kg_probe_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
