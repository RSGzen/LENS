"""Sweep the retrieval funnel's capacity knobs, with no model call and no API spend.

The funnel has four numbers and only three of them are capacity:

* ``k``  -- stage-A HNSW candidates (papers),
* ``n``  -- papers kept by the stage-A cross-encoder,
* ``C``  -- chunk candidates pulled from those papers (``deep_match`` LIMIT),
* ``C_TOP`` -- chunks kept by the stage-B cross-encoder.

``C_TOP`` is a **filter only when C > C_TOP**. At the defaults both are 100, so stage B
reorders the candidates and drops none -- the sweep reports that rather than pretending
otherwise.

Two modes:

* ``--tuples`` measures **capacity**: how many papers survive, how many chunks each
  paper gets, how large the bundle becomes, how long a pass takes. No judgements
  needed, which is why it can run before the instruments exist.
* ``--qrels`` measures **relevance** against a judged set: recall@n and precision@n.
  A pilot set is enough to choose an operating point; it is not enough to report nDCG.

Usage::

    python scripts/sweep_retrieval.py --tuples experiments/instruments/w2a-tuples.json \
        --queries experiments/campaigns/m11-w1-v2.json --out sweep.csv
    python scripts/sweep_retrieval.py --qrels experiments/instruments/pilot-qrels-aggregation.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from lens import config
from lens.db.connect import connect
from lens.embeddings import Embedder
from lens.reranking import Reranker
from lens.tools.retrieval import (
    deep_match,
    format_query,
    hnsw_config_sql,
    hnsw_ef_for,
    rerank_chunks,
    rerank_papers,
    shallow_match,
    vector_literal,
)


def load_qrels(path: Path) -> dict[str, Any]:
    """Return the judged set: query text plus ``{arxiv_id: grade}``."""
    record = json.loads(path.read_text(encoding="utf-8"))
    grades = record.get("grades") or {}

    if not grades:
        raise SystemExit(f"{path} has no grades")

    return {"query": str(record["query"]), "grades": {k: int(v) for k, v in grades.items()}}


def load_tuples(path: Path) -> list[dict[str, Any]]:
    """Return the ``(k, n, C, C_TOP)`` points to run, each with a name."""
    record = json.loads(path.read_text(encoding="utf-8"))
    points = record.get("points") or record

    if not isinstance(points, list) or not points:
        raise SystemExit(f"{path} has no points")

    for point in points:
        for key in ("k", "n", "C", "c_top"):
            if key not in point:
                raise SystemExit(f"{path}: point {point} is missing {key}")

        if point["c_top"] > point["C"]:
            raise SystemExit(f"{path}: point {point.get('name')} has c_top > C")

    return points


def load_queries(paths: Sequence[Path], ids: Sequence[str] | None) -> list[dict[str, str]]:
    """Return deduplicated queries from campaign files, optionally filtered by id."""
    seen: dict[str, dict[str, str]] = {}

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))

        for query in data.get("queries") or []:
            text = str(query["text"]).strip()

            if ids and query["id"] not in ids:
                continue

            seen.setdefault(text, {"id": str(query["id"]), "text": text})

    if not seen:
        raise SystemExit("no queries matched")

    return list(seen.values())


def buckets_for(text: str) -> dict[str, list[str]]:
    """Return a one-bucket query object: the sweep measures the funnel, not the segmenter."""
    return {"objectives": [text], "problems": [], "constraints": [], "requirements": []}


#: Papers per configuration that go into the judged pool. Ten because the metric is
#: nDCG@10: a paper ranked 11th cannot change it.
POOL_DEPTH = 10


def ann_recall(literal: str, k: int) -> float:
    """Return the ANN overlap with the exact top-k, at the configured ``ef_search``."""
    with connect(readonly=True) as conn, conn.cursor() as cur:
        cur.execute("set local enable_indexscan = off")
        cur.execute("set local enable_bitmapscan = off")
        cur.execute(
            "SELECT arxiv_id FROM papers WHERE abstract_embedding IS NOT NULL "
            f"ORDER BY abstract_embedding <=> %s::{config.EMBEDDING_PGVECTOR_TYPE} LIMIT %s",
            (literal, k),
        )
        exact = [row[0] for row in cur.fetchall()]
        conn.rollback()

        for statement in hnsw_config_sql(hnsw_ef_for(k)):
            cur.execute(statement)

        cur.execute(
            "SELECT arxiv_id FROM papers WHERE abstract_embedding IS NOT NULL "
            f"ORDER BY abstract_embedding <=> %s::{config.EMBEDDING_PGVECTOR_TYPE} LIMIT %s",
            (literal, k),
        )
        approx = [row[0] for row in cur.fetchall()]

    return len(set(approx) & set(exact)) / len(exact) if exact else 0.0


def run_point(
    conn: Any,
    query: dict[str, str],
    point: dict[str, Any],
    embedder: Embedder,
    reranker: Reranker,
) -> tuple[dict[str, Any], list[str]]:
    """Run one tuple and return ``(measurements, paper_ids)``.

    The paper ids come back because the relevance mode needs exactly the papers the
    funnel kept -- re-running stage A to obtain them would double the cross-encoder
    work for no reason.
    """
    buckets = buckets_for(query["text"])
    started = time.perf_counter()

    candidates = shallow_match(conn, buckets, embedder, k=int(point["k"]))
    papers = rerank_papers(conn, candidates, buckets, reranker, n=int(point["n"]))
    paper_ids = [row["arxiv_id"] for row in papers]
    chunk_ids = deep_match(conn, paper_ids, buckets, embedder, candidate_chunks=int(point["C"]))
    bundle = rerank_chunks(conn, chunk_ids, buckets, reranker, top_chunks=int(point["c_top"]))
    elapsed = time.perf_counter() - started

    chunks = bundle.get("chunks") or []
    per_paper: dict[str, int] = {}

    for chunk in chunks:
        arxiv = str(chunk.get("arxiv_id"))
        per_paper[arxiv] = per_paper.get(arxiv, 0) + 1

    counts = list(per_paper.values())

    row = {
        "query_id": query["id"],
        "k": int(point["k"]),
        "n": int(point["n"]),
        "C": int(point["C"]),
        "c_top": int(point["c_top"]),
        "candidates": len(candidates),
        "papers": len(paper_ids),
        "chunk_candidates": len(chunk_ids),
        "chunks": len(chunks),
        "papers_with_chunks": len(per_paper),
        "papers_without_chunks": max(0, len(paper_ids) - len(per_paper)),
        "chunks_per_paper_min": min(counts) if counts else 0,
        "chunks_per_paper_median": statistics.median(counts) if counts else 0,
        "bundle_chars": sum(len(str(chunk.get("chunk_text") or "")) for chunk in chunks),
        "seconds": round(elapsed, 1),
    }

    return row, paper_ids


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one row per tuple: the mean across queries, plus its worst case."""
    grouped: dict[tuple, list[dict[str, Any]]] = {}

    for row in rows:
        grouped.setdefault((row["k"], row["n"], row["C"], row["c_top"]), []).append(row)

    summary: list[dict[str, Any]] = []

    for (k, n, c_value, c_top), group in sorted(grouped.items()):
        summary.append(
            {
                "k": k,
                "n": n,
                "C": c_value,
                "c_top": c_top,
                "queries": len(group),
                "chunks_mean": round(statistics.mean(r["chunks"] for r in group), 1),
                "papers_with_chunks_mean": round(
                    statistics.mean(r["papers_with_chunks"] for r in group), 1
                ),
                "papers_starved_worst": max(r["papers_without_chunks"] for r in group),
                "chunks_min_worst": min(r["chunks_per_paper_min"] for r in group),
                "bundle_kchars_mean": round(
                    statistics.mean(r["bundle_chars"] for r in group) / 1000, 1
                ),
                "seconds_mean": round(statistics.mean(r["seconds"] for r in group), 1),
            }
        )

    return summary


def _log(handle: Any, message: str) -> None:
    """Write one timestamped line to the log, if there is one, and flush it.

    A sweep over 25 queries and six points runs for about an hour, so the log is the
    only place that says how far it got if the session dies. Line-buffered by hand:
    a crash must not take the record of where it crashed with it.
    """
    if handle is None:
        return

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    handle.write(f"{stamp} | {message}\n")
    handle.flush()


def summarise_point(rows: list[dict[str, Any]]) -> str:
    """Return a one-line summary of one point's rows."""
    if not rows:
        return "no rows"

    return (
        f"chunks_mean={statistics.mean(r['chunks'] for r in rows):.1f} "
        f"papers_with_chunks_mean="
        f"{statistics.mean(r['papers_with_chunks'] for r in rows):.1f} "
        f"starved_worst={max(r['papers_without_chunks'] for r in rows)} "
        f"min_ppr_worst={min(r['chunks_per_paper_min'] for r in rows)} "
        f"bundle_kchars={statistics.mean(r['bundle_chars'] for r in rows) / 1000:.1f} "
        f"secs_mean={statistics.mean(r['seconds'] for r in rows):.1f}"
    )


def capacity_mode(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the tuple grid over the given queries, optionally ranking the top papers.

    The rankings file is written **per point and flushed**, not collected and written
    at the end: a sweep over 25 queries takes about an hour, and a session that dies
    halfway should cost one point rather than the whole run. ``--log`` writes the same
    progress, timestamped, to a file opened ``w`` so a rerun starts a fresh record.
    """
    points = load_tuples(Path(args.tuples))

    if args.point:
        wanted = set(args.point)
        missing = wanted - {point.get("name") for point in points}

        if missing:
            raise SystemExit(f"no such point(s): {sorted(missing)}")

        points = [point for point in points if point.get("name") in wanted]

    queries = load_queries([Path(path) for path in args.queries], args.query_ids)
    embedder, reranker = Embedder(), Reranker()
    rows: list[dict[str, Any]] = []
    handle = None
    writer: Any = None
    log = None

    if args.log:
        log = Path(args.log).open("w", encoding="utf-8")
        _log(
            log,
            f"sweep start: {len(points)} points x {len(queries)} queries | "
            f"ef_search={config.HNSW_EF_SEARCH} | tuples={args.tuples} | queries={args.queries}",
        )

    if args.rankings:
        handle = Path(args.rankings).open("a" if args.append else "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=["config", "query_id", "rank", "doc_id"])

        if not (args.append and Path(args.rankings).stat().st_size > 0):
            writer.writeheader()

        handle.flush()

    print(f"{len(points)} tuples x {len(queries)} queries | ef_search={config.HNSW_EF_SEARCH}")

    try:
        with connect(readonly=True) as conn:
            if writer is not None and args.a1_dense:
                # A1 is retrieval only: the top papers by abstract cosine, no
                # cross-encoder. It is the Tier-2 baseline, so it has to be in the
                # pool that gets judged.
                started = time.perf_counter()

                for query in queries:
                    candidates = shallow_match(
                        conn, buckets_for(query["text"]), embedder, k=int(points[0]["k"])
                    )

                    writer.writerows(
                        {
                            "config": "a1-dense",
                            "query_id": query["id"],
                            "rank": rank,
                            "doc_id": doc_id,
                        }
                        for rank, doc_id in enumerate(candidates[:POOL_DEPTH], start=1)
                    )

                handle.flush()
                _log(
                    log,
                    f"a1-dense: {len(queries)} queries ranked, "
                    f"{time.perf_counter() - started:.0f}s",
                )
                print("  a1-dense: dense top-10 papers")

            for index, point in enumerate(points, start=1):
                started = time.perf_counter()
                name = point.get("name") or f"k{point['k']}-n{point['n']}"
                point_rows: list[dict[str, Any]] = []

                _log(
                    log,
                    f"point {index}/{len(points)} {name} | k={point['k']} n={point['n']} "
                    f"C={point['C']} c_top={point['c_top']} | {len(queries)} queries",
                )

                for position, query in enumerate(queries, start=1):
                    row, papers = run_point(conn, query, point, embedder, reranker)
                    rows.append(row)
                    point_rows.append(row)

                    if writer is not None:
                        writer.writerows(
                            {
                                "config": name,
                                "query_id": query["id"],
                                "rank": rank,
                                "doc_id": doc_id,
                            }
                            for rank, doc_id in enumerate(papers[:POOL_DEPTH], start=1)
                        )

                    _log(
                        log,
                        f"  {name} q{position}/{len(queries)} {query['id']} | "
                        f"candidates={row['candidates']} papers={row['papers']} "
                        f"chunks={row['chunks']} papers_with_chunks={row['papers_with_chunks']} "
                        f"starved={row['papers_without_chunks']} "
                        f"min_ppr={row['chunks_per_paper_min']} "
                        f"kchars={row['bundle_chars'] / 1000:.1f} | {row['seconds']}s",
                    )

                if handle is not None:
                    handle.flush()

                elapsed = time.perf_counter() - started
                _log(
                    log,
                    f"point {name} done in {elapsed:.0f}s | {summarise_point(point_rows)}",
                )
                print(
                    f"  [{index}/{len(points)}] {name} k={point['k']} "
                    f"n={point['n']} C={point['C']} c_top={point['c_top']} "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
    finally:
        if handle is not None:
            handle.close()

        _log(log, f"sweep end: {len(rows)} query rows")

        if log is not None:
            log.close()

    return rows


def relevance_mode(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the (k, n) grid against one judged query."""
    qrels = load_qrels(Path(args.qrels))
    query = {"id": str(qrels.get("query_id", "qrels")), "text": qrels["query"]}
    core = {paper for paper, grade in qrels["grades"].items() if grade >= 2}
    embedder, reranker = Embedder(), Reranker()
    literal = vector_literal(
        embedder.encode_documents(
            texts=[format_query(buckets=buckets_for(qrels["query"]))],
            task_prefix=config.EMBEDDING_QUERY_PREFIX,
            batch_size=1,
        )
    )
    rows: list[dict[str, Any]] = []

    with connect(readonly=True) as conn:
        for k in args.k:
            recall = ann_recall(literal, k)

            for n in args.n:
                point = {"k": k, "n": n, "C": config.RETRIEVAL_C, "c_top": config.RETRIEVAL_C_TOP}
                row, paper_ids = run_point(conn, query, point, embedder, reranker)
                row["ann_recall"] = round(recall, 3)
                found = len(core & set(paper_ids))
                row["recall_core"] = round(found / len(core), 3) if core else 0.0
                row["recall_all"] = round(
                    len(set(qrels["grades"]) & set(paper_ids)) / len(qrels["grades"]), 3
                )
                row["precision"] = round(
                    len(set(qrels["grades"]) & set(paper_ids)) / max(1, len(paper_ids)), 3
                )
                rows.append(row)

    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested mode, print a table, optionally write CSV."""
    parser = argparse.ArgumentParser(prog="sweep_retrieval", description=__doc__)
    parser.add_argument("--qrels", default=None, help="pilot qrels JSON (relevance mode)")
    parser.add_argument("--tuples", default=None, help="tuple JSON (capacity mode)")
    parser.add_argument(
        "--queries", nargs="*", default=[], help="campaign JSON files to take queries from"
    )
    parser.add_argument("--query-ids", nargs="*", default=None, help="only these query ids")
    parser.add_argument("--k", type=int, nargs="*", default=[50, 100, 200, 400])
    parser.add_argument("--n", type=int, nargs="*", default=[5, 10, 20, 50])
    parser.add_argument("--out", default=None, help="write the rows as CSV here")
    parser.add_argument(
        "--rankings",
        default=None,
        help="write 'config,query_id,rank,doc_id' for the top 10 papers of every point "
        "(the pooled candidate list, and score_retrieval's rankings input)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to --rankings instead of truncating it, so a point-per-run sweep "
        "accumulates into one pooled file",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="write a timestamped progress log here (opened 'w'): every query's "
        "measurements and every point's summary",
    )
    parser.add_argument(
        "--point",
        nargs="*",
        default=None,
        help="run only these tuple names, so a long sweep can be split into short runs",
    )
    parser.add_argument(
        "--a1-dense",
        action="store_true",
        help="also rank the A1 baseline: dense top-10 papers with no cross-encoder",
    )
    args = parser.parse_args(argv)

    if bool(args.qrels) == bool(args.tuples):
        parser.error("give exactly one of --qrels or --tuples")

    if args.tuples and not args.queries:
        parser.error("--tuples needs at least one --queries file")

    rows = relevance_mode(args) if args.qrels else capacity_mode(args)

    print()

    if args.qrels:
        print(
            f"{'k':>5} {'n':>4} {'ann':>5} {'recall core':>12} {'recall all':>11} {'precision':>10}"
        )

        for row in rows:
            print(
                f"{row['k']:>5} {row['n']:>4} {row['ann_recall']:>5.0%} "
                f"{row['recall_core']:>11.0%} {row['recall_all']:>10.0%} "
                f"{row['precision']:>10.0%}"
            )
    else:
        header = (
            f"{'k':>5} {'n':>4} {'C':>5} {'c_top':>6} {'chunks':>7} {'papers':>7} "
            f"{'starved':>8} {'min/ppr':>8} {'kchars':>7} {'secs':>6}"
        )
        print(header)

        for row in summarise(rows):
            print(
                f"{row['k']:>5} {row['n']:>4} {row['C']:>5} {row['c_top']:>6} "
                f"{row['chunks_mean']:>7} {row['papers_with_chunks_mean']:>7} "
                f"{row['papers_starved_worst']:>8} {row['chunks_min_worst']:>8} "
                f"{row['bundle_kchars_mean']:>7} {row['seconds_mean']:>6}"
            )

        print(
            "\nstarved = papers that received no chunk in the worst query; min/ppr = fewest "
            "chunks one paper received in the worst query."
        )

    if args.out:
        out = Path(args.out)

        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
