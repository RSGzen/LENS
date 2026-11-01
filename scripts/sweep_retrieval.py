"""Sweep the retrieval funnel's (k, n) against a judged query set, no model involved.

Why this exists: W2a asks which operating point the funnel should report at, and the
honest way to choose is to measure the funnel rather than to keep the defaults. This
runs stage A and the stage-A reranker only -- embed, HNSW, cross-encoder -- so a full
grid costs GPU seconds and no tokens, and every row can be re-run later against the
same judgements.

It reports two different things, and they answer different questions:

* **ANN recall@k** -- how much of the exact top-k the HNSW scan returned. This is a
  property of the index, not of the ranking, and it caps everything downstream.
* **recall@n of the judged set** -- how many of the papers a human marked relevant
  survive the funnel into the ``n`` that the deep stage will read.

A pilot qrels file (``experiments/instruments/pilot-qrels-*.json``) is enough to choose
an operating point. It is **not** enough to report nDCG: one query, few judgements, one
judge. The output says so.

Usage::

    python scripts/sweep_retrieval.py --qrels experiments/instruments/pilot-qrels-aggregation.json
    python scripts/sweep_retrieval.py --qrels <file> --k 50 100 200 400 --n 5 10 20 50
    python scripts/sweep_retrieval.py --qrels <file> --out sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from lens import config
from lens.db.connect import connect
from lens.embeddings import Embedder
from lens.reranking import Reranker
from lens.tools.retrieval import (
    format_query,
    hnsw_config_sql,
    hnsw_ef_for,
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


def exact_top(cur: Any, literal: str, k: int) -> list[str]:
    """Return the exact k nearest paper ids (index bypassed)."""
    cur.execute("set local enable_indexscan = off")
    cur.execute("set local enable_bitmapscan = off")
    cur.execute(
        "SELECT arxiv_id FROM papers WHERE abstract_embedding IS NOT NULL "
        f"ORDER BY abstract_embedding <=> %s::{config.EMBEDDING_PGVECTOR_TYPE} LIMIT %s",
        (literal, k),
    )

    return [row[0] for row in cur.fetchall()]


def approx_top(cur: Any, literal: str, k: int) -> list[str]:
    """Return the ANN k nearest paper ids at the configured ``ef_search`` rule."""
    for statement in hnsw_config_sql(hnsw_ef_for(k)):
        cur.execute(statement)

    cur.execute(
        "SELECT arxiv_id FROM papers WHERE abstract_embedding IS NOT NULL "
        f"ORDER BY abstract_embedding <=> %s::{config.EMBEDDING_PGVECTOR_TYPE} LIMIT %s",
        (literal, k),
    )

    return [row[0] for row in cur.fetchall()]


def run(qrels: dict[str, Any], ks: Sequence[int], ns: Sequence[int]) -> list[dict[str, Any]]:
    """Return one row per ``(k, n)`` pair."""
    buckets = {
        "objectives": [qrels["query"]],
        "problems": [],
        "constraints": [],
        "requirements": [],
    }
    literal = vector_literal(
        Embedder().encode_documents(
            texts=[format_query(buckets=buckets)],
            task_prefix=config.EMBEDDING_QUERY_PREFIX,
            batch_size=1,
        )
    )
    embedder, reranker = Embedder(), Reranker()
    grades = qrels["grades"]
    core = {paper for paper, grade in grades.items() if grade >= 2}
    rows: list[dict[str, Any]] = []

    with connect(readonly=True) as conn, conn.cursor() as cur:
        for k in ks:
            cur.execute("set local enable_indexscan = off")
            exact = exact_top(cur, literal, k)
            conn.rollback()
            approx = shallow_match(conn, buckets, embedder, k=k)
            ann_recall = len(set(approx) & set(exact)) / len(exact) if exact else 0.0

            for n in ns:
                kept = [
                    row["arxiv_id"] for row in rerank_papers(conn, approx, buckets, reranker, n=n)
                ]
                rows.append(
                    {
                        "k": k,
                        "n": n,
                        "ann_recall_at_k": round(ann_recall, 3),
                        "recall_at_n_core": round(len(core & set(kept)) / len(core), 3),
                        "recall_at_n_all": round(len(set(grades) & set(kept)) / len(grades), 3),
                        "precision_at_n": round(
                            len(set(grades) & set(kept)) / max(1, len(kept)), 3
                        ),
                        "core_found": sorted(core & set(kept)),
                        "core_missing": sorted(core - set(kept)),
                    }
                )
                conn.rollback()

    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Print the grid and optionally write it as CSV."""
    parser = argparse.ArgumentParser(prog="sweep_retrieval", description=__doc__)
    parser.add_argument("--qrels", required=True, help="pilot qrels JSON")
    parser.add_argument("--k", type=int, nargs="*", default=[50, 100, 200, 400])
    parser.add_argument("--n", type=int, nargs="*", default=[5, 10, 20, 50])
    parser.add_argument("--out", default=None, help="write the rows as CSV here")
    args = parser.parse_args(argv)

    qrels = load_qrels(Path(args.qrels))
    core = sum(1 for grade in qrels["grades"].values() if grade >= 2)
    print(f"query: {qrels['query'][:88]}...")
    print(
        f"judged: {len(qrels['grades'])} papers ({core} core) | ef_search={config.HNSW_EF_SEARCH}"
    )
    print()
    header = (
        f"{'k':>5} {'ann recall':>11} {'n':>4} "
        f"{'recall core':>12} {'recall all':>11} {'precision':>10}"
    )
    print(header)

    rows = run(qrels, args.k, args.n)

    for row in rows:
        print(
            f"{row['k']:>5} {row['ann_recall_at_k']:>10.0%} {row['n']:>4} "
            f"{row['recall_at_n_core']:>11.0%} {row['recall_at_n_all']:>10.0%} "
            f"{row['precision_at_n']:>10.0%}"
        )

    if args.out:
        out = Path(args.out)

        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nwrote {out}")

    print(
        "\nPilot judgements: one query, one judge, titles and abstracts only. Read this as a\n"
        "choice of operating point, not as an nDCG result."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
