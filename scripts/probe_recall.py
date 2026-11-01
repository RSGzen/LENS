"""Measure ANN recall: how much of the exact top-k the HNSW index actually returns.

Why this exists: `hnsw.ef_search` bounds how many candidates the index explores, and
the query path used to set it equal to the LIMIT. With the two equal, the scan
explores exactly as many nodes as it is asked to return, and recall drops. That was
not a theory -- on a recorded query the 4th-closest paper of 166,704 was missing from
a k=100 candidate set, so the cross-encoder never saw it and the answer was written
over the wrong evidence.

pgvector documents this check: compare the approximate result with an exact scan by
turning index use off in a transaction. This script does that at several `ef_search`
values and reports the overlap with the exact top-k.

Usage::

    python scripts/probe_recall.py --run m11-subtask-probe-v6.rlm-full.aggregation.r2
    python scripts/probe_recall.py --query "recursive summarisation of long documents" --k 100
    python scripts/probe_recall.py --run <id> --target 2512.24601 --min-recall 0.95

Exit code 2 when recall at the configured ``LENS_HNSW_EF_SEARCH`` is below
``--min-recall``, so a starved index fails a check instead of quietly degrading every
answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from lens import config
from lens.db.connect import connect
from lens.embeddings import Embedder
from lens.tools.retrieval import format_query, hnsw_config_sql, vector_literal

#: A stored run keeps the segmented query the funnel actually used.
QUERY_FILE = "work/query.json"


def _buckets_from_run(run_id: str) -> dict[str, Any]:
    """Return the recorded query buckets of one run."""
    from lens.agent import runstore

    path = Path(runstore.runs_root()) / run_id / QUERY_FILE

    if not path.is_file():
        raise SystemExit(f"no recorded query at {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _exact_top(cur: Any, literal: str, k: int) -> list[str]:
    """Return the exact k nearest paper ids, bypassing the index."""
    cur.execute("set local enable_indexscan = off")
    cur.execute("set local enable_bitmapscan = off")
    cur.execute(
        "SELECT arxiv_id FROM papers WHERE abstract_embedding IS NOT NULL "
        f"ORDER BY abstract_embedding <=> %s::{config.EMBEDDING_PGVECTOR_TYPE} LIMIT %s",
        (literal, k),
    )

    return [row[0] for row in cur.fetchall()]


def _approx_top(cur: Any, literal: str, k: int, ef_search: int) -> list[str]:
    """Return the approximate k nearest paper ids at one ``ef_search``."""
    for statement in hnsw_config_sql(max(ef_search, k)):
        cur.execute(statement)

    cur.execute(
        "SELECT arxiv_id FROM papers WHERE abstract_embedding IS NOT NULL "
        f"ORDER BY abstract_embedding <=> %s::{config.EMBEDDING_PGVECTOR_TYPE} LIMIT %s",
        (literal, k),
    )

    return [row[0] for row in cur.fetchall()]


def probe(text: str, k: int, efs: Sequence[int], target: str | None) -> list[dict[str, Any]]:
    """Return one row per ``ef_search`` value."""
    literal = vector_literal(
        Embedder().encode_documents(
            texts=[text], task_prefix=config.EMBEDDING_QUERY_PREFIX, batch_size=1
        )
    )
    rows: list[dict[str, Any]] = []

    with connect(readonly=True) as conn, conn.cursor() as cur:
        exact = _exact_top(cur, literal, k)
        conn.rollback()
        target_rank = exact.index(target) + 1 if target and target in exact else None

        for ef in efs:
            approx = _approx_top(cur, literal, k, ef)
            conn.rollback()
            shared = len(set(approx) & set(exact))
            rows.append(
                {
                    "ef_search": ef,
                    "recall": shared / len(exact) if exact else 0.0,
                    "shared": shared,
                    "k": k,
                    "target_rank_exact": target_rank,
                    "target_present": target in approx if target else None,
                }
            )

    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Print the recall table; exit 2 when the configured ``ef_search`` is short."""
    parser = argparse.ArgumentParser(prog="probe_recall", description=__doc__)
    parser.add_argument("--run", default=None, help="recorded run id (uses its query.json)")
    parser.add_argument("--query", default=None, help="a raw query string instead")
    parser.add_argument("--k", type=int, default=config.RETRIEVAL_K)
    parser.add_argument("--target", default=None, help="arxiv id expected to be near the top")
    parser.add_argument(
        "--ef", type=int, nargs="*", default=None, help="ef_search values to compare"
    )
    parser.add_argument("--min-recall", type=float, default=0.95)
    args = parser.parse_args(argv)

    if bool(args.run) == bool(args.query):
        parser.error("give exactly one of --run or --query")

    text = args.query or format_query(buckets=_buckets_from_run(args.run))
    efs = args.ef or sorted({40, 100, config.HNSW_EF_SEARCH, 1000})

    print(f"query: {text[:90]}{'...' if len(text) > 90 else ''}")
    print(f"k={args.k} | configured ef_search={config.HNSW_EF_SEARCH} | target={args.target}")
    print()
    print(f"{'ef_search':>9} {'recall@k':>9} {'shared':>7} {'target':>8}")

    rows = probe(text, args.k, efs, args.target)
    configured: float | None = None

    for row in rows:
        present = "-" if row["target_present"] is None else "yes" if row["target_present"] else "NO"
        print(f"{row['ef_search']:>9} {row['recall']:>8.0%} {row['shared']:>7} {present:>8}")

        if row["ef_search"] == max(config.HNSW_EF_SEARCH, args.k):
            configured = row["recall"]

    if configured is None:
        # The configured value may not be in the requested list; probe it too.
        configured = probe(text, args.k, [max(config.HNSW_EF_SEARCH, args.k)], args.target)[0][
            "recall"
        ]

    print()
    print(f"recall at the configured ef_search: {configured:.0%} (floor {args.min_recall:.0%})")

    if configured < args.min_recall:
        print("FAIL: the index is returning a degraded candidate set", file=sys.stderr)

        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
