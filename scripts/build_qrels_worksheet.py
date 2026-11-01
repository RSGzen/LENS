"""Turn a rankings CSV into a judging worksheet: one row per (query, paper) pair.

The pool is judged blind on title and abstract, in a shuffled, seed-fixed order with
the configuration hidden — otherwise a judge who knows the configuration is judging
the configuration. This script is the only producer of the worksheet, and it writes
the same columns `score_retrieval.py` reads, so the filled file is the qrels file.

Usage::

    python scripts/build_qrels_worksheet.py --rankings rankings.csv --out worksheet.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from lens.db.connect import connect

#: Characters of the abstract shown per candidate. Enough to judge relevance from
#: title and abstract without turning the worksheet into the corpus.
ABSTRACT_CHARS = 900

#: Fixed so the order is reproducible: the sheet is shuffled, not re-randomised.
SHUFFLE_SEED = 20261101


def load_rankings(path: Path, configs: Sequence[str] = ()) -> dict[str, set[str]]:
    """Return ``{query_id: {doc_id}}`` for the pooled candidates.

    ``configs`` restricts the pool to the configurations being judged: pooling every
    point in the grid would judge papers no shipped configuration returns, and the
    plan allows at most three.
    """
    pool: dict[str, set[str]] = {}
    wanted = set(configs)

    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = (row.get("query_id") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            config = (row.get("config") or "").strip()

            if not query_id or not doc_id:
                continue

            if wanted and config not in wanted:
                continue

            pool.setdefault(query_id, set()).add(doc_id)

    return pool


def fetch_papers(cur: Any, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Return ``{arxiv_id: {title, abstract}}`` for the given ids."""
    cur.execute(
        "SELECT arxiv_id, title, coalesce(abstract, '') FROM papers WHERE arxiv_id = ANY(%s)",
        (list(ids),),
    )

    return {
        str(arxiv_id): {"title": str(title or ""), "abstract": str(abstract or "")}
        for arxiv_id, title, abstract in cur.fetchall()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", required=True, help="config,query_id,rank,doc_id CSV")
    parser.add_argument("--out", required=True, help="worksheet CSV (fill the rel column)")
    parser.add_argument(
        "--config",
        nargs="*",
        default=None,
        help="pool only these configurations (the plan allows at most three)",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="campaign JSON with the query texts, so the sheet can show them",
    )
    args = parser.parse_args(argv)

    pool = load_rankings(Path(args.rankings), args.config or ())
    texts: dict[str, str] = {}

    if args.queries:
        import json

        for query in (
            json.loads(Path(args.queries).read_text(encoding="utf-8")).get("queries") or []
        ):
            texts[str(query["id"])] = str(query["text"]).strip()

    order = random.Random(SHUFFLE_SEED)
    rows: list[dict[str, Any]] = []

    with connect(readonly=True) as conn, conn.cursor() as cur:
        for query_id in sorted(pool):
            papers = fetch_papers(cur, sorted(pool[query_id]))
            ids = sorted(papers)
            order.shuffle(ids)

            for doc_id in ids:
                paper = papers[doc_id]
                rows.append(
                    {
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "rel": "",
                        "title": " ".join(paper["title"].split()),
                        "abstract": " ".join(paper["abstract"].split())[:ABSTRACT_CHARS],
                        "query": " ".join(texts.get(query_id, "").split()),
                    }
                )

            print(f"{query_id}: {len(ids)} candidate(s)")

    out = Path(args.out)

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["query_id", "doc_id", "rel", "title", "abstract", "query"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {out}: {len(rows)} pair(s) to judge")
    print("Fill `rel` with 1 (relevant) or 0 (not), then pass the same file to score_retrieval.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
