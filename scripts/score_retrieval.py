"""Retrieval metrics from pooled relevance judgments (local, no key, no LLM).

Inputs are two CSVs, so the numbers are reproducible from files rather than from
a live database:

``qrels.csv``      ``query_id, doc_id, rel``      (``rel`` graded 0-3)
``rankings.csv``   ``config, query_id, rank, doc_id``

Output is one row per configuration: nDCG@10 (primary), recall@10, recall@50,
MRR and precision@10. A document counts as relevant at ``rel >= 1``; nDCG uses
the graded value.

Why a separate script: retrieval quality is measured on retrieval output, which
needs no model call. Bundling it into the campaign runner would make every
metric change cost money.

Usage::

    python scripts/score_retrieval.py --qrels qrels.csv --rankings rankings.csv
    python scripts/score_retrieval.py --qrels qrels.csv --rankings rankings.csv --out scores.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

#: ``rel`` at or above this is "relevant" for recall / MRR / precision.
RELEVANT = 1


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Return ``{query_id: {doc_id: rel}}``; a duplicate pair keeps the max."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)

    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = (row.get("query_id") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()

            if not query_id or not doc_id:
                continue

            rel = int(float(row.get("rel") or 0))
            qrels[query_id][doc_id] = max(rel, qrels[query_id].get(doc_id, 0))

    return dict(qrels)


def load_rankings(path: str | Path) -> dict[tuple[str, str], list[str]]:
    """Return ``{(config, query_id): [doc_id, ...]}`` ordered by rank."""
    ranked: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)

    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            config = (row.get("config") or "").strip()
            query_id = (row.get("query_id") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()

            if not config or not query_id or not doc_id:
                continue

            ranked[(config, query_id)].append((int(float(row.get("rank") or 0)), doc_id))

    return {key: [doc for _rank, doc in sorted(values)] for key, values in ranked.items()}


def _gain(rel: int) -> float:
    """Exponential gain, the standard nDCG form: ``2**rel - 1``."""
    return float(2**rel - 1)


def dcg(rels: Sequence[int], k: int) -> float:
    """Discounted cumulative gain over the first ``k`` graded relevances."""
    return sum(_gain(rel) / math.log2(index + 2) for index, rel in enumerate(rels[:k]))


def ndcg_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int = 10) -> float:
    """``DCG@k / IDCG@k``; ``0.0`` when the query has no relevant document."""
    ideal = sorted(qrels.values(), reverse=True)

    if not ideal or _gain(ideal[0]) == 0:
        return 0.0

    actual = dcg([qrels.get(doc_id, 0) for doc_id in ranked], k)

    return actual / dcg(ideal, k)


def recall_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int) -> float:
    """Share of relevant documents that appear in the first ``k`` results."""
    relevant = {doc_id for doc_id, rel in qrels.items() if rel >= RELEVANT}

    if not relevant:
        return 0.0

    return len(relevant & set(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], qrels: dict[str, int]) -> float:
    """``1 / rank`` of the first relevant result, ``0.0`` if none is retrieved."""
    for index, doc_id in enumerate(ranked):
        if qrels.get(doc_id, 0) >= RELEVANT:
            return 1.0 / (index + 1)

    return 0.0


def precision_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int = 10) -> float:
    """Share of the first ``k`` results that are relevant (``k`` as the denominator)."""
    if k <= 0:
        return 0.0

    return sum(1 for doc_id in ranked[:k] if qrels.get(doc_id, 0) >= RELEVANT) / k


def score_config(
    rankings: dict[tuple[str, str], list[str]],
    qrels: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    """Score every configuration; a query absent from the rankings scores zero."""
    configs = sorted({config for config, _query in rankings})
    per_config: dict[str, list[dict[str, float]]] = defaultdict(list)

    for config in configs:
        for query_id, judged in qrels.items():
            ranked = rankings.get((config, query_id), [])
            per_config[config].append(
                {
                    "ndcg@10": ndcg_at_k(ranked, judged, 10),
                    "recall@10": recall_at_k(ranked, judged, 10),
                    "recall@50": recall_at_k(ranked, judged, 50),
                    "mrr": reciprocal_rank(ranked, judged),
                    "p@10": precision_at_k(ranked, judged, 10),
                }
            )

    rows: list[dict[str, object]] = []

    for config in configs:
        scores = per_config[config]
        row: dict[str, object] = {"config": config, "queries": len(scores)}

        for metric in ("ndcg@10", "recall@10", "recall@50", "mrr", "p@10"):
            values = [entry[metric] for entry in scores]
            row[metric] = round(sum(values) / len(values), 4) if values else 0.0

        rows.append(row)

    return rows


COLUMNS = ("config", "queries", "ndcg@10", "recall@10", "recall@50", "mrr", "p@10")


def write_scores(rows: Iterable[dict[str, object]], path: str | Path | None) -> str:
    """Write the score table as CSV and return it as text (also for the console)."""
    buffer: list[str] = [",".join(COLUMNS)]

    for row in rows:
        buffer.append(",".join(str(row.get(column, "")) for column in COLUMNS))

    text = "\n".join(buffer) + "\n"

    if path is not None:
        Path(path).write_text(text, encoding="utf-8")

    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score retrieval configurations on pooled qrels.")
    parser.add_argument("--qrels", required=True, help="CSV: query_id,doc_id,rel")
    parser.add_argument("--rankings", required=True, help="CSV: config,query_id,rank,doc_id")
    parser.add_argument("--out", default=None, help="write scores.csv here")
    args = parser.parse_args(argv)

    qrels = load_qrels(args.qrels)

    if not qrels:
        print(f"{args.qrels}: no judgments", file=sys.stderr)
        return 2

    rows = score_config(load_rankings(args.rankings), qrels)
    sys.stdout.write(write_scores(rows, args.out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
