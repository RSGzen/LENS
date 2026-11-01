"""Record the retrieval gate's P(sufficient) for a query set, offline.

The stage gate is not calibrated: its threshold is 0.5 because that is where a
probability becomes "more likely than not", not because anything was fitted. Fitting
it needs two things, and this script produces the first: **one probability per
query**, computed from the same state the runtime builds, at a chosen funnel
configuration. The second is a relevance judgement per query, which comes from the
pooled qrels the Tier-2 wave already produces — so the fit costs no labelling of its
own.

Why offline: the runtime's verdict is recorded once per run, and a run is expensive.
The state is a pure function of the funnel output, so the same verdict can be asked
again for a fraction of a cent, and swept over thresholds without re-running a query.

Usage::

    python scripts/regrade_stage.py --queries experiments/instruments/tier2-queries-v1.json \\
        --k 200 --n 20 --out <data_root>/campaigns/w2-stage.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from lens import config
from lens.agent.runtime import _retrieval_gate_summary
from lens.clients import LensClient
from lens.db.connect import connect
from lens.embeddings import Embedder
from lens.observability.trajectory import TraceCtx, Trajectory
from lens.reranking import Reranker
from lens.tools.evaluate import evaluate_stage
from lens.tools.retrieval import retrieval_funnel


def load_queries(path: Path) -> list[dict[str, str]]:
    """Return ``[{id, text}]`` from a campaign or instrument file."""
    record = json.loads(path.read_text(encoding="utf-8"))
    queries = record.get("queries") or []

    if not queries:
        raise SystemExit(f"{path} has no queries")

    return [{"id": str(query["id"]), "text": str(query["text"]).strip()} for query in queries]


def buckets_for(text: str) -> dict[str, list[str]]:
    """Return the one-bucket shape the funnel takes; the sweep measures the funnel."""
    return {"objectives": [text], "problems": [], "constraints": [], "requirements": []}


def score_query(
    conn: Any,
    query: dict[str, str],
    *,
    embedder: Embedder,
    reranker: Reranker,
    client: LensClient,
    k: int,
    n: int,
    candidate_chunks: int,
    top_chunks: int,
) -> dict[str, Any]:
    """Run the funnel once and return the gate's probability for that query."""
    started = time.perf_counter()
    bundle = retrieval_funnel(
        conn,
        buckets_for(query["text"]),
        embedder,
        reranker,
        k=k,
        n=n,
        candidate_chunks=candidate_chunks,
        top_chunks=top_chunks,
    )
    verdict = evaluate_stage(
        "retrieval",
        {"score": None, "summary": _retrieval_gate_summary(bundle)},
        client=client,
        trace_ctx=TraceCtx(trace_id=f"stage-{query['id']}", step_idx=0, actor="root"),
        query=query["text"],
    )

    return {
        "query_id": query["id"],
        "papers": len(bundle.get("papers") or {}),
        "chunks": len(bundle.get("chunks") or []),
        "p_sufficient": ""
        if verdict is None or verdict.confidence is None
        else (round(float(verdict.confidence), 3)),
        "verdict": "none" if verdict is None else verdict.verdict,
        "criterion": "" if verdict is None else (verdict.failing_criterion or "")[:160],
        "seconds": round(time.perf_counter() - started, 1),
    }


COLUMNS = ("query_id", "papers", "chunks", "p_sufficient", "verdict", "criterion", "seconds")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="campaign/instrument JSON")
    parser.add_argument("--k", type=int, default=config.RETRIEVAL_K)
    parser.add_argument("--n", type=int, default=config.RETRIEVAL_N)
    parser.add_argument("--candidate-chunks", type=int, default=config.RETRIEVAL_C)
    parser.add_argument("--top-chunks", type=int, default=config.RETRIEVAL_C_TOP)
    parser.add_argument("--out", default=None, help="write the rows as CSV here")
    parser.add_argument(
        "--query-ids", nargs="*", default=None, help="only these query ids (a smoke check)"
    )
    args = parser.parse_args(argv)

    queries = load_queries(Path(args.queries))

    if args.query_ids:
        queries = [query for query in queries if query["id"] in set(args.query_ids)]

    if not queries:
        print("no queries matched", file=sys.stderr)
        return 2

    embedder, reranker = Embedder(), Reranker()
    trajectory = Trajectory(run_id=f"stage-{time.strftime('%Y%m%d-%H%M%S')}")
    client = LensClient(trajectory=trajectory)
    rows: list[dict[str, Any]] = []

    print(
        f"{len(queries)} queries | k={args.k} n={args.n} "
        f"C={args.candidate_chunks} c_top={args.top_chunks}"
    )

    with connect(readonly=True) as conn:
        for index, query in enumerate(queries, start=1):
            row = score_query(
                conn,
                query,
                embedder=embedder,
                reranker=reranker,
                client=client,
                k=args.k,
                n=args.n,
                candidate_chunks=args.candidate_chunks,
                top_chunks=args.top_chunks,
            )
            rows.append(row)
            print(
                f"  [{index}/{len(queries)}] {row['query_id']}: "
                f"P(sufficient)={row['p_sufficient']} {row['verdict']} "
                f"({row['seconds']}s)"
            )

    if args.out:
        out = Path(args.out)

        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nwrote {out}")

    values = [row["p_sufficient"] for row in rows if row["p_sufficient"] != ""]
    failures = sum(1 for row in rows if row["verdict"] == "fail")
    print(
        f"P(sufficient): {len(values)} recorded, "
        f"{min(values) if values else 0:.2f}-{max(values) if values else 0:.2f}, "
        f"{failures} fail at the hard-coded 0.5"
    )
    print(f"gate calls logged in {trajectory.run_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
