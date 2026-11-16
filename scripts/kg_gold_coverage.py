r"""Gold-evidence coverage for the query-time KG (read-only except the build cache).

Builds the knowledge graph over **exactly the gold ``chunk_uuid`` set** of the I5
and I6 instruments and reports how much of that evidence survives with at least
one edge. This is the upper bound on what the graph could navigate to: the
production scope (the funnel's own chunks) is measured separately by
``scripts/kg_probe.py``.

Run::

    .\.venv\Scripts\python.exe scripts\kg_gold_coverage.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from lens.db.connect import connect
from lens.kg_extract import GlinerExtractor
from lens.observability.run_logging import setup_file_logging
from lens.tools.kg import build_kg

INSTRUMENTS = ("i5-gold-v1.jsonl", "compound-v1.jsonl")


def _instances(paths: list[Path]) -> list[dict[str, Any]]:
    """Load instruments; keep only instances that carry gold claims."""
    out = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("meta", {}).get("example"):
                continue
            claims = (row.get("gold") or {}).get("claims") or []
            if claims:
                out.append(row)
    return out


def main() -> None:
    setup_file_logging("lens.kg_gold_coverage", "kg_gold_coverage")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instrument-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "instruments",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    instances = _instances([args.instrument_dir / name for name in INSTRUMENTS])
    chunk_ids = sorted({c["chunk_uuid"] for inst in instances for c in inst["gold"]["claims"]})
    print(f"instances={len(instances)} gold_chunks={len(chunk_ids)}")

    with connect(readonly=True) as conn:
        t0 = time.perf_counter()
        result = build_kg(conn, chunk_ids, GlinerExtractor(), use_cache=False)
        elapsed = time.perf_counter() - t0

    graph = result["graph"]
    edges = graph.get("edges") or graph.get("links") or []
    edge_chunks = {e["chunk_uuid"] for e in edges if e.get("chunk_uuid")}
    print(f"build_kg: {elapsed:.1f}s kept={result['stats']['triples_kept']} edges={len(edges)}")
    print(f"stats: {result['stats']['stage_counts']} {result['stats']['ontology_dropped']}")

    rows = []
    for inst in instances:
        claims = inst["gold"]["claims"]
        covered = [c for c in claims if c["chunk_uuid"] in edge_chunks]
        rows.append(
            {
                "instance": inst["instance_id"],
                "claims": len(claims),
                "covered": len(covered),
            }
        )

    for row in rows:
        print(f"  {row['instance']:<44} claims={row['claims']:<2} covered={row['covered']}")

    total_claims = sum(r["claims"] for r in rows)
    total_covered = sum(r["covered"] for r in rows)
    print(
        f"gold_chunks_with_edge={len(set(chunk_ids) & edge_chunks)}/{len(chunk_ids)} "
        f"({100 * len(set(chunk_ids) & edge_chunks) / max(1, len(chunk_ids)):.1f}%) "
        f"claims_covered={total_covered}/{total_claims} "
        f"({100 * total_covered / max(1, total_claims):.1f}%) "
        f"instances_with_any={sum(1 for r in rows if r['covered'])}/{len(rows)}"
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "instances": rows,
                    "gold_chunks": len(chunk_ids),
                    "chunks_with_edge": len(set(chunk_ids) & edge_chunks),
                    "claims": total_claims,
                    "claims_covered": total_covered,
                    "stats": result["stats"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
