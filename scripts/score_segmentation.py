"""Score a four-bucket segmentation against the compound gold buckets.

Two modes:
  --live        call the segmenter once per non-example instance (small cost)
  --run <id>    read a recorded run's work/query.json (one instance)

Run:
    python scripts/score_segmentation.py --instrument experiments/instruments/compound-v1.jsonl \
        --live --out <data_root>/campaigns/segmentation-scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from lens import config
from lens.agent import runstore
from lens.agent.segmentation import segment_query
from lens.experiments.segmentation_score import (
    BUCKETS,
    bucket_union_scores,
    score_buckets,
)
from lens.observability.trajectory import TraceCtx, Trajectory


def _instances(path: Path) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        payload = json.loads(line)

        if not isinstance(payload, dict) or not payload.get("instance_id"):
            raise ValueError(f"{path}:{number}: an instance needs instance_id")

        if (payload.get("meta") or {}).get("example"):
            continue

        instances.append(payload)

    return instances


def _recorded(run_id: str) -> dict[str, list[str]]:
    path = runstore.runs_root() / run_id / "work" / "query.json"

    if not path.is_file():
        raise FileNotFoundError(f"{run_id} has no work/query.json")

    return json.loads(path.read_text(encoding="utf-8"))


def _live(instances: list[dict[str, Any]], run_id: str) -> dict[str, list[str]]:
    from lens.clients import LensClient

    config.require_api_key()
    trajectory = Trajectory(run_id)
    client = LensClient(trajectory=trajectory)
    trace = TraceCtx(run_id, 0, "root")
    produced: dict[str, list[str]] = {}

    for instance in instances:
        instance_id = str(instance["instance_id"])
        produced[instance_id] = segment_query(
            client, str(instance.get("query") or ""), trace_ctx=trace
        )

    return produced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--run", default=None, help="a recorded run id (one instance)")
    parser.add_argument("--instance", default=None, help="instance id for --run")
    parser.add_argument("--buckets-out", default=None, help="write the produced buckets here")
    parser.add_argument(
        "--buckets-in",
        default=None,
        help="re-score a saved buckets file, no model call",
    )
    args = parser.parse_args(argv)

    instances = _instances(Path(args.instrument))

    if args.instance:
        instances = [item for item in instances if item["instance_id"] == args.instance]

    if not instances:
        print("no instances to score", file=sys.stderr)
        return 1

    if args.buckets_in:
        produced = json.loads(Path(args.buckets_in).read_text(encoding="utf-8"))
    elif args.live:
        run_id = args.run or "segmentation-gold-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        produced = _live(instances, run_id)
    else:
        if not args.run:
            print("pass --live, --run <id> or --buckets-in <file>", file=sys.stderr)
            return 1

        if not args.instance:
            print("--run mode needs --instance (one run records one query)", file=sys.stderr)
            return 1

        buckets = _recorded(args.run)
        produced = {str(instance["instance_id"]): buckets for instance in instances}

    rows: list[dict[str, Any]] = []

    for instance in instances:
        instance_id = str(instance["instance_id"])
        gold = instance.get("gold_buckets") or {}
        got = produced.get(instance_id) or {}
        scores = score_buckets(gold, got)
        union = bucket_union_scores(gold, got)

        for bucket in BUCKETS:
            rows.append(
                {
                    "instance_id": instance_id,
                    "bucket": bucket,
                    **scores[bucket],
                    "matched_union": union[bucket]["matched_union"],
                    "recall_union": union[bucket]["recall_union"],
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.buckets_out:
        Path(args.buckets_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.buckets_out).write_text(
            json.dumps(produced, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    recalls = [row["recall"] for row in rows if row["recall"] is not None]
    precisions = [row["precision"] for row in rows if row["precision"] is not None]
    unions = [row["recall_union"] for row in rows if row["recall_union"] is not None]
    print(
        f"{len(instances)} instances · clause recall "
        f"{round(mean(recalls), 4) if recalls else 'n/a'} · bucket precision "
        f"{round(mean(precisions), 4) if precisions else 'n/a'} · union recall "
        f"{round(mean(unions), 4) if unions else 'n/a'}"
    )
    print(f"wrote {len(rows)} rows to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
