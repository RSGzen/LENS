"""Build and score a blind claim-truth worksheet.

The worksheet hides the verifier's status: the labeler judges whether the cited
excerpt supports the claim as written. A sidecar keeps the status so the scorer
can compare grounded and unverifiable claims after the labels are filled.

Run:
    python scripts/truth_sample.py build --run <run_id> --out worksheet.csv
    python scripts/truth_sample.py score --labels worksheet.csv --scores worksheet.scores.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

LABELS = ("supported", "not supported", "cannot judge")


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for ``successes / total``."""
    if total < 1:
        return (0.0, 0.0)

    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator

    return (max(0.0, centre - margin), min(1.0, centre + margin))


def build_rows(
    frames: list[dict[str, Any]],
    texts: dict[str, str],
    *,
    run_id: str,
    per_status: int = 10,
    seed: int = 7,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """Return ``(rows, sidecar)`` sampled evenly from grounded and unverifiable frames."""
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for frame in frames:
        verdict = frame.get("verdict") or {}
        status = str(verdict.get("status") or "")
        by_status[status].append(frame)

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    sidecar: dict[str, dict[str, Any]] = {}

    for status in ("grounded", "unverifiable"):
        candidates = list(by_status.get(status, []))
        rng.shuffle(candidates)

        for frame in candidates[:per_status]:
            claim_id = f"{run_id}:{len(rows):03d}"
            chunk_id = str(frame.get("chunk_id") or "")
            excerpt = texts.get(chunk_id, "")
            rows.append(
                {
                    "claim_id": claim_id,
                    "claim_text": str(frame.get("text") or ""),
                    "cited_excerpt": " ".join(excerpt.split())[:1200],
                    "label": "",
                }
            )
            sidecar[claim_id] = {
                "run_id": run_id,
                "status": status,
                "unit_index": frame.get("unit_index"),
                "chunk_id": chunk_id,
            }

    rng.shuffle(rows)

    return rows, sidecar


def score_rows(
    labels: dict[str, str], sidecar: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return support counts and rates by verifier status."""
    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"supported": 0, "not supported": 0, "cannot judge": 0}
    )

    for claim_id, label in labels.items():
        status = str((sidecar.get(claim_id) or {}).get("status") or "unknown")

        if label not in LABELS:
            continue

        buckets[status][label] += 1

    result: dict[str, dict[str, Any]] = {}

    for status, counts in buckets.items():
        decided = counts["supported"] + counts["not supported"]
        low, high = wilson(counts["supported"], decided)
        result[status] = {
            **counts,
            "decided": decided,
            "rate": None if decided < 1 else round(counts["supported"] / decided, 4),
            "ci": [round(low, 4), round(high, 4)],
        }

    return result


def _load_run(run_id: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    from lens.observability.citations import recorded_chunk_texts
    from lens.observability.trajectory import Trajectory

    trajectory = Trajectory(run_id)
    claims_path = trajectory.run_dir / "claims.json"

    if not claims_path.is_file():
        raise FileNotFoundError(f"{run_id} has no claims.json")

    payload = json.loads(claims_path.read_text(encoding="utf-8"))

    return list(payload.get("frames") or []), recorded_chunk_texts(trajectory)


def build(args: argparse.Namespace) -> int:
    frames, texts = _load_run(args.run)
    rows, sidecar = build_rows(
        frames, texts, run_id=args.run, per_status=args.per_status, seed=args.seed
    )

    if not rows:
        print("no claim frames to sample", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scores = Path(args.scores) if args.scores else out.with_suffix(".scores.json")
    scores.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(rows)} claims to {out}")
    print(f"sidecar {scores}")

    return 0


def score(args: argparse.Namespace) -> int:
    sidecar = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    labels: dict[str, str] = {}

    with Path(args.labels).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[str(row.get("claim_id") or "")] = str(row.get("label") or "").strip().lower()

    result = score_rows(labels, sidecar)

    if not result:
        print("no labels filled", file=sys.stderr)
        return 1

    for status, values in sorted(result.items()):
        print(
            f"{status}: supported {values['supported']} / decided {values['decided']} "
            f"= {values['rate']} (95 % CI {values['ci']}), "
            f"cannot judge {values['cannot judge']}"
        )

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build")
    build_parser.add_argument("--run", required=True)
    build_parser.add_argument("--out", required=True)
    build_parser.add_argument("--scores", default=None)
    build_parser.add_argument("--per-status", type=int, default=10)
    build_parser.add_argument("--seed", type=int, default=7)
    build_parser.set_defaults(func=build)

    score_parser = sub.add_parser("score")
    score_parser.add_argument("--labels", required=True)
    score_parser.add_argument("--scores", required=True)
    score_parser.set_defaults(func=score)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
