"""Build the bucket-coverage worksheet for the compound set.

Mechanical and gold checklist items are auto-scored (`auto`: pass/fail); the
structural items are marked `human` and left blank for the rater. A row is the
unit: one instance × one arm × one checklist item.

Run:
    python scripts/score_bucket_coverage.py --instrument experiments/instruments/compound-v1.jsonl \
        --campaign <campaign.csv> --out <data_root>/campaigns/compound-coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from lens.agent import runstore
from lens.experiments.claim_score import cited_ids
from lens.experiments.coverage_score import checklist_rows
from lens.observability.citations import claim_units

_ARXIV = re.compile(r"^\d{4}\.\d{4,5}$")


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


def _campaign_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = (row.get("query_id") or "").strip()
            if query_id:
                grouped[query_id].append(row)

    return dict(grouped)


def _answer(run_id: str) -> str:
    path = runstore.runs_root() / run_id / "answer.md"
    return path.read_text(encoding="utf-8", errors="replace").strip() if path.is_file() else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    instances = _instances(Path(args.instrument))
    campaign = _campaign_rows(Path(args.campaign))
    rows: list[dict[str, str]] = []

    for instance in instances:
        instance_id = str(instance["instance_id"])

        for campaign_row in campaign.get(instance_id, []):
            run_id = campaign_row.get("run_id") or ""
            arm = campaign_row.get("campaign_arm") or ""
            answer = _answer(run_id)
            units = claim_units(answer, include_uncited=True)
            cited = [token for token in cited_ids(answer) if _ARXIV.match(token)]

            rows.extend(
                checklist_rows(
                    instance,
                    answer=answer,
                    cited_arxiv_ids=cited,
                    claim_units=units,
                    arm=arm,
                    run_id=run_id,
                )
            )

    if not rows:
        print("no (instance, arm) checklist rows to build", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0, "human": 0})
    for row in rows:
        counts[row["arm"]][row["auto"]] += 1

    for arm, values in sorted(counts.items()):
        print(
            f"{arm}: auto pass {values['pass']} · fail {values['fail']} · human {values['human']}"
        )

    print(f"wrote {len(rows)} rows to {out}")
    print("fill the blank `label` on the `human` rows; the auto rows are already decided")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
