"""Score a W3 campaign with the release's own scorer and write `scores.csv`.

An empty answer or a failed run is **excluded, never scored zero** (`Eval D-31`):
a 93k-token prompt can push the flat arm to `finish_reason=length` with no visible
text, and counting that as a wrong answer would report a harness property as a
capability result. The per-task-type summary is what the bridge reports, because
the numeric instances ask for an exact count among ~3,182 records and a flat arm
that forfeits them cannot be read as a single blended mean.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lens.agent import runstore

SENTINEL = "FINAL_ANSWER:"


def read_answer(run_id: str) -> str:
    """Return the text after the sentinel, or the whole answer when it is absent."""
    path = runstore.runs_root() / run_id / "answer.md"

    if not path.is_file():
        return ""

    text = path.read_text(encoding="utf-8")

    return text.rsplit(SENTINEL, 1)[1].strip() if SENTINEL in text else text.strip()


def usable(status: str, stop_reason: str, answer: str) -> bool:
    """Return whether a run may enter an arm mean at all."""
    return bool(answer.strip()) and status != "failed" and stop_reason != "error"


def summarise(
    rows: Sequence[Mapping[str, Any]], answer_type_by_id: Mapping[str, str]
) -> dict[str, Any]:
    """Return per-arm and per-type means over the usable rows only."""
    by_arm: dict[str, list[float]] = {}
    by_type: dict[str, list[float]] = {}
    excluded = 0

    for row in rows:
        if not row["usable"]:
            excluded += 1
            continue
        by_arm.setdefault(row["arm"], []).append(float(row["score"]))
        answer_type = answer_type_by_id.get(row["instance_id"], "?")
        by_type.setdefault(answer_type, []).append(float(row["score"]))

    return {
        "arms": {arm: (len(scores), sum(scores) / len(scores)) for arm, scores in by_arm.items()},
        "types": {
            kind: (len(scores), sum(scores) / len(scores)) for kind, scores in by_type.items()
        },
        "excluded": excluded,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import oolong_scorer  # noqa: PLC0415 - scripts dir is on sys.path when run as a script

    instances = {
        line["instance_id"]: line
        for line in (
            json.loads(row)
            for row in Path(args.instrument).read_text(encoding="utf-8").splitlines()
            if row.strip()
        )
    }

    out_rows: list[dict[str, Any]] = []

    for row in csv.DictReader(open(args.campaign, encoding="utf-8")):
        instance = instances.get(row["query_id"])

        if instance is None:
            continue

        answer = read_answer(row["run_id"])
        scored = oolong_scorer.synth_process_response(
            {
                "id": instance["instance_id"],
                "answer": instance["gold"]["answer"],
                "answer_type": instance.get("answer_type"),
                "context_window_id": (instance.get("meta") or {}).get("context_window_id"),
                "dataset": (instance.get("meta") or {}).get("dataset"),
            },
            answer,
        )
        out_rows.append(
            {
                "run_id": row["run_id"],
                "arm": row["campaign_arm"],
                "instance_id": instance["instance_id"],
                "answer_type": instance.get("answer_type", ""),
                "score": scored["score"],
                "attempted_parse": scored["attempted_parse"],
                "parse_confidence": scored["parse_confidence"],
                "answer": scored["answer"],
                "status": row["status"],
                "run_status": row["stop_reason"] or "answer",
                "answer_chars": len(answer),
                "usable": usable(row["status"], row["stop_reason"], answer),
            }
        )

    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    summary = summarise(
        out_rows, {key: value.get("answer_type", "") for key, value in instances.items()}
    )
    print(f"excluded (empty or broken): {summary['excluded']}")
    for arm, (count, mean) in sorted(summary["arms"].items()):
        print(f"  {arm}: {count} usable, mean score {mean:.3f}")
    for kind, (count, mean) in sorted(summary["types"].items()):
        print(f"  {kind}: {count} usable, mean {mean:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
