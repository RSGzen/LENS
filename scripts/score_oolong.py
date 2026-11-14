"""Score a W3 campaign with the release's own scorer and write `scores.csv`."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lens.agent import runstore

SENTINEL = "FINAL_ANSWER:"


def read_answer(run_id: str) -> str:
    """Return the text after the sentinel, or the whole answer when it is absent."""
    path = runstore.runs_root() / run_id / "answer.md"

    if not path.is_file():
        return ""

    text = path.read_text(encoding="utf-8")

    return text.rsplit(SENTINEL, 1)[1].strip() if SENTINEL in text else text.strip()


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

    rows = list(csv.DictReader(open(args.campaign, encoding="utf-8")))
    out_rows: list[dict] = []

    for row in rows:
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
                "score": scored["score"],
                "attempted_parse": scored["attempted_parse"],
                "parse_confidence": scored["parse_confidence"],
                "answer": scored["answer"],
                "status": row["status"],
                "run_status": row["stop_reason"] or "answer",
                "answer_chars": len(answer),
            }
        )

    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    by_arm: dict[str, list[float]] = {}
    for row in out_rows:
        by_arm.setdefault(row["arm"], []).append(float(row["score"]))
    for arm, scores in sorted(by_arm.items()):
        print(f"{arm}: {len(scores)} runs, mean score {sum(scores) / len(scores):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
