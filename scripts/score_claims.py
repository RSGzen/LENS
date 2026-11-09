"""Score I5/I6 answers against a gold claim set (text-level, no key, no DB).

The instrument line carries ``gold.claims``; see ``Evaluation.md`` §6.5. The
answer is read from ``answer.md`` and the cited text from the run's recorded
payloads. The claim ledger is reported separately, with an explicit
``ledger_status``:

- ``ok``      the ledger describes this answer;
- ``missing`` no ``claims.json`` at all;
- ``empty``   a ledger with no units - the claim pass was skipped;
- ``stale``   a ledger whose units are not in the final answer.

``ledger_coverage`` is written **only** for ``ok``. A skipped or stale pass is
reported as absent, never as 0 %: a zero would enter an arm mean as a
measurement and drag it down, which is the failure `Eval D-24` and `D-25` name.

**A broken or empty run is excluded, never scored as zero** (`Eval D-31`). A run
that failed keeps its exception text in ``answer.md``, so "has an answer" is not
the test; the campaign row's ``status`` and ``stop_reason`` are. The raw counts
stay (they are facts about the run) and every **rate** is blanked, so no reader
can average one by accident.

Run:
    python scripts/score_claims.py --instrument <file.jsonl> \
        --campaign <campaign.csv> --out <scores.csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from lens.agent import runstore
from lens.experiments.claim_score import (
    citation_precision,
    citation_recall,
    claim_recall,
    gold_chunk_recall,
    ledger_status,
)
from lens.observability.citations import (
    expand_short_ids,
    recorded_chunk_ids,
    recorded_chunk_texts,
)
from lens.observability.trajectory import Trajectory


def _load_instances(path: Path) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        payload = json.loads(line)

        if not isinstance(payload, dict) or not payload.get("instance_id"):
            raise ValueError(f"{path}:{number}: an instance needs instance_id")

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


def _claims(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def score(instrument: Path, campaign: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for instance in _load_instances(instrument):
        instance_id = str(instance["instance_id"])
        gold_claims = list((instance.get("gold") or {}).get("claims") or [])

        for row in _campaign_rows(campaign).get(instance_id, []):
            run_id = row.get("run_id") or ""
            arm = row.get("campaign_arm") or ""
            answer = _answer(run_id)
            trajectory = Trajectory(run_id)
            texts = recorded_chunk_texts(trajectory)
            recorded = recorded_chunk_ids(trajectory)
            expanded = expand_short_ids(answer, recorded)
            claims = _claims(trajectory.run_dir / "claims.json")

            matched, _ = claim_recall(gold_claims, expanded)
            supported, judged = citation_precision(gold_claims, expanded, texts)
            present, gold_chunks = gold_chunk_recall(gold_claims, recorded)
            cited_gold, _ = citation_recall(gold_claims, expanded)
            units = int((claims or {}).get("units") or 0)
            coverage = float((claims or {}).get("coverage") or 0.0)
            status = ledger_status(answer, claims)

            # `D-31`: a run that broke, or whose answer is empty, must not enter an
            # arm mean as a zero. A failed run keeps its exception text in
            # `answer.md`, so "has an answer" is not the test - the campaign row's
            # own status is. The raw counts stay (they are facts); every **rate** is
            # blanked, so no reader can average one by accident.
            run_status = (row.get("status") or "").strip()
            run_stop = (row.get("stop_reason") or "").strip()
            broke = run_status == "failed" or run_stop == "error"
            empty = not answer.strip()
            usable = not broke and not empty

            rows.append(
                {
                    "instance_id": instance_id,
                    "arm": arm,
                    "run_id": run_id,
                    "run_status": run_status,
                    "stop_reason": run_stop,
                    "gold_claims": len(gold_claims),
                    "matched": matched,
                    "claim_recall": _rate(matched, len(gold_claims)) if usable else None,
                    "supported": supported,
                    "judged": judged,
                    "citation_precision": _rate(supported, judged) if usable else None,
                    "gold_chunks": gold_chunks,
                    "present": present,
                    "gold_chunk_recall": _rate(present, gold_chunks) if usable else None,
                    "cited_gold": cited_gold,
                    "citation_recall": _rate(cited_gold, gold_chunks) if usable else None,
                    "ledger_status": status,
                    "ledger_units": units,
                    "ledger_coverage": round(coverage, 4) if status == "ok" else "",
                    "ledger_stale": status == "stale",
                    "answer_chars": len(answer),
                    "missing": empty or broke,
                }
            )

    return rows


def summarise(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    for arm in sorted({row["arm"] for row in rows}):
        cells = [row for row in rows if row["arm"] == arm]
        usable = [cell for cell in cells if not cell["missing"]]
        claims = [cell["claim_recall"] for cell in usable if cell["claim_recall"] is not None]
        precision = [
            cell["citation_precision"] for cell in usable if cell["citation_precision"] is not None
        ]
        gold_recall = [
            cell["gold_chunk_recall"] for cell in usable if cell["gold_chunk_recall"] is not None
        ]
        stale = sum(1 for cell in cells if cell["ledger_status"] == "stale")
        missing_ledger = sum(1 for cell in cells if cell["ledger_status"] == "missing")
        empty_ledger = sum(1 for cell in cells if cell["ledger_status"] == "empty")
        measured = [cell["ledger_coverage"] for cell in cells if cell["ledger_coverage"] != ""]

        lines.append(
            f"{arm}: runs {len(cells)} · usable {len(usable)} · excluded {len(cells) - len(usable)}"
            f" · claim recall {round(mean(claims), 4) if claims else 'n/a'}"
            f" · citation precision {round(mean(precision), 4) if precision else 'n/a'}"
            f" · gold-chunk recall {round(mean(gold_recall), 4) if gold_recall else 'n/a'}"
            f" · ledger ok {len(measured)}/{len(cells)}"
            f" (stale {stale}, empty {empty_ledger}, missing {missing_ledger})"
            f" · mean ledger coverage "
            f"{round(mean(measured), 4) if measured else 'n/a'} over the ok rows"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = score(Path(args.instrument), Path(args.campaign))

    if not rows:
        print("no (instance, run) rows to score", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(summarise(rows))
    print(f"wrote {len(rows)} rows to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
