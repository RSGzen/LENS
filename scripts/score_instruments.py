"""Exact-match scoring for the in-house instruments (local, no key).

Each instrument file is JSONL, one instance per line::

    {"instance_id": "cag-25-001", "instrument": "cag-25",
     "question": "...", "gold": {"answer": "7", "working_ids": [...]},
     "answer_format": "integer", "meta": {...}}

The script joins instances to a campaign CSV (which carries ``run_id`` and
``query_id``), reads each run's ``answer.md`` from the runs root, and compares the
answer to the gold under the format the instance declares. A format the instance
does not state is scored as ``free`` (gold substring), which is the weakest rule
and therefore the one that cannot flatter a run.

The bridge instrument can be scored by the benchmark's own evaluator instead:
``--scorer-module mymodule`` calls ``mymodule.score(answer, instance) -> bool``.
That keeps a third-party scorer importable without vendoring it.

Usage::

    python scripts/score_instruments.py --instances cag-25.jsonl --campaign campaign.csv
    python scripts/score_instruments.py --instances bridge.jsonl --campaign c.csv \
        --scorer-module oolong_scorer --out bridge-scores.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from lens.agent import runstore

#: Text after the sentinel is the answer; without one, the whole reply is.
SENTINEL = "FINAL_ANSWER:"

_WHITESPACE = re.compile(r"\s+")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_INTEGER = re.compile(r"-?\d+")
_ID_SPLIT = re.compile(r"[,;\s]+")


def load_instances(path: str | Path) -> list[dict[str, Any]]:
    """Return the instances, skipping blank lines; a non-object line raises."""
    instances: list[dict[str, Any]] = []

    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        payload = json.loads(line)

        if not isinstance(payload, dict) or not payload.get("instance_id"):
            raise ValueError(f"{path}:{number}: an instance needs instance_id")

        instances.append(payload)

    return instances


def read_answer(run_id: str) -> str | None:
    """Return ``answer.md`` for a run, or ``None`` when the run is missing."""
    path = runstore.runs_root() / run_id / "answer.md"

    if not path.is_file():
        return None

    text = path.read_text(encoding="utf-8")

    if SENTINEL in text:
        text = text.rsplit(SENTINEL, 1)[1]

    return text.strip()


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, and strip punctuation at both ends."""
    return _WHITESPACE.sub(" ", text).strip().strip(".!?;:").lower()


def _as_number(text: str) -> float | None:
    match = _INTEGER.search(text.replace(",", ""))
    return float(match.group()) if match else None


def _as_ids(text: str) -> list[str]:
    return sorted(part.strip("[]()'\"") for part in _ID_SPLIT.split(text) if part.strip("[]()'\""))


def score_answer(answer: str, gold: Any, answer_format: str) -> tuple[bool, str]:
    """Return ``(correct, predicted)`` for one answer under one format."""
    text = answer.strip()

    if answer_format == "integer":
        predicted = _as_number(text)
        expected = _as_number(str(gold))

        return (predicted is not None and predicted == expected, str(predicted))

    if answer_format == "sorted-id-list":
        predicted = _as_ids(text)
        expected = _as_ids(str(gold))

        return (bool(predicted) and predicted == expected, ",".join(predicted))

    if answer_format == "uuid":
        match = _UUID.search(text)
        predicted = match.group().lower() if match else ""

        return (predicted != "" and predicted == str(gold).lower(), predicted)

    # ``free``: the gold must appear, normalised. Weakest rule, so it is the
    # default — an instance that wants more has to declare it.
    return (normalise(str(gold)) in normalise(text), normalise(text)[:200])


def campaign_rows(path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Group campaign rows by ``query_id`` (the instance id)."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = (row.get("query_id") or "").strip()

            if query_id:
                grouped[query_id].append(row)

    return dict(grouped)


def score_instances(
    instances: Iterable[dict[str, Any]],
    campaign: dict[str, list[dict[str, str]]],
    *,
    answers: Callable[[str], str | None] = read_answer,
    scorer: Callable[[str, dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Score every (instance, arm) cell that the campaign actually ran."""
    results: list[dict[str, Any]] = []

    for instance in instances:
        instance_id = str(instance["instance_id"])
        gold = (instance.get("gold") or {}).get("answer")
        answer_format = str(instance.get("answer_format") or "free")

        for row in campaign.get(instance_id, []):
            run_id = row.get("run_id") or ""
            answer = answers(run_id)

            if answer is None:
                results.append(
                    {
                        "instance_id": instance_id,
                        "arm": row.get("campaign_arm", ""),
                        "run_id": run_id,
                        "gold": gold,
                        "predicted": "",
                        "correct": 0,
                        "note": "missing answer.md",
                    }
                )
                continue

            if scorer is not None:
                correct, predicted, note = bool(scorer(answer, instance)), "", "custom scorer"
            else:
                correct, predicted = score_answer(answer, gold, answer_format)
                note = ""

            results.append(
                {
                    "instance_id": instance_id,
                    "arm": row.get("campaign_arm", ""),
                    "run_id": run_id,
                    "gold": gold,
                    "predicted": predicted,
                    "correct": int(correct),
                    "note": note,
                }
            )

    return results


COLUMNS = ("instance_id", "arm", "run_id", "gold", "predicted", "correct", "note")


def summarise(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accuracy per arm, plus the count behind it."""
    by_arm: dict[str, list[int]] = defaultdict(list)

    for row in results:
        by_arm[str(row["arm"])].append(int(row["correct"]))

    return [
        {
            "instance_id": f"%%{arm}",
            "arm": arm,
            "run_id": "",
            "gold": "",
            "predicted": "",
            "correct": round(sum(values) / len(values), 4) if values else 0.0,
            "note": f"accuracy over {len(values)} runs",
        }
        for arm, values in sorted(by_arm.items())
    ]


def write_rows(rows: Iterable[dict[str, Any]], path: str | Path | None) -> str:
    """Write the score table as CSV and return it as text."""
    buffer: list[str] = [",".join(COLUMNS)]

    for row in rows:
        buffer.append(",".join(str(row.get(column, "")).replace(",", ";") for column in COLUMNS))

    text = "\n".join(buffer) + "\n"

    if path is not None:
        Path(path).write_text(text, encoding="utf-8")

    return text


def load_scorer(module_name: str) -> Callable[[str, dict[str, Any]], bool]:
    """Import ``module.score(answer, instance)`` for a third-party evaluator."""
    module = importlib.import_module(module_name)
    scorer = getattr(module, "score", None)

    if not callable(scorer):
        raise SystemExit(f"{module_name} has no callable score(answer, instance)")

    return scorer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score instrument instances by exact match.")
    parser.add_argument("--instances", required=True, help="JSONL instance file")
    parser.add_argument("--campaign", required=True, help="campaign.csv from the runner")
    parser.add_argument(
        "--scorer-module", default=None, help="module exposing score(answer, instance)"
    )
    parser.add_argument("--out", default=None, help="write the score CSV here")
    parser.add_argument("--quiet", action="store_true", help="write only the summary")
    args = parser.parse_args(argv)

    instances = load_instances(args.instances)
    campaign = campaign_rows(args.campaign)
    scorer = load_scorer(args.scorer_module) if args.scorer_module else None

    results = score_instances(instances, campaign, scorer=scorer)

    if not results:
        print(
            f"{args.instances}: no campaign rows matched these instance ids",
            file=sys.stderr,
        )
        return 2

    rows = summarise(results) if args.quiet else [*summarise(results), *results]
    sys.stdout.write(write_rows(rows, args.out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
