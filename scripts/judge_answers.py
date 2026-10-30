"""Blind LLM judge for end-to-end answers (one call per answer; writes judge.csv).

The judge is a **measurement instrument, not a component**: it never runs inside a
query, it never sees the arm name, and its calls are logged to their own run
directory so its spend cannot be confused with a query's spend.

Facets are binary and come from a frozen rubric file, because a 1-5 scale agrees
poorly between raters and cannot be checked against a human sample:

    coverage        names what a domain reader expects for this query
    grounding       every factual claim carries a citation that resolves
    scope_honesty   states what was inspected; no global negative from a local look
    usability       reads as a synthesis a reviewer could act on

An unparseable reply is recorded as ``unknown`` with the raw text as the reason —
never converted to ``fail``, which would turn a schema drift into a quality drop.

Usage::

    python scripts/judge_answers.py --campaign campaign.csv --rubric rubric.json
    python scripts/judge_answers.py --campaign campaign.csv --rubric rubric.json \
        --arms rlm-full naive-flat --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lens.agent import runstore
from lens.clients import LensClient
from lens.observability.trajectory import TraceCtx, Trajectory

SENTINEL = "FINAL_ANSWER:"
MAX_ANSWER_CHARS = 6000

SYSTEM_PROMPT = """\
You grade one literature-evidence answer against a fixed rubric.

Rules
- Judge only what the answer says, against the question.
- A claim without a resolvable citation fails grounding, even when it is true.
- An answer that says it could not support something, and says what it inspected,
  passes scope honesty. An answer that states a global absence from a partial look
  fails it.
- Output one JSON object and nothing else.

Schema
{"coverage": "pass|fail|unknown", "grounding": "...", "scope_honesty": "...",
 "usability": "...", "reason": "one sentence per facet, same order"}
"""


def load_rubric(path: str | Path) -> dict[str, Any]:
    """Load the rubric file; a missing facet list raises rather than guessing."""
    rubric = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(rubric.get("facets"), dict) or not rubric["facets"]:
        raise ValueError(f"{path}: needs a non-empty 'facets' object")

    return rubric


def read_answer(run_id: str) -> str | None:
    """Return the answer text for a run, or ``None`` when it is missing."""
    path = runstore.runs_root() / run_id / "answer.md"

    if not path.is_file():
        return None

    text = path.read_text(encoding="utf-8")

    if SENTINEL in text:
        text = text.rsplit(SENTINEL, 1)[1]

    return text.strip()[:MAX_ANSWER_CHARS]


def user_prompt(question: str, answer: str, rubric: Mapping[str, Any]) -> str:
    """Build the grading request: question, answer, then the facet definitions."""
    facets = "\n".join(f"- {name}: {text}" for name, text in rubric["facets"].items())

    return (
        f"Question\n{question}\n\n"
        f"Answer\n{answer}\n\n"
        f"Facets\n{facets}\n\n"
        "Return the JSON object described in the instructions."
    )


def parse_verdicts(text: str, facets: Sequence[str]) -> dict[str, str]:
    """Return ``{facet: pass|fail|unknown}``; an unparseable reply is all-unknown."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end <= start:
        return {facet: "unknown" for facet in facets}

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {facet: "unknown" for facet in facets}

    verdicts: dict[str, str] = {}

    for facet in facets:
        value = str(payload.get(facet, "unknown")).strip().lower()
        verdicts[facet] = value if value in {"pass", "fail", "unknown"} else "unknown"

    return verdicts


def judge_rows(
    campaign: Sequence[Mapping[str, str]],
    rubric: Mapping[str, Any],
    *,
    client: LensClient,
    answers: Any = read_answer,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Grade every campaign row that has an answer; order is shuffled by ``seed``.

    The arm is dropped before the call: a judge that knows which system produced
    an answer is not measuring the answer.
    """
    facets = list(rubric["facets"])
    rows = [{key: str(value) for key, value in row.items()} for row in campaign]
    random.Random(seed).shuffle(rows)
    results: list[dict[str, Any]] = []

    for row in rows:
        run_id = row.get("run_id") or ""
        answer = answers(run_id)

        if answer is None:
            continue

        trace_ctx = TraceCtx(trace_id="judge", step_idx=len(results), actor="root")
        completion = client.chat(
            "root",
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt(
                        row.get("query") or row.get("query_id") or "", answer, rubric
                    ),
                },
            ],
            trace_ctx=trace_ctx,
            max_tokens=800,
        )
        verdicts = parse_verdicts(completion.text, facets)

        for facet in facets:
            results.append(
                {
                    "run_id": run_id,
                    "query_id": row.get("query_id", ""),
                    "facet": facet,
                    "verdict": verdicts[facet],
                    "raw": completion.text[:400] if verdicts[facet] == "unknown" else "",
                }
            )

    return results


COLUMNS = ("run_id", "query_id", "facet", "verdict", "raw")


def write_rows(rows: Sequence[Mapping[str, Any]], path: str | Path | None) -> str:
    """Write judge.csv and return it as text."""
    buffer: list[str] = [",".join(COLUMNS)]

    for row in rows:
        buffer.append(
            ",".join(
                str(row.get(column, "")).replace(",", ";").replace("\n", " ") for column in COLUMNS
            )
        )

    text = "\n".join(buffer) + "\n"

    if path is not None:
        Path(path).write_text(text, encoding="utf-8")

    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blind LLM judge for one campaign.")
    parser.add_argument("--campaign", required=True, help="campaign.csv from the runner")
    parser.add_argument("--rubric", required=True, help="rubric JSON")
    parser.add_argument("--arms", nargs="*", default=None, help="only these campaign arms")
    parser.add_argument("--seed", type=int, default=7, help="shuffle seed (blinding)")
    parser.add_argument("--out", default=None, help="write judge.csv here")
    parser.add_argument("--dry-run", action="store_true", help="print the rows, spend nothing")
    args = parser.parse_args(argv)

    rubric = load_rubric(args.rubric)

    with Path(args.campaign).open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)]

    if args.arms:
        wanted = set(args.arms)
        rows = [row for row in rows if (row.get("campaign_arm") or "") in wanted]

    if args.dry_run:
        print(f"would judge {len(rows)} answers; facets {sorted(rubric['facets'])}")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    trajectory = Trajectory(run_id=f"judge-{stamp}")
    client = LensClient(trajectory=trajectory)

    judgements = judge_rows(rows, rubric, client=client, seed=args.seed)
    sys.stdout.write(write_rows(judgements, args.out))
    print(f"judge calls logged in {trajectory.run_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
