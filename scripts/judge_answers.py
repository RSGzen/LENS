"""Blind LLM judge for end-to-end answers (one call per answer; writes judge.csv).

The judge is a **measurement instrument, not a component**: it never runs inside a
query, it never sees the arm name, and its calls are logged to their own run
directory so its spend cannot be confused with a query's spend.

Facets are binary and come from a frozen rubric file, because a 1-5 scale agrees
poorly between raters and cannot be checked against a human sample:

    coverage        names the gold items the instance lists
    grounding       every factual claim carries a citation that resolves
    scope_honesty   states what was inspected; no global negative from a local look
    usability       reads as a synthesis a reviewer could act on

An unparseable reply is recorded as ``unknown`` with the raw text as the reason —
never converted to ``fail``, which would turn a schema drift into a quality drop.

The campaign CSV carries a ``query_id`` but **no question text**, so the instrument
file is what supplies both the question and the gold. Without it the judge would
grade an answer against the string ``i5-f1-pope``.

Usage::

    python scripts/judge_answers.py --campaign campaign.csv \
        --instrument experiments/instruments/i5-gold-v1.jsonl \
        --rubric experiments/rubrics/tier3_v2.json
    python scripts/judge_answers.py ... --arms a0 a2 --seed 7
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
# The output budget covers the model's reasoning as well as its text. At 4,000 the
# first full run lost 7 of 36 verdicts to ``finish_reason=length`` with no visible
# text, so the cap is sized to let a long deliberation finish. Only generated
# tokens are billed.
MAX_JUDGE_TOKENS = 16000

SYSTEM_PROMPT = """\
You grade one literature-evidence answer against a fixed rubric.

Rules
- Judge the answer against the question and, where gold items are supplied, against
  those items. Do not judge the search that produced it.
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


def load_instrument(path: str | Path) -> dict[str, dict[str, Any]]:
    """Return ``{instance_id: {"query", "answer", "gold"}}`` for the judge.

    ``answer`` is what coverage is judged against (a count, a fact, or the id set),
    and ``gold`` is the supporting claim list. They are different on purpose: an
    instance can require the count while also listing comparison values that a
    correct answer may legitimately not contain.
    """
    instances: dict[str, dict[str, Any]] = {}

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)
        instance_id = str(row.get("instance_id") or "")

        if not instance_id:
            continue

        gold_block = row.get("gold") or {}
        claims = gold_block.get("claims") or []

        instances[instance_id] = {
            "query": str(row.get("query") or ""),
            "answer": str(gold_block.get("answer") or ""),
            "gold": [str(claim.get("value") or "") for claim in claims if claim.get("value")],
        }

    return instances


def read_answer(run_id: str) -> str | None:
    """Return the answer text for a run, or ``None`` when it is missing."""
    path = runstore.runs_root() / run_id / "answer.md"

    if not path.is_file():
        return None

    text = path.read_text(encoding="utf-8")

    if SENTINEL in text:
        text = text.rsplit(SENTINEL, 1)[1]

    return text.strip()[:MAX_ANSWER_CHARS]


def user_prompt(
    question: str,
    answer: str,
    rubric: Mapping[str, Any],
    gold: Sequence[str] = (),
    required: str = "",
) -> str:
    """Build the grading request: question, gold answer, supporting items, answer, facets."""
    facets = "\n".join(f"- {name}: {text}" for name, text in rubric["facets"].items())
    gold_block = ""

    if required:
        gold_block += f"Gold answer (what coverage must see)\n{required}\n\n"

    if gold:
        gold_block += (
            "Supporting items (the gold claims behind that answer; an answer is not "
            "required to repeat all of them)\n" + "\n".join(f"- {item}" for item in gold) + "\n\n"
        )

    return (
        f"Question\n{question}\n\n"
        f"{gold_block}"
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
    instances: Mapping[str, Mapping[str, Any]] | None = None,
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

        query_id = row.get("query_id", "")
        instance = (instances or {}).get(query_id) or {}
        question = instance.get("query") or row.get("query") or query_id

        trace_ctx = TraceCtx(trace_id="judge", step_idx=len(results), actor="root")
        completion = client.chat(
            "root",
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt(
                        question,
                        answer,
                        rubric,
                        instance.get("gold") or (),
                        instance.get("answer") or "",
                    ),
                },
            ],
            trace_ctx=trace_ctx,
            # The output budget covers the model's reasoning as well as its text, so
            # a cap sized for the verdict alone returns an empty reply with
            # finish_reason=length.
            max_tokens=MAX_JUDGE_TOKENS,
        )
        verdicts = parse_verdicts(completion.text, facets)

        for facet in facets:
            results.append(
                {
                    "run_id": run_id,
                    "query_id": query_id,
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
    parser.add_argument(
        "--instrument",
        default=None,
        help="instrument JSONL: supplies the question text and the gold items",
    )
    parser.add_argument("--arms", nargs="*", default=None, help="only these campaign arms")
    parser.add_argument("--seed", type=int, default=7, help="shuffle seed (blinding)")
    parser.add_argument("--out", default=None, help="write judge.csv here")
    parser.add_argument("--dry-run", action="store_true", help="print the rows, spend nothing")
    args = parser.parse_args(argv)

    rubric = load_rubric(args.rubric)
    instances = load_instrument(args.instrument) if args.instrument else {}

    with Path(args.campaign).open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)]

    if args.arms:
        wanted = set(args.arms)
        rows = [row for row in rows if (row.get("campaign_arm") or "") in wanted]

    if not instances:
        print(
            "warning: no --instrument, so each answer is graded against its query_id",
            file=sys.stderr,
        )

    if args.dry_run:
        print(f"would judge {len(rows)} answers; facets {sorted(rubric['facets'])}")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    trajectory = Trajectory(run_id=f"judge-{stamp}")
    client = LensClient(trajectory=trajectory)

    judgements = judge_rows(rows, rubric, client=client, instances=instances, seed=args.seed)
    sys.stdout.write(write_rows(judgements, args.out))
    print(f"judge calls logged in {trajectory.run_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
