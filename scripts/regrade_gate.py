"""Re-run the current final gate over recorded answers (no chat calls, JEV only).

Why this exists: changing how the gate is *asked* changes its verdicts, and the
cheap way to see that is to put the new question to answers that already exist.
Every answer, its citations and its chunk text are on disk or in the database, so a
regrade costs the decision layer only — about $0.0002 an answer — and no query is
re-run.

The recorded verdict is read from the run's own trajectory, so each row shows
*old → new* side by side. Nothing in the run directory is modified: the regrade
calls are logged to their own ``regrade-<timestamp>`` run, which keeps the probe's
spend and prompts separate from the run it is judging.

Usage::

    python scripts/regrade_gate.py --campaign <campaign.csv>
    python scripts/regrade_gate.py --runs run-a run-b --out regrade.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from lens.agent import runstore
from lens.clients import LensClient
from lens.db.connect import connect
from lens.observability.citations import cited_ids
from lens.observability.trajectory import TraceCtx, Trajectory
from lens.tools.evaluate import evaluate_stage
from lens.tools.retrieval import fetch_chunks

SENTINEL = "FINAL_ANSWER:"
EVIDENCE_CHUNK_CHARS = 2000


def read_answer(run_id: str) -> str:
    """Return the answer text of a run, or an empty string when there is none."""
    path = runstore.runs_root() / run_id / "answer.md"

    if not path.is_file():
        return ""

    text = path.read_text(encoding="utf-8")

    return text.rsplit(SENTINEL, 1)[1].strip() if SENTINEL in text else text.strip()


def recorded_final_verdict(run_id: str) -> tuple[str | None, float | None]:
    """Return the last final-gate verdict a run recorded, and its probability."""
    path = runstore.runs_root() / run_id / "trajectory.jsonl"

    if not path.is_file():
        return None, None

    verdict: str | None = None
    confidence: float | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("action") == "gate.verdict" and event.get("stage") == "final":
            verdict = event.get("verdict")
            confidence = event.get("confidence")

    return verdict, confidence


def evidence_for(conn: Any, citations: Sequence[str]) -> dict[str, str]:
    """Return ``chunk_uuid -> text`` for the cited chunks (full text, bounded)."""
    ids = [citation for citation in citations if "-" in citation][:64]

    if not ids:
        return {}

    rows = fetch_chunks(conn, ids)

    return {
        str(row["chunk_uuid"]): str(row.get("chunk_text") or "")[:EVIDENCE_CHUNK_CHARS]
        for row in rows
        if isinstance(row, dict) and row.get("chunk_uuid")
    }


def regrade_run(
    run_id: str,
    *,
    client: LensClient,
    conn: Any,
    arm: str = "",
    query_id: str = "",
) -> dict[str, Any]:
    """Score one recorded answer with the current gate and report old vs new."""
    answer = read_answer(run_id)
    citations = cited_ids(answer)
    old_verdict, old_confidence = recorded_final_verdict(run_id)

    if not answer.strip():
        return {
            "run_id": run_id,
            "arm": arm,
            "query_id": query_id,
            "old": old_verdict or "",
            "new": "skipped",
            "reason": "no answer.md",
        }

    verdict = evaluate_stage(
        "final",
        {
            "score": None,
            "summary": {
                "answer": answer,
                "citations": citations,
                "evidence": evidence_for(conn, citations),
                "evidence_truncated": False,
            },
        },
        client=client,
        trace_ctx=TraceCtx(trace_id="regrade", step_idx=0, actor="root"),
    )

    return {
        "run_id": run_id,
        "arm": arm,
        "query_id": query_id,
        "old": old_verdict or "",
        "old_confidence": "" if old_confidence is None else round(float(old_confidence), 3),
        "new": "none" if verdict is None else verdict.verdict,
        "new_confidence": ""
        if verdict is None or verdict.confidence is None
        else round(float(verdict.confidence), 3),
        "criterion": "" if verdict is None else (verdict.failing_criterion or "")[:200],
    }


COLUMNS = (
    "run_id",
    "arm",
    "query_id",
    "old",
    "old_confidence",
    "new",
    "new_confidence",
    "criterion",
    "reason",
)


def write_rows(rows: Sequence[dict[str, Any]], path: str | Path | None) -> str:
    """Write the comparison CSV and return it as text."""
    buffer: list[str] = [",".join(COLUMNS)]

    for row in rows:
        buffer.append(",".join(str(row.get(column, "")).replace(",", ";") for column in COLUMNS))

    text = "\n".join(buffer) + "\n"

    if path is not None:
        Path(path).write_text(text, encoding="utf-8")

    return text


def campaign_rows(path: str | Path) -> list[dict[str, str]]:
    """Return ``run_id / campaign_arm / query_id`` from a campaign CSV."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [
            {
                "run_id": row.get("run_id") or "",
                "arm": row.get("campaign_arm") or "",
                "query_id": row.get("query_id") or "",
            }
            for row in csv.DictReader(handle)
            if row.get("run_id")
        ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-score recorded answers with the current gate.")
    parser.add_argument("--campaign", action="append", default=[], help="campaign.csv (repeatable)")
    parser.add_argument("--runs", nargs="*", default=[], help="explicit run ids")
    parser.add_argument("--out", default=None, help="write the comparison CSV here")
    parser.add_argument("--limit", type=int, default=0, help="score at most this many runs")
    parser.add_argument("--dry-run", action="store_true", help="list what would be scored")
    args = parser.parse_args(argv)

    targets: list[dict[str, str]] = []

    for path in args.campaign:
        targets.extend(campaign_rows(path))

    targets.extend({"run_id": run_id, "arm": "", "query_id": ""} for run_id in args.runs)

    if args.limit:
        targets = targets[: args.limit]

    if not targets:
        print("nothing to regrade: pass --campaign or --runs", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"would regrade {len(targets)} answers (JEV only)")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    trajectory = Trajectory(run_id=f"regrade-{stamp}")
    client = LensClient(trajectory=trajectory)
    rows: list[dict[str, Any]] = []

    with connect() as conn:
        for target in targets:
            rows.append(
                regrade_run(
                    target["run_id"],
                    client=client,
                    conn=conn,
                    arm=target.get("arm", ""),
                    query_id=target.get("query_id", ""),
                )
            )
            print(f"  {target['run_id']}: {rows[-1].get('old')} -> {rows[-1].get('new')}")

    sys.stdout.write(write_rows(rows, args.out))
    print(f"regrade calls logged in {trajectory.run_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
