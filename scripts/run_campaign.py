"""Run a campaign: one ``python -m lens.run`` per (arm, query).

Why subprocesses, not an in-process loop
---------------------------------------
``lens.config`` is import-time constants and ``SandboxSession`` copies some of them
into the container's environment. An arm applied in-process would change the
Python side and leave the container and the ``Budget`` object on the old values -
two arms producing identical numbers for no visible reason. One process per cell
makes the arm *the* environment, and every run keeps its own ``manifest.json``.

Usage::

    python scripts/run_campaign.py experiments/campaigns/m9-smoke.json --dry-run
    python scripts/run_campaign.py experiments/campaigns/m9-smoke.json

``--dry-run`` prints the launch matrix and spends nothing, so an arm config is
validated before it costs money. Output: ``<DATA_ROOT>/campaigns/<name>-<ts>/``
with ``campaign.json`` + ``campaign.csv``.

A campaign **refuses to run without an image digest**: a tag is rebuilt in place,
so two waves of the same tag can be two different systems, and a campaign file that
cannot say which image produced its rows is not a measurement. ``--allow-missing-digest``
is the escape hatch for a machine without Docker, and the override is written into
``campaign.json`` so a reader sees it.

The only "scoring hook" has is citation resolvability (a run either backs its
citations or it does not); the OOLONG/nDCG metrics attach at against the same
CSV rather than through a plug-in registry with one plug-in.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lens import config
from lens.agent import runstore
from lens.clients import LensClient
from lens.experiments.campaign import build_jobs, load_campaign
from lens.experiments.polarity import PolarityResult, probe_polarity
from lens.observability.citations import resolve_citations
from lens.observability.replay import replay_summary, replay_turns
from lens.observability.trajectory import TraceCtx, Trajectory


def image_digest() -> str | None:
    """Return the local sandbox image id, or ``None`` when Docker is unavailable.

    A tag is not reproducible (it is rebuilt in place); the manifest records both
    . Best-effort: a campaign must run on a machine without Docker only if
    the runs themselves do not need it, so this never raises.
    """
    try:
        finished = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", config.SANDBOX_IMAGE],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if finished.returncode != 0:
        return None

    return finished.stdout.strip() or None


def check_polarity() -> PolarityResult:
    """Run the four JEV probes in their own run so the spend is auditable.

    The probe needs a client and a trajectory, which is why it lives here rather
    than in ``experiments.polarity``: that module stays network-free and testable,
    and this function is the only place that knows how to build the real objects.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"polarity-{stamp}"
    trajectory = Trajectory(run_id)
    client = LensClient(trajectory)
    trace = TraceCtx(trace_id=run_id, step_idx=0, actor="tool")

    return probe_polarity(lambda state, questions: client.jev(state, questions, trace_ctx=trace))


def run_jobs(
    jobs: list[dict[str, Any]],
    *,
    digest: str | None,
    dry_run: bool,
    csv_path: Path | None = None,
) -> list[dict]:
    """Execute the matrix and return one result row per job.

    A failing run does **not** abort the campaign: the row says ``exit_code`` and
    the next arm still runs. Partial evidence beats a campaign that stops at the
    first bad query. Each row is appended to ``csv_path`` as it finishes, so a
    wave that stops mid-way keeps the runs it already paid for.
    """
    rows: list[dict[str, Any]] = []

    for job in jobs:
        command = [sys.executable, "-m", "lens.run", job["query"], "--run-id", job["run_id"]]
        env = {**os.environ, **job["env"]}

        if digest:
            env["LENS_SANDBOX_DIGEST"] = digest

        if dry_run:
            print(
                "would run: " + " ".join(f'"{part}"' if " " in part else part for part in command)
            )
            print(f"  env: {job['env'] or '(inherited)'}")
            continue

        started_at = datetime.now(timezone.utc)
        finished = subprocess.run(command, cwd=config.REPO_ROOT, env=env, check=False)
        finished_at = datetime.now(timezone.utc)
        seconds = (finished_at - started_at).total_seconds()
        row = _row(
            job,
            finished.returncode,
            seconds,
            started_at=started_at,
            finished_at=finished_at,
        )
        rows.append(row)

        if csv_path is not None:
            _append_row(csv_path, row)

    return rows


def _claims_columns(run_dir: Path) -> tuple[int, float]:
    """Return ``(units, coverage)`` from ``claims.json`` (zeros when absent)."""
    path = run_dir / "claims.json"

    if not path.is_file():
        return 0, 0.0

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0, 0.0

    return int(payload.get("units") or 0), round(float(payload.get("coverage") or 0.0), 4)


def _task_counts(run_dir: Path) -> dict[str, int]:
    """Return task counts by status from ``tasks.json`` (empty when absent)."""
    path = run_dir / "tasks.json"

    if not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    tasks = payload.get("tasks") if isinstance(payload, dict) else payload

    if isinstance(tasks, dict):
        tasks = list(tasks.values())

    counts: dict[str, int] = {}

    for task in tasks or []:
        if isinstance(task, dict):
            status = str(task.get("status") or "pending")
            counts[status] = counts.get(status, 0) + 1

    return counts


def _gate_columns(trajectory_path: Path) -> dict[str, Any]:
    """Return the last retrieval and final gate verdicts recorded in a run.

    A rejected answer can be revised, so the final gate may appear more than once;
    the last verdict is the one the run accepted or stopped on.
    """
    columns: dict[str, Any] = {
        "retrieval_gate": "",
        "final_gate": "",
        "final_gate_p": "",
        "final_gate_decision": "",
    }

    if not trajectory_path.is_file():
        return columns

    for line in trajectory_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue

        if not isinstance(event, dict) or event.get("action") != "gate.verdict":
            continue

        stage = event.get("stage")

        if stage == "retrieval":
            columns["retrieval_gate"] = str(event.get("verdict") or "")
        elif stage == "final":
            confidence = event.get("confidence")
            columns["final_gate"] = str(event.get("verdict") or "")
            columns["final_gate_p"] = round(float(confidence), 4) if confidence is not None else ""
            columns["final_gate_decision"] = str((event.get("extra") or {}).get("decision") or "")

    return columns


def _row(
    job: dict[str, Any],
    exit_code: int,
    seconds: float,
    *,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> dict[str, Any]:
    """Build one CSV row from the run's own artifacts (never from stdout)."""
    run_dir = runstore.runs_root() / job["run_id"]
    manifest = runstore.read_manifest(run_dir)
    totals = (manifest.totals if manifest else None) or {}
    report = resolve_citations(job["run_id"]) if manifest else None
    counts = report.counts if report else {}
    turns = []
    trajectory_path = run_dir / "trajectory.jsonl"

    if manifest is not None and trajectory_path.is_file():
        turns = replay_turns(Trajectory(job["run_id"]))

    work = replay_summary(turns)
    claims_units, claims_coverage = _claims_columns(run_dir)
    tasks = _task_counts(run_dir)
    gates = _gate_columns(trajectory_path)
    cost = float(totals.get("cost_usd") or 0.0)
    tokens_in = int(totals.get("tokens_in") or 0)
    tokens_out = int(totals.get("tokens_out") or 0)
    # Both are **subsets** of the token counts, so they never change a cap; they
    # are recorded because a cost comparison between arms with different cache
    # behaviour or reasoning effort is not like-for-like without them.
    cached = int(totals.get("cached_tokens") or 0)
    reasoning = int(totals.get("reasoning_tokens") or 0)

    return {
        "campaign_arm": job["arm"],
        "query_id": job["query_id"],
        "expect": job["expect"] or "",
        "run_id": job["run_id"],
        "started_at": started_at.isoformat(timespec="seconds") if started_at else "",
        "finished_at": finished_at.isoformat(timespec="seconds") if finished_at else "",
        "exit_code": exit_code,
        "status": manifest.status if manifest else "missing",
        "iterations": manifest.iterations if manifest else 0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
        "cached_share": round(cached / tokens_in, 4) if tokens_in else 0.0,
        "reasoning_share": round(reasoning / tokens_out, 4) if tokens_out else 0.0,
        "stop_reason": (manifest.stop_reason if manifest else "") or "",
        "cost_usd": round(cost, 6),
        # The credit-load fee is part of the reported $/query.
        "cost_usd_with_fee": round(cost * config.CREDIT_FEE, 6),
        # Work counters. A sub-task is a chat call and a code block is an exec;
        # neither is a host-tool dispatch, and the tool column alone made a
        # 35-chunk, 9-sub-task run look idle.
        "host_tools": int(work.get("tool_calls") or 0),
        "sub_tasks": int(work.get("sub_tasks") or 0),
        "sub_calls": int(work.get("sub_calls") or 0),
        "code_blocks": int(work.get("code_blocks") or 0),
        "cited": counts.get("cited") or 0,
        "resolved": counts.get("resolved") or 0,
        "merged": counts.get("merged") or 0,
        "unresolved": counts.get("unresolved") or 0,
        "retrieval_gate": gates["retrieval_gate"],
        "final_gate": gates["final_gate"],
        "final_gate_p": gates["final_gate_p"],
        "final_gate_decision": gates["final_gate_decision"],
        "claims_units": claims_units,
        "claims_coverage": claims_coverage,
        "tasks_done": tasks.get("done", 0),
        "tasks_failed": tasks.get("failed", 0),
        "tasks_open": tasks.get("pending", 0) + tasks.get("in_progress", 0),
        "seconds": round(seconds, 1),
    }


def _append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one row, writing the header on first use.

    The CSV is where a wave's per-run numbers are aggregated. Writing it once at
    the end loses the summary when a long wave stops mid-way, and the run
    directories alone do not say what the matrix was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))

        if new:
            writer.writeheader()

        writer.writerow(row)


def _terminal(run_id: str) -> bool:
    """Return True when a previous attempt already wrote a final manifest."""
    try:
        manifest = runstore.read_manifest(runstore.runs_root() / run_id)
    except (OSError, ValueError):
        return False

    return manifest is not None and manifest.status in {"completed", "partial", "failed"}


def stop_reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return how many rows ended for each reason; ``unrecorded`` when absent.

    The cap-binding rate is the difference between "the model finished" and "a
    number we picked ended it", and it is the one property of a wave that cannot
    be read from the cost column. Counting it here means every campaign reports it
    without anyone remembering to.
    """
    counts: dict[str, int] = {}

    for row in rows:
        key = str(row.get("stop_reason") or "") or "unrecorded"
        counts[key] = counts.get(key, 0) + 1

    return counts


def _write_output(
    rows: list[dict[str, Any]],
    campaign: dict[str, Any],
    out: Path,
    *,
    digest: str | None,
    polarity: dict[str, Any] | None = None,
) -> Path:
    """Write ``campaign.json`` and, when not already appended, ``campaign.csv``.

    The image identity and the polarity probes live in the JSON header, not in
    every row: one campaign ran on one image, and a per-row copy invites an edit
    by hand. ``polarity: null`` means the preflight was skipped or refused to run.
    """
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "campaign": campaign["name"],
        "created_at": stamp,
        "image": {
            "tag": config.SANDBOX_IMAGE,
            "digest": digest,
            "digest_missing": digest is None,
        },
        "polarity": polarity,
        "stop_reasons": stop_reason_counts(rows),
        "rows": rows,
    }

    (out / "campaign.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = out / "campaign.csv"

    if rows and not csv_path.is_file():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    return out


def main(argv: list[str] | None = None) -> int:
    """Parse args, build the matrix, then run it (or print it)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign", help="path to a campaign JSON file")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix; spend nothing")
    parser.add_argument(
        "--allow-missing-digest",
        action="store_true",
        help="run without an image digest (a machine without Docker); recorded in campaign.json",
    )
    parser.add_argument(
        "--skip-polarity",
        action="store_true",
        help="do not run the four JEV polarity probes (skips a ~$0.00002 preflight)",
    )
    parser.add_argument(
        "--out", default=None, help="output directory (default: DATA_ROOT/campaigns)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a cell whose run directory already has a final manifest",
    )
    args = parser.parse_args(argv)

    # A campaign prints the queries it is about to run; the console is cp1252.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    campaign = load_campaign(args.campaign)
    jobs = build_jobs(campaign)

    if args.skip_existing:
        keep = []
        skipped = 0

        for job in jobs:
            if _terminal(job["run_id"]):
                skipped += 1
            else:
                keep.append(job)

        jobs = keep
        print(f"skipping {skipped} run(s) with a final manifest")

    digest = image_digest()

    print(f"campaign {campaign['name']}: {len(jobs)} run(s), {len(campaign['arms'])} arm(s)")
    print(f"image    {config.SANDBOX_IMAGE} digest={digest}")

    if digest is None and not args.dry_run and not args.allow_missing_digest:
        # A tag is rebuilt in place. Without the digest the rows cannot name the
        # system that produced them, so the wave is not a measurement.
        print(
            "\nrefusing to run: no image digest for "
            f"{config.SANDBOX_IMAGE} (is Docker running?)\n"
            "pass --allow-missing-digest to run anyway; the override is recorded",
            file=sys.stderr,
        )
        return 2

    print("")

    # Polarity before spend: if the decision layer no longer reads `noul` at 0.5,
    # every gate verdict this wave records is noise, and the probes cost $0.00002.
    polarity: PolarityResult | None = None

    if not args.dry_run and not args.skip_polarity:
        polarity = check_polarity()
        print(f"polarity {polarity.to_dict()['skewed'] or 'ok'} (4 probes)")

        if not polarity.ok:
            print(
                "\nrefusing to run: the decision layer did not read `noul` at 0.5 on "
                "probe(s) " + ", ".join(polarity.skewed) + "\n"
                "the gate's thresholds assume that mapping; check the endpoint's answer "
                "shape, or pass --skip-polarity and record why",
                file=sys.stderr,
            )
            return 3

    out_root = Path(args.out) if args.out else config.DATA_ROOT / "campaigns"
    out_dir = out_root / (
        f"{campaign['name']}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )

    rows = run_jobs(
        jobs,
        digest=digest,
        dry_run=args.dry_run,
        csv_path=None if args.dry_run else out_dir / "campaign.csv",
    )

    if args.dry_run:
        return 0

    written = _write_output(
        rows,
        campaign,
        out_dir,
        digest=digest,
        polarity=polarity.to_dict() if polarity is not None else None,
    )

    spent = sum(float(row["cost_usd"]) for row in rows)
    print("")
    print(f"wrote    {written}")
    print(
        f"spent    ${spent:.4f} raw "
        f"(${spent * config.CREDIT_FEE:.4f} with the {config.CREDIT_FEE - 1:.1%} credit fee)"
    )
    reasons = stop_reason_counts(rows)
    print("stopped  " + ", ".join(f"{name}={count}" for name, count in sorted(reasons.items())))

    return 0 if all(row["exit_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
