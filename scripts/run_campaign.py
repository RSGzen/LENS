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


def run_jobs(jobs: list[dict[str, Any]], *, digest: str | None, dry_run: bool) -> list[dict]:
    """Execute the matrix and return one result row per job.

    A failing run does **not** abort the campaign: the row says ``exit_code`` and
    the next arm still runs. Partial evidence beats a campaign that stops at the
    first bad query.
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

        started = datetime.now(timezone.utc)
        finished = subprocess.run(command, cwd=config.REPO_ROOT, env=env, check=False)
        seconds = (datetime.now(timezone.utc) - started).total_seconds()

        rows.append(_row(job, finished.returncode, seconds))

    return rows


def _row(job: dict[str, Any], exit_code: int, seconds: float) -> dict[str, Any]:
    """Build one CSV row from the run's own artifacts (never from stdout)."""
    manifest = runstore.read_manifest(runstore.runs_root() / job["run_id"])
    totals = (manifest.totals if manifest else None) or {}
    counts = resolve_citations(job["run_id"]).counts if manifest else {}
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
        "cited": counts.get("cited") or 0,
        "resolved": counts.get("resolved") or 0,
        "unresolved": counts.get("unresolved") or 0,
        "seconds": round(seconds, 1),
    }


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
    """Write ``campaign.json`` + ``campaign.csv``; return the directory.

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

    if rows:
        with (out / "campaign.csv").open("w", newline="", encoding="utf-8") as handle:
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
    args = parser.parse_args(argv)

    # A campaign prints the queries it is about to run; the console is cp1252.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    campaign = load_campaign(args.campaign)
    jobs = build_jobs(campaign)
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

    rows = run_jobs(jobs, digest=digest, dry_run=args.dry_run)

    if args.dry_run:
        return 0

    out = Path(args.out) if args.out else config.DATA_ROOT / "campaigns"
    written = _write_output(
        rows,
        campaign,
        out / f"{campaign['name']}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
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
