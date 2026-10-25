"""Run a campaign: one ``python -m lens.run`` per (arm, query) (M9 scaffold).

Spec: ``MVP Build Roadmap.md`` M9 item 3 ("experiment-runner scaffolding - arm
config, scoring hooks"; the campaign itself is M11).

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

The only "scoring hook" M9 has is citation resolvability (a run either backs its
citations or it does not); the OOLONG/nDCG metrics attach at M11 against the same
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
from lens.experiments.campaign import build_jobs, load_campaign
from lens.observability.citations import resolve_citations

def image_digest() -> str | None:
    """Return the local sandbox image id, or ``None`` when Docker is unavailable.

    A tag is not reproducible (it is rebuilt in place); the manifest records both
    (E-04). Best-effort: a campaign must run on a machine without Docker only if
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


def run_jobs(jobs: list[dict[str, Any]], *, digest: str | None, dry_run: bool) -> list[dict]:
    """Execute the matrix and return one result row per job.

    A failing run does **not** abort the campaign: the row says ``exit_code`` and
    the next arm still runs. Partial evidence beats a campaign that stops at the
    first bad query - which is exactly what M9 is looking for.
    """
    rows: list[dict[str, Any]] = []

    for job in jobs:
        command = [sys.executable, "-m", "lens.run", job["query"], "--run-id", job["run_id"]]
        env = {**os.environ, **job["env"]}

        if digest:
            env["LENS_SANDBOX_DIGEST"] = digest

        if dry_run:
            print("would run: " + " ".join(f'"{part}"' if " " in part else part for part in command))
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

    return {
        "campaign_arm": job["arm"],
        "query_id": job["query_id"],
        "expect": job["expect"] or "",
        "run_id": job["run_id"],
        "exit_code": exit_code,
        "status": manifest.status if manifest else "missing",
        "iterations": manifest.iterations if manifest else 0,
        "tokens_in": totals.get("tokens_in") or 0,
        "tokens_out": totals.get("tokens_out") or 0,
        "cost_usd": round(cost, 6),
        # R4: the 5.5 % credit-load fee is part of the reported $/query.
        "cost_usd_with_fee": round(cost * 1.055, 6),
        "cited": counts.get("cited") or 0,
        "resolved": counts.get("resolved") or 0,
        "unresolved": counts.get("unresolved") or 0,
        "seconds": round(seconds, 1),
    }


def _write_output(rows: list[dict[str, Any]], campaign: dict[str, Any], out: Path) -> Path:
    """Write ``campaign.json`` + ``campaign.csv``; return the directory."""
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"campaign": campaign["name"], "created_at": stamp, "rows": rows}

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
    parser.add_argument("--out", default=None, help="output directory (default: DATA_ROOT/campaigns)")
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
    print("")

    rows = run_jobs(jobs, digest=digest, dry_run=args.dry_run)

    if args.dry_run:
        return 0

    out = Path(args.out) if args.out else config.DATA_ROOT / "campaigns"
    written = _write_output(
        rows,
        campaign,
        out / f"{campaign['name']}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
    )

    spent = sum(float(row["cost_usd"]) for row in rows)
    print("")
    print(f"wrote    {written}")
    print(f"spent    ${spent:.4f} raw (${spent * 1.055:.4f} with the 5.5% credit fee)")

    return 0 if all(row["exit_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
