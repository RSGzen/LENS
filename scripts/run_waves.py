"""Run the remaining campaign waves in order, with a timestamped, resumable log.

One wave is one ``run_campaign.py`` invocation, so every cell keeps its own
manifest, CSV and image identity. This script adds what a single campaign cannot
know: the order, the expected size of each cell, the cumulative spend against a
checkpoint, and a record of when each cell started and ended.

A cell is a campaign file plus the number of runs it must expand to. The expected
count is checked before anything is spent, because the wave files are generated
from instruments and a stale or placeholder file otherwise runs happily and
measures nothing.

Usage::

    python scripts/run_waves.py experiments/campaigns/m11-waves.json --dry-run
    python scripts/run_waves.py experiments/campaigns/m11-waves.json --prior-spend 22.0
    python scripts/run_waves.py ... --through w3-bridge-subset

The plan stops at a ``stop_after`` cell unless it is the one named by ``--through``:
the staged-spend rule is a decision point, not a batch boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lens.config as config
from lens.experiments.campaign import build_jobs, load_campaign

#: Fields a cell must carry. ``checkpoint_usd`` and ``stop_after`` are optional.
REQUIRED = ("id", "campaign", "expect")

#: Free space the next wave needs. A run directory is a few megabytes; the blob
#: store is shared and deduplicated, so this is a floor, not an estimate.
MIN_FREE_BYTES = 5 * 1024**3


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate a wave plan; raise ``ValueError`` on a bad shape."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("plan must be a JSON object")

    if not payload.get("name") or not isinstance(payload.get("waves"), list):
        raise ValueError("plan needs a name and a waves list")

    if not payload["waves"]:
        raise ValueError("plan has no waves")

    seen: set[str] = set()

    for cell in payload["waves"]:
        if not isinstance(cell, dict):
            raise ValueError("a wave must be an object")

        missing = [key for key in REQUIRED if key not in cell]
        if missing:
            raise ValueError(f"wave is missing {missing}")

        ident = cell["id"]
        if not isinstance(ident, str) or not ident:
            raise ValueError(f"bad wave id {ident!r}")
        if ident in seen:
            raise ValueError(f"duplicate wave id {ident!r}")
        seen.add(ident)

        if int(cell["expect"]) <= 0:
            raise ValueError(f"wave {ident!r} needs a positive expect")

    return payload


def cell_status(cell: dict[str, Any]) -> dict[str, Any]:
    """Return ``{present, runs, expected, ok, reason}`` for one cell, spending nothing."""
    path = config.REPO_ROOT / str(cell["campaign"])

    if not path.is_file():
        return {
            "present": False,
            "runs": 0,
            "expected": int(cell["expect"]),
            "ok": False,
            "reason": "campaign file missing",
        }

    try:
        jobs = build_jobs(load_campaign(path))
    except (OSError, ValueError) as exc:
        return {
            "present": True,
            "runs": 0,
            "expected": int(cell["expect"]),
            "ok": False,
            "reason": f"invalid campaign: {exc}",
        }

    expected = int(cell["expect"])

    return {
        "present": True,
        "runs": len(jobs),
        "expected": expected,
        "ok": len(jobs) == expected,
        "reason": "" if len(jobs) == expected else f"expands to {len(jobs)}, expected {expected}",
    }


def recorded_arms(campaign: dict[str, Any]) -> list[str]:
    """Return the arms that do not set ``LENS_RECORDED=1``."""
    return [
        str(arm.get("name"))
        for arm in campaign.get("arms") or []
        if str((arm.get("env") or {}).get("LENS_RECORDED", "")) != "1"
    ]


def summarise_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Return the counts a cell's log line reports, from its own CSV."""
    statuses: dict[str, int] = {}
    cost = 0.0

    for row in rows:
        status = str(row.get("status") or "missing")
        statuses[status] = statuses.get(status, 0) + 1
        cost += float(row.get("cost_usd") or 0.0)

    failed = statuses.get("failed", 0) + statuses.get("missing", 0)

    return {
        "ran": len(rows),
        "ok": statuses.get("completed", 0),
        "partial": statuses.get("partial", 0),
        "failed": failed,
        "cost": cost,
        "statuses": statuses,
    }


def checkpoint_decision(cumulative: float, limit: float | None) -> str:
    """Return ``stop`` when the staged-spend limit is crossed, else ``continue``."""
    if limit is not None and cumulative > limit:
        return "stop"

    return "continue"


def _newest_campaign_dir(root: Path) -> Path | None:
    if not root.is_dir():
        return None

    dirs = [path for path in root.iterdir() if path.is_dir()]

    return max(dirs, key=lambda path: path.name) if dirs else None


def _read_rows(root: Path) -> list[dict[str, str]]:
    """Read the CSV of the newest campaign under one cell's output directory."""
    campaign_dir = _newest_campaign_dir(root)

    if campaign_dir is None:
        return []

    path = campaign_dir / "campaign.csv"

    if not path.is_file():
        return []

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cell_existing_cost(cell: dict[str, Any]) -> float:
    """Return what the cell's already-finished runs cost, so a resume is not free.

    ``run_campaign --skip-existing`` drops a finished cell from the new CSV, so the
    rows alone would under-report the campaign's spend after a crash.
    """
    from lens.agent import runstore

    try:
        campaign = load_campaign(config.REPO_ROOT / str(cell["campaign"]))
    except (OSError, ValueError):
        return 0.0

    total = 0.0

    for job in build_jobs(campaign):
        try:
            manifest = runstore.read_manifest(runstore.runs_root() / job["run_id"])
        except (OSError, ValueError):
            continue

        if manifest is not None and manifest.status in {"completed", "partial", "failed"}:
            total += float(((manifest.totals or {}).get("cost_usd")) or 0.0)

    return total


def _log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

    print(line, flush=True)


def _freeze_commit() -> str:
    try:
        payload = json.loads((config.REPO_ROOT / "freeze.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "missing"

    return str(payload.get("commit") or "missing")[:7]


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _image_digest() -> str | None:
    finished = _docker("image", "inspect", "--format", "{{.Id}}", config.SANDBOX_IMAGE)

    return finished.stdout.strip() if finished.returncode == 0 else None


def _running_containers() -> list[str]:
    finished = _docker("ps", "--format", "{{.Names}}")

    if finished.returncode != 0:
        return []

    names = [line.strip() for line in finished.stdout.splitlines() if line.strip()]

    return [name for name in names if config.SANDBOX_SERVICE in name]


def _key_remaining() -> float | None:
    """Return the API key's remaining spend limit, or ``None`` when unknown."""
    try:
        import requests

        response = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {config.require_api_key()}"},
            timeout=20,
        )
        response.raise_for_status()
        data = (response.json() or {}).get("data") or {}
        value = data.get("limit_remaining")

        return float(value) if value is not None else None
    except Exception:  # noqa: BLE001 - a key probe must not block a valid run
        return None


def preflight(plan: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Return the reasons a wave must not start, plus the facts the log records."""
    problems: list[str] = []

    freeze = subprocess.run(
        [sys.executable, "scripts/freeze_manifest.py", "--check"],
        cwd=config.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    if freeze.returncode != 0:
        problems.append("freeze check failed (dirty tree or a different commit)")

    digest = _image_digest()

    if digest is None:
        problems.append(f"no image digest for {config.SANDBOX_IMAGE} (is Docker running?)")

    running = _running_containers()

    if running:
        problems.append(f"sandbox container already running: {', '.join(running)}")

    free = shutil.disk_usage(config.DATA_ROOT).free

    if free < MIN_FREE_BYTES:
        problems.append(f"only {free / 1024**3:.1f} GB free under {config.DATA_ROOT}")

    estimate = float(plan.get("spend_estimate_usd") or 0.0)
    remaining = _key_remaining()

    if estimate and remaining is not None and remaining < estimate:
        problems.append(
            f"API key has ${remaining:.2f} left, below the ${estimate:.2f} estimate "
            "for the remaining waves"
        )

    facts = {
        "freeze": freeze.returncode == 0,
        "digest": digest,
        "running": running,
        "key_remaining": remaining,
        "free_gb": round(free / 1024**3, 1),
    }

    return problems, facts


def _run_cell(cell: dict[str, Any], out_root: Path) -> tuple[int, float]:
    """Run one campaign and return ``(exit_code, wall_seconds)``."""
    command = [
        sys.executable,
        "scripts/run_campaign.py",
        str(cell["campaign"]),
        "--out",
        str(out_root),
        "--skip-existing",
    ]
    started = datetime.now(timezone.utc)
    finished = subprocess.run(command, cwd=config.REPO_ROOT, check=False)

    return finished.returncode, (datetime.now(timezone.utc) - started).total_seconds()


def _print_table(plan: dict[str, Any]) -> bool:
    print(f"plan {plan['name']}: {len(plan['waves'])} cell(s)")
    all_ok = True

    for cell in plan["waves"]:
        status = cell_status(cell)
        marker = "ok" if status["ok"] else "FAIL"
        print(
            f"  {marker:>4}  {cell['id']:<20} expect={status['expected']:<4} "
            f"runs={status['runs']:<4} {status['reason']}"
        )
        all_ok = all_ok and status["ok"]

    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan", help="path to a wave plan JSON file")
    parser.add_argument("--dry-run", action="store_true", help="validate the plan; spend nothing")
    parser.add_argument("--through", default=None, help="stop after this cell id")
    parser.add_argument("--prior-spend", type=float, default=0.0, help="spend already made")
    parser.add_argument(
        "--continue-past-checkpoint",
        action="store_true",
        help="keep going after a staged-spend limit (record why)",
    )
    parser.add_argument("--out", default=None, help="root for the wave directories")
    args = parser.parse_args(argv)

    plan = load_plan(args.plan)

    if args.through and args.through not in {cell["id"] for cell in plan["waves"]}:
        print(f"unknown cell {args.through!r}", file=sys.stderr)
        return 2

    plan_ok = _print_table(plan)

    if args.dry_run:
        return 0 if plan_ok else 1

    if not plan_ok:
        print("refusing to run: the plan does not expand to its expected counts", file=sys.stderr)
        return 2

    problems, facts = preflight(plan)

    if problems:
        for problem in problems:
            print(f"refusing to run: {problem}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out) if args.out else config.DATA_ROOT / "campaigns"
    wave_dir = out_root / f"waves-{stamp}"
    log_path = wave_dir / "waves.log"
    freeze = _freeze_commit()
    cumulative = float(args.prior_spend)
    expected = sum(int(cell["expect"]) for cell in plan["waves"])

    key = facts["key_remaining"]
    key_text = f"${key:.2f}" if isinstance(key, float) else "unknown"

    _log(
        log_path,
        f"{_now()} plan_start name={plan['name']} cells={len(plan['waves'])} "
        f"expect={expected} prior=${cumulative:.2f} freeze={freeze} "
        f"digest={facts['digest']} key={key_text} disk={facts['free_gb']}GB "
        f"through={args.through or '(all)'}",
    )

    consecutive_failures = 0

    for cell in plan["waves"]:
        ident = cell["id"]
        status = cell_status(cell)

        if not status["ok"]:
            _log(log_path, f"{_now()} abort id={ident} reason={status['reason']!r}")
            return 2

        _log(
            log_path,
            f"{_now()} cell_start id={ident} campaign={cell['campaign']} "
            f"expect={status['expected']} freeze={freeze}",
        )
        # Measured before the cell runs: after it, every row it just wrote looks
        # like a finished run and the cell would be counted twice.
        prior_cost = cell_existing_cost(cell)
        code, wall = _run_cell(cell, wave_dir / ident)
        rows = _read_rows(wave_dir / ident)
        summary = summarise_rows(rows)
        cumulative += prior_cost + summary["cost"]
        _log(
            log_path,
            f"{_now()} cell_end id={ident} exit={code} ran={summary['ran']} "
            f"ok={summary['ok']} partial={summary['partial']} failed={summary['failed']} "
            f"cost=${summary['cost']:.4f} prior=${prior_cost:.4f} wall={wall:.0f}s "
            f"cumulative=${cumulative:.2f}",
        )

        consecutive_failures = consecutive_failures + 1 if code != 0 else 0

        if consecutive_failures >= 2:
            _log(log_path, f"{_now()} abort reason='two consecutive cells failed'")
            return 4

        limit = cell.get("checkpoint_usd")

        if limit is not None:
            decision = checkpoint_decision(cumulative, float(limit))

            if decision == "stop" and not args.continue_past_checkpoint:
                _log(
                    log_path,
                    f"{_now()} checkpoint id={ident} limit=${float(limit):.2f} "
                    f"cumulative=${cumulative:.2f} decision=stop",
                )
                return 3

        if cell.get("stop_after") and ident != args.through:
            _log(
                log_path,
                f"{_now()} checkpoint id={ident} cumulative=${cumulative:.2f} decision=pause",
            )
            return 0

        if ident == args.through:
            break

    _log(log_path, f"{_now()} plan_end cumulative=${cumulative:.2f} log={log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
