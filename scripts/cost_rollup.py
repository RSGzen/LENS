#!/usr/bin/env python
"""Add up what a campaign actually cost: the cells, the judge, and any regrade.

Why this exists
---------------
A campaign's true cost is not the number printed at the end of
``scripts/run_campaign.py``. That number is the CSV sum, and it excludes two things
that spend real money after the cells finish:

1. the **judge** (``scripts/judge_answers.py``), one call per answer, logged to its
   own ``judge-<timestamp>`` run;
2. a **regrade** (``scripts/regrade_gate.py``), decision-layer only, logged to
   ``regrade-<timestamp>``.

All three are real spend and each lands in a different artifact, so a table that
quotes only one of them is optimistic by whatever the other two cost.

Usage::

    python scripts/cost_rollup.py <campaign dir or campaign.json>
    python scripts/cost_rollup.py <campaign> --runs judge-20261031T120000 regrade-...
    python scripts/cost_rollup.py <campaign> --no-discover      # cells only

Discovery is by **run-id prefix and age**: a ``judge-*``/``regrade-*`` run whose log
is at least as new as the campaign is counted, because that is when such a run can
belong to the wave. Older ones are someone else's spend. ``--runs`` adds ids the
prefix rule cannot see, and ``--no-discover`` turns the heuristic off.

Output: one markdown table to stdout, and ``cost_rollup.md`` beside the campaign when
``--out`` is given. Nothing is written into a run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from lens import config
from lens.agent import runstore
from lens.observability.trajectory import Trajectory

#: Run-id prefixes that spend money *after* the cells, in the order they run.
COMPANION_PREFIXES = ("judge-", "regrade-")


def load_campaign(path: str | Path) -> dict[str, Any]:
    """Return the campaign payload, whether given the directory or the JSON file."""
    path = Path(path)

    if path.is_dir():
        for name in ("campaign.json", "campaign.csv"):
            if (path / name).is_file():
                path = path / name
                break
        else:
            raise FileNotFoundError(f"no campaign.json or campaign.csv in {path}")

    if path.suffix == ".csv":
        # A CSV alone has the cells but not the creation time or the image. The
        # roll-up still works; discovery then falls back to "everything recent".
        import csv

        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {"campaign": path.stem, "created_at": None, "rows": rows}

    return json.loads(path.read_text(encoding="utf-8"))


def run_cost(run_id: str) -> float | None:
    """Return the logged cost of one run, or ``None`` when it has no log.

    Never constructs a :class:`Trajectory` for a run that does not exist: the
    constructor creates the directory, so a typo would leave an empty run behind
    and a roll-up must not do that.
    """
    run_dir = runstore.runs_root() / run_id

    if not (run_dir / config.TRAJECTORY_FILENAME).is_file():
        return None

    return float(Trajectory(run_id).totals()["cost_usd"])


def discover_companions(created_at: str | None) -> list[str]:
    """Return ``judge-*``/``regrade-*`` runs at least as new as the campaign."""
    root = runstore.runs_root()

    if not root.is_dir():
        return []

    floor = _parse(created_at)
    found: list[str] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith(COMPANION_PREFIXES):
            continue

        log = entry / config.TRAJECTORY_FILENAME
        if not log.is_file():
            continue

        if floor is not None and log.stat().st_mtime < floor:
            continue

        found.append(entry.name)

    return found


def _parse(stamp: str | None) -> float | None:
    """Return a POSIX timestamp for an ISO string, or ``None`` when absent/bad."""
    if not stamp:
        return None

    try:
        return datetime.fromisoformat(stamp).timestamp()
    except ValueError:
        return None


def build_rollup(campaign: dict[str, Any], extra_runs: list[str]) -> dict[str, Any]:
    """Return the per-source cost table plus the total."""
    cells = sum(float(row.get("cost_usd") or 0.0) for row in campaign.get("rows") or [])

    companions: list[tuple[str, float]] = []
    for run_id in extra_runs:
        cost = run_cost(run_id)
        if cost is not None:
            companions.append((run_id, cost))

    return {"cells": cells, "companions": companions}


def render(rollup: dict[str, Any], campaign_name: str) -> str:
    """Return the markdown table a reader is given."""
    lines = [
        f"# Cost roll-up — {campaign_name}",
        "",
        "| Source | USD | Note |",
        "|---|---|---|",
        f"| campaign cells | {rollup['cells']:.6f} | sum of the CSV `cost_usd` |",
    ]

    for run_id, cost in rollup["companions"]:
        lines.append(f"| `{run_id}` | {cost:.6f} | judge or regrade run |")

    raw = rollup["cells"] + sum(cost for _run, cost in rollup["companions"])
    lines.append(f"| **total raw** | **{raw:.6f}** | what the API billed |")
    lines.append(
        f"| **total with fee** | **{raw * config.CREDIT_FEE:.6f}** | "
        f"x{config.CREDIT_FEE} credit load |"
    )
    lines.append("")

    if not rollup["companions"]:
        lines.append(
            "No judge or regrade run was counted. If this wave was scored, pass its "
            "run id with `--runs` — the total is otherwise missing real spend."
        )
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the roll-up and optionally write ``cost_rollup.md``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign", help="campaign directory, campaign.json or campaign.csv")
    parser.add_argument(
        "--runs", nargs="*", default=[], help="extra run ids to add (judge / regrade)"
    )
    parser.add_argument(
        "--no-discover",
        action="store_true",
        help="count only --runs; do not look for judge-*/regrade-* runs",
    )
    parser.add_argument("--out", default=None, help="write cost_rollup.md here")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    campaign = load_campaign(args.campaign)
    extra = list(args.runs)

    if not args.no_discover:
        extra += [run for run in discover_companions(campaign.get("created_at"))]

    rollup = build_rollup(campaign, extra)
    text = render(rollup, str(campaign.get("campaign") or "campaign"))
    print(text)

    if args.out:
        directory = Path(args.out)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cost_rollup.md").write_text(text, encoding="utf-8")
        print(f"wrote {directory / 'cost_rollup.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
