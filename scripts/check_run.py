"""Assert the M9 gate on one recorded run (read-only, no key, no DB, no Docker).

Spec: ``MVP Build Roadmap.md`` M9 verify clause ("reproducible command; cost
reported") and §6 hard gate ("within caps, >= 4 event types, ``tasks.json``
non-empty, ``answer.md`` written, ``cost_usd == trajectory.totals()``").

Usage::

    python scripts/check_run.py <run_id>
    LENS_RUNS_ROOT=D:\\demo-runs python scripts/check_run.py <run_id>

The runs root comes from ``config`` (``LENS_RUNS_ROOT``), so the same command
works on an imported bundle. Exit code 0 = every check passed. This is the same
assertion set M11 needs per run, which is why it is a command and not a paragraph
in a log: a gate that is not executable drifts.

The container check is tolerant: Docker may not be installed on the machine that
reads a bundle, so it prints SKIP rather than FAIL.
"""

from __future__ import annotations

import argparse
import json
import subprocess

from pathlib import Path
from typing import Any

from lens import config
from lens.agent import runstore
from lens.observability.citations import resolve_citations
from lens.observability.trajectory import Trajectory

REQUIRED_CALL_TYPES = 4


def _check(name: str, ok: bool | None, detail: str) -> bool:
    """Print one PASS/FAIL/SKIP line and return whether it counts as passing."""
    label = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    print(f"[{label}] {name}: {detail}")
    return ok is not False


def _events(path: Path) -> list[dict[str, Any]]:
    """Return the parsed JSONL events (a corrupt line is skipped, never fatal)."""
    events: list[dict[str, Any]] = []

    if not path.is_file():
        return events

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            event = json.loads(line)
        except ValueError:
            continue

        if isinstance(event, dict):
            events.append(event)

    return events


def _ledger_size(run_dir: Path) -> int:
    """Return the number of tasks in ``tasks.json`` (0 when absent or corrupt)."""
    path = run_dir / "tasks.json"

    if not path.is_file():
        return 0

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0

    tasks = payload.get("tasks") if isinstance(payload, dict) else payload

    return len(tasks) if isinstance(tasks, (list, dict)) else 0


def _containers_running() -> bool | None:
    """Return whether a sandbox/relay container is still up, or ``None`` if unknown."""
    try:
        finished = subprocess.run(
            [
                "docker",
                "ps",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if finished.returncode != 0:
        return None

    running = set(finished.stdout.split())
    expected = {config.SANDBOX_SERVICE, config.SANDBOX_RELAY_SERVICE}

    return bool(running & expected)


def check_run(run_id: str) -> bool:
    """Run every M9 assertion; return True when the run passes the gate."""
    run_dir = runstore.runs_root() / run_id

    print(f"run      {run_id}")
    print(f"run_dir  {run_dir}")
    print("")

    manifest = runstore.read_manifest(run_dir)

    if manifest is None:
        _check("manifest readable", False, f"no manifest.json under {run_dir}")
        return False

    trajectory = Trajectory(run_id)
    totals = trajectory.totals()
    events = _events(trajectory.trajectory_path)

    call_types = sorted({e.get("call_type") for e in events if e.get("call_type")})
    answer_path = run_dir / "answer.md"
    answer = answer_path.read_text(encoding="utf-8") if answer_path.is_file() else ""
    ledger_size = _ledger_size(run_dir)
    running = _containers_running()
    report = resolve_citations(run_id)
    counts = report.counts

    results = [
        _check(
            "status finished",
            manifest.status != "running",
            f"status={manifest.status}",
        ),
        _check(
            f"event types >= {REQUIRED_CALL_TYPES}",
            len(call_types) >= REQUIRED_CALL_TYPES,
            f"{len(call_types)}: {call_types}",
        ),
        _check("ledger non-empty", ledger_size > 0, f"{ledger_size} task(s)"),
        _check("answer.md written", bool(answer.strip()), f"{len(answer)} chars"),
        _check(
            "cost matches totals()",
            abs(
                float((manifest.totals or {}).get("cost_usd") or 0.0)
                - float(totals.get("cost_usd") or 0.0)
            )
            < 1e-9,
            f"manifest ${float((manifest.totals or {}).get('cost_usd') or 0.0):.4f}"
            f" vs log ${float(totals.get('cost_usd') or 0.0):.4f}",
        ),
        _check(
            "citations resolved",
            counts["unresolved"] == 0,
            f"cited {counts['cited']} · resolved {counts['resolved']}"
            f" · unresolved {counts['unresolved']}",
        ),
        _check(
            "sandbox stopped",
            None if running is None else not running,
            "no container running"
            if running is None
            else f"{config.SANDBOX_SERVICE}/{config.SANDBOX_RELAY_SERVICE} present={running}",
        ),
    ]

    # The numbers the log and the thesis report need (R4: metric + source row).
    turn_tokens = [e.get("tokens_in") or 0 for e in events if e.get("call_type") == "chat"]
    budgets = [
        (e.get("extra") or {}).get("budget")
        for e in events
        if isinstance(e.get("extra"), dict)
    ]
    stop = [b["stop_reason"] for b in budgets if isinstance(b, dict) and b.get("stop_reason")]

    print("")
    print(f"iterations   {manifest.iterations}")
    print(f"tokens       in {totals.get('tokens_in') or 0} / out {totals.get('tokens_out') or 0}")
    print(f"calls        {totals.get('calls') or 0} billable / {totals.get('tool_calls') or 0} tool")
    print(f"turn tokens  {turn_tokens}")
    print(f"cap binding  {stop[-1] if stop else 'none (FINAL_ANSWER)'}")
    print(f"arm          {json.dumps((manifest.extra or {}).get('arm') or {}, sort_keys=True)}")

    passed = all(results)
    print("")
    print("RESULT       " + ("PASS" if passed else "FAIL"))

    return passed


def main(argv: list[str] | None = None) -> int:
    """Parse args and return an exit code (0 = the M9 gate passes)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", help="run id (a directory under the runs root)")
    args = parser.parse_args(argv)

    return 0 if check_run(args.run_id) else 1


if __name__ == "__main__":
    raise SystemExit(main())
