"""Assert the gate on one recorded run (read-only: no key, no DB, no model call).

Docker is *optional*: the "sandbox stopped" check is skipped when ``docker`` is not
available, so the same command works on an imported bundle.

Usage::

    python scripts/check_run.py <run_id>
    LENS_RUNS_ROOT=D:\\demo-runs python scripts/check_run.py <run_id>

The runs root comes from ``config`` (``LENS_RUNS_ROOT``), so the same command
works on an imported bundle. Exit code 0 = every check passed. This is the same
assertion set needs per run, which is why it is a command and not a paragraph
in a log: a gate that is not executable drifts.

The container check is tolerant: Docker may not be installed on the machine that
reads a bundle, so it prints SKIP rather than FAIL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def _ledger_counts(run_dir: Path) -> dict[str, int]:
    """Return task counts by status (empty when ``tasks.json`` is absent or corrupt)."""
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
        if not isinstance(task, dict):
            continue

        status = str(task.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1

    return counts


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


def _event_shape(mode: str) -> tuple[int, frozenset[str] | None, bool]:
    """Return ``(required_types, allowed_types, ledger_required)`` for a run mode.

    The gate checks the shape the mode can produce. A flat run has no loop (so no
    ``exec`` and no ledger) and a document run has no corpus (so no tool calls);
    asserting the full loop's shape against either would fail every baseline run
    for being a baseline.
    """
    shapes: dict[str, tuple[int, frozenset[str] | None, bool]] = {
        "rlm": (REQUIRED_CALL_TYPES, None, True),
        "naive": (2, frozenset({"chat", "tool", "note"}), False),
        "document": (2, frozenset({"chat", "exec", "note"}), False),
    }

    return shapes.get(mode, shapes["rlm"])


def check_run(run_id: str) -> bool:
    """Run every assertion; return True when the run passes the gate."""
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
    ledger_counts = _ledger_counts(run_dir)
    ledger_size = sum(ledger_counts.values())
    running = _containers_running()
    report = resolve_citations(run_id)
    counts = report.counts

    if ledger_size:
        open_count = ledger_counts.get("pending", 0) + ledger_counts.get("in_progress", 0)
        print(
            f"[INFO] ledger: {ledger_counts.get('done', 0)} done, "
            f"{ledger_counts.get('failed', 0)} failed, {open_count} open"
        )

    for written, full in report.merged:
        print(f"[INFO] merged id repaired: {written} -> {full}")

    # The gate checks the shape the mode can produce; see ``_event_shape``.
    mode = ((manifest.extra or {}).get("arm") or {}).get("mode", {}).get("run") or "rlm"
    required_types, allowed, ledger_required = _event_shape(mode)
    event_types = (
        call_types if allowed is None else [name for name in call_types if name in allowed]
    )

    if counts["cited"] == 0:
        # Not a FAIL: a negative-control query can legitimately cite nothing. The
        # campaign CSV carries the count, and an answerable query with 0 citations
        # is exactly what a reader must be able to spot.
        print("[WARN] no citations in the answer: unsupported unless this query is unanswerable")

    # Quote fidelity is a quality signal, not a validity check: a run is not invalid
    # because the model retyped a quote, and only the sentence the claim rests on is
    # lost. Printed so a campaign table can carry it, never raised to FAIL.
    try:
        from lens.observability.citations import recorded_chunk_texts, verify_quoted_spans

        texts = recorded_chunk_texts(trajectory)
        answer = (run_dir / "answer.md").read_text(encoding="utf-8")
        in_cited, elsewhere, total = verify_quoted_spans(answer, texts)

        print(
            f"[INFO] quote fidelity: {in_cited}/{total} quoted spans verbatim in the chunk "
            f"they cite, {elsewhere} found under another chunk, {total - in_cited - elsewhere} "
            "found nowhere"
        )
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not fail the gate
        print(f"[INFO] quote fidelity unavailable ({type(exc).__name__})")

    try:
        claims_path = run_dir / "claims.json"

        if claims_path.is_file():
            ledger = json.loads(claims_path.read_text(encoding="utf-8"))
            claim_counts = ledger.get("counts") or {}
            print(
                f"[INFO] claims: {ledger.get('units', 0)} units, "
                f"{claim_counts.get('grounded', 0)} grounded, "
                f"{claim_counts.get('contradicted', 0)} contradicted, "
                f"{claim_counts.get('unverifiable', 0)} unverifiable, "
                f"coverage {float(ledger.get('coverage') or 0):.0%}"
            )
        else:
            print("[INFO] claims: no ledger for this run")
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not fail the gate
        print(f"[INFO] claims unavailable ({type(exc).__name__})")

    results = [
        _check(
            "status finished",
            manifest.status != "running",
            f"status={manifest.status}",
        ),
        _check(
            f"event types >= {required_types}",
            len(event_types) >= required_types,
            f"{len(event_types)}: {event_types}",
        ),
        _check(
            "ledger non-empty",
            ledger_size > 0 if ledger_required else None,
            f"{ledger_size} task(s)" if ledger_required else f"{mode} run: no ledger by design",
        ),
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
            f" · merged {counts['merged']} · unresolved {counts['unresolved']}",
        ),
        _check(
            "sandbox stopped",
            None if running is None else not running,
            "no container running"
            if running is None
            else f"{config.SANDBOX_SERVICE}/{config.SANDBOX_RELAY_SERVICE} present={running}",
        ),
    ]

    # The numbers a report needs: the metric, and where it came from.
    turn_tokens = [e.get("tokens_in") or 0 for e in events if e.get("call_type") == "chat"]
    budgets = [
        (e.get("extra") or {}).get("budget") for e in events if isinstance(e.get("extra"), dict)
    ]
    stop = [b["stop_reason"] for b in budgets if isinstance(b, dict) and b.get("stop_reason")]

    print("")
    print(f"iterations   {manifest.iterations}")
    print(f"tokens       in {totals.get('tokens_in') or 0} / out {totals.get('tokens_out') or 0}")
    print(
        f"calls        {totals.get('calls') or 0} billable / {totals.get('tool_calls') or 0} tool"
    )
    print(f"turn tokens  {turn_tokens}")
    print(f"cap binding  {stop[-1] if stop else 'none (FINAL_ANSWER)'}")
    print(f"arm          {json.dumps((manifest.extra or {}).get('arm') or {}, sort_keys=True)}")

    passed = all(results)
    print("")
    print("RESULT       " + ("PASS" if passed else "FAIL"))

    return passed


def main(argv: list[str] | None = None) -> int:
    """Parse args and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", help="run id (a directory under the runs root)")
    args = parser.parse_args(argv)

    # The arm block and the answer are model/config prose; the console is cp1252.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    return 0 if check_run(args.run_id) else 1


if __name__ == "__main__":
    raise SystemExit(main())
