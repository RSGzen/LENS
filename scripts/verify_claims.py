"""Verify claim frames over recorded runs and write the campaign ledger.

Offline by design: the script reads ``answer.md`` and the run's recorded tool
payloads, calls the sub model to decompose each claim sentence, and checks every
frame against the chunk it cites. Nothing is written into the run: the
extractor logs to its own scratch run, and the ledger goes to a separate output
directory.

The two numbers the ledger exists for are coverage (how many claim units a
mechanical check could reach) and status (grounded / contradicted /
unverifiable). A low coverage is a finding, not a failure.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from lens import config
from lens.agent import runstore
from lens.clients import LensClient
from lens.observability.citations import claim_units, recorded_chunk_texts
from lens.observability.claim_extract import extract_frames
from lens.observability.claims import ClaimReport, verify_units
from lens.observability.trajectory import TraceCtx, Trajectory

CSV_FIELDS = [
    "run_id",
    "unit_index",
    "predicate",
    "subject",
    "object",
    "benchmark",
    "metric",
    "value",
    "unit",
    "direction",
    "chunk_id",
    "status",
    "reason",
    "evidence",
    "quote",
    "text",
]


def _stamp() -> str:
    """Return a filename-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _selected_ids(args: argparse.Namespace) -> list[str]:
    """Return the run ids to check, in the order given, without duplicates."""
    ids = list(dict.fromkeys(args.run or []))

    if args.all:
        ids += [
            manifest.run_id
            for manifest in runstore.list_runs()
            if manifest.status in {"completed", "partial"}
        ]

    return list(dict.fromkeys(ids))


def collect(run_id: str, client: LensClient, trace_ctx: TraceCtx) -> ClaimReport | None:
    """Return the claim report for one recorded run, or ``None`` without an answer."""
    trajectory = Trajectory(run_id)
    answer_path = trajectory.run_dir / "answer.md"

    if not answer_path.is_file():
        return None

    answer = answer_path.read_text(encoding="utf-8", errors="replace")
    units = claim_units(answer, include_uncited=True)

    if not units:
        return ClaimReport(run_id=run_id, units=0)

    texts = recorded_chunk_texts(trajectory)
    frames, errors = extract_frames(client, units, trace_ctx, texts=texts)
    report = verify_units(units, frames, texts, run_id=run_id)
    report.errors = errors

    return report


def write_csv(reports: list[ClaimReport], path: Path) -> Path:
    """Write one row per frame and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for report in reports:
            for row in report.rows():
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})

    return path


def write_markdown(reports: list[ClaimReport], path: Path) -> Path:
    """Write the human view: per-run counts, then every contradicted frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = _totals(reports)
    lines = [
        "# Claim ledger",
        "",
        f"runs {totals['runs']} | units {totals['units']} | frames {totals['frames']} | "
        f"grounded {totals['grounded']} | contradicted {totals['contradicted']} | "
        f"unverifiable {totals['unverifiable']} | coverage {totals['coverage']:.1%}",
        "",
        "| run | units | frames | grounded | contradicted | unverifiable | coverage |",
        "|---|---|---|---|---|---|---|",
    ]

    for report in reports:
        counts = report.counts
        lines.append(
            f"| {report.run_id} | {report.units} | {len(report.frames)} | "
            f"{counts['grounded']} | {counts['contradicted']} | "
            f"{counts['unverifiable']} | {report.coverage:.1%} |"
        )

    contradicted = [
        (report.run_id, frame, verdict)
        for report in reports
        for frame, verdict in zip(report.frames, report.verdicts)
        if verdict.status == "contradicted"
    ]

    if contradicted:
        lines += ["", "## Contradicted", ""]

        for run_id, frame, verdict in contradicted:
            lines.append(
                f"- **{run_id}** unit {frame.unit_index}: {frame.text}  \n"
                f"  {verdict.reason}  \n"
                f"  quote: {frame.quote}"
            )

    errors = [(report.run_id, error) for report in reports for error in report.errors]

    if errors:
        lines += ["", "## Extractor errors", ""]
        lines += [f"- {run_id}: {error}" for run_id, error in errors]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return path


def _totals(reports: list[ClaimReport]) -> dict[str, int | float]:
    """Return the aggregate counts over every report."""
    units = sum(report.units for report in reports)
    counts = {
        status: sum(report.counts[status] for report in reports)
        for status in ("grounded", "contradicted", "unverifiable")
    }

    return {
        "runs": len(reports),
        "units": units,
        "frames": sum(len(report.frames) for report in reports),
        **counts,
        "coverage": 0.0 if units < 1 else sum(report.covered_units for report in reports) / units,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="run id; repeatable")
    parser.add_argument(
        "--all",
        action="store_true",
        help="every completed or partial run in the store",
    )
    parser.add_argument("--out", default=None, help="CSV path")
    parser.add_argument("--md", default=None, help="Markdown path")
    parser.add_argument("--batch-size", type=int, default=4, help="claim units per call")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ids = _selected_ids(args)

    if not ids:
        print("choose --run <id> or --all", file=sys.stderr)
        return 2

    stamp = _stamp()
    out = Path(args.out) if args.out else config.DATA_ROOT / "campaigns" / f"claims-{stamp}.csv"
    md = Path(args.md) if args.md else out.with_suffix(".md")

    # One scratch run holds the extractor's own calls, so the recorded runs stay
    # untouched and the ledger can still be traced to the prompts that built it.
    scratch = Trajectory(f"claims-verify-{stamp}")
    client = LensClient(trajectory=scratch)
    trace_ctx = TraceCtx(trace_id=scratch.run_id, step_idx=1, actor="sub")

    reports: list[ClaimReport] = []

    for run_id in ids:
        report = collect(run_id, client, trace_ctx)

        if report is None:
            print(f"[skip] {run_id}: no answer.md")
            continue

        reports.append(report)
        counts = report.counts
        print(
            f"[run] {run_id}: units {report.units} frames {len(report.frames)} "
            f"grounded {counts['grounded']} contradicted {counts['contradicted']} "
            f"unverifiable {counts['unverifiable']} coverage {report.coverage:.1%}"
        )

        for error in report.errors:
            print(f"       error: {error}")

    if not reports:
        print("no runs with an answer", file=sys.stderr)
        return 1

    write_csv(reports, out)
    write_markdown(reports, md)

    totals = _totals(reports)
    print(
        f"\nruns {totals['runs']} | units {totals['units']} | frames {totals['frames']} | "
        f"grounded {totals['grounded']} | contradicted {totals['contradicted']} | "
        f"unverifiable {totals['unverifiable']} | coverage {totals['coverage']:.1%}"
    )
    print(f"csv {out}")
    print(f"md  {md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
