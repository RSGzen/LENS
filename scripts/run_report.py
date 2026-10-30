"""Write a static, printable run report - read-only, no key, no DB, no Docker.

Usage::

    python scripts/run_report.py <run_id> [<run_id> ...]
    LENS_RUNS_ROOT=D:\\imported python scripts/run_report.py <run_id>

Writes ``<run_dir>/report.md`` and prints the path. Never raises on a damaged
bundle: a missing blob shows as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lens.agent import runstore
from lens.observability.citations import resolve_citations, write_sources_md
from lens.observability.replay import Turn, replay_summary, replay_turns
from lens.observability.trajectory import Trajectory

CODE_CHARS = 1500
OBSERVATION_CHARS = 800


def _fence(text: str, language: str = "") -> str:
    """Return ``text`` inside a fence that its own backticks cannot break."""
    body = (text or "").rstrip()
    marker = "```"

    while marker in body:
        marker += "`"

    return f"{marker}{language}\n{body}\n{marker}"


def _pct(part: int | None, whole: int | None) -> str:
    """Return ``part`` as a percentage of ``whole``, or ``n/a`` when ``whole`` is 0."""
    if not whole:
        return "n/a"

    return f"{100 * (part or 0) / whole:.0f} %"


def _turn_block(turn: Turn) -> list[str]:
    """Render one turn: completion, code, tool calls, observation, meters."""
    lines = [f"### Turn {turn.turn}", ""]
    meters = [
        f"tokens_in {turn.tokens_in or 0}"
        + (f" ({turn.cached_tokens} cached)" if turn.cached_tokens else ""),
        f"tokens_out {turn.tokens_out or 0}"
        + (f" ({turn.reasoning_tokens} reasoning)" if turn.reasoning_tokens else ""),
        f"cost ${(turn.cost_usd or 0.0) + (turn.jev_cost_usd or 0.0):.6f}",
        f"latency {turn.latency_ms or 0} ms",
    ]

    if turn.gate:
        meters.append(f"gate {turn.gate.get('verdict')} (tier {turn.gate.get('tier')})")

    if turn.ok is False:
        meters.append("**block failed**")

    if turn.ok is None and turn.code:
        meters.append("**interrupted**")

    lines += ["`" + " · ".join(meters) + "`", ""]

    if turn.tools:
        lines += ["| tool | latency ms | gate | replayed |", "|---|---|---|---|"]

        for tool in turn.tools:
            lines.append(
                f"| `{tool.name}` | {tool.latency_ms or 0} | "
                f"{(tool.gate or {}).get('verdict', '-')} | {tool.replayed} |"
            )

        lines.append("")

    if turn.code:
        code = turn.code

        if len(code) > CODE_CHARS:
            code = code[:CODE_CHARS] + "\n# … truncated for the report"

        lines += [_fence(code, "python"), ""]

    if turn.observation:
        observation = turn.observation

        if len(observation) > OBSERVATION_CHARS:
            observation = observation[:OBSERVATION_CHARS] + " … (truncated)"

        lines += ["**Observation (what the model saw)**", "", _fence(observation), ""]

    if turn.emits:
        lines += [f"emits: {', '.join(turn.emits)}", ""]

    if turn.error:
        lines += [f"error: `{json.dumps(turn.error)[:300]}`", ""]

    if turn.missing_refs:
        lines += [f"missing refs: {len(turn.missing_refs)}", ""]

    return lines


def write_report(run_id: str) -> Path:
    """Write ``<run_dir>/report.md`` for ``run_id`` and return its path."""
    run_dir = runstore.runs_root() / run_id
    manifest = runstore.read_manifest(run_dir)
    trajectory = Trajectory(run_id)
    turns = replay_turns(trajectory)
    summary = replay_summary(turns)
    citations = resolve_citations(run_id)
    totals = trajectory.totals()

    lines = [
        f"# Run report — {run_id}",
        "",
        "> Static rendering of the recorded trajectory. The same content is what",
        "> the M10b viewer shows turn by turn; no model was called to produce it.",
        "",
        "| field | value |",
        "|---|---|",
        f"| status | {manifest.status if manifest else 'missing'} |",
        f"| query | {(manifest.query if manifest else '')} |",
        f"| iterations | {manifest.iterations if manifest else 0} |",
        f"| cost | ${float(totals.get('cost_usd') or 0.0):.6f} raw "
        f"(${float(totals.get('cost_usd') or 0.0) * 1.055:.6f} with the 5.5 % fee) |",
        f"| calls | {totals.get('calls') or 0} billable · {totals.get('tool_calls') or 0} tool |",
        f"| tokens | in {totals.get('tokens_in') or 0} · out {totals.get('tokens_out') or 0} "
        f"· gate {summary['jev_tokens_in']} |",
        f"| cache | {summary['cached_tokens']} of {summary['tokens_in']} chat input "
        f"tokens ({_pct(summary['cached_tokens'], summary['tokens_in'])}) served from the "
        "provider cache |",
        f"| reasoning | {summary['reasoning_tokens']} of {summary['tokens_out']} chat output "
        f"tokens ({_pct(summary['reasoning_tokens'], summary['tokens_out'])}) |",
        f"| turns | {summary['turns']} |",
        f"| citations | cited {citations.counts['cited']} · "
        f"resolved {citations.counts['resolved']} · "
        f"unresolved {citations.counts['unresolved']} |",
        f"| missing refs | {summary['missing_refs']} |",
        "",
    ]

    if manifest and manifest.extra.get("arm"):
        lines += ["## Arm", "", _fence(json.dumps(manifest.extra["arm"], indent=2), "json"), ""]

    lines += ["## Timeline", ""]

    for turn in turns:
        lines += _turn_block(turn)

    answer_path = run_dir / "answer.md"
    lines += ["## Answer", ""]

    if answer_path.is_file():
        lines += [answer_path.read_text(encoding="utf-8").rstrip(), ""]
    else:
        lines += ["*(no answer.md)*", ""]

    if citations.citations:
        lines += [
            "## Citations",
            "",
            "| cited id | paper | section | turn | status |",
            "|---|---|---|---|---|",
        ]

        for citation in citations.citations:
            status = "OK" if citation.resolved else "**unresolved**"
            lines.append(
                f"| `{citation.token}` | {citation.arxiv_id or '-'} | "
                f"{citation.section_type or '-'} | "
                f"{citation.turn if citation.turn is not None else '-'} | {status} |"
            )

        lines.append("")

        if citations.unresolved:
            lines += ["Unresolved: " + ", ".join(f"`{t}`" for t in citations.unresolved), ""]

    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The citation trace beside the answer : `answer.md` stays the
    # model's own words, `sources.md` is the derived join against the recorded
    # payloads, and unresolved ids are listed rather than hidden.
    write_sources_md(citations, run_dir / "sources.md")

    return path


def main(argv: list[str] | None = None) -> int:
    """Render one report per run id; print the paths."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_ids", nargs="+", help="run ids under the runs root")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    for run_id in args.run_ids:
        print(write_report(run_id))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
