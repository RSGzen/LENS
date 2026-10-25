"""Package marker and console entry point for LENS."""

from __future__ import annotations


def main() -> None:
    """Print where to start; ``python -m lens.run`` runs a query."""
    print(
        "LENS - Literature Evidence & Narrative Synthesis.\n"
        "Run a query:      python -m lens.run \"<query>\"\n"
        "List saved runs:  python -m lens.run --list\n"
        "Check a run:      python scripts/check_run.py <run_id>"
    )
