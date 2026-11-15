"""Build a second OOLONG rung (262k) beside the frozen 128k bridge.

``--suffix`` names the rung; the documents, instrument and campaigns all carry it,
so the 131k files are never overwritten and both rungs can be frozen side by side.

The flat arm's inline budget has to hold the longer document or the comparison
measures the budget instead of the arm, so ``--max-tokens`` sets it, the campaign
carries it as ``LENS_FLAT_DOCUMENT_TOKENS``, and ``check_context`` validates the
same number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lens.experiments.oolong import (
    build_instance,
    check_context,
    split_context,
    write_document,
)

ARMS = [
    {
        "name": "a0-doc",
        "env": {
            "LENS_RECORDED": "1",
            "LENS_EVIDENCE": "document",
            "LENS_MODE": "naive",
            "LENS_PROMPT_VARIANT": "naive",
            "LENS_GATE": "0",
            "LENS_SUB_CALLS": "0",
        },
        "note": "A0-doc: the whole document in one prompt, one call. No loop, no tools.",
    },
    {
        "name": "a2-doc",
        "env": {
            "LENS_RECORDED": "1",
            "LENS_EVIDENCE": "document",
            "LENS_MODE": "rlm",
            "LENS_PROMPT_VARIANT": "document",
            "LENS_SUB_CALLS": "0",
            "LENS_KG_ENABLED": "0",
            "LENS_KG_PROMPT": "0",
            "LENS_GATE": "1",
        },
        "note": (
            "A2-doc: REPL over the same file. Sub-calls are 0 because the document prompt "
            "advertises no sub-task and sub_extract hydrates ids from the database."
        ),
    },
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one OOLONG rung")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--suffix", required=True, help="e.g. 262k")
    parser.add_argument("--max-tokens", type=int, required=True, help="flat inline budget")
    parser.add_argument("--instruments", default=r"D:\lens\experiments\instruments")
    parser.add_argument("--campaigns", default=r"D:\lens\experiments\campaigns")
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in Path(args.raw).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: row["id"])
    instruments = Path(args.instruments)
    tag = args.suffix

    by_window: dict[int, int] = {}
    for row in rows:
        window = int(row["context_window_id"])
        if window in by_window:
            continue
        records = split_context(row["context_window_text"])
        check_context(row["context_window_text"], records, budget=args.max_tokens)
        path = write_document(records, instruments / f"oolong-ctx-{tag}-{window}.jsonl")
        by_window[window] = len(records)
        print(f"document {path.name}: {len(records)} records, {path.stat().st_size / 1e6:.1f} MB")

    lines: list[dict] = []
    for position, row in enumerate(rows, start=1):
        lines.append(
            build_instance(
                row,
                document_file=f"oolong-ctx-{tag}-{int(row['context_window_id'])}.jsonl",
                subset=position % 5 == 0,
                seed=20261115,
            )
        )

    instrument_path = instruments / f"oolong-bridge-50-{tag}.jsonl"
    instrument_path.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8"
    )
    print(f"instrument: {len(lines)} instances -> {instrument_path.name}")

    arms = [
        {**arm, "env": {**arm["env"], "LENS_FLAT_DOCUMENT_TOKENS": str(args.max_tokens)}}
        if arm["name"] == "a0-doc"
        else arm
        for arm in ARMS
    ]

    def query(line: dict) -> dict:
        return {
            "id": line["instance_id"],
            "expect": "answerable",
            "text": line["query"],
            "env": {
                "LENS_DOCUMENTS": str(instruments / line["evidence"]["file"]),
            },
        }

    campaign = {
        "name": f"m11-w3-{tag}",
        "repeats": 1,
        "note": (
            f"OOLONG bridge, second rung: **{tag}** context, document mode, 50 instances x 2 arms. "
            "The 131k rung is frozen beside it; this one raises the flat arm's inline budget to "
            f"{args.max_tokens} so the flat arm is not truncated by a number chosen for a shorter "
            "document. Same release, same scorer, same two arms. Not pooled with the 131k rung: a "
            "different input length is a different measurement."
        ),
        "arms": arms,
        "queries": [query(line) for line in lines],
    }
    path = Path(args.campaigns) / f"m11-w3-{tag}.json"
    path.write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path.name}: {len(campaign['queries'])} x 2 = {len(campaign['queries']) * 2} runs")

    # a two-instance dry check with the same budget
    dry = {
        "name": f"m11-w3-{tag}-dry",
        "repeats": 1,
        "note": (
            f"Dry check for the {tag} rung: two instances, both arms, unrecorded. Checks that the "
            "document loads in both prompts, that the flat arm drops no record at the raised "
            "budget, and that the answer reaches the release's parser."
        ),
        "arms": [{**arm, "env": {**arm["env"], "LENS_RECORDED": "0"}} for arm in arms],
        "queries": campaign["queries"][:2],
    }
    dry_path = Path(args.campaigns) / f"m11-w3-{tag}-dry.json"
    dry_path.write_text(json.dumps(dry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{dry_path.name}: 2 x 2 = 4 runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
