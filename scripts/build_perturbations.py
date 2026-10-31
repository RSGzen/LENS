"""Build claim variants for the calibration set: corrupted twins, and a repeat check.

The negative class is the expensive half of a calibration. A **corrupted twin** is
cheap and certain: take a claim a human marked ``supported``, swap one number for a
different one, and the new claim is unsupported **by construction** — provided the
new number does not appear in any of its excerpts, which is checked by string
search rather than assumed. Claims with no number cannot be corrupted this way and
are reported, never invented.

``--repeat-check`` measures the other half of the noise: the same claim asked twice
in two separate calls. A verifier whose own answer moves by ±0.15 between identical
requests cannot be calibrated more finely than that.

Without ``--ask`` the script only writes the variants (free). With ``--ask`` it
scores each one through the decision layer, roughly $0.0002 a variant.

Usage::

    python scripts/build_perturbations.py --labels labels.csv --out variants.csv
    python scripts/build_perturbations.py --labels labels.csv --out variants.csv --ask
    python scripts/build_perturbations.py --labels labels.csv --out variants.csv --repeat-check 20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from lens.clients import LensClient
from lens.observability.trajectory import TraceCtx, Trajectory
from lens.tools.evaluate import build_claim_question

POSITIVE = "supported"
NEGATIVE = "unsupported"
_CLAIM = "claim-01"
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_COLUMNS = ("pair_id", "kind", "claim_text", "label", "cited_ids", "score")


def load_worksheet(path: str | Path) -> list[dict[str, str]]:
    """Return the labelled rows; unlabelled rows are skipped."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if (row.get("your_label") or "").strip().lower() in {POSITIVE, "partial", NEGATIVE}
        ]


def load_sidecar(path: str | Path) -> dict[str, Any]:
    """Return the score sidecar keyed by ``pair_id``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def excerpts_of(row: dict[str, str]) -> dict[str, str]:
    """Return ``id -> excerpt`` for the rows's cited ids, in order."""
    ids = (row.get("cited_ids") or "").split()
    texts = [row.get(f"excerpt_{index}") or "" for index in (1, 2, 3)]

    return {chunk_id: text for chunk_id, text in zip(ids, texts) if text}


def corrupt(claim_text: str, excerpts: Sequence[str]) -> tuple[str, bool]:
    """Return ``(corrupted_claim, ok)``: one number swapped for an absent one.

    ``ok`` is False when the claim has no number, or every candidate replacement
    already appears in the excerpts — in which case the claim cannot be turned into
    a clean negative and must be skipped rather than half-corrupted.
    """
    match = _NUMBER.search(claim_text or "")

    if match is None:
        return claim_text, False

    original = match.group(1)
    try:
        value = float(original)
    except ValueError:
        return claim_text, False

    haystack = " ".join(excerpts)

    for candidate in (value + 7, value * 2, value + 1, value - 1):
        if candidate <= 0 or candidate == value:
            continue

        replacement = f"{candidate:g}"

        if replacement in haystack:
            continue

        return f"{claim_text[: match.start(1)]}{replacement}{claim_text[match.end(1) :]}", True

    return claim_text, False


def build_variants(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    """Return ``(variants, skipped)``: the originals plus their corrupted twins."""
    variants: list[dict[str, Any]] = []
    skipped = 0

    for row in rows:
        label = (row.get("your_label") or "").strip().lower()
        variants.append(
            {
                "pair_id": row.get("pair_id") or "",
                "kind": "original",
                "claim_text": row.get("claim_text") or "",
                "label": POSITIVE if label == POSITIVE else NEGATIVE,
                "cited_ids": row.get("cited_ids") or "",
                "excerpts": excerpts_of(row),
                "score": "",
            }
        )

        if label != POSITIVE:
            continue

        corrupted, ok = corrupt(row.get("claim_text") or "", list(excerpts_of(row).values()))

        if not ok:
            skipped += 1
            continue

        variants.append(
            {
                "pair_id": row.get("pair_id") or "",
                "kind": "corrupted",
                "claim_text": corrupted,
                "label": NEGATIVE,
                "cited_ids": row.get("cited_ids") or "",
                "excerpts": excerpts_of(row),
                "score": "",
            }
        )

    return variants, skipped


def ask(client: LensClient, claim_text: str, excerpts: dict[str, str]) -> float | None:
    """Ask the claim question once; return P(unsupported) or ``None``."""
    if not excerpts:
        return None

    state = json.dumps(
        {"query": None, "stage": "claim-probe", "summary": {"evidence": excerpts}},
        ensure_ascii=False,
    )
    result = client.jev(
        state,
        {_CLAIM: build_claim_question(_CLAIM, claim_text, list(excerpts))},
        trace_ctx=TraceCtx(trace_id="calibration", step_idx=0, actor="root"),
    )
    answer = (getattr(result, "answers", None) or {}).get(_CLAIM) or {}
    probability = answer.get("noul")

    return float(probability) if isinstance(probability, (int, float)) else None


def score_variants(variants: list[dict[str, Any]], client: LensClient) -> int:
    """Fill ``score`` on every variant; return how many were scored."""
    scored = 0

    for variant in variants:
        excerpts = dict(variant.get("excerpts") or {})
        probability = ask(client, variant["claim_text"], excerpts)

        if probability is None:
            continue

        variant["score"] = f"{probability:.4f}"
        scored += 1

    return scored


def repeat_check(rows: Sequence[dict[str, str]], client: LensClient, count: int) -> list[float]:
    """Ask the same claim twice in separate calls; return the absolute deltas."""
    deltas: list[float] = []

    for row in rows[:count]:
        excerpts = {cid: text for cid, text in excerpts_of(row).items() if text}
        first = ask(client, row.get("claim_text") or "", excerpts)
        second = ask(client, row.get("claim_text") or "", excerpts)

        if first is not None and second is not None:
            deltas.append(abs(first - second))

    return deltas


def write_variants(variants: Sequence[dict[str, Any]], path: str | Path) -> str:
    """Write the variants CSV and return it as text."""
    buffer: list[str] = [",".join(_COLUMNS)]

    for variant in variants:
        buffer.append(
            ",".join(
                str(variant.get(column, "")).replace(",", ";").replace("\n", " ")
                for column in _COLUMNS
            )
        )

    text = "\n".join(buffer) + "\n"
    Path(path).write_text(text, encoding="utf-8")

    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build corrupted claim twins for calibration.")
    parser.add_argument("--labels", required=True, help="labelled worksheet CSV")
    parser.add_argument("--out", required=True, help="variants CSV path")
    parser.add_argument("--ask", action="store_true", help="score the variants (spends JEV)")
    parser.add_argument("--repeat-check", type=int, default=0, help="test-retest on N claims")
    args = parser.parse_args(argv)

    rows = load_worksheet(args.labels)

    if not rows:
        print(f"{args.labels}: no labelled rows (your_label is empty?)", file=sys.stderr)
        return 2

    variants, skipped = build_variants(rows)
    positives = sum(1 for variant in variants if variant["label"] == POSITIVE)
    negatives = len(variants) - positives
    print(f"rows labelled: {len(rows)} | variants: {len(variants)} (+{positives}/-{negatives})")
    print(f"claims with no clean numeric corruption (skipped): {skipped}")

    if args.ask or args.repeat_check:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        trajectory = Trajectory(run_id=f"calibration-{stamp}")
        client = LensClient(trajectory=trajectory)

        if args.ask:
            scored = score_variants(variants, client)
            print(f"scored {scored} variants; calls logged in {trajectory.run_dir}")

        if args.repeat_check:
            deltas = repeat_check(rows, client, args.repeat_check)

            if deltas:
                print(
                    f"test-retest over {len(deltas)} claims: mean |delta| {mean(deltas):.3f}, "
                    f"max {max(deltas):.3f}"
                )

    write_variants(variants, args.out)
    print(f"variants: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
