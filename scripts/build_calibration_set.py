"""Sample recorded claims into a blind labelling worksheet, plus a sidecar of scores.

Why a worksheet and a sidecar: the calibration is only honest if the labels are
blind to the score, so the model's P(unsupported) and the source run never appear
in the file a human edits. They are joined afterwards, by ``pair_id``.

Sampling is **stratified by score band** (the same bands the calibration plan
uses), because a uniform sample from 206 claims would be dominated by the 0-0.3
band where the gate is already confident, and the threshold decision lives in the
middle. The consequence is stated in the output: the sampled prevalence is *not*
the population prevalence.

Usage::

    python scripts/build_calibration_set.py --regrade regrade-20261031-021751
    python scripts/build_calibration_set.py --regrade <id> --per-band 30 --out labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from lens.agent import runstore

#: Score bands, low to high. The middle two hold the decision.
BANDS: tuple[tuple[float, float], ...] = ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01))

#: Excerpts shown per claim: the cited chunks, at most this many.
MAX_EXCERPTS = 3
EXCERPT_CHARS = 1200


def band_of(score: float) -> int:
    """Return the index of the band ``score`` falls in."""
    for index, (low, high) in enumerate(BANDS):
        if low <= score < high:
            return index

    return len(BANDS) - 1


def _load(ref: str, blobs: Path) -> str:
    """Return the text of one content-addressed blob."""
    return (blobs / ref.split(":", 1)[1]).read_text(encoding="utf-8")


def collect_claims(regrade_run: str) -> list[dict[str, Any]]:
    """Return every ``(run, claim, citations, score, excerpts)`` from a regrade log.

    Each ``final`` gate call in the log is one answer: the ``gate.verdict`` event
    carries the per-claim scores, and the matching JEV request blob carries the
    claim text and the evidence map. The trace id is the source run id.
    """
    run_dir = runstore.runs_root() / regrade_run
    trajectory = run_dir / "trajectory.jsonl"
    blobs = runstore.runs_root() / "blobs"

    if not trajectory.is_file():
        raise FileNotFoundError(f"no trajectory in {run_dir}")

    events = [
        json.loads(line)
        for line in trajectory.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verdicts = [
        event
        for event in events
        if event.get("action") == "gate.verdict" and event.get("stage") == "final"
    ]
    requests = [event for event in events if event.get("call_type") == "jev"]

    claims: list[dict[str, Any]] = []

    for verdict, request in zip(verdicts, requests, strict=False):
        scores = (verdict.get("extra") or {}).get("claim_scores") or {}

        if not scores:
            continue

        body = json.loads(_load(request["prompt_ref"], blobs))
        state = json.loads(body["state"]) if isinstance(body.get("state"), str) else {}
        summary = state.get("summary") or {}
        evidence = summary.get("evidence") or {}
        questions = body.get("questions") or {}

        for name, probability in scores.items():
            instructions = (questions.get(name) or {}).get("instructions") or ""
            text = instructions.rsplit("Claim: ", 1)[-1].strip()
            cited = [chunk_id for chunk_id in evidence if chunk_id and chunk_id[:8] in instructions]
            excerpts = [evidence[chunk_id][:EXCERPT_CHARS] for chunk_id in cited][:MAX_EXCERPTS]
            claims.append(
                {
                    "run_id": verdict.get("trace_id") or "",
                    "claim_id": name,
                    "claim_text": " ".join(text.split()),
                    "cited_ids": cited,
                    "excerpts": excerpts,
                    "score": float(probability),
                    "aggregate": verdict.get("confidence"),
                }
            )

    return claims


def stratified_sample(
    claims: Sequence[dict[str, Any]], per_band: int, seed: int = 7
) -> list[dict[str, Any]]:
    """Return at most ``per_band`` claims from each band, shuffled.

    A band that cannot fill its quota contributes what it has; the shortfall is
    reported by the caller, never silently made up from another band (that would
    quietly change the sampling design).
    """
    buckets: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(BANDS))}

    for claim in claims:
        buckets[band_of(float(claim["score"]))].append(claim)

    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []

    for index in range(len(BANDS)):
        rng.shuffle(buckets[index])
        sample.extend(buckets[index][:per_band])

    rng.shuffle(sample)

    return sample


def write_worksheet(sample: Sequence[dict[str, Any]], path: Path) -> Path:
    """Write the blind worksheet: claim, citations, excerpts, blank label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "pair_id",
        "claim_text",
        "cited_ids",
        "excerpt_1",
        "excerpt_2",
        "excerpt_3",
        "your_label",
        "note",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for index, claim in enumerate(sample, start=1):
            excerpts = list(claim["excerpts"]) + ["", "", ""]
            writer.writerow(
                {
                    "pair_id": f"pair-{index:03d}",
                    "claim_text": claim["claim_text"],
                    "cited_ids": " ".join(claim["cited_ids"]),
                    "excerpt_1": excerpts[0],
                    "excerpt_2": excerpts[1],
                    "excerpt_3": excerpts[2],
                    "your_label": "",
                    "note": "",
                }
            )

    return path


def write_sidecar(sample: Sequence[dict[str, Any]], path: Path) -> Path:
    """Write the scores the human must not see, keyed by the same ``pair_id``."""
    payload = {
        f"pair-{index:03d}": {
            "run_id": claim["run_id"],
            "claim_id": claim["claim_id"],
            "score": claim["score"],
            "aggregate": claim["aggregate"],
            "band": band_of(float(claim["score"])),
            "cited_ids": claim["cited_ids"],
        }
        for index, claim in enumerate(sample, start=1)
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the blind calibration worksheet.")
    parser.add_argument("--regrade", required=True, help="regrade run id (carries claim scores)")
    parser.add_argument("--per-band", type=int, default=30, help="claims per score band")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True, help="worksheet CSV path")
    args = parser.parse_args(argv)

    claims = collect_claims(args.regrade)

    if not claims:
        print(f"{args.regrade}: no per-claim scores in that log", file=sys.stderr)
        return 2

    sample = stratified_sample(claims, args.per_band)
    sidecar = Path(args.out).with_suffix(".scores.json")

    write_worksheet(sample, Path(args.out))
    write_sidecar(sample, sidecar)

    counts = {index: 0 for index in range(len(BANDS))}

    for claim in sample:
        counts[band_of(float(claim["score"]))] += 1

    print(f"claims available: {len(claims)} | sampled: {len(sample)}")
    print("per band: " + ", ".join(f"{BANDS[i][0]}-{BANDS[i][1]}: {counts[i]}" for i in counts))
    print(f"worksheet: {args.out}")
    print(f"scores (do not open until labelling is done): {sidecar}")
    print("note: the sample is stratified, so its prevalence is not the population's.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
