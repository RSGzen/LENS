"""Fit the gate's thresholds on labelled claims, and print the decision table.

The unit of truth is a **claim**: a human label plus the probability the gate gave
that same claim. A rule is judged by what its *answer-level* decision does to each
labelled claim:

- a ``supported`` claim that the rule rejects is a **false reject** (the gate
  refuses a good answer);
- an ``unsupported`` claim that the rule accepts is a **false accept** (the bad
  claim slips through) — the only kind that reaches a reader.

Metrics per rule and threshold: TPR (bad claims caught), TNR (good claims kept),
precision of a rejection (TP / all rejections), and the pass rate on the sample.

The decision rule is pre-registered and printed with the result:

    among the candidates with pass rate >= 0.25, TNR >= 0.6 and TPR >= 0.5, take
    the highest rejection precision; if none reaches 0.8, keep the current
    thresholds and report the gate as uncalibrated.

Confidence intervals are Wilson, reported as spread descriptors — no significance
test is performed anywhere in this project.

Usage::

    python scripts/fit_thresholds.py --labels labels.csv
    python scripts/fit_thresholds.py --labels labels.csv --variants variants.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

POSITIVE = "supported"
NEGATIVE = "unsupported"
PARTIAL = "partial"

#: Thresholds swept. 0.5 and 0.7 are the values currently in the code.
TAUS: tuple[float, ...] = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9)

#: Claim threshold held fixed while the aggregate threshold is swept.
FIXED_CLAIM_TAU = 0.7

#: Pre-registered decision constraints.
#:
#: ``MIN_TPR`` is there because precision alone is gameable: a rule that rejects
#: almost nothing scores perfect precision. Without a floor, "only reject the one
#: obvious case" wins every table.
MIN_PASS_RATE = 0.25
MIN_TPR = 0.5
MIN_TNR = 0.6
TARGET_PRECISION = 0.8

CURRENT = {"rule": "both", "tau_agg": 0.5, "tau_claim": 0.7}


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for ``successes / total`` (0..1)."""
    if total <= 0:
        return 0.0, 0.0

    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator

    return max(0.0, centre - margin), min(1.0, centre + margin)


def truth_of(label: str) -> int | None:
    """Return 1 for a supported claim, 0 for anything else, ``None`` if unlabelled.

    ``partial`` counts as **not supported**: the gate's question is binary, and a
    claim that only half-appears in its excerpt is exactly the case a reader would
    be misled by.
    """
    value = (label or "").strip().lower()

    if value == POSITIVE:
        return 1

    if value in {NEGATIVE, PARTIAL}:
        return 0

    return None


def load_rows(labels: str | Path, variants: str | Path | None = None) -> list[dict[str, Any]]:
    """Return the labelled rows, with the gate's scores joined where available."""
    worksheet = Path(labels)
    sidecar_path = worksheet.with_suffix(".scores.json")
    sidecar: dict[str, Any] = (
        json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.is_file() else {}
    )
    rows: list[dict[str, Any]] = []

    with worksheet.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            truth = truth_of(row.get("your_label") or "")

            if truth is None:
                continue

            score = sidecar.get(row.get("pair_id") or "", {})
            claim_probability = score.get("score")

            if claim_probability is None:
                continue

            rows.append(
                {
                    "pair_id": row.get("pair_id") or "",
                    "kind": "original",
                    "truth": truth,
                    "claim": float(claim_probability),
                    "aggregate": None
                    if score.get("aggregate") is None
                    else float(score["aggregate"]),
                }
            )

    if variants is not None and Path(variants).is_file():
        with Path(variants).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row.get("kind") or "") != "corrupted":
                    continue

                truth = truth_of(row.get("label") or "")

                if truth is None or not row.get("score"):
                    continue

                rows.append(
                    {
                        "pair_id": row.get("pair_id") or "",
                        "kind": "corrupted",
                        "truth": truth,
                        "claim": float(row["score"]),
                        "aggregate": None,
                    }
                )

    return rows


def decide(row: dict[str, Any], rule: str, tau_agg: float, tau_claim: float) -> bool:
    """Return whether the rule accepts this claim's answer."""
    aggregate = row.get("aggregate")

    if rule == "aggregate":
        return aggregate is not None and aggregate < tau_agg

    if rule == "claim":
        return row["claim"] < tau_claim

    accepted = row["claim"] < tau_claim

    return accepted and (aggregate is None or aggregate < tau_agg)


def evaluate(
    rows: Sequence[dict[str, Any]], rule: str, tau_agg: float, tau_claim: float
) -> dict[str, Any]:
    """Return the confusion counts and metrics for one rule at one threshold pair."""
    positives = [row for row in rows if row["truth"] == 1]
    negatives = [row for row in rows if row["truth"] == 0]
    true_positive = sum(1 for row in negatives if not decide(row, rule, tau_agg, tau_claim))
    true_negative = sum(1 for row in positives if decide(row, rule, tau_agg, tau_claim))
    false_reject = len(positives) - true_negative
    rejected = true_positive + false_reject
    accepted = true_negative + (len(negatives) - true_positive)

    tpr = true_positive / len(negatives) if negatives else 0.0
    tnr = true_negative / len(positives) if positives else 0.0
    precision = true_positive / rejected if rejected else 0.0

    return {
        "rule": rule,
        "tau_agg": tau_agg,
        "tau_claim": tau_claim,
        "positives": len(positives),
        "negatives": len(negatives),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_reject": false_reject,
        "tpr": round(tpr, 3),
        "tnr": round(tnr, 3),
        "tpr_ci": [round(value, 3) for value in wilson(true_positive, len(negatives))],
        "tnr_ci": [round(value, 3) for value in wilson(true_negative, len(positives))],
        "precision": round(precision, 3),
        "precision_ci": [round(value, 3) for value in wilson(true_positive, rejected)],
        "pass_rate": round(accepted / len(rows), 3) if rows else 0.0,
    }


def sweep(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every rule × threshold candidate the plan considers."""
    results: list[dict[str, Any]] = []

    for tau in TAUS:
        results.append(evaluate(rows, "aggregate", tau, 1.1))
        results.append(evaluate(rows, "claim", 0.0, tau))
        results.append(evaluate(rows, "both", tau, FIXED_CLAIM_TAU))

    return results


def choose(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered decision rule to the candidate table."""
    eligible = [
        result
        for result in results
        if result["pass_rate"] >= MIN_PASS_RATE
        and result["tnr"] >= MIN_TNR
        and result["tpr"] >= MIN_TPR
    ]

    if not eligible:
        return {
            "chosen": None,
            "reason": "no candidate met the pass-rate, TNR and TPR constraints",
        }

    best = max(eligible, key=lambda result: (result["precision"], result["tnr"]))

    if best["precision"] < TARGET_PRECISION:
        return {
            "chosen": None,
            "best": best,
            "reason": (
                f"best rejection precision {best['precision']:.2f} is below the "
                f"{TARGET_PRECISION:.1f} target: keep the current thresholds and report the "
                "gate as uncalibrated, with this precision measured"
            ),
        }

    return {"chosen": best, "reason": "pre-registered rule satisfied"}


COLUMNS = (
    "rule",
    "tau_agg",
    "tau_claim",
    "positives",
    "negatives",
    "tpr",
    "tpr_ci",
    "tnr",
    "tnr_ci",
    "precision",
    "pass_rate",
)


def format_table(results: Iterable[dict[str, Any]]) -> str:
    """Return the candidate table as fixed-width text."""
    header = (
        f"{'rule':9} {'tau_a':>5} {'tau_c':>5} {'pos':>4} {'neg':>4} "
        f"{'TPR':>5} {'TPR 95%':>13} {'TNR':>5} {'TNR 95%':>13} {'prec':>5} {'pass':>5}"
    )
    lines = [header, "-" * len(header)]

    for result in results:
        lines.append(
            f"{result['rule']:9} {result['tau_agg']:5.2f} {result['tau_claim']:5.2f} "
            f"{result['positives']:4d} {result['negatives']:4d} {result['tpr']:5.2f} "
            f"{str(result['tpr_ci']):>13} {result['tnr']:5.2f} {str(result['tnr_ci']):>13} "
            f"{result['precision']:5.2f} {result['pass_rate']:5.2f}"
        )

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit gate thresholds on labelled claims.")
    parser.add_argument("--labels", required=True, help="labelled worksheet CSV")
    parser.add_argument("--variants", default=None, help="scored variants CSV (corrupted twins)")
    parser.add_argument("--out", default=None, help="write calibration.json here")
    args = parser.parse_args(argv)

    rows = load_rows(args.labels, args.variants)

    if not rows:
        print(f"{args.labels}: no labelled rows with a score", file=sys.stderr)
        return 2

    results = sweep(rows)
    decision = choose(results)
    print(format_table(results))
    print()
    print(
        f"positives {sum(1 for r in rows if r['truth'] == 1)} · "
        f"negatives {sum(1 for r in rows if r['truth'] == 0)} · "
        f"corrupted twins included: {sum(1 for r in rows if r['kind'] == 'corrupted')}"
    )
    print(f"decision: {decision['reason']}")

    if decision.get("chosen"):
        best = decision["chosen"]
        print(
            f"chosen: rule={best['rule']} tau_agg={best['tau_agg']} tau_claim={best['tau_claim']} "
            f"(TPR {best['tpr']}, TNR {best['tnr']}, precision {best['precision']})"
        )
    elif decision.get("best"):
        best = decision["best"]
        print(
            f"best available: rule={best['rule']} tau_agg={best['tau_agg']} "
            f"tau_claim={best['tau_claim']} precision={best['precision']}"
        )

    if args.out:
        payload = {
            "rows": len(rows),
            "positives": sum(1 for r in rows if r["truth"] == 1),
            "negatives": sum(1 for r in rows if r["truth"] == 0),
            "current": CURRENT,
            "decision": decision,
            "candidates": results,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
