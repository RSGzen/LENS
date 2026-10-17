r"""M7.2 — score an annotated gold-sample worksheet.

Triples (K-04):
  .\.venv\Scripts\python.exe scripts\score_gold.py \
      --mode triples --csv gold_triples.csv --out gold_triples_report.json

Entities (K-05):
  .\.venv\Scripts\python.exe scripts\score_gold.py \
      --mode entities --csv gold_entities.csv \
      --predictions gold_entities_predictions.json --out gold_entities_report.json

Only rows with a filled ``gold_relation_ok`` (triples) / ``gold_entities`` (entities)
are scored; blank rows are ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lens.kg_gold import (
    read_entities_worksheet,
    read_triples_worksheet,
    score_entities,
    score_triples,
)


def _print_triples(report: dict) -> None:
    lo, hi = report["triple_precision_ci"]
    print(f"n={report['n']}")
    print(f"triple precision: {report['triple_precision']:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    print(f"relation precision: {report['relation_precision']:.3f}")
    print(f"head-type precision: {report['head_type_precision']:.3f}")
    print(f"tail-type precision: {report['tail_type_precision']:.3f}")
    print("by relation:")
    for relation, values in report["by_relation"].items():
        print(f"  {values['correct']:>3}/{values['n']:<3} {values['precision']:.3f}  {relation}")


def _print_entities(report: dict) -> None:
    print(f"gold entities: {report['gold_entities']} | predicted: {report['predicted_entities']}")
    print(f"recall (typed): {report['recall_typed']:.3f}")
    print(f"recall (untyped): {report['recall_untyped']:.3f}")
    print(f"precision (untyped): {report['precision_untyped']:.3f}")
    print("by type:")
    for kind, values in report["by_type"].items():
        print(f"  {values['matched']:>3}/{values['gold']:<3} {values['recall']:.3f}  {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a gold-sample worksheet")
    parser.add_argument("--mode", choices=("triples", "entities"), default="triples")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--predictions", help="entity mode: predictions JSON")
    parser.add_argument("--out", help="optional report JSON path")
    args = parser.parse_args()

    if args.mode == "triples":
        report = score_triples(read_triples_worksheet(args.csv))
        _print_triples(report)
    else:
        if not args.predictions:
            raise SystemExit("--predictions is required for entity mode")
        predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
        report = score_entities(read_entities_worksheet(args.csv), predictions)
        _print_entities(report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
