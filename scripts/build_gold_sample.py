r"""M7.2 — build a small gold-sample worksheet (K-04 triples or K-05 entities).

Triples (K-04):
  .\.venv\Scripts\python.exe scripts\build_gold_sample.py \
      --kg D:\lens-data\logs\kg_grid_....json --mode triples --n 100 --out gold_triples.csv

Entities (K-05):
  .\.venv\Scripts\python.exe scripts\build_gold_sample.py \
      --kg D:\lens-data\logs\kg_grid_....json --mode entities --n 20 \
      --chunks D:\lens-data\cache\kg\chunks_C200.json \
      --out gold_entities.csv --predictions gold_entities_predictions.json

The entity worksheet contains **text only**; the KG predictions are written to a
separate JSON so the annotator is not biased. Annotate ``gold_entities`` as
``Name:Type; Name:Type`` (Type optional).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lens.db.connect import connect
from lens.kg_gold import (
    entities_from_payload,
    load_payload,
    sample_items,
    stratified_sample,
    triples_from_payload,
    write_entities_worksheet,
    write_triples_worksheet,
)
from lens.tools.retrieval import fetch_chunks


def _entity_rows(chunk_ids: list[str]) -> list[dict]:
    with connect(readonly=True) as conn:
        rows = fetch_chunks(conn, chunk_ids)
    by_id = {row["chunk_uuid"]: row for row in rows}
    return [
        {
            "chunk_uuid": cid,
            "arxiv_id": by_id[cid]["arxiv_id"],
            "section_type": by_id[cid]["section_type"],
            "chunk_text": by_id[cid]["chunk_text"],
        }
        for cid in chunk_ids
        if cid in by_id
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a K-04/K-05 gold worksheet")
    parser.add_argument("--kg", required=True, help="saved KG JSON (probe/grid/cache)")
    parser.add_argument("--mode", choices=("triples", "entities"), default="triples")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunks", help="chunk-id JSON (entity mode)")
    parser.add_argument("--out", required=True, help="CSV worksheet path")
    parser.add_argument("--predictions", help="entity mode: predictions JSON path")
    args = parser.parse_args()

    payload = load_payload(args.kg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "triples":
        triples = triples_from_payload(payload)
        sampled = stratified_sample(triples, args.n, seed=args.seed)
        write_triples_worksheet(sampled, out)
        print(f"triples worksheet: {out} ({len(sampled)} rows)")
        print("annotation: fill gold_relation_ok / gold_head_type_ok / gold_tail_type_ok with y/n")
        return

    if not args.chunks:
        raise SystemExit("--chunks is required for entity mode")
    chunk_ids = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    sampled_ids = sample_items(chunk_ids, args.n, seed=args.seed)
    rows = _entity_rows(sampled_ids)
    write_entities_worksheet(rows, out)

    predictions_path = Path(args.predictions) if args.predictions else out.with_suffix(".predictions.json")
    predictions = entities_from_payload(payload)
    filtered = {cid: predictions.get(cid, []) for cid in sampled_ids}
    predictions_path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"entity worksheet: {out} ({len(rows)} chunks)")
    print(f"predictions (do not open while labelling): {predictions_path}")
    print("annotation: fill gold_entities as 'Name:Type; Name:Type' (Type optional)")


if __name__ == "__main__":
    main()
