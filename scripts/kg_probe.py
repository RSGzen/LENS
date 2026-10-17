r"""M7 K-05 probe — GLiNER-family recall/quality sample over a small paper set.

Wiring only: read-only connect (``lens_ro``), pick a deterministic set of ``--papers``
(default 20) papers, pull their chunks, and run ``build_kg`` through the configured
extractor. Prints per-stage timing, triple/entity/relation counts, and a sample of
triples, and dumps the full node-link graph to a timestamped JSON under
``config.LENS_LOGS_ROOT_PATH`` for manual gold comparison.

The K-05 decision (Roadmap §4 M7): if recall on the sample is low, switch the
extractor variant (pipeline <-> joint) or fall back to few-shot LLM extraction; the
losing dependency family is then removed.

Prereqs: ``lens-pg`` running with M3+M4 loaded. This is a **live** probe (loads
GLiNER/GLiREL; no API key, zero cost).

Run:  .\.venv\Scripts\python.exe scripts\kg_probe.py --papers 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lens import config
from lens.db.connect import connect
from lens.kg_extract import GlinerExtractor
from lens.observability.run_logging import setup_file_logging
from lens.tools.kg import build_kg

DEFAULT_PAPERS = 20


def _sample_chunk_ids(conn, papers: int) -> list[str]:
    """Deterministic sample: the first ``papers`` arxiv_ids, all their chunks."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT arxiv_id FROM papers ORDER BY arxiv_id LIMIT %s",
            (papers,),
        )
        arxiv_ids = [row[0] for row in cur.fetchall()]

        cur.execute(
            "SELECT chunk_uuid::text FROM chunks "
            "WHERE arxiv_id = ANY(%s) ORDER BY arxiv_id, section_order",
            (arxiv_ids,),
        )
        return [row[0] for row in cur.fetchall()]


def main() -> None:
    setup_file_logging("lens.kg_probe", "kg_probe")

    parser = argparse.ArgumentParser(description="M7 K-05 KG extraction probe")
    parser.add_argument("--papers", type=int, default=DEFAULT_PAPERS)
    args = parser.parse_args()

    print(
        f"probe: papers={args.papers} | ner={config.KG_NER_MODEL_ID} "
        f"rel={config.KG_REL_MODEL_ID} | ontology={config.KG_ONTOLOGY_VERSION} "
        f"| min_score={config.KG_MIN_SCORE}"
    )

    extractor = GlinerExtractor()

    with connect(readonly=True) as conn:
        chunk_ids = _sample_chunk_ids(conn, args.papers)
        print(f"sample: {len(chunk_ids)} chunks over {args.papers} papers")

        t0 = time.perf_counter()
        result = build_kg(conn, chunk_ids, extractor)
        elapsed = time.perf_counter() - t0

    stats = result["stats"]
    summary = result["summary"]
    print(f"build_kg: {elapsed:.1f}s  kg_id={result['kg_id']}  stats={stats}")
    print(f"entities by type: {summary['entity_types']}")
    print(f"relations: {summary['relations']}")
    print("top entities:")
    for node in summary["top_entities"]:
        print(f"  {node['degree']:>4}  {node['type']:<8}  {node['entity']}")

    triples = [
        edge
        for edge in result["graph"].get("links", result["graph"].get("edges", []))
    ]
    print(f"triples: {len(triples)}")
    for edge in triples[:20]:
        print(f"  {edge}")

    out_path = Path(config.LENS_LOGS_ROOT_PATH) / f"kg_probe_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
