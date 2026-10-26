r"""M7.3 KG-level ablation: ``off`` / ``raw`` / ``postprocess`` / ``verified``.

This measures the **KG arms** — triples, type/relation composition, cost, latency.
The RLM answer-quality ablation is part of the **M11 evaluation campaign** (it
requires the M8 harness); this is the KG-side companion and the evidence for
whether the KG earns its place.

Arms:
  off          ``kg_enabled=False`` (empty graph, no extraction)
  raw          extraction + ontology filter + dedup only
  postprocess  span filter + coref + suffix/gazetteer/relation typing + directionality
  verified     postprocess + budgeted JEV direction verifier

Run:
  .\.venv\Scripts\python.exe scripts\kg_ablation.py \
      --chunks D:\lens-data\cache\kg\chunks_C200.json --verify --verify-budget 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lens import config
from lens.db.connect import connect
from lens.kg_extract import ENTITY_LABEL_SETS, GlinerExtractor
from lens.observability.run_logging import setup_file_logging
from lens.tools.kg import build_kg


def _gpu_stats() -> dict:
    try:
        import torch
    except Exception:
        return {"device": "unavailable"}
    if torch.cuda.is_available():
        return {
            "device": torch.cuda.get_device_name(0),
            "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        }
    return {"device": "cpu"}


def _reset_peak_vram() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def main() -> None:
    setup_file_logging("lens.kg_ablation", "kg_ablation")

    parser = argparse.ArgumentParser(description="M7.3 KG-level ablation")
    parser.add_argument("--chunks", required=True, help="saved chunk-id JSON")
    parser.add_argument("--labels", choices=tuple(ENTITY_LABEL_SETS), default="descriptive")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument(
        "--verify", action="store_true", help="include the JEV-verified arm (live API)"
    )
    parser.add_argument("--verify-budget", type=int, default=config.KG_VERIFY_BUDGET)
    parser.add_argument("--out-dir", default=config.LENS_LOGS_ROOT_PATH)
    args = parser.parse_args()

    chunk_ids = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    print(
        f"ablation: chunks={args.chunks} ({len(chunk_ids)}) labels={args.labels} "
        f"min_score={args.min_score} verify={args.verify} "
        f"verify_budget={args.verify_budget}"
    )

    extractor = GlinerExtractor(
        entity_labels=ENTITY_LABEL_SETS[args.labels],
        threshold=args.min_score,
    )

    verifier = None
    if args.verify:
        from lens.clients import LensClient
        from lens.kg_verify import DirectionVerifier
        from lens.observability.trajectory import TraceCtx, Trajectory

        run_id = time.strftime("kgablation_%Y%m%d_%H%M%S")
        trajectory = Trajectory(run_id)
        client = LensClient(trajectory)
        trace = TraceCtx(trace_id=run_id, step_idx=0, actor="tool")
        verifier = DirectionVerifier(
            lambda state, questions: client.jev(state, questions, trace_ctx=trace),
            budget=args.verify_budget,
        )

    arms = [
        ("off", {"kg_enabled": False}),
        ("raw", {"postprocess": False, "span_filter": False, "coref": False, "verifier": None}),
        (
            "postprocess",
            {"postprocess": True, "span_filter": True, "coref": True, "verifier": None},
        ),
        (
            "verified",
            {"postprocess": True, "span_filter": True, "coref": True, "verifier": verifier},
        ),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []

    with connect(readonly=True) as conn:
        for name, kwargs in arms:
            _reset_peak_vram()
            t0 = time.perf_counter()
            result = build_kg(conn, chunk_ids, extractor, use_cache=False, **kwargs)
            elapsed = time.perf_counter() - t0

            result["stats"]["timing"] = {
                "overall_s": round(elapsed, 3),
                "chunks_per_s": round(len(chunk_ids) / elapsed, 2) if elapsed else 0.0,
            }
            result["stats"]["gpu"] = _gpu_stats()
            (out_dir / f"kg_ablation_{stamp}_{name}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            stats = result["stats"]
            summary = result["summary"]
            fired = {k: v for k, v in summary["relations"].items() if v}
            rows.append(
                {
                    "arm": name,
                    "chunks": stats["chunks"],
                    "raw": stats["triples_raw"],
                    "kept": stats["triples_kept"],
                    "types": summary["entity_types"],
                    "type_sources": summary.get("type_sources"),
                    "relations": fired,
                    "verify": stats.get("verify"),
                    "seconds": elapsed,
                    "chunks_per_s": stats["timing"]["chunks_per_s"],
                    "gpu": stats["gpu"],
                }
            )
            print(
                f"[{name}] chunks={stats['chunks']} raw={stats['triples_raw']} "
                f"kept={stats['triples_kept']} types={summary['entity_types']} "
                f"rels={fired} verify={stats.get('verify')} "
                f"t={elapsed:.1f}s"
            )

    summary_path = out_dir / f"kg_ablation_{stamp}_summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== KG ablation summary ===")
    for row in rows:
        print(
            f"{row['arm']:<11} kept={row['kept']:>4} raw={row['raw']:>4} "
            f"types={row['types']} rels={row['relations']} "
            f"verify={row['verify']} t={row['seconds']:.1f}s"
        )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
