r""" experiment grid (b)/(c) — extractor labels/threshold/models x chunk sets.

Does no sampling itself: pass one or more saved chunk-id JSON files produced by
``kg_probe.py --save-chunks`` (e.g. the ``C=100`` and ``C=200`` query sets). For each
(chunk set x NER model x label set x min_score) it runs a fresh ``build_kg`` (cache
bypassed), prints a compact row, and writes each result JSON for ``kg_inspect.py``.

Run:
  .\.venv\Scripts\python.exe scripts\kg_grid.py --chunks chunks_C100.json chunks_C200.json
  .\.venv\Scripts\python.exe scripts\kg_grid.py --chunks chunks_C200.json \
      --labels descriptive scierc --min-scores 0.3 0.5 \
      --ner-models urchade/gliner_medium-v2.1 urchade/gliner_large-v2.1
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
    setup_file_logging("lens.kg_grid", "kg_grid")

    parser = argparse.ArgumentParser
    parser.add_argument(
        "--chunks",
        nargs="+",
        required=True,
        help="saved chunk-id JSON files (one per retrieval scope)",
    )
    parser.add_argument("--ner-models", nargs="+", default=[config.KG_NER_MODEL_ID])
    parser.add_argument("--rel-model", default=config.KG_REL_MODEL_ID)
    parser.add_argument(
        "--labels", nargs="+", choices=tuple(ENTITY_LABEL_SETS), default=list(ENTITY_LABEL_SETS)
    )
    parser.add_argument("--min-scores", nargs="+", type=float, default=[0.3, 0.5])
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="disable the post-process (pre/post comparison)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="enable budgeted JEV direction verification (live API)",
    )
    parser.add_argument("--verify-budget", type=int, default=config.KG_VERIFY_BUDGET)
    parser.add_argument("--out-dir", default=config.LENS_LOGS_ROOT_PATH)
    args = parser.parse_args()
    postprocess = not args.no_postprocess

    print(
        f"grid: chunks={args.chunks} ner={args.ner_models} rel={args.rel_model}\n"
        f"      labels={args.labels} min_scores={args.min_scores} "
        f"postprocess={postprocess} verify={args.verify} "
        f"verify_budget={args.verify_budget}"
    )

    verifier = None
    if args.verify:
        from lens.clients import LensClient
        from lens.kg_verify import DirectionVerifier
        from lens.observability.trajectory import TraceCtx, Trajectory

        run_id = time.strftime("kggrid_%Y%m%d_%H%M%S")
        trajectory = Trajectory(run_id)
        client = LensClient(trajectory)
        trace = TraceCtx(trace_id=run_id, step_idx=0, actor="tool")
        verifier = DirectionVerifier(
            lambda state, questions: client.jev(state, questions, trace_ctx=trace),
            budget=args.verify_budget,
        )
        print(f"verify: JEV run={run_id}")

    # Load each checkpoint pair once; share the models across label/threshold arms.
    model_cache: dict[tuple[str, str], tuple[object, object]] = {}

    def models(ner_id: str, rel_id: str):
        key = (ner_id, rel_id)
        if key not in model_cache:
            base = GlinerExtractor(ner_model_id=ner_id, rel_model_id=rel_id)
            _ = base.ner_model  # trigger download/load
            _ = base.rel_model
            model_cache[key] = (base.ner_model, base.rel_model)
        return model_cache[key]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []

    with connect(readonly=True) as conn:
        for chunk_file in args.chunks:
            chunk_ids = json.loads(Path(chunk_file).read_text(encoding="utf-8"))
            for ner_id in args.ner_models:
                ner, rel = models(ner_id, args.rel_model)
                for label_name in args.labels:
                    for min_score in args.min_scores:
                        extractor = GlinerExtractor(
                            ner_model_id=ner_id,
                            rel_model_id=args.rel_model,
                            entity_labels=ENTITY_LABEL_SETS[label_name],
                            threshold=min_score,
                            ner_model=ner,
                            rel_model=rel,
                        )
                        _reset_peak_vram()
                        t0 = time.perf_counter()
                        result = build_kg(
                            conn,
                            chunk_ids,
                            extractor,
                            use_cache=False,
                            postprocess=postprocess,
                            verifier=verifier,
                        )
                        elapsed = time.perf_counter() - t0

                        result["stats"]["timing"] = {
                            "overall_s": round(elapsed, 3),
                            "chunks_per_s": round(len(chunk_ids) / elapsed, 2) if elapsed else 0.0,
                        }
                        result["stats"]["gpu"] = _gpu_stats()

                        tag = (
                            f"{Path(chunk_file).stem}__{Path(ner_id).name}"
                            f"__{label_name}__{min_score}"
                        )
                        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)
                        out_path = out_dir / f"kg_grid_{stamp}_{len(rows):02d}_{safe}.json"
                        out_path.write_text(
                            json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )

                        stats = result["stats"]
                        summary = result["summary"]
                        fired = {k: v for k, v in summary["relations"].items() if v}
                        rows.append(
                            {
                                "tag": tag,
                                "chunks": stats["chunks"],
                                "raw": stats["triples_raw"],
                                "kept": stats["triples_kept"],
                                "types": summary["entity_types"],
                                "type_sources": summary.get("type_sources"),
                                "relations": fired,
                                "seconds": elapsed,
                                "chunks_per_s": stats["timing"]["chunks_per_s"],
                                "gpu": stats["gpu"],
                                "verify": stats.get("verify"),
                                "file": out_path.name,
                            }
                        )
                        print(
                            f"[{len(rows):02d}] {tag}: chunks={stats['chunks']} "
                            f"raw={stats['triples_raw']} kept={stats['triples_kept']} "
                            f"types={summary['entity_types']} rels={fired} "
                            f"sources={summary.get('type_sources')} "
                            f"verify={stats.get('verify')} "
                            f"t={elapsed:.1f}s ({stats['timing']['chunks_per_s']:.1f} chunks/s) "
                            f"gpu={stats['gpu'].get('device')} "
                            f"peak={stats['gpu'].get('vram_peak_mb')}MB"
                        )

    print("\n=== summary (kept triples) ===")
    for row in rows:
        print(
            f"{row['kept']:>4} kept | {row['chunks']:>3} chunks | "
            f"{row['types']} | {row['relations']} | {row['tag']}"
        )


if __name__ == "__main__":
    main()
