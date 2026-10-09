r"""M4 pre-run benchmark — measure real chunks/s before the full-corpus load.

Why this exists: the M3 full run settled at ~23 papers/s vs a 114.8/s short
benchmark (≈ 5x gap; GPU saturated at 100%). The full M4 load could range from
~10 h to ~3 days depending on throughput, so measure on a sample first and use
the projection to decide whether to optimise the embed path (Roadmap §4 M3/M4,
§10; M3 evidence row 29).

Wiring only. Measures, over ``SAMPLE_SIZE`` manifest papers:
  * parse + tokenize + chunk  -> papers/s, chunks/s;
  * embed the produced chunks -> chunks/s, tokens/s (nomic path);
and extrapolates to the full manifest (chunk volume measured, not assumed).

Prereqs: the M1 manifest exists; M3 deps installed. No DB writes.

Run:  .\\.venv\\Scripts\\python.exe scripts\\benchmark_chunks.py
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from lens import config
from lens.chunking import chunk_tei
from lens.dataset.manifest import load_manifest
from lens.embeddings import Embedder

SAMPLE_SIZE = 200


def main() -> None:
    manifest = load_manifest(
        Path(config.LENS_MANIFESTS_ROOT_PATH) / config.MANIFEST_FILENAME
    )
    papers = manifest["papers"]
    total_papers = manifest["count"]

    random.seed(config.PAPER_START_YEAR)  # stable sample across runs
    sample = random.sample(papers, min(SAMPLE_SIZE, len(papers)))

    embedder = Embedder()
    tokenizer = embedder.model.tokenizer

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))

    # --- parse + chunk (CPU) ------------------------------------------------
    xml_dir = Path(config.GROBID_XML_DIR_PATH)
    t0 = time.perf_counter()
    chunks = []
    for entry in sample:
        chunks.extend(
            chunk_tei(
                xml_dir / entry["tei_xml"],
                count_tokens,
                config.CHUNK_TOKEN_THRESHOLD,
                config.CHUNK_OVERLAP_TOKENS,
            )
        )
    parse_s = time.perf_counter() - t0

    if not chunks:
        raise SystemExit("no chunks produced — is the chunking logic implemented?")

    texts = [c.chunk_text for c in chunks]
    total_tokens = sum(c.token_count for c in chunks)

    # --- embed (GPU) --------------------------------------------------------
    t1 = time.perf_counter()
    embedder.encode_documents(
        texts,
        task_prefix=config.EMBEDDING_DOC_PREFIX,
        batch_size=config.EMBEDDING_CHUNK_BATCH_SIZE,
    )
    embed_s = time.perf_counter() - t1

    chunks_per_paper = len(chunks) / len(sample)
    est_chunks = chunks_per_paper * total_papers

    print(f"sample papers          : {len(sample)}")
    print(f"chunks                 : {len(chunks)}  ({chunks_per_paper:.1f}/paper)")
    print(f"tokens (sample)        : {total_tokens}")
    print(f"parse+chunk            : {parse_s:.2f}s  ({len(sample) / parse_s:.1f} papers/s)")
    print(f"embed                  : {embed_s:.2f}s  ({len(chunks) / embed_s:.1f} chunks/s, "
          f"{total_tokens / embed_s:.0f} tokens/s)")
    print(f"batch size             : {config.EMBEDDING_CHUNK_BATCH_SIZE} chunks")
    print(f"est. full chunk volume : {est_chunks:,.0f} chunks")
    print(f"est. embed-only time   : {est_chunks / (len(chunks) / embed_s) / 3600:.1f} h")


if __name__ == "__main__":
    main()
