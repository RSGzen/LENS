"""M3 runner — load the full manifest into ``papers``.

Wiring only: read the M1 manifest, embed every abstract, upsert.

Prereqs: the ``lens-pg`` container is running with the schema applied
(``python scripts/setup_db.py``); the M1 manifest exists under the manifests root.

Run:  .\\.venv\\Scripts\\python.exe scripts\\load_papers.py
"""

from __future__ import annotations

import time
from pathlib import Path

from lens import config
from lens.dataset.manifest import load_manifest
from lens.dataset.papers import load_papers
from lens.db.connect import connect
from lens.embeddings import Embedder


def main() -> None:
    manifest_path = Path(config.LENS_MANIFESTS_ROOT_PATH) / config.MANIFEST_FILENAME
    manifest = load_manifest(manifest_path)

    embedder = Embedder()

    process_chunk_size = config.PROCESS_CHUNK_SIZE
    embedding_batch_size = config.EMBEDDING_ABSTRACT_BATCH_SIZE

    start = time.perf_counter()
    with connect(autocommit=True) as conn:
        count = load_papers(conn=conn,
                            entries=manifest["papers"],
                            process_chunk_size=process_chunk_size,
                            encode_batch_size=embedding_batch_size,
                            embedder=embedder)
    elapsed_s = time.perf_counter() - start

    rate = count / elapsed_s if elapsed_s > 0 else 0.0
    print(
        f"loaded {count} papers in {elapsed_s:.1f}s "
        f"({rate:.1f} papers/s; embedding_version={config.EMBEDDING_VERSION})"
    )


if __name__ == "__main__":
    main()
