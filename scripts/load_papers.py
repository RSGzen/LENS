"""M3 runner — load the full manifest into ``papers``.

Wiring only: read the M1 manifest, embed every abstract, upsert.

Prereqs: the ``lens-pg`` container is running with the schema applied
(``python scripts/setup_db.py``); the M1 manifest exists under the manifests root.

Run:  .\\.venv\\Scripts\\python.exe scripts\\load_papers.py
"""

from __future__ import annotations

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

    with connect(autocommit=True) as conn:
        count = load_papers(conn, manifest["papers"], embedder)

    print(f"loaded {count} papers (embedding_version={config.EMBEDDING_VERSION})")


if __name__ == "__main__":
    main()
