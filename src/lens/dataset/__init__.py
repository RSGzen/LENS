"""Dataset layer: the arXiv cs.AI manifest + the full-corpus paper load.

- :func:`build_manifest` / :func:`load_manifest` — the manifest (single-pass
  scan -> ``manifest.json`` + a JSONL checkpoint).
- :func:`load_papers` — the full-manifest load into ``papers`` with abstract
  embeddings (no subset;.
"""

from lens.dataset.manifest import build_manifest, load_manifest
from lens.dataset.papers import load_papers

__all__ = ["build_manifest", "load_manifest", "load_papers"]
