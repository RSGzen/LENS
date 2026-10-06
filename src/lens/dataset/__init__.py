"""Dataset layer: the M1 arXiv cs.AI manifest + the M3 full-corpus paper load.

Spec: ``MVP Build Roadmap.md`` §4 (M1, M3), §2.1 (assets); ``Tech Stack (Draft).md``
§3 (``papers`` columns), §4f (manifest), §4g (DB layer).

- :func:`build_manifest` / :func:`load_manifest` — the M1 manifest (single-pass
  scan -> ``manifest.json`` + a JSONL checkpoint).
- :func:`load_papers` — the M3 full-manifest load into ``papers`` with abstract
  embeddings (no subset; advisor decision 6 Oct 2026).
"""

from lens.dataset.manifest import build_manifest, load_manifest
from lens.dataset.papers import load_papers

__all__ = ["build_manifest", "load_manifest", "load_papers"]
