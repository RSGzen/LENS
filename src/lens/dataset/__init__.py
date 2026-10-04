"""Dataset layer: the M1 arXiv cs.AI manifest (full 2015–2026 corpus).

Spec: ``MVP Build Roadmap.md`` §4 (M1), §2.1 (assets); ``Tech Stack (Draft).md`` §4f.

Exposes :func:`build_manifest` (single-pass scan -> ``manifest.json`` + a JSONL
checkpoint) and :func:`load_manifest`.
"""

from lens.dataset.manifest import build_manifest, load_manifest

__all__ = ["build_manifest", "load_manifest"]
