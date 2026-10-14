"""Host retrieval tools (M5): the T1 funnel + T2 fetch-by-id.

Spec: ``MVP Build Roadmap.md`` §4 (M5); ``Thesis Drafts/Tech Stack (Draft).md``
§2 (stages 10-11), §3 (HNSW tuning), §4i (retrieval / hydration / asset contract).

- :func:`shallow_match` / :func:`rerank_papers` / :func:`deep_match` /
  :func:`rerank_chunks` — the graded ``k -> n -> C -> c`` funnel.
- :func:`fetch_papers` / :func:`fetch_chunks` — typed fetch-by-id look-ups (T2).

These run **host-side** on a DB connection; the RLM reaches them through
``tool(name, **kwargs)`` and never issues SQL itself (§4i, LOG-32).
"""

from lens.tools.retrieval import (
    deep_match,
    fetch_chunks,
    fetch_papers,
    rerank_chunks,
    rerank_papers,
    shallow_match,
)

__all__ = [
    "deep_match",
    "fetch_chunks",
    "fetch_papers",
    "rerank_chunks",
    "rerank_papers",
    "shallow_match",
]
