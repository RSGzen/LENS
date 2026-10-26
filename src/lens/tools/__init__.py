"""Host retrieval tools : the T1 funnel + T2 fetch-by-id.

- :func:`shallow_match` / :func:`rerank_papers` / :func:`deep_match` /
  :func:`rerank_chunks` — the graded ``k -> n -> C -> c`` funnel.
- :func:`fetch_papers` / :func:`fetch_chunks` — typed fetch-by-id look-ups (T2).

These run **host-side** on a DB connection; the RLM reaches them through
``tool(name, **kwargs)`` and never issues SQL itself.
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
