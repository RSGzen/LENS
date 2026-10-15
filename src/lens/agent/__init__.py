"""Host agent layer: query segmentation (M6) and, later, the orchestrator (M8).

Spec: ``MVP Build Roadmap.md`` section 4 (M6/M8); ``Thesis Drafts/Tech Stack
(Draft).md`` sections 1-2, 4a, 4e.

- :mod:`lens.agent.segmentation` -- M6: one ``gpt-5-mini`` structured-JSON call
  turns a raw query into the 4 retrieval buckets, emitted as ``query.json``.
"""

from lens.agent.segmentation import segment_query, write_query_json

__all__ = ["segment_query", "write_query_json"]
