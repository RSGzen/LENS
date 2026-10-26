"""Host agent layer: query segmentation and, later, the orchestrator.

- :mod:`lens.agent.segmentation` -- : one ``gpt-5-mini`` structured-JSON call
  turns a raw query into the 4 retrieval buckets, emitted as ``query.json``.
"""

from lens.agent.segmentation import segment_query, write_query_json

__all__ = ["segment_query", "write_query_json"]
