"""Observability layer: content-addressed blobs + one-schema JSONL trajectory.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §5 (trajectory + blobs) and §4a
(Transport / Observability layering).
"""

from lens.observability.trajectory import Actor, CallType, TraceCtx, Trajectory

__all__ = ["Actor", "CallType", "TraceCtx", "Trajectory"]
