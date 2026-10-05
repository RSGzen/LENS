"""Database layer (M2): pgvector connection + schema migration.

Spec: ``MVP Build Roadmap.md`` §4 (M2); ``Thesis Drafts/Tech Stack (Draft).md``
§3 (the authoritative ``papers`` / ``chunks`` schema).

Exposes:
  - :func:`connect` — open a psycopg connection (admin or read-only ``lens_ro``).
  - :func:`apply_schema` / :func:`drop_schema` — idempotent migration + teardown.
  - :func:`tables_exist` — the M2 existence check.
"""

from lens.db.connect import connect
from lens.db.schema import apply_schema, drop_schema, tables_exist

__all__ = ["connect", "apply_schema", "drop_schema", "tables_exist"]
