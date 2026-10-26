"""Database layer : pgvector connection + schema migration.

Exposes:
  - :func:`connect` — open a psycopg connection (admin or read-only ``lens_ro``).
  - :func:`apply_schema` / :func:`drop_schema` — idempotent migration + teardown.
  - :func:`tables_exist` — the existence check.
"""

from lens.db.connect import connect
from lens.db.schema import apply_schema, drop_schema, tables_exist

__all__ = ["connect", "apply_schema", "drop_schema", "tables_exist"]
