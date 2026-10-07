"""M3/M5 runner — (re)build the pgvector HNSW indexes after a bulk load.

Wiring only. pgvector recommends creating HNSW indexes **after** loading the
data: it keeps row inserts fast during the load and the one-shot build is
cheaper than maintaining the graph row-by-row.

The build is forced **single-process** (``max_parallel_maintenance_workers = 0``).
The ``lens-pg`` container's ``/dev/shm`` is small (Docker default 64 MB), and a
large ``maintenance_work_mem`` with parallel HNSW workers tries to resize a
dynamic shared-memory segment and fails with ``psycopg.errors.DiskFull``
(pgvector README: match ``--shm-size`` to ``maintenance_work_mem``, or avoid
parallel). Single-process keeps ``maintenance_work_mem`` in the backend's local
memory, so a larger value is safe.

Prereqs: the ``lens-pg`` container is running and the load has completed.

Run:  .\\.venv\\Scripts\\python.exe scripts\\build_indexes.py
"""

from __future__ import annotations

from lens.db.connect import connect
from lens.db.schema import apply_schema


def main() -> None:
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SET max_parallel_maintenance_workers = 0")
            cur.execute("SET maintenance_work_mem = '1GB'")
        apply_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT indexname FROM pg_indexes WHERE indexname LIKE '%_hnsw'")
            print("hnsw indexes:", [row[0] for row in cur.fetchall()])
            cur.execute("SELECT count(*), count(abstract_embedding) FROM papers")
            print("rows, with_embedding:", cur.fetchone())


if __name__ == "__main__":
    main()
