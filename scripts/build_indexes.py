r"""/ runner — (re)build the pgvector HNSW indexes after a bulk load.

Wiring only. pgvector recommends creating HNSW indexes **after** loading the
data: it keeps row inserts fast during the load and the one-shot build is
cheaper than maintaining the graph row-by-row.

The build is forced **single-process** (``max_parallel_maintenance_workers = 0``).
The ``lens-pg`` container's ``/dev/shm`` default is small, and a large
``maintenance_work_mem`` with parallel HNSW workers tries to resize a dynamic
shared-memory segment and fails with ``psycopg.errors.DiskFull`` (pgvector README:
match ``--shm-size`` to ``maintenance_work_mem``, or avoid parallel). Single-process
keeps ``maintenance_work_mem`` in the backend's local memory, so a larger value is
safe. 4 GB holds the graph for ~5 M x 512-d halfvec nodes without spilling.

**Storage matters as much as memory.** HNSW build is random-I/O bound. PGDATA on a
Windows bind mount (``-v D:\lens-data\postgres:/var/lib/postgresql/data``) is
served over the WSL2 host-filesystem boundary and is ~100x slower than a Docker
**named volume** (native ext4). On a bind mount a 4.7 M-row build can run for days;
on a named volume it is minutes-to-~1 h. See `Task Update Logs/.

Stdout (including the wall-clock line) is mirrored to a timestamped log under
``config.LENS_LOGS_ROOT_PATH`` for evidence.

Prereqs: the ``lens-pg`` container is running and the load has completed.

Run:  .\\.venv\\Scripts\\python.exe scripts\\build_indexes.py
"""

from __future__ import annotations

import time

from lens.db.connect import connect
from lens.db.schema import apply_schema
from lens.observability.run_logging import setup_file_logging


def main() -> None:
    setup_file_logging("lens.build_indexes", "build_indexes")

    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SET max_parallel_maintenance_workers = 0")
            cur.execute("SET maintenance_work_mem = '4GB'")

        start = time.perf_counter()
        apply_schema(conn)  # CREATE INDEX IF NOT EXISTS -> builds any missing HNSW index
        elapsed_s = time.perf_counter() - start

        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) "
                "FROM pg_indexes WHERE indexname LIKE '%_hnsw' ORDER BY indexname"
            )
            print("hnsw indexes:", cur.fetchall())
            cur.execute("SELECT count(*), count(abstract_embedding) FROM papers")
            print("papers rows, with_embedding:", cur.fetchone())
            cur.execute("SELECT count(*), count(chunk_embedding) FROM chunks")
            print("chunks rows, with_embedding:", cur.fetchone())

        print(f"index (re)build wall-clock: {elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
