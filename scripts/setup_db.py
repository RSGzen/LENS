"""M2 runner — apply the pgvector schema to the local LENS database.

Wiring only (no logic): open an admin connection and call :func:`apply_schema`.

Prereq: the ``lens-pg`` container is running (see ``lens/db/schema.py``).

Run:  .\\.venv\\Scripts\\python.exe scripts\\setup_db.py
"""

from __future__ import annotations

from lens.db.connect import connect
from lens.db.schema import apply_schema, tables_exist


def main() -> None:
    with connect(autocommit=True) as conn:
        apply_schema(conn)
        state = tables_exist(conn)
    print(f"schema applied: {state}")


if __name__ == "__main__":
    main()
