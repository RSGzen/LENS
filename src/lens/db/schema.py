"""M2 — idempotent pgvector schema migration for the LENS database.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §3 (the authoritative ``papers`` /
``chunks`` figure schemas + HNSW indexes); ``MVP Build Roadmap.md`` §4 (M2).

Bring the DB up first (Docker Desktop must be running)::

    docker run -d --name lens-pg --restart unless-stopped ^
        -e POSTGRES_USER=lens -e POSTGRES_PASSWORD=lens -e POSTGRES_DB=lens ^
        -p 5433:5432 -v D:\\lens-data\\postgres:/var/lib/postgresql/data ^
        pgvector/pgvector:pg16

Host port **5433** (not 5432) is deliberate on this machine: a native
``postgresql-x64-17`` service already owns host 5432, so connections to
``localhost:5432`` never reach the container. ``LENS_DB_PORT=5433`` in ``.env``
points the harness at the published port; on a clean machine the default 5432
works and the flags can stay ``-p 5432:5432``.

This is a **one-time** container creation. Afterwards ``docker start lens-pg``
(or the ``--restart`` flag + Docker Desktop) is enough — do not re-run it, the
name would collide. A ``docker-compose.yaml`` covering the DB **and** the M8
sandbox is deferred to M8 (that is where a second service appears); one
container does not need it.

Then ``python scripts/setup_db.py`` (which calls :func:`apply_schema`).

The migration is **safe to run repeatedly** (``IF NOT EXISTS`` / a ``pg_roles``
guard) because M3/M4 and every fresh session may re-run it.
``tests/test_schema.py`` is the contract (needs the container, no API key).
"""

from __future__ import annotations

import psycopg
from psycopg import sql

from lens import config


def apply_schema(conn: psycopg.Connection) -> None:
    """Idempotently create the full LENS schema on ``conn``.

    ``conn`` is an admin/owner connection. Steps:

    1. ``CREATE EXTENSION IF NOT EXISTS vector;`` (pgvector; the
       ``pgvector/pgvector:pg16`` image ships it).
    2. ``CREATE TABLE IF NOT EXISTS papers`` with the columns (Tech Stack §3):
       ``arxiv_id TEXT PRIMARY KEY`` · ``title TEXT`` · ``authors TEXT`` ·
       ``categories TEXT[]`` · ``arxiv_doi TEXT`` · ``journal_ref TEXT`` ·
       ``current_ver TEXT`` · ``update_date DATE`` · ``abstract TEXT`` ·
       ``abstract_embedding vector(<config.EMBEDDING_DIM>)`` ·
       ``embedding_version TEXT``.
    3. ``CREATE TABLE IF NOT EXISTS chunks``:
       ``chunk_uuid UUID PRIMARY KEY`` (default ``gen_random_uuid()`` — core in
       PG13+, no extension) · ``arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id)`` ·
       ``section_type TEXT`` · ``chunk_text TEXT`` ·
       ``chunk_embedding vector(<config.EMBEDDING_DIM>)`` · ``token_count INT`` ·
       ``section_order INT`` · ``embedding_version TEXT``.
    4. HNSW indexes on both embeddings with cosine distance::

           CREATE INDEX IF NOT EXISTS papers_abstract_embedding_hnsw
               ON papers USING hnsw (abstract_embedding vector_cosine_ops);
           CREATE INDEX IF NOT EXISTS chunks_chunk_embedding_hnsw
               ON chunks USING hnsw (chunk_embedding vector_cosine_ops);

    5. The read-only role ``config.LENS_DB_RO_USER`` with ``LOGIN PASSWORD``
       ``config.LENS_DB_RO_PASSWORD``. ``CREATE ROLE`` has no ``IF NOT EXISTS``,
       so its presence is checked against ``pg_roles`` first; the name and
       password are composed with :mod:`psycopg.sql` (``Identifier`` /
       ``Literal``) — a role name is an identifier, not a bindable value.
    6. ``GRANT SELECT ON papers, chunks TO <role>`` — read access **only**. The
       grant is idempotent and runs on every call, so a pre-existing role keeps
       access even if the tables were dropped and recreated. No
       INSERT/UPDATE/DELETE is granted — that denial is the M2 verify clause;
       ``CONNECT``/``USAGE`` already come from PUBLIC defaults.

    Do not commit here; the caller controls the transaction (``autocommit`` or an
    explicit commit). Running this twice must raise nothing and change nothing.
    """

    embedding_dim = int(config.EMBEDDING_DIM)
    role_name = config.LENS_DB_RO_USER
    role_password = str(config.LENS_DB_RO_PASSWORD)

    # psycopg.sql composition keeps the role name an identifier and the password a
    # literal (a role name cannot be passed as a plain query parameter). The
    # vector width is a literal inside vector(...), not a column constraint.
    create_papers_query = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            categories TEXT[],
            arxiv_doi TEXT,
            journal_ref TEXT,
            current_ver TEXT,
            update_date DATE,
            abstract TEXT,
            abstract_embedding vector({dims}),
            embedding_version TEXT
        )
        """
    ).format(dims=sql.Literal(embedding_dim))

    create_chunks_query = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
            section_type TEXT,
            chunk_text TEXT,
            chunk_embedding vector({dims}),
            token_count INT,
            section_order INT,
            embedding_version TEXT
        )
        """
    ).format(dims=sql.Literal(embedding_dim))

    create_role_query = sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD {}").format(
        sql.Identifier(role_name), sql.Literal(role_password)
    )

    grant_select_query = sql.SQL("GRANT SELECT ON papers, chunks TO {}").format(
        sql.Identifier(role_name)
    )

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cur.execute(create_papers_query)
        cur.execute(create_chunks_query)

        cur.execute(
            "CREATE INDEX IF NOT EXISTS papers_abstract_embedding_hnsw "
            "ON papers USING hnsw (abstract_embedding vector_cosine_ops)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS chunks_chunk_embedding_hnsw "
            "ON chunks USING hnsw (chunk_embedding vector_cosine_ops)"
        )

        # Guard the role creation (CREATE ROLE has no IF NOT EXISTS).
        cur.execute("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = %s", (role_name,))
        if cur.fetchone() is None:
            cur.execute(create_role_query)

        # GRANT is idempotent: run it every time, so an already-existing role
        # regains SELECT if the tables were dropped and recreated.
        cur.execute(grant_select_query)


def drop_schema(conn: psycopg.Connection) -> None:
    """Reverse :func:`apply_schema` for tests/dev; idempotent.

    Drop in FK-safe order: ``DROP TABLE IF EXISTS chunks;`` then
    ``papers`` (chunks references papers), then ``DROP ROLE IF EXISTS
    <config.LENS_DB_RO_USER>``. Table drops remove that role's table-level
    grant, so the role drop should succeed; if PostgreSQL still objects, revoke
    the leftover privilege first.
    """
    role_name = config.LENS_DB_RO_USER

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS chunks")
        cur.execute("DROP TABLE IF EXISTS papers")

        cur.execute(
            sql.SQL("DROP ROLE IF EXISTS {}").format(
                sql.Identifier(role_name)
            )
        )


def tables_exist(conn: psycopg.Connection) -> dict[str, bool]:
    """Return ``{"papers": bool, "chunks": bool}`` for the current database.

    One lookup against ``information_schema.tables``; the M2 existence check used
    by tests and ``scripts/setup_db.py``.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name IN ('papers', 'chunks')"
        )
        found = {row[0] for row in cur.fetchall()}

    return {"papers": "papers" in found, "chunks": "chunks" in found}
