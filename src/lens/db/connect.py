"""M2 — psycopg connection helper for the local LENS pgvector database.

Spec: ``MVP Build Roadmap.md`` §4 (M2); ``Thesis Drafts/Tech Stack (Draft).md``
§3 (store layer). Connection settings live in :mod:`lens.config` (env-overridable);
nothing here reads or logs the API key — this is the database, not OpenRouter.

Two roles exist (M2 verify clause):
  - **admin/owner** — the ``LENS_DB_USER`` connection; runs the migration.
  - **read-only ``lens_ro``** — used by the host tool layer on the RLM's behalf;
    SELECT only. The sandbox itself holds no DB credentials (LOG-32).
"""

from __future__ import annotations

from typing import Any

import psycopg

from lens import config


def connect(
    *,
    readonly: bool = False,
    autocommit: bool = False,
    **overrides: Any,
) -> psycopg.Connection:
    """
    Open a psycopg connection to the LENS database.

    Contract
    --------
    - ``readonly=False`` uses ``config.LENS_DB_USER`` / ``LENS_DB_PASSWORD``;
      ``readonly=True`` uses ``config.LENS_DB_RO_USER`` / ``LENS_DB_RO_PASSWORD``
      (the migration creates that role — see :func:`lens.db.schema.apply_schema`).
    - Host / port / dbname come from ``config.LENS_DB_HOST`` / ``_PORT`` / ``_NAME``.
    - ``overrides`` (``host``, ``port``, ``dbname``, ``user``, ``password``)
      replace those values — tests point at a throwaway database this way.
    - Return ``psycopg.connect(**conninfo, autocommit=autocommit)``. Do **not**
      open a cursor or run anything; the caller owns the connection lifecycle.

    Note: the DSN string carries a password in memory only. Never print it.
    """

    conninfo: dict[str, Any] = {
        "host": config.LENS_DB_HOST,
        "port": config.LENS_DB_PORT,
        "dbname": config.LENS_DB_NAME,
        "user": config.LENS_DB_RO_USER if readonly else config.LENS_DB_USER,
        "password": config.LENS_DB_RO_PASSWORD if readonly else config.LENS_DB_PASSWORD,
    }
    conninfo.update(overrides)

    return psycopg.connect(**conninfo, autocommit=autocommit)
