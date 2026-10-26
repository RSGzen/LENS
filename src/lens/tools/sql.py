"""Gated T3 read-only SQL escape hatch (dev / exploration only).

T3 exists so a developer can inspect the corpus ad hoc. It is **not** part of the
graded path: it is disabled by default and forced off when ``recorded=True`` so a
recorded run cannot bypass the ``k -> n -> C -> c`` funnel.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from lens import config

# Only read-only statements are permitted. The real boundary is the `lens_ro`
# role + a read-only transaction; this is a cheap first line of defence.
SQL_ALLOWED_PREFIXES: tuple[str, ...] = ("select", "with")
SQL_FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "call",
    "do",
)


def sql_is_allowed(
    *,
    recorded: bool = config.LENS_RECORDED,
    allow_sql: bool = config.ALLOW_SQL,
) -> bool:
    """Return ``False`` for recorded runs, otherwise the developer opt-in.

    Recorded wins over the opt-in: a recorded run is an experiment, and an
    experiment that can bypass the ``k -> n -> C -> c`` funnel is not the
    experiment the describes.
    """
    return bool(allow_sql) and not recorded


def validate_sql(sql: str) -> str:
    """Return a stripped single read-only statement or raise ``ValueError``.

    Reject: blank SQL, multiple statements (``;`` followed by more text),
    non-``SELECT``/``WITH`` statements, and forbidden keywords.
    """
    text = (sql or "").strip()

    if not text:
        raise ValueError("empty SQL")

    # One trailing semicolon is harmless; anything after a semicolon is a second
    # statement, which is how "SELECT 1; DROP TABLE papers" gets in.
    statement = text[:-1].strip() if text.endswith(";") else text

    if ";" in statement:
        raise ValueError("only one statement is allowed")

    lowered = statement.lower()

    if not lowered.startswith(SQL_ALLOWED_PREFIXES):
        raise ValueError(
            f"only {' / '.join(prefix.upper() for prefix in SQL_ALLOWED_PREFIXES)} "
            "statements are allowed"
        )

    for keyword in SQL_FORBIDDEN_KEYWORDS:
        # Word boundaries: a column called ``updated_at`` must not trip ``update``.
        if re.search(rf"\b{keyword}\b", lowered):
            raise ValueError(f"forbidden keyword {keyword.upper()!r} in SQL")

    return statement


def query_sql(
    conn: Any,
    sql: str,
    params: Sequence[Any] | None = None,
    *,
    row_cap: int = config.SQL_ROW_CAP,
    timeout_ms: int = config.SQL_TIMEOUT_MS,
) -> list[dict[str, Any]]:
    """Execute one validated read-only query and return at most ``row_cap`` rows.

    ``conn`` must be the **``lens_ro``** connection (``db.connect(readonly=True)``)
    with ``autocommit=False``: ``SET TRANSACTION READ ONLY`` is a no-op under
    autocommit, and the role's grants are the real boundary — the prefix/ keyword
    checks below are only a cheap first line of defence.

    Steps: validate; open a cursor; ``SET LOCAL statement_timeout``; set the
    transaction read-only; execute; fetch at most ``row_cap + 1`` to detect
    truncation; map rows to dicts by column name.

    The row cap is enforced by **not fetching** beyond it rather than by trusting
    the query: ``fetchmany(row_cap)`` bounds what this process ever holds, so a
    ``SELECT *`` on a 4.6 M-row table cannot exhaust memory.
    """
    statement = validate_sql(sql)
    limit = max(1, int(row_cap))

    with conn.cursor() as cursor:
        # First statement in the transaction. ``set_config(..., true)`` is the
        # parameterisable form of ``SET LOCAL`` — ``SET`` itself takes no binds.
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute("SELECT set_config('statement_timeout', %s, true)", (str(int(timeout_ms)),))
        cursor.execute(statement, tuple(params) if params else None)

        columns = [
            getattr(description, "name", str(description))
            for description in cursor.description or []
        ]
        fetched = cursor.fetchmany(limit)

    rows: list[dict[str, Any]] = []

    for row in fetched:
        if isinstance(row, Mapping):
            rows.append(dict(row))
        else:
            rows.append(dict(zip(columns, row)))

    return rows
