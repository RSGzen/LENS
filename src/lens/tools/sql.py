"""Gated T3 read-only SQL escape hatch (dev / exploration only).

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4/§4i (tiering).

T3 exists so a developer can inspect the corpus ad hoc. It is **not** part of the
graded path: it is disabled by default and forced off when ``recorded=True`` so a
recorded run cannot bypass the ``k -> n -> C -> c`` funnel.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    """Return ``False`` for recorded runs, otherwise the developer opt-in."""
    raise NotImplementedError("M8 scaffold: implement sql_is_allowed")


def validate_sql(sql: str) -> str:
    """Return a stripped single read-only statement or raise ``ValueError``.

    Reject: blank SQL, multiple statements (``;`` followed by more text),
    non-``SELECT``/``WITH`` statements, and forbidden keywords.
    """
    raise NotImplementedError("M8 scaffold: implement validate_sql")


def query_sql(
    conn: Any,
    sql: str,
    params: Sequence[Any] | None = None,
    *,
    row_cap: int = config.SQL_ROW_CAP,
    timeout_ms: int = config.SQL_TIMEOUT_MS,
) -> list[dict[str, Any]]:
    """Execute one validated read-only query and return at most ``row_cap`` rows.

    Steps: validate; open a cursor; ``SET LOCAL statement_timeout``; set the
    transaction read-only; execute; fetch at most ``row_cap + 1`` to detect
    truncation; map rows to dicts by column name.
    """
    raise NotImplementedError("M8 scaffold: implement query_sql")
