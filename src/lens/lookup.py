"""Read-only corpus lookups, run as a child process by the UI.

The UI process never opens a database connection: the Live page shells out to the
CLI for the same reason, and a page that connects would hold a pool open for as long
as a browser tab is. This module is that child. It reads two things and prints one
JSON object:

    python -m lens.lookup paper 2407.21049
    python -m lens.lookup chunk <chunk_uuid> [<chunk_uuid> ...]

Output shape, always one object, never a traceback:

    {"ok": true,  "found": [{"column": value, ...}], "missing": [...]}
    {"ok": false, "error": "..."}

``missing`` lists the ids that were asked for and are not in the table, so "no such
paper" is a result rather than an empty list a caller has to interpret.

Embedding columns are not selected. They are the largest thing in both tables, no
reader can use a 512-float vector, and returning them would make a one-row lookup
megabytes wide.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Sequence

#: Paper columns a reader can use. ``abstract_embedding`` is deliberately absent.
PAPER_COLUMNS: tuple[str, ...] = (
    "arxiv_id",
    "title",
    "authors",
    "categories",
    "arxiv_doi",
    "journal_ref",
    "current_ver",
    "update_date",
    "abstract",
    "embedding_version",
)

#: Chunk columns a reader can use; ``chunk_embedding`` is absent for the same reason.
CHUNK_COLUMNS: tuple[str, ...] = (
    "chunk_uuid",
    "arxiv_id",
    "section_type",
    "chunk_text",
    "token_count",
    "section_order",
    "embedding_version",
)

#: An arXiv id: ``YYMM.NNNN`` or ``YYMM.NNNNN``, optionally versioned.
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")

#: A chunk id as the corpus stores it.
CHUNK_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

#: One call is a person pasting ids; a larger list is a script, and it belongs in SQL.
MAX_CHUNK_IDS = 50

#: Characters of ``abstract`` and ``chunk_text`` returned. Both are selected whole in
#: the query and trimmed here, so the cap is a display decision rather than a query one.
TEXT_MAX_CHARS = 4000


def _rows(cur: Any) -> list[dict[str, Any]]:
    """Return the fetched rows as dicts keyed by column name."""
    columns = [column[0] for column in cur.description]

    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _jsonable(value: Any) -> Any:
    """Coerce a database value to something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    return str(value)


def _trim(record: dict[str, Any]) -> dict[str, Any]:
    """Drop embedding columns that slipped in, coerce values, cap long text."""
    trimmed: dict[str, Any] = {}

    for key, value in record.items():
        if key.endswith("_embedding"):
            continue

        text = _jsonable(value)

        if isinstance(text, str) and len(text) > TEXT_MAX_CHARS:
            text = text[:TEXT_MAX_CHARS] + f"… [{len(text) - TEXT_MAX_CHARS:,} more characters]"

        trimmed[key] = text

    return trimmed


def paper_row(conn: Any, arxiv_id: str) -> dict[str, Any] | None:
    """Return one paper's human-readable columns, or ``None`` when it is absent."""
    columns = ", ".join(PAPER_COLUMNS)

    with conn.cursor() as cur:
        cur.execute(f"SELECT {columns} FROM papers WHERE arxiv_id = %s", (arxiv_id,))
        rows = _rows(cur)

    return _trim(rows[0]) if rows else None


def chunk_rows(conn: Any, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Return the chunk rows for ``chunk_ids`` that exist, in the order asked."""
    if not chunk_ids:
        return []

    columns = ", ".join(CHUNK_COLUMNS)
    placeholders = ", ".join(["%s"] * len(chunk_ids))

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {columns} FROM chunks WHERE chunk_uuid IN ({placeholders})",
            tuple(chunk_ids),
        )
        rows = [_trim(row) for row in _rows(cur)]

    order = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}

    return sorted(rows, key=lambda row: order.get(str(row.get("chunk_uuid")), len(order)))


def _fail(message: str) -> dict[str, Any]:
    """Return the one failure shape the caller parses."""
    return {"ok": False, "error": message}


def lookup_paper(conn: Any, arxiv_id: str) -> dict[str, Any]:
    """Validate one arXiv id and return the lookup envelope."""
    cleaned = (arxiv_id or "").strip()

    if not ARXIV_ID_RE.match(cleaned):
        return _fail(f"{cleaned!r} is not an arXiv id (YYMM.NNNN, optionally versioned)")

    try:
        row = paper_row(conn, cleaned)
    except Exception as exc:  # noqa: BLE001 - the caller shows the message, never a traceback
        return _fail(f"query failed: {type(exc).__name__}")

    if row is None:
        return {"ok": True, "found": [], "missing": [cleaned]}

    return {"ok": True, "found": [row], "missing": []}


def lookup_chunks(conn: Any, chunk_ids: Sequence[str]) -> dict[str, Any]:
    """Validate a list of chunk ids and return the lookup envelope."""
    cleaned = [chunk_id.strip().lower() for chunk_id in chunk_ids if chunk_id.strip()]

    if not cleaned:
        return _fail("no chunk ids given")

    if len(cleaned) > MAX_CHUNK_IDS:
        return _fail(f"at most {MAX_CHUNK_IDS} ids per lookup, got {len(cleaned)}")

    invalid = [chunk_id for chunk_id in cleaned if not CHUNK_ID_RE.match(chunk_id)]

    if invalid:
        return _fail("not a chunk id: " + ", ".join(invalid[:3]))

    try:
        rows = chunk_rows(conn, cleaned)
    except Exception as exc:  # noqa: BLE001 - the caller shows the message, never a traceback
        return _fail(f"query failed: {type(exc).__name__}")

    found_ids = {str(row.get("chunk_uuid")) for row in rows}
    missing = [chunk_id for chunk_id in cleaned if chunk_id not in found_ids]

    return {"ok": True, "found": rows, "missing": missing}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, query, print one JSON object."""
    parser = argparse.ArgumentParser(prog="lens.lookup", description=__doc__)
    sub = parser.add_subparsers(dest="what", required=True)

    paper = sub.add_parser("paper", help="one paper by arXiv id")
    paper.add_argument("arxiv_id")

    chunk = sub.add_parser("chunk", help="chunks by id")
    chunk.add_argument("chunk_ids", nargs="+")

    args = parser.parse_args(argv)

    from lens.db.connect import connect

    try:
        with connect(readonly=True) as conn:
            if args.what == "paper":
                payload = lookup_paper(conn, args.arxiv_id)
            else:
                payload = lookup_chunks(conn, args.chunk_ids)
    except Exception as exc:  # noqa: BLE001 - a missing database is a message, not a crash
        payload = _fail(f"database unavailable: {type(exc).__name__}")

    json.dump(payload, sys.stdout, ensure_ascii=False, default=str)
    sys.stdout.write("\n")

    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
