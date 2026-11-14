"""Turn OOLONG release rows into bridge instances and their documents.

The release gives one **context window** per row - a single string of
`Date: ... || User: ... || Instance: ...` lines - and one question about it. The
bridge needs two things a row does not carry: a document the flat arm can be
handed inline, and a record per line so the loop can count over the file.

``split_context`` is the whole conversion: one record per non-empty line, with the
intro line kept as ``rec-0000`` so the byte content survives except for the
newline delimiter. ``check_context`` is what makes that claim checkable - it
refuses a context whose record count disagrees with its line count, or whose text
carries a ``|| Label:`` fragment (which would hand the model the answer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from lens.agent.runtime import count_tokens

INSTRUMENT = "oolong-bridge"
LABEL_FRAGMENT = "|| Label:"
#: The flat document arm's inline budget; a context over it would be truncated and
#: the two arms would stop seeing the same evidence.
FLAT_DOCUMENT_TOKENS = 140_000


def split_context(text: str) -> list[dict[str, str]]:
    """Return one ``{id, text, label}`` record per non-empty line of ``text``."""
    records: list[dict[str, str]] = []

    for index, line in enumerate(line for line in text.splitlines() if line.strip()):
        records.append({"id": f"rec-{index:04d}", "text": line, "label": ""})

    return records


def check_context(text: str, records: list[Mapping[str, str]]) -> None:
    """Raise when the records do not account for every line, or leak a label."""
    lines = [line for line in text.splitlines() if line.strip()]

    if len(records) != len(lines):
        raise ValueError(f"{len(records)} records for {len(lines)} lines")

    joined = "\n".join(str(record["text"]) for record in records)

    if LABEL_FRAGMENT in joined:
        raise ValueError("context carries a label fragment; the labelled variant was used")

    if joined != "\n".join(lines):
        raise ValueError("records do not reproduce the context's non-empty lines")

    tokens = count_tokens(joined)

    if tokens > FLAT_DOCUMENT_TOKENS:
        raise ValueError(f"context is {tokens} tokens, over the {FLAT_DOCUMENT_TOKENS} budget")


def document_name(context_window_id: int) -> str:
    """Return the document file name for one context window."""
    return f"oolong-ctx-{context_window_id}.jsonl"


def build_instance(
    row: Mapping[str, Any],
    *,
    document_file: str,
    subset: bool,
    seed: int,
) -> dict[str, Any]:
    """Return one instrument line for a release row.

    ``query`` is the release's question **verbatim**: the scorer parses the answer
    against that wording, so any edit would change what is being measured.
    """
    return {
        "instance_id": f"oolong-tc-{row['id']}",
        "instrument": INSTRUMENT,
        "query": row["question"],
        "evidence": {"kind": "doc_chunks", "file": document_file},
        "gold": {"answer": row["answer"], "claims": []},
        "answer_type": row["answer_type"],
        "answer_format": "free",
        "meta": {
            "dataset": row["dataset"],
            "task": row["task"],
            "context_len": int(row["context_len"]),
            "context_window_id": int(row["context_window_id"]),
            "num_labels": int(row["num_labels"]),
            "subset": subset,
            "seed": seed,
        },
    }


def write_document(records: list[Mapping[str, str]], path: str | Path) -> Path:
    """Write a document file: one JSON object per line, UTF-8, no BOM."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return target
