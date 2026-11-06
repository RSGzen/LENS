"""Extract claim frames from answer sentences with the sub model.

The extractor is host-side. It is not a root tool: the root writes prose, and
the ledger is built from that prose after the fact so a model cannot grade its
own answer. The host still validates everything the model proposes: an id the
unit did not cite is dropped, and a quote the chunk does not contain is
unverifiable.

The reply is one strict JSON object. Batching is by claim unit, not by chunk:
six sentences fit in the reply cap and one call keeps the latency down.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from lens.observability.claims import PREDICATES, ClaimFrame, parse_frames

#: Sentence units one call carries. Six short sentences fit the input and the
#: output cap; a batch that grows past this truncates the reply mid-object.
DEFAULT_BATCH_SIZE = 6

#: Output tokens for one batch. The model's reasoning shares this budget, so the
#: cap carries the frames plus the thinking that precedes them.
DEFAULT_MAX_TOKENS = 8000

SYSTEM = (
    "You decompose a claim sentence into typed frames. Use only the text and "
    "the chunk ids given. Copy each quote character for character from the "
    "cited chunk. Reply with JSON only."
)


def _schema() -> dict[str, Any]:
    """Return the strict schema for one extraction reply."""
    properties = {
        "unit_index": {"type": "integer"},
        "predicate": {"type": "string", "enum": list(PREDICATES)},
        "subject": {"type": "string"},
        "object": {"type": ["string", "null"]},
        "benchmark": {"type": ["string", "null"]},
        "metric": {"type": ["string", "null"]},
        "value": {"type": ["string", "null"]},
        "unit": {"type": ["string", "null"]},
        "direction": {"type": ["string", "null"], "enum": [">", "<", "=", None]},
        "quote": {"type": "string"},
        "chunk_id": {"type": ["string", "null"]},
    }

    return {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["frames"],
        "additionalProperties": False,
    }


def _strip_fence(text: str) -> str:
    """Drop a surrounding markdown code fence, if the model added one."""
    body = (text or "").strip()

    if not body.startswith("```"):
        return body

    newline = body.find("\n")
    body = "" if newline < 0 else body[newline + 1 :]

    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]

    return body.strip()


def _instruction(units: Sequence[tuple[str, Sequence[str]]], indices: Sequence[int]) -> str:
    """Render the user message for one batch of claim units."""
    payload = [
        {
            "unit_index": index,
            "text": units[index][0],
            "chunk_ids": [str(chunk_id) for chunk_id in units[index][1]],
        }
        for index in indices
    ]

    return (
        "Decompose every numbered claim into one or more frames.\n"
        "A frame binds one predicate to its roles: 'Model A scores 56.0 on OOLONG' "
        "is measures(subject=Model A, benchmark=OOLONG, metric=accuracy, value=56.0).\n"
        "Rules:\n"
        "- Copy the quote character for character from the cited chunk. Never "
        "paraphrase, shorten or join sentences.\n"
        "- chunk_id must be one of that unit's ids; use null only when it lists none.\n"
        "- Use predicate=other when none of the named predicates fits.\n"
        "- Emit two frames rather than one frame with two predicates.\n"
        'Reply with JSON only: {"frames": [...]}\n\n'
        "Units\n" + json.dumps(payload, ensure_ascii=False)
    )


def extract_frames(
    client: Any,
    units: Sequence[tuple[str, Sequence[str]]],
    trace_ctx: Any,
    *,
    budget: Any | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[list[ClaimFrame], list[str]]:
    """Return ``(frames, errors)`` for every claim unit in ``units``.

    ``units`` is ``[(claim_text, [chunk_id, ...]), ...]`` in the answer's order.
    A failed batch is an error entry, never an exception: one unavailable call
    must not lose the frames the other batches produced.
    """
    frames: list[ClaimFrame] = []
    errors: list[str] = []
    size = max(1, int(batch_size))

    for start in range(0, len(units), size):
        indices = list(range(start, min(start + size, len(units))))

        try:
            result = client.chat(
                "sub",
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _instruction(units, indices)},
                ],
                trace_ctx=trace_ctx,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "claim_frames",
                        "strict": True,
                        "schema": _schema(),
                    },
                },
                max_tokens=max_tokens,
                extra={"claim_frames": len(indices)},
            )
        except Exception as exc:  # noqa: BLE001 - one failed batch is a visible error
            errors.append(f"claim extraction failed: {type(exc).__name__}")
            continue

        if budget is not None:
            budget.record_chat(result)
            budget.record_sub_task()

        try:
            payload = json.loads(_strip_fence(result.text))
        except (TypeError, ValueError):
            errors.append("claim extractor reply was not JSON")
            continue

        batch_frames, batch_errors = parse_frames(payload, units, allowed_units=set(indices))
        frames.extend(batch_frames)
        errors.extend(batch_errors)

    return frames, errors


__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_MAX_TOKENS", "extract_frames"]
