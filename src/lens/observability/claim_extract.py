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
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from lens import config
from lens.observability.citations import claim_units, expand_short_ids
from lens.observability.claims import (
    PREDICATES,
    ClaimFrame,
    ClaimReport,
    align_quotes,
    parse_frames,
    quoted_spans,
    split_long_units,
    strip_self_reported_gaps,
    verify_units,
)

#: Sentence units one call carries. Six short sentences keep the reply's JSON
#: small; the model's reasoning, not the frames, is what fills the cap.
DEFAULT_BATCH_SIZE = 6

#: Output tokens for one batch. Reasoning shares this budget: at 8,000 the model
#: spent every token thinking and returned empty text, so the cap carries the
#: reasoning plus the frames. Only generated tokens are billed, so unused
#: headroom costs nothing.
DEFAULT_MAX_TOKENS = 16000

#: Cited chunks one unit carries and characters per chunk. The model needs the
#: sentence, not the paper; two chunks cover a claim that cites two sources.
MAX_CHUNKS_PER_UNIT = 2
MAX_CHUNK_CHARS = 2000

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


def _instruction(
    units: Sequence[tuple[str, Sequence[str]]],
    indices: Sequence[int],
    texts: Mapping[str, str] | None = None,
    answer_quotes: Mapping[int, Sequence[str]] | None = None,
) -> str:
    """Render the user message for one batch of claim units.

    The cited chunk text travels with the unit: a quote has to be copied from
    the chunk, and a model that can only see the answer can only quote the
    answer, which is a paraphrase by construction. ``answer_quotes`` carries the
    spans the answer displays, so the frame can use the same sentence the reader
    sees instead of a different one from the chunk.
    """
    payload = []

    for index in indices:
        chunks = []

        for chunk_id in list(units[index][1])[:MAX_CHUNKS_PER_UNIT]:
            body = (texts or {}).get(str(chunk_id), "")

            if body:
                chunks.append({"id": str(chunk_id), "text": body[:MAX_CHUNK_CHARS]})

        payload.append(
            {
                "unit_index": index,
                "text": units[index][0],
                "chunk_ids": [str(chunk_id) for chunk_id in units[index][1]],
                "chunks": chunks,
                "answer_quotes": list((answer_quotes or {}).get(index) or ()),
            }
        )

    return (
        "Decompose every numbered claim into one or more frames.\n"
        "A frame binds one predicate to its roles: 'Model A scores 56.0 on OOLONG' "
        "is measures(subject=Model A, benchmark=OOLONG, metric=accuracy, value=56.0).\n"
        "Predicate guide\n"
        "- reports: the cited text states a finding, property, limitation or claim. "
        "Subject is the named method, system or paper; object is what it states.\n"
        "- measures: the text states a numeric result. Bind subject, benchmark, metric, value.\n"
        "- compares: the text states a direction between two named things. Bind subject, "
        "object and direction (> or <).\n"
        "- uses / evaluated_on: the text states a method-dataset or method-metric relation.\n"
        "- addresses: the text names the problem or task a method targets.\n"
        "- other: only when none of the above fits.\n"
        "Rules:\n"
        "- The ``chunks`` array holds the cited text. Copy the quote character for "
        "character from there. Never paraphrase, shorten or join sentences.\n"
        "- If ``answer_quotes`` is not empty, the frame's quote must be one of those "
        "spans, copied exactly; the reader sees that quote, so do not choose a "
        "different sentence from the chunk.\n"
        "- When the unit contains a quoted span and an id, emit a reports frame "
        "with that quote; do not return other with an empty quote.\n"
        "- The subject must be a name that appears in the quote. Never use a section "
        "heading or a label such as 'Empirical results' as the subject.\n"
        "- If no cited chunk states the claim, emit one frame with predicate=other "
        "and an empty quote; do not manufacture support.\n"
        "- chunk_id must be one of that unit's ids; use null only when it lists none.\n"
        "- Emit two frames rather than one frame with two predicates.\n"
        'Reply with JSON only: {"frames": [...]}\n\n'
        "Units\n" + json.dumps(payload, ensure_ascii=False)
    )


def extract_frames(
    client: Any,
    units: Sequence[tuple[str, Sequence[str]]],
    trace_ctx: Any,
    *,
    texts: Mapping[str, str] | None = None,
    answer_quotes: Mapping[int, Sequence[str]] | None = None,
    usage: list[dict[str, Any]] | None = None,
    budget: Any | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[list[ClaimFrame], list[str]]:
    """Return ``(frames, errors)`` for every claim unit in ``units``.

    ``units`` is ``[(claim_text, [chunk_id, ...]), ...]`` in the answer's order.
    ``texts`` maps ``chunk_uuid -> chunk_text`` and is sent with each unit so a
    quote can be copied from the chunk rather than from the answer.
    ``answer_quotes`` maps a unit index to the spans the answer displays; when it
    is non-empty the extractor must copy one of them. ``usage`` collects one
    ``{cost_usd, tokens_out}`` entry per call, so the report can state what the
    extractor spent. A failed batch is an error entry, never an exception: one
    unavailable call must not lose the frames the other batches produced.

    Batches are independent, so they run concurrently: the pass was sequential and
    a 29-unit answer spent eight calls end to end. Results are collected in batch
    order, so the frames stay in the answer's own order.
    """
    frames: list[ClaimFrame] = []
    errors: list[str] = []
    size = max(1, int(batch_size))
    accounting = threading.Lock()

    def run_batch(indices: list[int], depth: int) -> tuple[list[ClaimFrame], list[str]]:
        """Call the extractor once; split a capped batch once and merge."""
        try:
            result = client.chat(
                "sub",
                [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": _instruction(units, indices, texts, answer_quotes),
                    },
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
            return [], [f"claim extraction failed: {type(exc).__name__}"]

        if budget is not None:
            with accounting:
                budget.record_chat(result)
                budget.record_sub_task()

        if usage is not None:
            with accounting:
                usage.append(
                    {"cost_usd": float(result.cost_usd), "tokens_out": int(result.tokens_out)}
                )

        body = _strip_fence(result.text)

        # The cap is spent on reasoning before any JSON arrives. Split the batch
        # once, so the retry has half the units and twice the per-unit headroom.
        cut = (
            result.finish_reason == "length"
            or not body
            or (body.startswith("{") and not body.rstrip().endswith("}"))
        )

        if cut and len(indices) > 1 and depth == 0:
            middle = len(indices) // 2
            left_frames, left_errors = run_batch(indices[:middle], 1)
            right_frames, right_errors = run_batch(indices[middle:], 1)

            return (
                [*left_frames, *right_frames],
                [
                    "claim extractor batch hit the output cap; retried as halves",
                    *left_errors,
                    *right_errors,
                ],
            )

        if not body:
            return [], ["claim extractor returned no text: the output cap was spent on reasoning"]

        if body.startswith("{") and not body.rstrip().endswith("}"):
            return [], ["claim extractor reply was cut off before its JSON closed"]

        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            return [], ["claim extractor reply was not JSON"]

        parsed, parsed_errors = parse_frames(payload, units, allowed_units=set(indices))

        return parsed, parsed_errors

    batches = [
        list(range(start, min(start + size, len(units)))) for start in range(0, len(units), size)
    ]

    if len(batches) <= 1:
        results = [run_batch(batches[0], 0)] if batches else []
    else:
        workers = max(1, min(len(batches), config.CLAIM_WORKERS))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_batch, indices, 0) for indices in batches]
            results = [future.result() for future in futures]

    for batch_frames, batch_errors in results:
        frames.extend(batch_frames)
        errors.extend(batch_errors)

    return frames, errors


def verify_answer(
    answer: str,
    texts: dict[str, str],
    client: Any,
    trace_ctx: Any,
    *,
    run_id: str = "",
    budget: Any | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ClaimReport:
    """Extract and verify every claim unit in one answer.

    ``texts`` maps ``chunk_uuid -> chunk_text`` for everything the run retrieved,
    so the check is offline against recorded evidence. A trailing self-reported-gap
    section is excluded first: those lines are the answer's own limitations, not
    findings, and counting them as claims understates coverage. An answer with no
    claim unit returns an empty report without a model call. Frames are pointed at
    the answer's own quoted spans before verification, so the ledger and the reader
    are looking at the same sentence.
    """
    full = answer or ""
    answer = strip_self_reported_gaps(full)
    removed = answer != full
    expanded = expand_short_ids(answer, texts)
    units = claim_units(expanded, include_uncited=True)
    units = [(text, [str(value) for value in ids if str(value) in texts]) for text, ids in units]
    units = split_long_units(units)

    if not units:
        return ClaimReport(run_id=run_id, units=0, self_reported_gaps_removed=removed)

    answer_quotes = {index: quoted_spans(text) for index, (text, _ids) in enumerate(units)}
    usage: list[dict[str, Any]] = []
    frames, errors = extract_frames(
        client,
        units,
        trace_ctx,
        texts=texts,
        answer_quotes=answer_quotes,
        usage=usage,
        budget=budget,
        batch_size=batch_size,
        max_tokens=max_tokens,
    )
    frames = align_quotes(frames, answer_quotes)
    report = verify_units(units, frames, texts, run_id=run_id)
    report.errors = errors
    report.self_reported_gaps_removed = removed
    report.cost_usd = sum(float(item["cost_usd"]) for item in usage)
    report.tokens_out = sum(int(item["tokens_out"]) for item in usage)

    return report


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_TOKENS",
    "extract_frames",
    "verify_answer",
]
