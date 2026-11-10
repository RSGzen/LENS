r"""-- query segmentation: prompt -> 4 buckets -> ``query.json``.

Decision
-------------------------------------
Segmentation is **one** the sub model structured-JSON call (``role=sub``), *not*
a per-clause JEV ``choice`` pass. The four buckets become ``query.json``. JEV stays
the *decision* layer (stage gate / ``noul`` sufficiency verifier / routing) and is
**not** used to generate the buckets here.

Buckets (``objectives`` / ``problems`` / ``constraints`` / ``requirements``)
---------------------------------------------------------------------------
Each bucket is a **list** of clause texts : a multi-clause prompt keeps its
structure, and a bucket with no match is an empty list. The mapping is emitted in
:data:`BUCKET_ORDER` -- owns the dict build order
(``config.QUERY_BUCKET_ORDER`` was dropped). :func:`lens.tools.retrieval.format_query`
consumes the mapping directly: it flattens every bucket's clauses, in insertion
order, into the single query string, so the clause text must stay the user's own
wording -- with one exception, :func:`retain_identifiers`, which copies an arXiv
identifier the segmenter dropped into ``constraints`` because losing it silently
removes the only way to reach a named paper.

Cost / identity
----------------------
The transport (``clients.LensClient.chat``) logs ``usage.cost`` and the dated
``model`` snapshot of this call to the trajectory; this module never logs cost
itself.

``tests/test_segmentation.py`` is the contract (no key, zero cost).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lens.clients import LensClient
from lens.observability.trajectory import TraceCtx

# The 4 canonical buckets, in the order the mapping is built. owns
# this order; ``retrieval.format_query`` joins in insertion order, so it is also
# the clause order of the embedded query.
BUCKET_ORDER = ("objectives", "problems", "constraints", "requirements")

# An arXiv identifier: four digits, a dot, four or five digits. The shape is what
# the corpus and every tool key on, so an identifier that survives segmentation is
# the one thing a retrieval arm needs in order to fetch a named paper.
IDENTIFIER = re.compile(r"\b\d{4}\.\d{4,5}\b")

# System prompt: fixes the 4 buckets, forbids rewriting (so the embedded query
# stays the user's own wording -- : plain clause join, no LLM rewriting),
# and forbids repeats / nesting.
SEGMENTATION_SYSTEM_PROMPT = (
    "You segment a research query into four buckets for literature retrieval.\n"
    "Return ONLY a JSON object with exactly these keys, each an array of short "
    "clause strings taken from the user's query:\n"
    "  objectives   - what the user wants to achieve or answer.\n"
    "  problems     - the problem, phenomenon, or pain point the query is about.\n"
    "  constraints  - constraints and limitations (scope, budget, method, time).\n"
    "  requirements - special requirements (must-haves, formats, evidence types).\n"
    "Assign each clause to exactly one bucket - never repeat a clause or put the "
    "same wording in two buckets, and do not restate a shorter phrase already "
    "covered by a longer clause. Preserve the user's own wording; do not rewrite, "
    "expand, or invent clauses. A bucket with no matching clause is an empty array."
)

# Strict JSON-schema response_format (the ``sub`` role supports response_format).
# Every property is required and additionalProperties is false, as strict mode
# demands; the value type is a list of strings for each bucket.
_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}
SEGMENTATION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_segmentation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {name: _STRING_ARRAY for name in BUCKET_ORDER},
            "required": list(BUCKET_ORDER),
            "additionalProperties": False,
        },
    },
}


# ------------------------------------------------------------------ messages
def build_segmentation_messages(query: str) -> list[dict[str, str]]:
    """Build the two-message payload for the segmentation call.

    Returns ``[{"role": "system", "content": SEGMENTATION_SYSTEM_PROMPT},
    {"role": "user", "content": query}]``. The user content is the raw query --
    no rewriting.
    """
    system_query_payload = {"role": "system", "content": SEGMENTATION_SYSTEM_PROMPT}

    user_query_payload = {"role": "user", "content": query}

    return [system_query_payload, user_query_payload]


# ------------------------------------------------------------------ parsing
def parse_segmentation_json(text: str) -> dict[str, Any]:
    """Parse the model's reply into a mapping.

    Strict ``response_format`` should return bare JSON, but a dev/fallback model
    may wrap it in a `````json ... ````` fence -- strip a surrounding fence if
    present, then ``json.loads``. Raise :class:`ValueError` on unparseable text.
    """
    # Extract the JSON object: from the first "{" to the last "}" (greedy, so a
    # surrounding ```json fence or stray text is ignored).
    match = re.search(r"(\{.*\})", text, re.DOTALL)

    try:
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return data

        raise ValueError("No JSON body is found from segmentation LLM response.")

    except Exception as e:
        raise ValueError(f"Unparseable text from segmentation LLM response. Error code: {e}")


def retain_identifiers(query: str, buckets: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return ``buckets`` with every arXiv identifier from ``query`` present.

    A segmenter that drops an identifier deletes the only way to ask for that
    paper: the retrieval query loses it, the model never sees it, and no tool can
    fetch it. The rule is deterministic and additive — an identifier the buckets
    already contain anywhere is left alone, and a missing one is copied verbatim
    into ``constraints`` (the bucket for scope and method limits).

    The clause is *not* user wording, which the segmentation contract otherwise
    preserves: it is an identifier lifted from the query, chosen because losing it
    is a silent capability loss rather than a phrasing choice.
    """
    named = sorted(set(IDENTIFIER.findall(query or "")))

    if not named:
        return {name: list(buckets.get(name) or []) for name in BUCKET_ORDER}

    present = set(IDENTIFIER.findall(json.dumps(dict(buckets))))

    missing = [identifier for identifier in named if identifier not in present]

    if not missing:
        return {name: list(buckets.get(name) or []) for name in BUCKET_ORDER}

    kept = {name: list(buckets.get(name) or []) for name in BUCKET_ORDER}

    kept["constraints"].append("arXiv " + ", ".join(missing))

    return kept


def normalize_buckets(raw: Mapping[str, Any]) -> dict[str, list[str]]:
    """Coerce ``raw`` into the fixed 4-bucket schema.

    - Return a dict in :data:`BUCKET_ORDER`.
    - A missing or ``None`` bucket -> ``[]``.
    - A scalar value -> a single-element list.
    - Strip each clause; drop empty / whitespace-only clauses.
    - **Drop duplicate clauses across all buckets**, compared case-insensitively
      on whitespace-normalized text, keeping the first occurrence in
      :data:`BUCKET_ORDER`. The join is global, so a repeated clause
      would otherwise double-weight that phrase in the embedded query.
    - Ignore keys outside the four buckets.
    """
    # Match raw keys case-/whitespace-insensitively to the 4 buckets.
    by_bucket: dict[str, Any] = {}
    for key, value in raw.items():
        cleaned_key = "".join(key.split()).lower()
        if cleaned_key in BUCKET_ORDER:
            by_bucket[cleaned_key] = value

    query_buckets: dict[str, list[str]] = {name: [] for name in BUCKET_ORDER}
    seen: set[str] = set()

    # Iterate in BUCKET_ORDER (not raw order) so the "first occurrence kept" is
    # deterministic: objectives -> problems -> constraints -> requirements.
    for name in BUCKET_ORDER:
        value = by_bucket.get(name)
        if not value:
            continue

        values = [value] if isinstance(value, str) else value
        for item in values:
            clause = str(item).strip()
            if not clause:
                continue

            dedupe_key = " ".join(clause.split()).lower()
            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            query_buckets[name].append(clause)

    return query_buckets


def validate_buckets(raw: Any) -> dict[str, list[str]]:
    """Return the four buckets, or raise with the fix the caller must make.

    The host segmenter always produces this shape; a root-segmented run writes it
    itself, so the tool boundary rejects a bare string or ``{"query": ...}``
    instead of letting it silently become the retrieval query. Matching is
    case-insensitive on the key, and the returned dict is
    :func:`normalize_buckets` (stripped, de-duplicated, in bucket order).
    """
    if not isinstance(raw, Mapping):
        raise ValueError("buckets must be a dict with the keys " + ", ".join(BUCKET_ORDER))

    by_key = {"".join(str(key).split()).lower(): value for key, value in raw.items()}
    missing = [name for name in BUCKET_ORDER if name not in by_key]
    extra = sorted(set(by_key) - set(BUCKET_ORDER))

    if missing or extra:
        raise ValueError(
            "buckets must have exactly "
            + ", ".join(BUCKET_ORDER)
            + f"; missing={missing} extra={extra}"
        )

    for name in BUCKET_ORDER:
        value = by_key[name]

        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValueError(f"bucket {name!r} must be a list of strings")

        if any(not isinstance(item, str) for item in value):
            raise ValueError(f"bucket {name!r} must contain only strings")

    return normalize_buckets(raw)


# ------------------------------------------------------------------ segmentation
def segment_query(
    client: LensClient,
    query: str,
    *,
    trace_ctx: TraceCtx,
) -> dict[str, list[str]]:
    """Segment ``query`` into the 4 buckets via one the sub model call.

    Steps:
      1. Reject a blank ``query`` with :class:`ValueError` (trust boundary).
      2. ``messages = build_segmentation_messages(query)``.
      3. ``result = client.chat(role="sub", messages=messages,
         trace_ctx=trace_ctx, response_format=SEGMENTATION_RESPONSE_FORMAT)``.
         The transport logs ``usage.cost`` + the dated ``model`` snapshot, tagged
         ``extra["source"] = "segmentation"`` so the event is not mistaken for a
         loop turn at ``step_idx = 0``.
      4. Return ``retain_identifiers(query, normalize_buckets(...))``, so an
         identifier the model dropped is still in the buckets the root sees.

    Returns the bucket mapping ready for ``retrieval.format_query`` / ``query.json``.
    """
    if not query.strip():
        raise ValueError("Blank query string")

    messages = build_segmentation_messages(query)

    result = client.chat(
        role="sub",
        messages=messages,
        trace_ctx=trace_ctx,
        response_format=SEGMENTATION_RESPONSE_FORMAT,
        # Segmentation is not a loop turn: it runs once, before the loop, and its
        # tokens belong to no turn. The tag is what lets a reader or a fold tell.
        extra={"source": "segmentation"},
    )

    result_json = parse_segmentation_json(result.text)

    query_buckets = normalize_buckets(result_json)

    return retain_identifiers(query, query_buckets)


# ------------------------------------------------------------------ output
def write_query_json(buckets: Mapping[str, Any], path: str | Path) -> Path:
    """Write the segmentation output to ``query.json`` (UTF-8, ``indent=2``).

    The sandbox reads this file as its ``query.json`` context file. Use
    ``ensure_ascii=False`` so non-ASCII query text round-trips losslessly: there is
    no normalisation step on the query path, so this file is the text as segmented.
    Returns the path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(buckets, f, ensure_ascii=False, indent=2)

    return Path(path)
