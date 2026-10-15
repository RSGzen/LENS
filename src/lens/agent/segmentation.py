r"""M6 -- query segmentation: prompt -> 4 buckets -> ``query.json``.

Spec: ``MVP Build Roadmap.md`` section 4 (M6); ``Thesis Drafts/Tech Stack
(Draft).md`` sections 1-2 (stage 9 / the pipeline diagram), 4e (planned
``agent/segmentation.py``).

Decision (LOG-49, amends frozen D-02)
-------------------------------------
Segmentation is **one** ``gpt-5-mini`` structured-JSON call (``role=sub``), *not*
a per-clause JEV ``choice`` pass. The four buckets become ``query.json``. JEV stays
the *decision* layer (stage gate / ``noul`` sufficiency verifier / routing) and is
**not** used to generate the buckets here.

Buckets (``objectives`` / ``problems`` / ``constraints`` / ``requirements``)
---------------------------------------------------------------------------
Each bucket is a **list** of clause texts (LOG-49): a multi-clause prompt keeps its
structure, and a bucket with no match is an empty list. The mapping is emitted in
:data:`BUCKET_ORDER` -- M6 owns the dict build order (``config.QUERY_BUCKET_ORDER``
was dropped). :func:`lens.tools.retrieval.format_query` consumes the mapping
directly: it flattens every bucket's clauses, in insertion order, into the single
query string (a **plain clause join** -- no rewriting, LOG-48), so the clause text
must stay the user's own wording.

Cost / identity (E-04)
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

# The 4 canonical buckets, in the order the mapping is built (LOG-49). M6 owns
# this order; ``retrieval.format_query`` joins in insertion order, so it is also
# the clause order of the embedded query.
BUCKET_ORDER = ("objectives", "problems", "constraints", "requirements")

# System prompt: fixes the 4 buckets, forbids rewriting (so the embedded query
# stays the user's own wording -- LOG-48: plain clause join, no LLM rewriting),
# and forbids repeats / nesting (one bucket per clause -- B-lite, LOG-53).
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
    no rewriting (LOG-48).
    """
    system_query_payload = {"role": "system",
                            "content": SEGMENTATION_SYSTEM_PROMPT}

    user_query_payload = {"role": "user",
                          "content": query}

    return [system_query_payload, user_query_payload]


# ------------------------------------------------------------------- parsing
def parse_segmentation_json(text: str) -> dict[str, Any]:
    """Parse the model's reply into a mapping.

    Strict ``response_format`` should return bare JSON, but a dev/fallback model
    may wrap it in a `````json ... ````` fence -- strip a surrounding fence if
    present, then ``json.loads``. Raise :class:`ValueError` on unparseable text
    (F-02: the caller decides whether to retry).
    """
    # Extract the JSON object: from the first "{" to the last "}" (greedy, so a
    # surrounding ```json fence or stray text is ignored).
    match = re.search(r'(\{.*\})', text, re.DOTALL)

    try:
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return data
        
        raise ValueError(f"No JSON body is found from segmentation LLM response.")

    except Exception as e:
        raise ValueError(f"Unparseable text from segmentation LLM response. Error code: {e}")


def normalize_buckets(raw: Mapping[str, Any]) -> dict[str, list[str]]:
    """Coerce ``raw`` into the fixed 4-bucket schema.

    - Return a dict in :data:`BUCKET_ORDER` (LOG-49).
    - A missing or ``None`` bucket -> ``[]``.
    - A scalar value -> a single-element list.
    - Strip each clause; drop empty / whitespace-only clauses.
    - **Drop duplicate clauses across all buckets**, compared case-insensitively
      on whitespace-normalized text, keeping the first occurrence in
      :data:`BUCKET_ORDER`. The join is global (LOG-48), so a repeated clause
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


# --------------------------------------------------------------- segmentation
def segment_query(
    client: LensClient,
    query: str,
    *,
    trace_ctx: TraceCtx,
) -> dict[str, list[str]]:
    """Segment ``query`` into the 4 buckets via one ``gpt-5-mini`` call.

    Steps:
      1. Reject a blank ``query`` with :class:`ValueError` (trust boundary).
      2. ``messages = build_segmentation_messages(query)``.
      3. ``result = client.chat(role="sub", messages=messages,
         trace_ctx=trace_ctx, response_format=SEGMENTATION_RESPONSE_FORMAT)``.
         The transport logs ``usage.cost`` + the dated ``model`` snapshot (E-04).
      4. Return ``normalize_buckets(parse_segmentation_json(result.text))``.

    Returns the bucket mapping ready for ``retrieval.format_query`` / ``query.json``.
    """
    if not query.strip():
        raise ValueError("Blank query string")

    messages = build_segmentation_messages(query)

    result = client.chat(role="sub",
                         messages=messages,
                         trace_ctx=trace_ctx,
                         response_format=SEGMENTATION_RESPONSE_FORMAT)

    result_json = parse_segmentation_json(result.text)

    query_buckets = normalize_buckets(result_json)

    return query_buckets


# --------------------------------------------------------------------- output
def write_query_json(buckets: Mapping[str, Any], path: str | Path) -> Path:
    """Write the segmentation output to ``query.json`` (UTF-8, ``indent=2``).

    The sandbox reads this file as its ``query.json`` context file (Tech Stack
    section 1). Use ``ensure_ascii=False`` so non-ASCII query text round-trips
    losslessly (the ``normalize_text`` policy, Tech Stack 4i). Returns the path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(buckets, f, ensure_ascii=False, indent=2)

    return Path(path)

