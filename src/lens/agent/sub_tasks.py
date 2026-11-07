"""Typed depth-1 sub-tasks: fixed instruction, strict JSON, host-stamped ids.

A sub-task is one call to the sub model the root makes from the sandbox. The root
supplies **parameters**, never the instruction: the system message, the user
template, the JSON schema and the output cap all come from the registry below.

That is the whole point of typing them. The previous helper took a free-form
prompt and an optional system message, so the root had to invent a role, an
output format and its own id bookkeeping on every call — and in recorded runs it
never made one. Three rules are enforced by the host instead, because a model
cannot be trusted with them:

- **Ids are stamped by the host.** A record whose ``id`` was not in the request is
  dropped, so a hallucinated id can never become evidence.
- **The answer shape is a schema.** ``response_format`` travels with the request
  and the reply is parsed and validated, so the sandbox receives a dict rather
  than prose it has to parse inside a code block.
- **The count is bounded.** ``config.MAX_SUB_TASKS`` calls per query; past that the
  call is refused with a structured error and the run continues.

A sub-task answers *questions about text*. It is not a judge: whether a claim is
supported is the decision layer's job, and a second uncalibrated opinion would
make the gate's measured behaviour stop describing the system.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lens import config

#: Fields ``sub_extract`` will pull out of a chunk. A whitelist, not a free
#: string: the schema has to name them, and the prompt block advertises them.
EXTRACT_FIELDS: tuple[str, ...] = (
    "claims",
    "method",
    "dataset",
    "metric",
    "numbers",
    "limitations",
)

#: Field that carries **verbatim** spans. Always returned, whatever else is
#: requested: it is the only part of a record an answer can cite, and a paraphrase
#: cannot be checked against the chunk the gate is shown.
QUOTES_FIELD = "quotes"

#: Shortest span worth keeping. A quote below this cannot carry a claim, and it is
#: the same floor the silver-label check uses for a supported claim.
MIN_QUOTE_CHARS = 25

#: Longest and fewest quotes kept per record. The output cap is the real limit: an
#: unconstrained quote list is what truncated the first recorded call, and three
#: sentences carry a record's claims without spending the reply on them.
MAX_QUOTE_CHARS = 400
MAX_QUOTES_PER_RECORD = 3

#: Ids one RPC call accepts. Larger than ``max_items`` on purpose: the host splits
#: them into batches of ``max_items`` and runs those **concurrently**, so the root
#: pays one round of latency instead of one per batch. Measured: a batch of four
#: chunks costs 13–41 s of wall time, and a run that read 32 chunks spent 8 calls
#: in sequence.
MAX_IDS_PER_CALL = 12

#: Characters of one item sent to a sub-task. A chunk is at most ~512 tokens
#: (~2,000 characters); the margin covers a chunk that ran long without letting
#: one item crowd out the others in a multi-item call.
ITEM_CHARS = 2400

#: Label the model may always return, so a schema-valid answer exists even when
#: the text does not decide the question.
UNSURE_LABEL = "unsure"

_ARRAY_OF_STRING = {"type": "array", "items": {"type": "string"}}


@dataclass(frozen=True)
class SubTask:
    """One registered sub-task.

    ``source`` names where the item text comes from: ``"ids"`` (the host hydrates
    ``chunk_ids`` from the database), ``"text"`` (the caller passes the text, which
    is the only option in document mode), or ``"either"``.

    ``signature`` carries the **types**, and ``result_shape`` the exact JSON the
    caller receives. Both are advertised because guessing them is expensive: five
    code blocks in three recorded runs died on ``KeyError: 'chunk_id'`` and
    ``AttributeError: 'str' object has no attribute 'get'``, i.e. on the record key
    and on ``quotes`` being a list of strings. The host already refused the wrong
    parameters; the return value had no contract at all.
    """

    name: str
    function: str
    signature: str
    result_shape: str
    result_note: str
    params: tuple[str, ...]
    summary: str
    system: str
    instruction: str
    result_key: str
    max_output_tokens: int
    max_items: int
    source: str
    build_schema: Callable[[Mapping[str, Any]], dict[str, Any]]


# ------------------------------------------------------------------ schemas


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap ``properties`` in the strict schema shape: all required, no extras.

    Strict mode rejects a schema that omits a property from ``required``, so the
    "field not stated in the text" case is expressed as an explicit ``null``
    type rather than by leaving the key out.
    """
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def _records_schema(fields: Sequence[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {"id": {"type": "string"}}

    for field in fields:
        if field == "method":
            properties[field] = {"type": ["string", "null"]}
        else:
            properties[field] = {"type": ["array", "null"], "items": {"type": "string"}}

    return _object({"records": {"type": "array", "items": _object(properties)}})


def _extract_schema(params: Mapping[str, Any]) -> dict[str, Any]:
    requested = params.get("fields")

    fields = EXTRACT_FIELDS if not requested else tuple(str(name) for name in requested)

    # ``quotes`` is appended rather than merged into a set so the schema — and
    # therefore the request bytes — is identical from one call to the next.
    return _records_schema((*[name for name in fields if name != QUOTES_FIELD], QUOTES_FIELD))


def _summarise_schema(_params: Mapping[str, Any]) -> dict[str, Any]:
    return _object(
        {
            "summaries": {
                "type": "array",
                "items": _object({"id": {"type": "string"}, "summary": {"type": "string"}}),
            }
        }
    )


def _classify_schema(params: Mapping[str, Any]) -> dict[str, Any]:
    labels = [str(label) for label in params.get("labels") or ()]

    return _object(
        {
            "labels": {
                "type": "array",
                "items": _object(
                    {
                        "id": {"type": "string"},
                        "label": {"type": "string", "enum": [*labels, UNSURE_LABEL]},
                    }
                ),
            }
        }
    )


# ------------------------------------------------------------------ registry


SUB_TASKS: dict[str, SubTask] = {
    "extract": SubTask(
        name="extract",
        function="sub_extract",
        signature="sub_extract(chunk_ids: list[str], fields: list[str] | None = None) -> dict",
        result_shape=(
            '{"records": [{"id": "<the chunk id you passed>", "claims": ["..."], '
            '"quotes": ["..."]}], "missing_ids": [...], "errors": [...]}'
        ),
        result_note=(
            "the key is ``id`` (not ``chunk_id``) and ``quotes`` is a list of "
            "**strings** — not of dicts"
        ),
        params=("chunk_ids", "fields"),
        summary=(
            "structured fields for a batch of chunks: claims, method, dataset, "
            "metric, numbers, limitations"
        ),
        system=(
            "You extract structured records from research text. Answer only from the text "
            "given: no outside knowledge, and no inference beyond what is written. "
            "Reply with JSON only."
        ),
        instruction=(
            "Extract these fields for every item: <<FIELDS>>.\n"
            'Reply with JSON only: {"records": [{"id": "...", "claims": ["..."], '
            '"quotes": ["..."]}]}\n'
            "\n"
            "Field guide\n"
            "- claims: statements the text makes, one short sentence each, in the text's "
            "own words.\n"
            "- quotes: sentences copied **character for character** from the item, each at "
            "least 25 characters and at most 400, at most three per item, that together "
            "state the most important claims. Copy; do not paraphrase, shorten with an "
            "ellipsis, or join two sentences. An empty list is correct when the text states "
            "nothing quotable.\n"
            "- method: the approach the text describes; null when it names none.\n"
            "- dataset: datasets the text evaluates on.\n"
            "- metric: metrics the text reports.\n"
            "- numbers: numeric results verbatim, with their units.\n"
            "- limitations: limitations the text states.\n"
            "Use null or an empty list when the text does not state a field. Never infer. "
            "A quote the caller cannot find in the item is discarded, so an invented or "
            "rewritten quote costs you the whole record.\n"
            "\n"
            "Items\n"
            "<<ITEMS>>"
        ),
        result_key="records",
        # 1600 tokens was the cap for four items and it silently truncated: every
        # recorded call that asked for all six fields plus quotes came back as
        # cut-off JSON, parsed to zero records, and the root read the chunks by hand
        # instead. A truncated reply is worse than a smaller one, because it looks
        # like "the text stated nothing".
        #
        # The cap now also has to cover the model's **reasoning**, which shares the
        # output budget: with reasoning at high effort a 3,000-token cap was spent
        # entirely on thinking and the reply came back empty with
        # finish_reason=length. Only tokens actually generated are billed, so the
        # headroom costs nothing when it is unused.
        #
        # 8,000 still truncated: a live run spent all 8,000 on reasoning and
        # returned cut-off JSON in 21 of 25 calls. 16,000 leaves room for the
        # reply after the thinking.
        max_output_tokens=16000,
        max_items=4,
        source="ids",
        build_schema=_extract_schema,
    ),
    "summarise": SubTask(
        name="summarise",
        function="sub_summarise",
        signature=(
            "sub_summarise(chunk_ids: list[str] | None = None, text: str | None = None, "
            "focus: str = '', max_words: int = 120) -> dict"
        ),
        result_shape=(
            '{"summaries": [{"id": "<the chunk id you passed>", "summary": "..."}], '
            '"missing_ids": [...], "errors": [...]}'
        ),
        result_note="",
        params=("chunk_ids", "text", "focus", "max_words"),
        summary="one short summary per item, focused on what the caller asks for "
        "(a paraphrase: for orientation, not citable)",
        system=(
            "You summarise research text. Report what the text says, keep its numbers and "
            "the names it uses, and add nothing. Reply with JSON only."
        ),
        instruction=(
            "Summarise every item for this focus: <<FOCUS>>\n"
            "At most <<MAX_WORDS>> words per item. Keep the numbers and the named systems.\n"
            'Reply with JSON only: {"summaries": [{"id": "...", "summary": "..."}]}\n'
            "Do not evaluate, compare, or recommend: report what the text says.\n"
            "\n"
            "Items\n"
            "<<ITEMS>>"
        ),
        result_key="summaries",
        max_output_tokens=4000,
        max_items=3,
        source="either",
        build_schema=_summarise_schema,
    ),
    "classify": SubTask(
        name="classify",
        function="sub_classify",
        signature="sub_classify(chunk_ids: list[str], question: str, labels: list[str]) -> dict",
        result_shape=(
            '{"labels": [{"id": "<the chunk id you passed>", "label": "<one of your '
            'labels>"}], "missing_ids": [...], "errors": [...]}'
        ),
        result_note="",
        params=("chunk_ids", "question", "labels"),
        summary="one label per chunk from a fixed label set, for triage before reading",
        system=(
            "You label research text against one question. Use only the labels you are "
            "given, and reply with JSON only."
        ),
        instruction=(
            "Answer this question for every item, using exactly one label from: <<LABELS>>\n"
            "Question: <<QUESTION>>\n"
            'Reply with JSON only: {"labels": [{"id": "...", "label": "..."}]}\n'
            'Use "' + UNSURE_LABEL + '" when the text does not decide it. Do not explain.\n'
            "\n"
            "Items\n"
            "<<ITEMS>>"
        ),
        result_key="labels",
        # Unused in every recorded run, but it shares the same budget rule: the
        # reasoning is paid for before a single label is written.
        max_output_tokens=3000,
        max_items=8,
        source="ids",
        build_schema=_classify_schema,
    ),
}


def get_task(name: str) -> SubTask | None:
    """Return the registered sub-task ``name``, or ``None``."""
    return SUB_TASKS.get(name)


#: Per-task parameter rules, applied before any model call. The root writes the
#: call from generated code, so its parameters are untrusted input: a limit here is
#: what keeps one call from sending the whole corpus.
MAX_FOCUS_CHARS = 400
MAX_LABELS = 6
MAX_WORDS_CEILING = 400


def split_batches(items: Sequence[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
    """Return ``items`` cut into consecutive batches of at most ``size``.

    The fan-out unit is the batch, so this is the one place that decides how many
    model calls one RPC becomes. ``size`` is the task's ``max_items``: a batch bigger
    than that overflows the reply's output cap, which reads to the caller as "the
    text stated nothing".
    """
    if size < 1:
        raise ValueError(f"batch size must be positive, got {size}")

    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def validate_params(task: SubTask, params: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalised parameters for ``task``, or raise :class:`ValueError`.

    Only the parameters the task declares are accepted: an unknown key is a
    mistake in the caller that would otherwise be silently ignored.
    """
    unknown = sorted(set(params) - set(task.params))

    if unknown:
        raise ValueError(f"{task.function} does not take {unknown}")

    normalised: dict[str, Any] = {}

    if "chunk_ids" in params:
        normalised["chunk_ids"] = _chunk_ids(params.get("chunk_ids"), MAX_IDS_PER_CALL)

    if task.source == "ids" and "chunk_ids" not in normalised:
        raise ValueError(f"{task.function} needs chunk_ids")

    if task.source == "either" and "chunk_ids" not in normalised:
        text = params.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{task.function} needs chunk_ids or text")

        normalised["text"] = text.strip()[: ITEM_CHARS * task.max_items]

    if task.name == "extract":
        normalised["fields"] = _fields(params.get("fields"))

    if task.name == "summarise":
        focus = params.get("focus") or "the text's main claims"

        if not isinstance(focus, str):
            raise ValueError("focus must be a string")

        normalised["focus"] = focus.strip()[:MAX_FOCUS_CHARS]

        words = params.get("max_words") or 120

        if not isinstance(words, int) or isinstance(words, bool) or words < 20:
            raise ValueError("max_words must be an integer of at least 20")

        normalised["max_words"] = min(words, MAX_WORDS_CEILING)

    if task.name == "classify":
        question = params.get("question")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("classify needs a question")

        normalised["question"] = question.strip()[:MAX_FOCUS_CHARS]

        labels = params.get("labels")

        if not isinstance(labels, list) or not 2 <= len(labels) <= MAX_LABELS:
            raise ValueError(f"labels must be a list of 2 to {MAX_LABELS} strings")

        cleaned = []

        for label in labels:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("every label must be a non-empty string")

            cleaned.append(label.strip())

        if len(set(cleaned)) != len(cleaned):
            raise ValueError("labels must be unique")

        normalised["labels"] = cleaned

    return normalised


def _chunk_ids(raw: Any, limit: int) -> list[str]:
    """Return 1..``limit`` non-empty chunk ids, or raise :class:`ValueError`."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("chunk_ids must be a non-empty list")

    if len(raw) > limit:
        raise ValueError(f"at most {limit} chunk_ids per call, got {len(raw)}")

    ids: list[str] = []

    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("every chunk id must be a non-empty string")

        ids.append(value.strip())

    return ids


def _fields(raw: Any) -> tuple[str, ...]:
    """Return the requested extract fields, or the full whitelist."""
    if raw is None:
        return EXTRACT_FIELDS

    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("fields must be a non-empty list")

    requested: list[str] = []

    for value in raw:
        if not isinstance(value, str) or value not in EXTRACT_FIELDS:
            raise ValueError(f"unknown field {value!r} (one of {list(EXTRACT_FIELDS)})")

        requested.append(value)

    return tuple(requested)


def render_items(items: Sequence[tuple[str, str]]) -> str:
    """Render ``(id, text)`` pairs as the ``<<ITEMS>>`` block."""
    blocks = [f"[{item_id}]\n{text.strip()[:ITEM_CHARS]}" for item_id, text in items]

    return "\n\n".join(blocks)


def render_instruction(task: SubTask, params: Mapping[str, Any], items: str) -> str:
    """Fill the task's user template from normalised parameters and the item block."""
    text = task.instruction.replace("<<ITEMS>>", items)

    if task.name == "extract":
        fields = params.get("fields") or EXTRACT_FIELDS
        text = text.replace("<<FIELDS>>", ", ".join(fields))

    if task.name == "summarise":
        text = text.replace("<<FOCUS>>", str(params.get("focus")))
        text = text.replace("<<MAX_WORDS>>", str(params.get("max_words")))

    if task.name == "classify":
        labels = [str(label) for label in params.get("labels") or ()]
        text = text.replace("<<LABELS>>", ", ".join(labels))
        text = text.replace("<<QUESTION>>", str(params.get("question")))

    if "<<" in text:
        raise ValueError(f"unfilled placeholder in the {task.name} instruction")

    return text


def response_format(task: SubTask, params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``response_format`` payload for one call."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"sub_task_{task.name}",
            "strict": True,
            "schema": task.build_schema(params),
        },
    }


# ------------------------------------------------------------------ prompt


def render_prompt_block() -> str:
    """Return the instruction-block section that documents the registered sub-tasks.

    Generated from the registry so the prompt cannot advertise a sub-task the host
    does not implement, or miss one it does.
    """
    sub_model = config.MODEL_REGISTRY["sub"].id
    lines = [
        "Sub-tasks",
        f"- Typed calls to a smaller model: ``{sub_model}``, a fixed instruction and a",
        "  strict JSON answer. They cannot see your variables, your tools or the",
        "  corpus, only what you pass them. Each returns a dict, already parsed.",
        "  Call them as ``rlm_api.<name>(...)``.",
    ]

    for task in SUB_TASKS.values():
        lines.append(
            f"- ``rlm_api.{task.signature}`` — {task.summary}. "
            f"At most {MAX_IDS_PER_CALL} chunk ids per call: the host reads them in "
            f"concurrent batches of {task.max_items}, so one call is one round of "
            "latency rather than one per batch."
        )
        # The return value had no contract, and the model guessed one: two runs died on
        # a record key and two more on treating a quote as a dict.
        note = f" — {task.result_note}" if task.result_note else ""
        lines.append(f"  Returns ``{task.result_shape}``{note}.")

        if task.name == "extract":
            lines.append(
                "  ``fields`` is a subset of: "
                + ", ".join(EXTRACT_FIELDS)
                + ". Every record also carries ``quotes``: verbatim sentences with the id "
                "they came from, checked host-side, and the only part of a record you may "
                'cite. Ask for ``fields=["claims"]`` when you are collecting evidence: '
                "each extra field spends the reply's output cap, and a truncated reply "
                "parses to no records at all."
            )

    lines.extend(
        [
            # Measured: the root read 100 retrieved chunks through this keyhole, so it
            # keyword-filtered in code instead of reading, and never called a sub-task.
            f"- **When to use them:** the observation you get back is capped at "
            f"{config.TOOL_RESULT_SUMMARY_CHARS:,} characters (about four chunk "
            "excerpts). To read more chunks than that, do not print their text: call "
            "``rlm_api.sub_extract`` once with up to 12 ids — the host runs the batches "
            "concurrently — keep the records in a variable, and print only the parts you "
            "need. Keep the ``quotes`` field: those spans are checked against their chunk "
            "host-side, so they are citable as they stand, which a summary or a "
            "paraphrase never is.",
            f"- At most {config.MAX_SUB_TASKS} sub-task calls per query.",
            "- The ids in a result are stamped from the ids you passed, so a record",
            "  cannot invent evidence. A sub-task answer is a derived record: cite the",
            "  chunk ids you passed in, never the sub-task itself.",
            "- Do not ask a sub-task whether a claim is supported. The gate decides that,",
            "  and its verdict is the one that counts.",
            "- One code block has a time limit, so batch: a call takes a few seconds and a",
            "  loop of them will overrun it.",
            "- A result carries ``errors`` and ``missing_ids``. Print both when they are "
            "non-empty: a reply cut off at the output cap reads as an empty result, and "
            "the only way to tell the two apart is that the error says so.",
        ]
    )

    return "\n".join(lines)


# ------------------------------------------------------------------ parsing


def verify_quotes(
    records: Sequence[Mapping[str, Any]], items: Sequence[tuple[str, str]]
) -> tuple[list[dict[str, Any]], int]:
    """Return ``(records, dropped)`` with every non-verbatim quote removed.

    A sub-task is a model, and a model asked to copy a sentence sometimes rewrites
    it. A reworded quote is worse than no quote: the root would cite it as verbatim
    and the gate would find the words are not in the chunk. The text is host-side, so
    the check is exact — whitespace-normalised containment, one contiguous span, no
    ellipsis joining — and a quote that fails is **removed**, never repaired.
    """
    text_by_id = {item_id: " ".join(text.split()) for item_id, text in items}
    dropped = 0
    cleaned: list[dict[str, Any]] = []

    for record in records:
        body = dict(record)
        quotes = body.get(QUOTES_FIELD)

        if quotes is None:
            cleaned.append(body)
            continue

        text = text_by_id.get(str(body.get("id") or ""), "")
        kept: list[str] = []

        for quote in quotes if isinstance(quotes, list) else []:
            span = " ".join(str(quote).split())

            if (
                MIN_QUOTE_CHARS <= len(span) <= MAX_QUOTE_CHARS
                and span in text
                and len(kept) < MAX_QUOTES_PER_RECORD
            ):
                kept.append(span)
            else:
                dropped += 1

        body[QUOTES_FIELD] = kept
        cleaned.append(body)

    return cleaned, dropped


def _strip_fence(text: str) -> str:
    """Drop a surrounding markdown code fence, if the model added one."""
    body = text.strip()

    if not body.startswith("```"):
        return body

    newline = body.find("\n")
    body = "" if newline < 0 else body[newline + 1 :]

    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]

    return body.strip()


def parse_result(
    task: SubTask, text: str, allowed_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return ``(records, missing_ids, errors)`` for one sub-task reply.

    Tolerant on purpose: a malformed or partial answer must reach the root as a
    result it can see and work around, not as an exception that kills the turn.
    Ids are the one thing that is not negotiated — a record naming an id that was
    not in the request is dropped.
    """
    allowed = list(allowed_ids)
    body = _strip_fence(text)

    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        # A reply that stopped mid-object was cut off at the output cap, and that is
        # a different thing from a malformed one: it says the request was too big,
        # which the root can fix by asking for fewer fields or fewer ids.
        if body.startswith("{") and not body.rstrip().endswith("}"):
            return (
                [],
                allowed,
                [
                    "reply was cut off before its JSON closed (the output cap bound): "
                    "ask for fewer fields or fewer chunk ids, or retry"
                ],
            )

        return [], allowed, ["reply was not JSON"]

    if not isinstance(payload, Mapping):
        return [], allowed, ["reply was not a JSON object"]

    raw = payload.get(task.result_key)

    if not isinstance(raw, list):
        return [], allowed, [f"reply had no {task.result_key!r} array"]

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()

    for entry in raw:
        if not isinstance(entry, Mapping):
            errors.append("a record was not an object")

            continue

        record_id = entry.get("id")

        if not isinstance(record_id, str) or record_id not in allowed:
            errors.append(f"dropped a record with an id that was not requested: {record_id!r}")

            continue

        if record_id in seen:
            errors.append(f"dropped a duplicate record for {record_id}")

            continue

        seen.add(record_id)
        records.append(_coerce(task, dict(entry)))

    missing = [record_id for record_id in allowed if record_id not in seen]

    return records, missing, errors


def _coerce(task: SubTask, record: dict[str, Any]) -> dict[str, Any]:
    """Normalise one record to plain strings, so callers never handle ``None`` lists."""

    if task.name != "extract":
        return record

    for field in EXTRACT_FIELDS:
        if field not in record:
            continue

        value = record[field]

        if field == "method":
            record[field] = None if value is None else str(value)

            continue

        if value is None:
            record[field] = []

        elif isinstance(value, str):
            record[field] = [value]

        elif isinstance(value, list):
            record[field] = [str(item) for item in value]

        else:
            record[field] = [str(value)]

    return record
