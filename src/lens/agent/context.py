"""Root-context assembly, observation formatting, and compaction.

The root context is deliberately thin: a system prompt, a tool manifest (names,
signatures, docstrings — not source), ``query.json``, the budget snapshot, and
rolling working memory. Large tool results stay in sandbox variables or are
offloaded with ``emit()``; only printed summaries become observations.

File ownership: ``query.json`` is written by ****
(:func:`lens.agent.segmentation.write_query_json`) — one writer per file. This
module writes the other sandbox context files (``AGENTS.md``, ``tools.md``,
``budget.md``, optional ``ontology.md``) and never rewrites the query.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from lens import config
from lens.agent.sub_tasks import render_prompt_block

# The root system prompt. It is the first bytes of every request (so it must stay
# byte-stable for prefix caching), and the tool manifest, the buckets and the
# budget snapshot are injected through the placeholders below - never by string
# formatting, so literal JSON braces in the payload stay safe.
# Placeholders: <<TOOL_MANIFEST>>, <<QUERY_JSON>>, <<BUDGET>>
SYSTEM_PROMPT_TEMPLATE = """\
You are the root RLM in LENS, a literature-evidence synthesis harness.

Role
- Answer the research query from the arXiv cs.AI corpus.
- You write Python. The code runs in a persistent sandbox namespace.
- Use the host tools for retrieval<<BEGIN:kg>>
  and the knowledge graph<<END:kg>>. Do not invent evidence.
- The sandbox has no shell, no network, and no API key.

Tools
<<TOOL_MANIFEST>>

Calling tools
- Each tool above is also bound as a plain function in the namespace, so
  ``retrieval_funnel(buckets=buckets)`` works. ``rlm_api.tool("retrieval_funnel", ...)``
  is the generic form and behaves identically — both go through one dispatch path.
  Keyword arguments bind by name; an omitted argument uses the tool's default.
- ``buckets`` (the segmented query) is already defined in the namespace. Pass it
  as-is; do not redefine it.

Sandbox modules
<<BEGIN:kg>>
- ``kg_neighbors``, ``kg_triples``, ``kg_subgraph`` are bound as plain functions in
  the namespace (no import needed): ``kg_neighbors(graph, entity=...)``,
  ``kg_triples(graph, head=...)``, ``kg_subgraph(graph, entities)``. Call
  ``build_kg`` **only when you will traverse the graph** with these helpers: building
  it takes about 20 s, and a graph you never read adds nothing to the answer. Build it
  once, then read it.
- Edges carry ``direction_verified`` / ``direction_confidence``. ``true`` means the
  decision layer confirmed the direction of the relation; unverified edges were
  kept but not confirmed, so do not state one as a fact.
- ``kg_query`` is the same module, importable if you prefer
  ``kg_query.kg_neighbors(...)``.
<<END:kg>>
- ``networkx``, ``numpy``, ``pandas`` are importable. There is no shell, no
  network, and no database driver.

Payload shapes — do not guess them
- ``retrieval_funnel`` -> ``dict``: ``chunks`` (a list of chunk rows, best first)
  and ``papers`` (a dict ``{arxiv_id: {"title", "authors", "year", "abstract"}}``).
- ``fetch_papers`` -> ``list[dict]``: paper rows.
- ``fetch_chunks`` -> ``list[dict]``: chunk rows.
- A chunk row has ``chunk_uuid``, ``arxiv_id``, ``section_type``, ``chunk_text``.

Search the variable, not the digest you printed
- The observation is a truncated summary; the full payload stays in the namespace.
  Filter it in code — ``hits = [c for c in bundle["chunks"]
  if "term" in c["chunk_text"].lower()]`` — then print ``len(hits)`` and the ids.
  A grep-style miss on a printed digest is not evidence of absence.
- ``retrieval_funnel`` **already returns the chunk text** (``chunk_text`` on every
  row). Do not call ``fetch_chunks`` for ids you got from it; ``fetch_chunks(ids)``
  takes many ids per call.
- **Run the funnel, then refine it only if you learned something.** The first call
  is the retrieval; keep its result in a variable (``bundle``) and reuse it. If what
  you read shows the query needs terms or papers the buckets do not name, copy them
  first — ``mine = copy.deepcopy(buckets)``, then add or remove entries on ``mine`` —
  and call ``retrieval_funnel(buckets=mine)``. Never edit ``buckets`` itself: it is
  the query as segmented, it is what the run records, and every later turn refers to
  it. Calling the funnel again with unchanged buckets returns the same chunks, so
  that is a wasted turn rather than a new retrieval.

Coverage is a duty, not advice
- The query arrives split into objectives, problems, constraints and requirements in
  ``query.json`` (also in the namespace as ``buckets``). Before you answer, name the
  chunk ids that evidence **each** objective. An objective with no id is a gap: either
  narrow the claim to what you have, or refine a copy of the buckets with the terms you
  learned from the chunks, run the funnel once more, and read the new chunks the same
  way. Do not smooth a gap over with a broader sentence — that is the sentence the
  final gate rejects.
- One funnel returns about a hundred chunks and you have ten turns, so you cannot read
  them inline: read them in batches, a few chunks at a time, and keep the verbatim
  quotes rather than a paraphrase of them.
- A **comparative** claim ("A aggregates outside the model, B inside it") may cite two
  to four chunks and quote one sentence from each. The quote rule is about evidence,
  not about one sentence per claim: a query that asks how methods **differ** cannot be
  answered with single-chunk bullets only. Label such a claim honestly and cite every
  chunk it draws on.

Work incrementally
- The namespace persists: every variable, function and ledger task id from an
  earlier turn still exists. Each turn's block contains **only the new step** —
  never re-define ``buckets``, re-create ledger tasks, or re-paste
  helper functions you already wrote.
- Read the chunks before you classify them: print or sub-task the ``chunk_text`` of
  the chunks you intend to cite. A keyword match is a lead, not a finding.

Query buckets (query.json)
<<QUERY_JSON>>

Budget (read-only)
<<BUDGET_LIMITS>>
- Your current usage is appended to each turn as a ``[budget]`` line. It is not
  part of this instruction block, so the block itself never changes and the
  provider can cache it.

Output contract
1. Reply with exactly one fenced ```python block per turn, or with a final answer.
2. Print a compact summary of what you learned — at most ~15 lines. Large values
   belong in a variable or in rlm_api.emit(value, label=...); never print whole
   papers or raw tool payloads.
3. **Retrieve first.** Run `retrieval_funnel`<<BEGIN:kg>> (and `build_kg` only if you will
   traverse the graph)<<END:kg>> and keep the returned chunks in a variable.
   Answering from memory is a failure: cite the real `chunk_uuid` + `arxiv_id` for
   every claim. "See the retrieved evidence" or "[above]" is not a citation. Write
   the **full** 36-character `chunk_uuid` in the answer, even if your digests print
   a shortened one.
4. **Quote before you assert.** For every factual claim, be able to point at the
   sentence in a retrieved chunk that states it. If you cannot quote it, do not
   write it as a finding: put it under what you could not support. A claim whose
   citation only shares the topic is an unsupported claim, and the gate will name it.
   A comparison drawn from several chunks is a claim too: cite all of them.
   A span may appear in quotation marks **only if you received it from a record's
   ``quotes`` array**: copy it unchanged (``…`` marks an omission), and copy the
   ``id`` from the record instead of retyping the uuid. A sentence you did not
   receive as a quote is a claim — write it as prose with its citation, never in
   quotation marks. Retyping is where quotes drift, and a drifted quote is one a
   reader cannot find in the chunk.

5. **A negative finding is a claim too.** If you conclude "no retrieved chunk
   contains X", cite the `chunk_uuid` values you inspected and say how many — the
   counts are the evidence. An uncited "not found" is indistinguishable from not
   having looked. **Report the scope you inspected**, never a global absence:
   "12 chunks of 2511.02817, no match" is checkable; "0 mentions globally" is not,
   and the final gate is right to reject it.
6. To finish, output a line beginning with ``FINAL_ANSWER:`` followed by the answer.
   A normal assistant message without code is NOT a final answer.
7. **Never print ``FINAL_ANSWER:`` inside a ```python block**, and never write it
   in the same message as a block: an answer written before its block ran cannot
   use that block's output, so it is discarded and you are asked again. Answer in
   a message of its own, after you have read the output you need.
8. Record your work in the ledger, not only in prose: ``ledger_create("<what this
   step must establish>")`` when you start a step, ``ledger_update(task_id,
   "done")`` (or ``"failed"``) when it ends, and ``ledger_add_evidence(task_id,
   [chunk_uuid, ...])`` when you have the ids. Create each task once and keep its
   id in a variable. The ledger is what a reviewer reads to see which claims you
   actually supported.
9. **Finish the reading and the printing in the same turn.** When a block completes
   your last extraction and its ledger updates, print in that same block the quotes
   you will cite — one line per quote, ``id: "quote"``, **up to ten quotes** (which
   is what the 3 000-character observation holds). The next turn then writes the
   answer. A turn that only re-prints material you already hold costs a model call
   and buys nothing, and printing fewer quotes than you hold costs coverage.
10. When asked to wrap up (budget nudge), answer from what you already have and
   list the claims you could not support. A labelled partial answer is a valid
   result; an over-budget run is not. On the **last permitted turn** the harness
   asks for the answer only: a code block written then does not run.

<<SUB_TASKS>>

Guardrails
- Stay inside the budget. If a cap is reached, return the best partial answer and
  list what is still unsupported.
<<BEGIN:gate>>
- A ``[gate]`` line at the end of an observation is the decision layer's verdict on
  the stage that just finished. It is not calibrated: use it to decide whether to
  check coverage in the chunks you hold, or to refine a **copy** of the buckets.
  Calling the funnel again with the same buckets returns the same chunks.
<<END:gate>>
"""


THIN_SYSTEM_PROMPT = """\
You are the root RLM in LENS, a literature-evidence synthesis harness.

Role
- Answer the research query from the arXiv cs.AI corpus.
- You write Python in a persistent sandbox. Tools are bound as plain functions.
  There is no shell, no network and no key.

Tools
<<TOOL_MANIFEST>>

Payload shapes
- ``retrieval_funnel(buckets)`` -> ``{"chunks": [...], "papers": {...}}``; a chunk
  row has ``chunk_uuid``, ``arxiv_id``, ``section_type``, ``chunk_text``.
- ``fetch_papers`` / ``fetch_chunks`` return rows; ``ledger_*`` and ``backtrack``
  are the work record.

How to work
- <<RETRIEVAL_RULE>>
- The observation is a truncated digest; the payload is in the namespace. Filter
  in code, print counts and ids.
- Read the chunk text before citing it. A keyword hit is a lead, not a finding.
- Record steps in the ledger; it is the reviewer's view of what you supported.

Sub-tasks
- ``rlm_api.sub_extract(chunk_ids, fields=None)`` returns
  ``{"records": [{"id", "quotes", ...}], "missing_ids", "errors"}``. Up to 12 ids
  per call; the host batches. ``quotes`` are verbatim and citable.
<<BEGIN:kg>>
- ``build_kg(chunk_ids)`` exists for relation questions; use it only if you will
  traverse the graph.
<<END:kg>>

Coverage and evidence
- Name the chunk ids that evidence each objective. An objective with no id is a
  gap: narrow the claim or refine a copy of the buckets.
- Cite the full 36-character ``chunk_uuid`` (and the arXiv id) for every claim.
  A prefix is not an id.
- For every factual claim, point at the sentence in a retrieved chunk that states
  it. If you cannot, put it under what you could not support.
- A comparative claim may cite two to four chunks, one sentence from each.
- Quote only a span you received from a record's ``quotes`` array, copied
  unchanged, with the id copied from the record. A sentence you did not receive
  as a quote is prose with its citation.
- A negative finding is a claim: cite the ids you inspected and say how many.
  Never claim a global absence.

Output
- One fenced ```python block per turn, or a final answer.
- Print at most ~15 lines; large values go to ``rlm_api.emit``.
- Finish with ``FINAL_ANSWER:`` as plain text, never inside the code block and
  never in the same message as a block.
- On a budget nudge, answer from what you have and list what you could not support.
- On the last permitted turn, answer only: a code block written then does not run.

Query buckets
<<QUERY_JSON>>

Budget
<<BUDGET_LIMITS>>
- Live usage is the ``[budget]`` line at the end of each turn.
- A ``[gate]`` line is an uncalibrated signal, not an instruction.
"""

NAIVE_SYSTEM_PROMPT = """\
You answer one research question from a fixed evidence set.

You have no tools, no code, and no second turn: read the evidence, then answer.

Rules
- Answer only from the evidence below. Background knowledge may not add facts the
  evidence does not contain.
- Cite the id of every item a claim comes from, in square brackets, next to the
  claim: [<id>]. A claim without an id is not evidence.
- If the evidence does not answer the question, say so, and state how many items
  you inspected and what you looked for. Never invent an answer or an id.
- Reply with the answer only: no preamble, no description of method.
"""

DOCUMENT_SYSTEM_PROMPT = """\
You are the root RLM in a literature-evidence harness, working on a fixed document.

Role
- Answer the research question from the records in the evidence file.
- You write Python. The code runs in a persistent sandbox namespace.
- There is no retrieval tool: the whole evidence set is in the file, and the job
  is to read it and aggregate it correctly.

Question
<<QUESTION>>

Tools
<<TOOL_MANIFEST>>

Evidence file
- The evidence file is ``<<WORKSPACE>>/documents.jsonl``: one JSON object per line,
  with ``id``, ``text`` and ``label``. Read it with
  ``pandas.read_json("<<WORKSPACE>>/documents.jsonl", lines=True)`` or the standard
  library. ``WORKSPACE`` is also defined in the namespace.
- The sandbox's working directory is **not** the workspace, so a bare
  ``documents.jsonl`` will not open: use the absolute path.
- Count explicitly. When the question asks for an aggregate, compute it in code
  and print the aggregate **and** what it was computed over (records inspected,
  labels counted, ids matched).

Query buckets (query.json)
<<QUERY_JSON>>

Budget (read-only)
<<BUDGET_LIMITS>>

Output contract
1. Reply with exactly one fenced ```python block per turn, or with a final answer.
2. Print a compact summary — at most ~15 lines. Large values stay in a variable.
3. **Every count is a claim.** State the number, the field counted, and how many
   records were inspected.
4. To finish, output a line beginning with ``FINAL_ANSWER:`` followed by the answer.
   A normal assistant message without code is NOT a final answer.
5. Never print ``FINAL_ANSWER:`` inside a ```python block.
6. When asked to wrap up, answer from what you already have and list what is still
   unsupported.
"""

_TEMPLATES: dict[str, str] = {
    "full": SYSTEM_PROMPT_TEMPLATE,
    "thin": THIN_SYSTEM_PROMPT,
    "naive": NAIVE_SYSTEM_PROMPT,
    "document": DOCUMENT_SYSTEM_PROMPT,
}

#: The one sentence the retrieval rule contributes. The full prompt keeps its
#: longer paragraph; the thin prompt takes one of these, so the rule can be
#: varied without rewriting the template.
_RETRIEVAL_RULES: dict[str, str] = {
    "once": (
        "Retrieve once with `retrieval_funnel(buckets=buckets)`; do not run the funnel again."
    ),
    "refine": (
        "Retrieve once. If a named objective has no chunk, copy the buckets "
        "(`mine = copy.deepcopy(buckets)`), edit the copy, and retrieve again."
    ),
}

# Marked, optional blocks. ``build_system_prompt`` keeps or drops them from what
# the run actually registered, so the instructions can never advertise a tool the
# harness does not have.
_BLOCK = re.compile(r"<<BEGIN:(\w+)>>\n?(?P<body>.*?)\n?<<END:\1>>", re.DOTALL)


def _render_blocks(template: str, keep: Mapping[str, bool]) -> str:
    """Keep or drop every ``<<BEGIN:name>>…<<END:name>>`` block, then strip markers."""

    def replace(match: re.Match[str]) -> str:
        return match.group("body") if keep.get(match.group(1), True) else ""

    return _collapse_blank_lines(_BLOCK.sub(replace, template))


def _collapse_blank_lines(text: str) -> str:
    """Collapse runs of blank lines, so an empty optional section leaves no gap."""
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text


def build_system_prompt(
    tool_manifest: str,
    query_buckets: Mapping[str, Any],
    budget_limits: Mapping[str, Any],
    *,
    variant: str | None = None,
    kg: bool | None = None,
    sub_calls: bool | None = None,
    question: str = "",
) -> str:
    """Fill the template for ``variant`` (default: the configured one).

    ``budget_limits`` is the **constant** half of the budget snapshot (the caps),
    never the live usage: this string is the first bytes of every request, so a
    field that changes per turn would invalidate the provider's prefix cache.
    Live usage travels in the per-turn tail (:func:`budget_usage_line`).

    ``kg`` and ``sub_calls`` default to the configured flags, so the instruction
    block describes the harness that exists. Replacement is ``str.replace``, not
    ``str.format``: the payloads contain JSON braces.
    """
    chosen = (variant or config.PROMPT_VARIANT).strip().lower()

    if chosen not in _TEMPLATES:
        raise ValueError(f"unknown prompt variant {chosen!r} (one of {sorted(_TEMPLATES)})")

    text = _render_blocks(
        _TEMPLATES[chosen],
        {
            "kg": (config.KG_ENABLED and config.KG_PROMPT) if kg is None else kg,
            "gate": config.GATE_ENABLED,
        },
    )

    sub_calls_on = config.SUB_CALLS_ENABLED if sub_calls is None else sub_calls

    prompt = (
        text.replace("<<SUB_TASKS>>", render_prompt_block() if sub_calls_on else "")
        .replace("<<TOOL_MANIFEST>>", tool_manifest)
        .replace("<<QUERY_JSON>>", json.dumps(dict(query_buckets), ensure_ascii=False, indent=2))
        .replace("<<BUDGET_LIMITS>>", _format_limits(budget_limits))
        .replace("<<WORKSPACE>>", config.SANDBOX_WORKSPACE_PATH.rstrip("/"))
        .replace("<<QUESTION>>", question or "(not provided)")
        .replace("<<RETRIEVAL_RULE>>", _RETRIEVAL_RULES[config.RETRIEVAL_RULE])
    )

    if "<<" in prompt or ">>" in prompt:
        raise ValueError("unfilled placeholder in the system prompt")

    return _collapse_blank_lines(prompt)


def _format_limits(limits: Mapping[str, Any]) -> str:
    """Render the caps as a short, stable list (identical for the whole run)."""
    iterations = limits.get("iterations")
    tokens = limits.get("tokens")
    cost = limits.get("cost_usd")
    block_s = limits.get("exec_block_s")
    backtracks = limits.get("backtracks")
    sub_tasks = limits.get("sub_tasks")

    return "\n".join(
        [
            f"- {iterations:,} iterations per query"
            if isinstance(iterations, int)
            else "- iterations: not set",
            f"- {tokens:,} tokens per query (input + output, sub-tasks included)"
            if isinstance(tokens, int)
            else "- tokens: not set",
            f"- ${float(cost):.2f} per query" if cost is not None else "- cost: not set",
            f"- {float(block_s):.0f} s per code block (not a run total)"
            if block_s is not None
            else "- block time: not set",
            f"- {backtracks} backtracking retries"
            if isinstance(backtracks, int)
            else "- backtracks: not set",
            f"- {sub_tasks} sub-task calls per query"
            if isinstance(sub_tasks, int)
            else "- sub-tasks: not set",
        ]
    )


def budget_usage_line(snapshot: Mapping[str, Any]) -> str:
    """Return the per-turn usage line that rides at the **end** of the prompt.

    Two reasons it is a tail line and not part of the system prompt: the numbers
    change after every call, and a changing system prompt means no provider can
    reuse the prefix (measured on a recording: 80 % of the input was the same
    re-sent prefix, and none of it was billed as cached).

    Everything is read from ``snapshot()``; ``left`` is ignored because it is
    clamped and hides an overshoot.
    """
    limits = snapshot.get("limits") or {}
    used = snapshot.get("used") or {}

    iterations = int(used.get("iterations") or 0)
    max_iterations = int(limits.get("iterations") or 0)
    tokens_in = int(used.get("tokens_in") or 0)
    tokens_out = int(used.get("tokens_out") or 0)
    max_tokens = int(limits.get("tokens") or 0)
    cost = float(used.get("cost_usd") or 0.0)
    max_cost = float(limits.get("cost_usd") or 0.0)
    backtracks = int(used.get("backtracks") or 0)
    max_backtracks = int(limits.get("backtracks") or 0)

    return (
        f"[budget] turn {iterations}/{max_iterations} · "
        f"tokens {tokens_in + tokens_out:,}/{max_tokens:,} "
        f"(in {tokens_in:,} + out {tokens_out:,}) · "
        f"cost ${cost:.4f}/${max_cost:.2f} · "
        f"backtracks {backtracks}/{max_backtracks}"
    )


def build_messages(
    *,
    system_prompt: str,
    memory: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the root message list: one system message + rolling memory.

    ``memory`` is a **sequence** of message dicts (not a dict). Each entry is
    copied, so mutating the returned list cannot edit the caller's history, and
    extra keys (e.g. ``name``) survive instead of being dropped.
    """
    messages = [{"role": "system", "content": system_prompt}]

    for message in memory or []:
        messages.append(dict(message))

    return messages


# tiktoken is imported lazily: the host has it (GLiNER needs it), the sandbox
# does not, and ``--help`` should not pay for it.
_ENCODING: Any = None


def count_tokens(text: str) -> int:
    """Estimate the token count of ``text`` for context-budget decisions.

    Uses ``tiktoken``'s ``o200k_base`` (the GPT-4o/GPT-5 family encoding) with a
    documented fallback of ``len // 4`` when tiktoken is unavailable. For a model
    with a different tokenizer this is an approximation, in the conservative
    direction on prose; it only decides when to compact, because actual usage comes
    from ``CallResult``. Compare a turn's estimate with its reported ``tokens_in``
    before trusting the margin.
    """
    global _ENCODING

    if not text:
        return 0

    if _ENCODING is None:
        try:
            import tiktoken

            # ``get_encoding`` takes an *encoding* name; ``encoding_for_model``
            # takes a model name ("gpt-4o") and raises on "o200k_base".
            _ENCODING = tiktoken.get_encoding("o200k_base")
        except Exception:  # tiktoken missing, or the encoding renamed upstream
            _ENCODING = False

    return len(_ENCODING.encode(text)) if _ENCODING else max(1, len(text) // 4)


# A chat format adds a few structural tokens per message (role markers,
# separators). 4 is the documented ballpark for the OpenAI chat format.
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Estimate the prompt size of a message list.

    **One estimator for both decisions** — the compaction trigger and the target
    size. Summing per message + framing overhead approximates what the API
    tokenises; ``count_tokens(json.dumps(messages))`` counts JSON escaping and the
    ``"role"``/``"content"`` keys, which the model never sees, and inflates
    non-proportionally for code blocks and markdown. Mixing the two would let a
    "compacted" list immediately trip the trigger again.
    """
    return sum(
        count_tokens(str(message.get("content") or "")) + MESSAGE_OVERHEAD_TOKENS
        for message in messages
    )


def context_trigger_tokens(remaining_tokens: int) -> int:
    """Return the prompt size at which compaction runs, from the budget that remains.

    Derived from the budget rather than from the model's window because the window
    is not what binds: with a ten-turn cap the largest prompt a run reaches is a few
    tens of thousands of tokens against a 400,000-token window. The orchestrator
    calls this **once**, at the start of the loop, and keeps the number: a trigger
    recomputed every turn shrank as the run spent down, so it fired again on each of
    the last turns and every firing re-wrote the start of the context, which is the
    part the provider caches.
    """
    if remaining_tokens <= 0:
        return config.CONTEXT_TRIGGER_FLOOR

    return max(
        config.CONTEXT_TRIGGER_FLOOR,
        int(remaining_tokens * config.CONTEXT_TRIGGER_RATIO_OF_REMAINING),
    )


def compact_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    target: int | None = None,
    start: int = 0,
) -> list[dict[str, Any]]:
    """Return a compacted message list when the estimated size crosses ``budget``.

    ``target`` (default ``budget``) is the size a compaction shrinks to, and
    ``start`` is how many non-system messages an earlier compaction already dropped
    (see :func:`compaction_start`, which the orchestrator calls to keep it sticky).

    Policy: keep the system message and the most recent turns; replace or evict
    older observations. Compaction must be deterministic and must never drop the
    final-answer instruction.

    How it shrinks:

    - the **target** is a fixed size (the trigger, unless the caller asks for less
      headroom). A budget derived from the *current* bloated size would grow with
      the problem and never converge.
    - the **system message is never a candidate** — it carries the output
      contract, the tool manifest and the final-answer instruction.
    - the **oldest** messages are evicted until the rest fits the target, so the
      turn the model is about to act on always survives.
    - the newest message is always kept, even if it alone exceeds the target: an
      empty history is worse than a slightly over-budget one.
    - one **omission marker** records how many turns went and that the sandbox
      namespace still holds their data, even if it pushes the list slightly over
      the budget. It is a ``user`` message rather than text
      appended to the system message, so the system prefix stays byte-identical
      and prefix caching keeps hitting.

    The size test is :func:`estimate_tokens` — the **same estimator** the caller
    uses to decide *when* to compact, so the two decisions cannot disagree.
    """
    system = [dict(message) for message in messages if message.get("role") == "system"]
    rest = [dict(message) for message in messages if message.get("role") != "system"]

    begin = compaction_start(
        rest,
        system_tokens=estimate_tokens(system),
        start=start,
        trigger=budget,
        target=budget if target is None else target,
    )
    kept = rest[begin:]

    if begin:
        # Inserted even when it pushes the list a few tokens over budget. The
        # marker is ~30 tokens and it is the only trace that compaction happened:
        # skipping it when the budget was tight made a compacted run look like a run
        # whose model suddenly wrote shorter turns, to the analyst and to the model.
        kept.insert(0, omission_marker(begin))

    return system + kept


def omission_marker(dropped: int) -> dict[str, str]:
    """Return the ``user`` message that stands in for ``dropped`` evicted messages."""
    return {
        "role": "user",
        "content": f"[{dropped} earlier messages omitted; sandbox variables persist across turns]",
    }


def compaction_start(
    memory: Sequence[Mapping[str, Any]],
    *,
    system_tokens: int,
    start: int,
    trigger: int,
    target: int,
) -> int:
    """Return the index of the first ``memory`` message to keep.

    **Sticky**: the index never moves back, and it only moves forward when the kept
    history (from ``start``) is over ``trigger`` — then far enough that it fits
    ``target``. Between two compactions the prompt is the previous prompt plus the
    new turn, byte for byte, so the provider's prefix cache keeps hitting. Sliding
    the window every turn instead re-wrote the message right after the system prompt
    on every turn: measured, the cached share fell from 94 % to the system prompt
    alone (2,176 of 17,553 tokens) on both compacted turns.

    The newest message is always kept, even when it alone exceeds ``target``.
    """
    costs = [
        count_tokens(str(message.get("content") or "")) + MESSAGE_OVERHEAD_TOKENS
        for message in memory
    ]
    begin = min(max(0, start), max(0, len(memory) - 1))
    size = system_tokens + sum(costs[begin:])

    if size <= trigger:
        return begin

    while begin < len(memory) - 1 and size > target:
        size -= costs[begin]
        begin += 1

    return begin


WRAP_UP_NUDGE_TEMPLATE = """\
Budget at {pct:.0f}% ({which}). Stop exploring: emit FINAL_ANSWER now from the
evidence you already hold, and list the claims you could not support."""

#: Appended instead of the nudge when this is the last root call the iteration cap
#: permits. It is imperative because the advisory nudge was not enough: a recorded
#: run spent its final turn on a failed ledger call and wrote no answer at all.
LAST_TURN_INSTRUCTION = """\
This is the last turn the iteration cap permits. Reply with FINAL_ANSWER: and no
code block: a block written now will not run. Use the evidence you already hold,
and list the claims you could not support."""

# The caps that can stop the loop, in the order stop_reason() reports them.
# exec_seconds is deliberately absent: it is a per-block limit enforced by the
# runner, not a run budget.
_WARNING_CAPS: tuple[str, ...] = ("iterations", "tokens", "cost_usd", "backtracks")


def budget_warning(
    snapshot: Mapping[str, Any],
    *,
    ratio: float = config.CONTEXT_WARN_RATIO,
) -> str | None:
    """Return the wrap-up nudge when any cap is at/over ``ratio``, else ``None``.

    This is the difference between a *degraded* run and a *truncated* one: the
    root gets one turn of warning to land the plane, so hitting a hard cap
    produces a labelled partial answer instead of a half-finished thought.

    Input is ``Budget.snapshot()``: ``{"limits": {...}, "used": {...}, "left":
    {...}, "stop_reason": ...}`` — read ``limits`` and ``used``, never ``left``
    (which is clamped and loses the overshoot).

    ``{pct}`` is the fraction **consumed** of the binding cap, in percent;
    ``{which}`` names that cap **with the absolute burn** (``tokens
    255,000/300,000``), because the percentage tells the model how urgent it is
    and the absolute numbers tell it what it can still afford. The token cap
    counts **input + output**, so both are summed.

    When several caps are over the threshold, the **most consumed** one is
    reported. Returns ``None`` when every cap is still below it — the orchestrator
    then appends nothing, so the prompt stays byte-identical (prefix cacheable).
    """
    limits = snapshot.get("limits") or {}
    used = snapshot.get("used") or {}

    amounts: dict[str, tuple[float, float]] = {
        "iterations": (
            float(used.get("iterations") or 0),
            float(limits.get("iterations") or 0),
        ),
        "tokens": (
            float((used.get("tokens_in") or 0) + (used.get("tokens_out") or 0)),
            float(limits.get("tokens") or 0),
        ),
        "cost_usd": (
            float(used.get("cost_usd") or 0.0),
            float(limits.get("cost_usd") or 0.0),
        ),
        "backtracks": (
            float(used.get("backtracks") or 0),
            float(limits.get("backtracks") or 0),
        ),
    }

    best: tuple[float, str, float, float] | None = None

    for key in _WARNING_CAPS:
        consumed, limit = amounts[key]

        if limit <= 0:
            continue  # a cap of zero is "disabled", not "instantly exhausted"

        fraction = consumed / limit
        if fraction >= ratio and (best is None or fraction > best[0]):
            best = (fraction, key, consumed, limit)

    if best is None:
        return None

    fraction, key, consumed, limit = best
    if key == "cost_usd":
        which = f"cost ${consumed:.4f}/${limit:.4f}"
    else:
        which = f"{key} {consumed:,.0f}/{limit:,.0f}"

    return WRAP_UP_NUDGE_TEMPLATE.format(pct=fraction * 100, which=which)


def truncate_observation(text: str, max_chars: int = config.TOOL_RESULT_SUMMARY_CHARS) -> str:
    """Truncate a printed observation with an explicit marker.

    The marker goes at the **end**, where the text actually stops: the model reads
    the head first, and a notice belongs at the cut, not in front of the result.
    Text that already fits is returned **unchanged** — prefixing "Truncated:" to a
    short observation would be a lie.

    The marker is always emitted, so the result can exceed ``max_chars`` by the
    marker's length when ``max_chars`` is smaller than the marker itself (only
    reachable with an absurdly small cap; production uses 3,000).
    """
    if len(text) <= max_chars:
        return text

    marker = f"\n…[truncated: {len(text)} chars total]"
    keep = max(0, max_chars - len(marker))

    return text[:keep] + marker


# A fenced block: optional language tag, then anything (non-greedy), then the
# closing fence. The closing fence is REQUIRED: a missing one almost always means
# the response was cut off, and executing half a script is worse than retrying.
_FENCE_RE = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\r?\n(?P<code>.*?)(?:\r?\n)?```",
    re.DOTALL,
)

# The prompt asks for ```python, but models also emit ```py, ```python3, ```Python
# or a bare ```. Anything else (```json, ```bash) must never be executed.
_PYTHON_LANGS: frozenset[str] = frozenset({"", "python", "py", "python3"})


def extract_python_block(text: str) -> str | None:
    """Return the body of the first fenced ```python block, or ``None``.

    Matches a fenced block whose language tag is empty or python-ish
    (``python`` / ``py`` / ``python3``, case-insensitive) and returns the first
    one. Newlines at the block's edges are stripped but **indentation is
    preserved** (never use ``str.strip()`` here).
    """
    for match in _FENCE_RE.finditer(text or ""):
        language = (match.group("lang") or "").strip().lower()

        if language in _PYTHON_LANGS:
            return match.group("code").strip("\n")

    return None


def extract_final_answer(text: str) -> str | None:
    """Return the text after the last ``FINAL_ANSWER:`` that sits **outside** a fence.

    Fenced blocks are removed before the search, for two reasons found in the
    smoke :

    - A sentinel printed from Python (``print("FINAL_ANSWER: ...")``) belongs to the
      block, not to the conversation. It must not end a run.
    - The opposite failure was just as damaging: a model that ran code **and** then
      appended a real answer after the fence never finished, because "code wins"
      sent the turn back to the loop until the iteration cap, and ``answer.md`` came
      out empty.

    Removing fenced text separates those two cases without guessing. The **last**
    outside-the-fence marker wins, because the model may quote the sentinel while
    reasoning ("I will emit FINAL_ANSWER: when done").
    """
    marker = config.FINAL_ANSWER_MARKER
    outside = _FENCE_RE.sub(" ", text or "")

    if marker not in outside:
        return None

    answer = outside.rsplit(marker, 1)[-1].strip()

    return answer or None


def visible_text(text: str) -> str:
    """Return the prose part of a completion: fenced blocks removed.

    Used when the loop has no turn left to ask again: the reply is not executed,
    so the model's own words are kept as a labelled partial answer. A block is
    not an answer, and a block written after the final turn cannot inform one.
    """
    prose = _FENCE_RE.sub(" ", text or "")

    return re.sub(r"\n{3,}", "\n\n", prose).strip()


# Static: no per-run numbers (those live in budget.md) and no tool list (that is
# tools.md), so this file cannot drift from the system prompt or go stale. The RLM
# re-reads it after compaction or on resume, when the prompt may have been
# truncated — it is the workspace's copy of the rules, not a second prompt.
AGENTS_MD = """\
# You are the root RLM in a LENS run

## Role
Answer the research query from the arXiv cs.AI corpus. You write Python; it runs
in a persistent sandbox namespace, so variables survive between turns.

## Hard rules
- **Retrieve before you answer.** Call `retrieval_funnel` once (and `build_kg` only
  when you will traverse the graph), keep the result, and cite the full
  `chunk_uuid`s you actually read.
- **Work incrementally.** Variables persist; never re-run the funnel with unchanged
  buckets or re-write code from an earlier turn. To retrieve again, copy the buckets
  (`copy.deepcopy(buckets)`) and edit the copy — `buckets` is the segmented query.
- An answer with no citation is treated as unsupported by the final gate —
  "see the retrieved evidence" or "[above]" is not a citation.
- **Search variables, not printed digests.** The observation is a truncated
  summary; the payload is in the namespace. Filter in code and print counts/ids.
  A grep miss on your own digest is not evidence of absence — and an uncited
  "not found" is indistinguishable from not having looked.
- No shell, no network, no API key. The only effects are the bound tool functions
  (`retrieval_funnel(buckets=buckets)`), `rlm_api.tool(name, **kwargs)` (the same
  dispatch), `rlm_api.sub_extract()` / `sub_summarise()` / `sub_classify()`,
  `rlm_api.emit()`, `rlm_api.note()` and
  `rlm_api.budget()`.
- Never invent evidence. Cite `chunk_uuid` + `arxiv_id` for every claim. A span in
  quotation marks must be one you received from a record's `quotes` array, copied
  unchanged, with the `id` copied from the record — not retyped.
- One fenced ```python block per turn. Print a compact digest (<= 15 lines);
  large values go to `rlm_api.emit(value, label="name")`.
- Finish with a line starting `FINAL_ANSWER:` **as plain text**, never inside the
  code block: code wins over the sentinel, so an answer printed from Python is
  discarded and the turn keeps working.
- Record steps in the ledger (`ledger_create` / `ledger_update` /
  `ledger_add_evidence`). A reviewer reads it to see which claims you supported.
- **Finish the reading and the printing in the same block.** When a block completes
  your last extraction and its ledger updates, print the quotes you will cite in
  that same block (one line per quote, `id: "quote"`, up to ten). A turn that only
  re-prints what you already hold costs a model call; printing fewer quotes than
  you hold costs coverage.

## Files in /workspace
| File | What it is |
|---|---|
| `query.json` | the query split into objectives / problems / constraints / requirements |
| `tools.md` | the host tools you can call, with signatures |
| `budget.md` | this run's caps (snapshot at start; `rlm_api.budget()` gives live values) |
| `emits/` | values you have offloaded with `emit()` |
"""


#: The KG traversal helpers and the signature each one is documented with. The
#: signatures are the module's real ones: ``relation``/``direction`` and
#: ``head``/``relation``/``tail`` are **keyword-only**, and the default direction is
#: ``"out"`` — documenting them otherwise would send the model into a ``TypeError``.
KG_HELPER_SIGNATURES: dict[str, str] = {
    "kg_neighbors": '(graph, entity, *, relation=None, direction="out")',
    "kg_triples": "(graph, *, head=None, relation=None, tail=None)",
    "kg_subgraph": "(graph, entities)",
}


#: Appended to ``tools.md`` when the KG is enabled: the traversal helpers are not
#: RPC tools, so without this block the only file the model is told to read for
#: tools lists ``build_kg`` alone.
KG_HELPERS_MD = """\
# Knowledge graph (sandbox helpers)

`build_kg(chunk_ids)` returns a payload whose `graph` is node-link JSON. Load it
with `networkx.node_link_graph(payload["graph"])`; the helpers below are bound as
plain functions in the namespace (importing `kg_query` works too):

| helper | signature |
|---|---|
| `kg_neighbors` | `kg_neighbors(graph, entity, *, relation=None, direction="out")` |
| `kg_triples` | `kg_triples(graph, *, head=None, relation=None, tail=None)` |
| `kg_subgraph` | `kg_subgraph(graph, entities)` |

`direction` is `"out"` (the entity is the head), `"in"` or `"both"`.

`ontology.md` in this workspace lists the entity types and the relations, with
the head and tail types each relation allows. Read it before traversing: a
relation name that is not in that file is not in the graph.

Names are matched case-insensitively against the graph's normalised node keys, so
`kg_neighbors(graph, "BERT")` finds the node `build_kg` stored for BERT.

Every edge carries `direction_verified` / `direction_confidence` /
`direction_source`. `true` means the decision layer confirmed the direction of the
relation; `jev_uncertain` and `unverified` mean the edge was kept but not
confirmed. Prefer a confirmed edge when two claims disagree, and do not state an
unverified relation as a fact.
"""


def kg_bindings() -> str:
    """Return sandbox code that binds the KG helpers as plain functions.

    Why this exists: the prompt told the model that ``kg_query`` was "KG traversal
    on the graph variable", but the helpers were not bound like the host tools and
    were absent from ``tools.md``, so ``kg_neighbors(G, "BERT")`` was a ``NameError``
    and a model that read the tool list concluded the graph had one entry point:
    build it, never read it.

    Each wrapper also counts calls and **notes the first one** through
    ``rlm_api.note``. That note is the use signal the KG on/off ablation was missing:
    without it, "the graph was available" and "the graph was traversed" were
    indistinguishable in the trajectory.

    The import is guarded: an image without ``kg_query`` must still seed, because a
    run that never touches the graph is legitimate.
    """
    signatures = ",\n".join(
        f"    {name!r}: {signature!r}" for name, signature in KG_HELPER_SIGNATURES.items()
    )

    return "\n".join(
        [
            "try:",
            "    import kg_query as _kg_query",
            "except Exception:",
            "    _kg_query = None",
            "try:",
            "    import rlm_api as _rlm_api",
            "except Exception:",
            "    _rlm_api = None",
            "KG_HELPERS = {",
            signatures,
            "}",
            "_KG_USED = {'count': 0}",
            "",
            "def _kg_bind(_name, _signature):",
            "    _fn = getattr(_kg_query, _name)",
            "",
            "    def _call(*args, **kwargs):",
            "        _KG_USED['count'] += 1",
            "        if _KG_USED['count'] == 1 and _rlm_api is not None:",
            "            try:",
            "                _rlm_api.note('kg used: ' + _name)",
            "            except Exception:",
            "                pass",
            "        return _fn(*args, **kwargs)",
            "",
            "    _call.__name__ = _name",
            "    _call.__doc__ = 'KG helper ' + _name + _signature",
            "    return _call",
            "",
            "if _kg_query is not None:",
            "    for _name, _signature in KG_HELPERS.items():",
            "        globals()[_name] = _kg_bind(_name, _signature)",
        ]
    )


def tool_bindings(parameters: Mapping[str, Sequence[str]]) -> str:
    """Return the sandbox code that binds each host tool as a plain function.

    Why this exists: the manifest the prompt shows prints
    ``shallow_match(buckets: dict[str, Any]) -> list[str]``, so the model writes
    ``shallow_match(buckets=...)``. The façade only exposed ``rlm_api.tool(...)``,
    so that call was a ``NameError`` on every turn — the model then reported
    "retrieval unavailable" and never retrieved anything.

    The block defines one module-level name per tool and maps positional
    arguments onto the tool's real parameter names, then delegates to
    ``rlm_api.tool`` — one dispatch path, one trajectory event, no duplicated
    logic in the sandbox. Keyword arguments pass straight through, and an
    omitted argument keeps the host-side default.
    """
    rows = ",\n".join(f"    {name!r}: {tuple(names)!r}" for name, names in parameters.items())

    return "\n".join(
        [
            # Imported under its real name: the docs and AGENTS.md tell the model to
            # call `rlm_api.emit()/budget()` and the sub-tasks, and an alias meant
            # `NameError: name 'rlm_api' is not defined`.
            "import rlm_api",
            "TOOL_PARAMETERS = {",
            rows,
            "}",
            "",
            "def _bind_tool(_name):",
            "    _keys = TOOL_PARAMETERS[_name]",
            "",
            "    def _call(*args, **kwargs):",
            "        if len(args) > len(_keys):",
            "            raise TypeError(",
            '                f"{_name} accepts at most {len(_keys)} positional argument(s)"',
            "            )",
            "        kwargs.update(zip(_keys, args))",
            "        return rlm_api.tool(_name, **kwargs)",
            "",
            "    _call.__name__ = _name",
            "    _call.__doc__ = f\"Host tool {_name}({', '.join(_keys)}).\"",
            "    return _call",
            "",
            "for _name in TOOL_PARAMETERS:",
            "    globals()[_name] = _bind_tool(_name)",
        ]
    )


def write_context_files(
    workspace: str | Path,
    *,
    tool_manifest: str,
    budget_snapshot: Mapping[str, Any],
    ontology_summary: Mapping[str, Any] | None = None,
    kg_helpers: bool = False,
) -> dict[str, Path]:
    """Write the sandbox context files and return their paths.

    Files: ``AGENTS.md`` (role + rules), ``tools.md`` (the manifest the prompt is
    built from, so the RLM can re-read it from disk), ``budget.md``,
    ``ontology.md`` only when ``ontology_summary`` is given (KG enabled), and a
    "Knowledge graph (sandbox helpers)" section appended to ``tools.md`` when
    ``kg_helpers`` is set.

    **Not** ``query.json``: owns that file
    (:func:`lens.agent.segmentation.write_query_json`) and writes it into the same
    workspace. Two writers for one file is how an artifact silently diverges from
    the context the model saw.

    Every ``.md`` is Markdown the model can read: prose for ``AGENTS.md``, a
    fenced JSON block for the data files (still valid Markdown, still parseable).
    Use UTF-8 and ``ensure_ascii=False`` so query text round-trips losslessly.
    Paths returned are **host** paths; the RLM sees the same files under
    ``config.SANDBOX_WORKSPACE_PATH``.
    """
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    agents_path = root / "AGENTS.md"
    agents_path.write_text(AGENTS_MD, encoding="utf-8")
    paths["AGENTS.md"] = agents_path

    tools_path = root / "tools.md"
    text = tool_manifest
    if kg_helpers:
        text = text.rstrip() + "\n\n" + KG_HELPERS_MD
    tools_path.write_text(text, encoding="utf-8")
    paths["tools.md"] = tools_path

    def _fenced(payload: Mapping[str, Any]) -> str:
        return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n"

    budget_path = root / "budget.md"
    budget_path.write_text(
        "# Budget\n\nSnapshot at run start; call `rlm_api.budget()` for live values.\n\n"
        + _fenced(budget_snapshot),
        encoding="utf-8",
    )
    paths["budget.md"] = budget_path

    if ontology_summary is not None:
        ontology_path = root / "ontology.md"
        ontology_path.write_text(
            "# Ontology\n\nEntity and relation types present in the knowledge graph.\n\n"
            + _fenced(ontology_summary),
            encoding="utf-8",
        )
        paths["ontology.md"] = ontology_path

    return paths
