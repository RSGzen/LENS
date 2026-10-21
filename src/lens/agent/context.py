"""Root-context assembly, observation formatting, and compaction.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a (context discipline) and §5.

The root context is deliberately thin: a system prompt, a tool manifest (names,
signatures, docstrings — not source), ``query.json``, the budget snapshot, and
rolling working memory. Large tool results stay in sandbox variables or are
offloaded with ``emit()``; only printed summaries become observations.

File ownership: ``query.json`` is written by **M6**
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

# Draft v0. The student owns the final wording; placeholders are replaced by
# ``build_system_prompt`` so literal JSON braces in the prompt stay safe.
# Placeholders: <<TOOL_MANIFEST>>, <<QUERY_JSON>>, <<BUDGET>>
SYSTEM_PROMPT_TEMPLATE = """\
You are the root RLM in LENS, a literature-evidence synthesis harness.

Role
- Answer the research query from the arXiv cs.AI corpus.
- You write Python. The code runs in a persistent sandbox namespace.
- Use the host tools for retrieval and the knowledge graph. Do not invent evidence.
- The sandbox has no shell, no network, and no API key.

Tools
<<TOOL_MANIFEST>>

Sandbox modules
- ``kg_query`` — KG traversal on the graph variable you build from build_kg():
  kg_neighbors(G, entity, relation=None, direction="out|in|both"),
  kg_triples(G, head=None, relation=None, tail=None), kg_subgraph(G, entities).
- ``networkx``, ``numpy``, ``pandas`` are importable. There is no shell, no
  network, and no database driver.

Query buckets (query.json)
<<QUERY_JSON>>

Budget (read-only)
<<BUDGET>>

Output contract
1. Reply with exactly one fenced ```python block per turn, or with a final answer.
2. Print a compact summary of what you learned — at most ~15 lines. Large values
   belong in a variable or in rlm_api.emit(value, label=...); never print whole
   papers or raw tool payloads.
3. Cite evidence by chunk_uuid and arxiv_id in the final answer.
4. To finish, output a line beginning with ``FINAL_ANSWER:`` followed by the answer.
   A normal assistant message without code is NOT a final answer.
5. When asked to wrap up (budget nudge), answer from what you already have and
   list the claims you could not support. A labelled partial answer is a valid
   result; an over-budget run is not.

Guardrails
- No recursion: rlm_api.sub_call() is a plain GPT-5-mini call and cannot use tools.
  Use it for long summarisation — root output tokens are the expensive ones.
- Stay inside the budget. If a cap is reached, return the best partial answer and
  list what is still unsupported.
"""


def build_system_prompt(
    tool_manifest: str,
    query_buckets: Mapping[str, Any],
    budget_snapshot: Mapping[str, Any],
) -> str:
    """Fill ``SYSTEM_PROMPT_TEMPLATE`` without interpreting JSON braces."""
    prompt = (
        SYSTEM_PROMPT_TEMPLATE
        .replace("<<TOOL_MANIFEST>>", tool_manifest)
        .replace("<<QUERY_JSON>>", json.dumps(query_buckets, ensure_ascii=False, indent=2))
        .replace("<<BUDGET>>", json.dumps(budget_snapshot, ensure_ascii=False, indent=2))
    )

    if "<<" in prompt or ">>" in prompt:
        raise ValueError("unfilled placeholder in the system prompt")
    
    return prompt


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

    Uses ``tiktoken``'s ``o200k_base`` (the gpt-5 family encoding) with a
    documented fallback of ``len // 4`` when tiktoken is unavailable. This is an
    **estimate** for compaction decisions, not a billing value — actual usage
    comes from ``CallResult``.
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


def compact_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    token_limit: int = config.CONTEXT_TOKEN_LIMIT,
    ratio: float = config.CONTEXT_COMPACT_RATIO,
) -> list[dict[str, Any]]:
    """Return a compacted message list when the estimated size crosses the limit.

    Policy: keep the system message and the most recent turns; replace or evict
    older observations. Compaction must be deterministic and must never drop the
    final-answer instruction.

    How it shrinks:

    - the **target is the trigger point** (``token_limit * ratio``), so there is
      headroom before the next compaction. A budget derived from the *current*
      bloated size would grow with the problem and never converge.
    - the **system message is never a candidate** — it carries the output
      contract, the tool manifest and the final-answer instruction.
    - messages are taken **newest-first** until the target is reached, so the turn
      the model is about to act on always survives, and stale observations are
      evicted before the (cheap) code blocks that explain which variables exist.
    - the newest message is always kept, even if it alone exceeds the target: an
      empty history is worse than a slightly over-budget one.
    - one **omission marker** records how many turns went and that the sandbox
      namespace still holds their data. It is a ``user`` message rather than text
      appended to the system message, so the system prefix stays byte-identical
      and prefix caching keeps hitting.

    The size test is :func:`estimate_tokens` — the **same estimator** the caller
    uses to decide *when* to compact, so the two decisions cannot disagree.
    """
    budget = max(1, int(token_limit * ratio))

    if estimate_tokens(messages) <= budget:
        return [dict(message) for message in messages]  # copies: never mutate the caller

    system = [dict(message) for message in messages if message.get("role") == "system"]
    rest = [dict(message) for message in messages if message.get("role") != "system"]

    used = estimate_tokens(system)
    kept: list[dict[str, Any]] = []

    for message in reversed(rest):
        cost = count_tokens(str(message.get("content") or "")) + MESSAGE_OVERHEAD_TOKENS

        if kept and used + cost > budget:
            break

        kept.append(message)
        used += cost

    kept.reverse()

    dropped = len(rest) - len(kept)
    if dropped:
        marker = {
            "role": "user",
            "content": (
                f"[{dropped} earlier messages omitted; "
                "sandbox variables persist across turns]"
            ),
        }

        if used + estimate_tokens([marker]) <= budget:
            kept.insert(0, marker)

    return system + kept


WRAP_UP_NUDGE_TEMPLATE = """\
Budget at {pct:.0f}% ({which}). Stop exploring: emit FINAL_ANSWER now from the
evidence you already hold, and list the claims you could not support."""


# The caps that can stop the loop, in the order stop_reason() reports them.
# exec_seconds is deliberately absent: it is a per-block limit enforced by the
# runner, not a run budget (LOG-96).
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
    counts **input + output** (LOG-96), so both are summed.

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


def truncate_observation(
    text: str, max_chars: int = config.TOOL_RESULT_SUMMARY_CHARS
) -> str:
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
    """Return the text after ``FINAL_ANSWER:``, or ``None`` if absent.

    The **last** occurrence wins: the model may quote the sentinel in its
    reasoning ("I will emit FINAL_ANSWER: when done") before actually using it.
    The caller must still prefer a code block over this, so a stray mention cannot
    end a turn that also contains executable code.
    """
    marker = config.FINAL_ANSWER_MARKER

    if marker not in text:
        return None

    return text.rsplit(marker, 1)[-1].strip()


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
- No shell, no network, no API key. The only effects are `rlm_api.tool()`,
  `rlm_api.sub_call()`, `rlm_api.emit()`, `rlm_api.note()` and `rlm_api.budget()`.
- Never invent evidence. Cite `chunk_uuid` + `arxiv_id` for every claim.
- One fenced ```python block per turn. Print a compact digest (<= 15 lines);
  large values go to `rlm_api.emit(value, label="name")`.
- Finish with a line starting `FINAL_ANSWER:`. A message without code is not an
  answer.

## Files in /workspace
| File | What it is |
|---|---|
| `query.json` | the query split into objectives / problems / constraints / requirements |
| `tools.md` | the host tools you can call, with signatures |
| `budget.md` | this run's caps (snapshot at start; `rlm_api.budget()` gives live values) |
| `ontology.md` | KG entity/relation types — only when the KG is enabled |
| `emits/` | values you have offloaded with `emit()` |
"""


def write_context_files(
    workspace: str | Path,
    *,
    tool_manifest: str,
    budget_snapshot: Mapping[str, Any],
    ontology_summary: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the sandbox context files and return their paths.

    Files: ``AGENTS.md`` (role + rules), ``tools.md`` (the manifest the prompt is
    built from, so the RLM can re-read it from disk), ``budget.md``, and
    ``ontology.md`` only when ``ontology_summary`` is given (KG enabled).

    **Not** ``query.json``: M6 owns that file
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
    tools_path.write_text(tool_manifest, encoding="utf-8")
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
