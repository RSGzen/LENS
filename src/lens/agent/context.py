"""Root-context assembly, observation formatting, and compaction.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §4a (context discipline) and §5.

The root context is deliberately thin: a system prompt, a tool manifest (names,
signatures, docstrings — not source), ``query.json``, the budget snapshot, and
rolling working memory. Large tool results stay in sandbox variables or are
offloaded with ``emit()``; only printed summaries become observations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lens import config

# Draft v0. The student owns the final wording; placeholders are replaced by
# ``build_system_prompt`` so literal JSON braces in the prompt stay safe.
SYSTEM_PROMPT_TEMPLATE = """\
You are the root RLM in LENS, a literature-evidence synthesis harness.

Role
- Answer the research query from the arXiv cs.AI corpus.
- You write Python. The code runs in a persistent sandbox namespace.
- Use the host tools for retrieval and the knowledge graph. Do not invent evidence.
- The sandbox has no shell, no network, and no API key.

Tools
<<TOOL_MANIFEST>>

Query buckets (query.json)
<<QUERY_JSON>>

Budget (read-only)
<<BUDGET>>

Output contract
1. Reply with exactly one fenced ```python block per turn, or with a final answer.
2. Print a compact summary of what you learned. Large values belong in a variable
   or in rlm_api.emit(value, label=...); do not print whole papers.
3. Cite evidence by chunk_uuid and arxiv_id in the final answer.
4. To finish, output a line beginning with ``FINAL_ANSWER:`` followed by the answer.
   A normal assistant message without code is NOT a final answer.

Guardrails
- No recursion: rlm_api.sub_call() is a plain GPT-5-mini call and cannot use tools.
- Stay inside the budget. If a cap is reached, return the best partial answer and
  list what is still unsupported.
"""


def build_system_prompt(
    tool_manifest: str,
    query_buckets: Mapping[str, Any],
    budget_snapshot: Mapping[str, Any],
) -> str:
    """Fill ``SYSTEM_PROMPT_TEMPLATE`` without interpreting JSON braces."""
    raise NotImplementedError("M8 scaffold: implement build_system_prompt")


def build_messages(
    *,
    system_prompt: str,
    memory: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the root message list: one system message + rolling memory."""
    raise NotImplementedError("M8 scaffold: implement build_messages")


def count_tokens(text: str) -> int:
    """Estimate the token count of ``text`` for context-budget decisions.

    Use a local tokenizer (``tiktoken``) with a documented fallback. This is an
    estimate, not a billing value; actual usage comes from ``CallResult``.
    """
    raise NotImplementedError("M8 scaffold: implement count_tokens")


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
    """
    raise NotImplementedError("M8 scaffold: implement compact_messages")


def truncate_observation(text: str, max_chars: int = config.TOOL_RESULT_SUMMARY_CHARS) -> str:
    """Truncate a printed observation with an explicit marker."""
    raise NotImplementedError("M8 scaffold: implement truncate_observation")


def extract_python_block(text: str) -> str | None:
    """Return the body of the first fenced ```python block, or ``None``."""
    raise NotImplementedError("M8 scaffold: implement extract_python_block")


def extract_final_answer(text: str) -> str | None:
    """Return the text after ``FINAL_ANSWER:``, or ``None`` if absent."""
    raise NotImplementedError("M8 scaffold: implement extract_final_answer")


def write_context_files(
    workspace: str | Path,
    *,
    query_buckets: Mapping[str, Any],
    tool_manifest: str,
    budget_snapshot: Mapping[str, Any],
    ontology_summary: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the sandbox context files and return their paths.

    Required files: ``AGENTS.md`` (role + rules), ``query.json``, ``tools.md``,
    ``budget.md``, and optionally ``ontology.md``. Use UTF-8 and
    ``ensure_ascii=False`` so query text round-trips losslessly.
    """
    raise NotImplementedError("M8 scaffold: implement write_context_files")
