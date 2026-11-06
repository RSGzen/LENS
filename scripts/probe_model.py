#!/usr/bin/env python
"""Probe a chat model before a campaign is committed to it.

Four things a recorded run quietly depends on are checked here, on the real
transport, for a few cents:

1. **What the listing says** about the model's reasoning support, so a setting is
   chosen from the provider's answer rather than from an assumption.
2. **The two parsing rules of the loop**: a turn comes back with a fenced python
   block, and a final turn carries the ``FINAL_ANSWER:`` sentinel.
3. **The prompt cache**: the same request twice, and whether the second one is
   served (partly) from cache. A baseline that cannot cache is compared to a loop
   that does, so the cache share belongs next to every cost number.
4. **The sub-task contract**: a strict JSON schema round-trips into records with
   host-stamped ids. A provider that ignores the schema turns every sub-task into
   a parse failure, and the failure is silent until a run spends a turn on it.

Spend is cents. The run directory is left in place, so every request body and
every reply is inspectable with ``scripts/run_report.py``.

    python scripts/probe_model.py
    python scripts/probe_model.py --model deepseek/deepseek-v4.1-flash --provider wafer/us
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from lens import config
from lens.agent.context import extract_final_answer, extract_python_block
from lens.agent.sub_tasks import (
    get_task,
    parse_result,
    render_instruction,
    render_items,
    response_format,
    validate_params,
)
from lens.clients import LensClient
from lens.observability.trajectory import TraceCtx, Trajectory

#: A stand-in for the system prompt's output rules. The real prompt is longer; the
#: two rules that break here are the fenced block and the sentinel, and both are
#: stated exactly as the runtime states them.
SYSTEM = (
    "You solve a task by writing Python. Reply with one ```python block. "
    "Nothing outside the block unless you are finishing. To finish, write "
    "FINAL_ANSWER: <answer> on its own line."
)
TASK = "Compute the sum of 2 and 2, and finish with the value."

#: Two short items shaped like chunks, for the sub-task check.
ITEMS = [
    (
        "11111111-1111-5111-8111-111111111111",
        "We evaluate on ChartQA and report 83.4% accuracy. The method uses a "
        "retrieval module that groups chunks into longer contexts.",
    ),
    (
        "22222222-2222-5222-8222-222222222222",
        "Our model reaches 81.2% accuracy on ChartQA. Latency is 1.4 s per query.",
    ),
]


def _listing(model_id: str) -> dict[str, Any]:
    """Return the model's entry from the provider's model list, or an empty dict."""
    url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/models"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    for entry in payload.get("data") or []:
        if isinstance(entry, dict) and entry.get("id") == model_id:
            return {
                "reasoning": entry.get("reasoning"),
                "context_length": entry.get("context_length"),
                "pricing": entry.get("pricing"),
            }

    return {"error": f"{model_id} is not in the listing"}


def _code_turn(client: LensClient, ctx: TraceCtx) -> dict[str, Any]:
    """One turn that must produce a fenced block and a sentinel."""
    result = client.chat(
        "root",
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASK},
        ],
        trace_ctx=ctx,
        max_tokens=config.ROOT_MAX_OUTPUT_TOKENS,
    )

    block = extract_python_block(result.text)
    final = extract_final_answer(result.text)

    return {
        "chars": len(result.text or ""),
        "has_code_block": block is not None,
        "has_sentinel": final is not None,
        "finish_reason": result.finish_reason,
        "reasoning_tokens": result.reasoning_tokens,
        "reasoning_text_chars": len(result.reasoning_text or ""),
        "cached_tokens": result.cached_tokens,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "preview": (result.text or "")[:400],
        "_sentinel_value": final,
    }


#: A standing instruction block big enough to cross the provider's cache
#: granularity (most providers only cache a prefix once it is several hundred
#: tokens). The real system prompt is this size, so the measurement matches it.
FILLER = (
    "Context note. The corpus is a snapshot of arXiv cs.AI papers, split into "
    "chunks of at most 512 tokens. Retrieval returns papers first and chunks "
    "second; every identifier is a UUID that resolves to one chunk of one paper. "
    "A claim is only citable when a verbatim span of the chunk is quoted beside "
    "its identifier. " * 40
)


def _cache_turn(client: LensClient, ctx: TraceCtx) -> dict[str, Any]:
    """Repeat the identical request and read the cache fields off the second call."""
    messages = [
        {"role": "system", "content": SYSTEM + "\n\n" + FILLER},
        {"role": "user", "content": "Reply with the single word: ready."},
    ]
    first = client.chat("root", messages, trace_ctx=ctx, max_tokens=64)
    second = client.chat("root", messages, trace_ctx=ctx, max_tokens=64)

    return {
        "first_cached": first.cached_tokens,
        "second_cached": second.cached_tokens,
        "second_tokens_in": second.tokens_in,
        "hit": second.cached_tokens > 0,
        "cost_usd": first.cost_usd + second.cost_usd,
    }


def _sub_task_turn(client: LensClient, ctx: TraceCtx) -> dict[str, Any]:
    """One call shaped exactly like a sub-task, parsed by the real parser."""
    task = get_task("extract")

    if task is None:
        return {"error": "no extract sub-task is registered"}

    allowed = [item_id for item_id, _text in ITEMS]
    params = validate_params(task, {"chunk_ids": allowed, "fields": ["claims"]})
    instruction = render_instruction(task, params, render_items(ITEMS))

    result = client.chat(
        "sub",
        [
            {"role": "system", "content": task.system},
            {"role": "user", "content": instruction},
        ],
        trace_ctx=ctx,
        response_format=response_format(task, params),
        max_tokens=task.max_output_tokens,
    )

    records, missing, errors = parse_result(task, result.text, allowed)

    return {
        "records": len(records),
        "missing_ids": missing,
        "errors": errors,
        "finish_reason": result.finish_reason,
        "tokens_out": result.tokens_out,
        "reasoning_tokens": result.reasoning_tokens,
        "cost_usd": result.cost_usd,
        "parsed": bool(records) and not errors,
        "preview": (result.text or "")[:400],
    }


def _status(value: Any) -> str:
    """Render a boolean check as the word a reader scans for."""
    return "PASS" if value else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.MODEL_REGISTRY["root"].id)
    parser.add_argument(
        "--provider",
        default=",".join(config.MODEL_REGISTRY["root"].provider_only),
        help="provider slug to pin (empty: no pin)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"probe-{stamp}"
    trajectory = Trajectory(run_id)
    client = LensClient(trajectory)

    print(f"model     {args.model}")
    print(f"provider  {args.provider or '(none)'}")
    print(f"run       {trajectory.run_dir}")

    listing = _listing(args.model)
    print(f"\nlisting   {json.dumps(listing, indent=2)}")

    reasoning = config.MODEL_REGISTRY["root"].reasoning
    print(f"reasoning {json.dumps(dict(reasoning or {}))}")

    code = _code_turn(client, TraceCtx(run_id, 0, "root"))
    cache = _cache_turn(client, TraceCtx(run_id, 1, "root"))
    sub = _sub_task_turn(client, TraceCtx(run_id, 2, "sub"))

    print(f"\ncode turn    {_status(code['has_code_block'] and code['has_sentinel'])}")
    print(f"  block {code['has_code_block']}  sentinel {code['has_sentinel']}")
    print(f"  finish_reason {code['finish_reason']}  chars {code['chars']}")
    print(
        f"  reasoning {code['reasoning_tokens']} tokens, "
        f"{code['reasoning_text_chars']} chars of text"
    )
    print(f"\ncache        {_status(cache['hit'])}")
    print(
        f"  first {cache['first_cached']} cached, second {cache['second_cached']} "
        f"of {cache['second_tokens_in']} input"
    )
    print(f"\nsub-task     {_status(sub['parsed'])}")
    print(f"  records {sub['records']}  errors {sub['errors']}")
    print(f"  finish_reason {sub['finish_reason']}  output {sub['tokens_out']} tokens")

    spent = sum(
        float(part.get("cost_usd") or 0.0) for part in (code, cache, sub) if isinstance(part, dict)
    )
    print(f"\nspend        ${spent:.6f}")

    report = {
        "created_at": stamp,
        "run_id": run_id,
        "model": args.model,
        "provider_pin": [part for part in args.provider.split(",") if part],
        "reasoning": dict(reasoning or {}),
        "listing": listing,
        "code_turn": code,
        "cache": cache,
        "sub_task": sub,
        "cost_usd": spent,
    }

    out = args.out or (config.DATA_ROOT / "campaigns" / f"model-probe-{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"written      {out}")

    passed = bool(
        code["has_code_block"] and code["has_sentinel"] and sub["parsed"] and cache["hit"]
    )

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
