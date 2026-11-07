"""Probe the sub model's reasoning effort on the two structured calls it serves.

The probe replays the **exact prompt that failed at 16,000 output tokens** in a
recorded run: one ``sub_extract`` batch and one claim-extraction batch. Each is
re-run at high / medium / low / off so the effect of reasoning is separated from
the cap, the batch size and the prompt. If the source run has no failed batch,
the probe falls back to a normal four-chunk extract batch and four claim units.

Usage::

    python scripts/probe_sub_reasoning.py
    python scripts/probe_sub_reasoning.py --efforts high,low --run <run_id>

The probe writes one JSON file under ``<data_root>/campaigns`` and prints a
table. It is a live probe (≈ $0.01-0.05), not part of the no-key suite.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lens.config as config
import lens.observability.trajectory as trajectory_module
from lens.agent.sub_tasks import (
    get_task,
    parse_result,
    render_instruction,
    render_items,
    response_format,
    validate_params,
    verify_quotes,
)
from lens.clients import LensClient
from lens.observability.citations import claim_units, recorded_chunk_texts
from lens.observability.claim_extract import DEFAULT_MAX_TOKENS, SYSTEM, _instruction, _schema
from lens.observability.claims import parse_frames, split_long_units, verify_units
from lens.observability.trajectory import TraceCtx, Trajectory

DEFAULT_RUN = "m11-postv10-check.rlm-full-kg-nofiles.aggregation.r2"

EFFORTS: dict[str, dict[str, Any]] = {
    "high": {"enabled": True, "effort": "high"},
    "medium": {"enabled": True, "effort": "medium"},
    "low": {"enabled": True, "effort": "low"},
    "off": {"enabled": False},
}


def _load_prompt(trajectory: Trajectory, event: dict[str, Any]) -> dict[str, Any]:
    return json.loads(trajectory.load_blob(event["prompt_ref"]))


def _failed_event(events: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("call_type") != "chat" or event.get("finish_reason") != "length":
            continue

        if kind == "sub" and event.get("actor") == "sub":
            return event

        if kind == "claim" and (event.get("extra") or {}).get("claim_frames"):
            return event

    return None


def _sub_inputs(
    body: dict[str, Any], texts: dict[str, str]
) -> tuple[list[dict[str, Any]], Any, Any, list[tuple[str, str]]]:
    user = body["messages"][-1]["content"]
    ids = re.findall(r"^\[([0-9a-fA-F-]{36})\]$", user, re.M)
    items = [(chunk_id, texts[chunk_id]) for chunk_id in ids if chunk_id in texts]

    return body["messages"], body.get("response_format"), body.get("max_tokens"), items


def _claim_inputs(
    body: dict[str, Any],
) -> tuple[list[dict[str, Any]], Any, Any, list[tuple[str, list[str]]], list[int]]:
    user = body["messages"][-1]["content"]
    payload = json.loads(user.split("Units\n", 1)[1])
    units = [
        (
            " ".join(str(item.get("text") or "").split()),
            [str(value) for value in item.get("chunk_ids") or []],
        )
        for item in payload
    ]
    indices = [int(item["unit_index"]) for item in payload]

    return body["messages"], body.get("response_format"), body.get("max_tokens"), units, indices


def _sample_sub(
    texts: dict[str, str],
) -> tuple[list[dict[str, Any]], Any, Any, list[tuple[str, str]]]:
    task = get_task("extract")
    assert task is not None
    ids = [chunk_id for chunk_id, body in texts.items() if len(body.strip()) >= 400][:4]
    params = validate_params(task, {"chunk_ids": ids, "fields": ["claims"]})
    items = [(chunk_id, texts[chunk_id]) for chunk_id in params["chunk_ids"]]
    messages = [
        {"role": "system", "content": task.system},
        {"role": "user", "content": render_instruction(task, params, render_items(items))},
    ]

    return messages, response_format(task, params), task.max_output_tokens, items


def _sample_claim(
    answer: str, texts: dict[str, str]
) -> tuple[list[dict[str, Any]], Any, Any, list[tuple[str, list[str]]], list[int]]:
    units = split_long_units(claim_units(answer))[:4]
    indices = list(range(len(units)))
    quotes = {index: [] for index in indices}
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _instruction(units, indices, texts, quotes)},
    ]
    response = {
        "type": "json_schema",
        "json_schema": {"name": "claim_frames", "strict": True, "schema": _schema()},
    }

    return messages, response, DEFAULT_MAX_TOKENS, units, indices


def _run_sub(
    client: LensClient,
    messages: list[dict[str, Any]],
    response: Any,
    max_tokens: int | None,
    items: list[tuple[str, str]],
    stamp: str,
) -> dict[str, Any]:
    task = get_task("extract")
    assert task is not None
    result = client.chat(
        "sub",
        messages,
        trace_ctx=TraceCtx(stamp, 0, "sub"),
        response_format=response,
        max_tokens=max_tokens,
        extra={"probe": "sub_extract"},
    )
    records, missing, errors = parse_result(task, result.text, [item_id for item_id, _ in items])
    records, dropped = verify_quotes(records, items)

    return {
        "call": "sub_extract",
        "items": len(items),
        "finish_reason": result.finish_reason,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "reasoning_tokens": result.reasoning_tokens,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
        "records": len(records),
        "missing": len(missing),
        "errors": errors,
        "quotes_dropped": dropped,
        "quote_count": sum(len(record.get("quotes") or []) for record in records),
    }


def _run_claim(
    client: LensClient,
    messages: list[dict[str, Any]],
    response: Any,
    max_tokens: int | None,
    units: list[tuple[str, list[str]]],
    indices: list[int],
    texts: dict[str, str],
    stamp: str,
) -> dict[str, Any]:
    result = client.chat(
        "sub",
        messages,
        trace_ctx=TraceCtx(stamp, 0, "sub"),
        response_format=response,
        max_tokens=max_tokens,
        extra={"probe": "claim"},
    )

    try:
        payload = json.loads(result.text or "")
    except ValueError:
        payload = None

    # The prompt's unit_index values are global; rebuild a full list so
    # parse_frames can address them, then remap the frames to the batch for the
    # report.
    full = [("", []) for _ in range(max(indices) + 1)]
    for offset, unit in zip(indices, units):
        full[offset] = unit

    frames, frame_errors = parse_frames(payload, full, allowed_units=set(indices))
    remap = {old: new for new, old in enumerate(indices)}
    frames = [replace(frame, unit_index=remap[frame.unit_index]) for frame in frames]
    report = verify_units(units, frames, texts)

    return {
        "call": "claim",
        "items": len(units),
        "finish_reason": result.finish_reason,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "reasoning_tokens": result.reasoning_tokens,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
        "frames": len(frames),
        "frame_errors": frame_errors,
        "counts": report.counts,
        "coverage": report.coverage,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=DEFAULT_RUN, help="recorded run to take the inputs from")
    parser.add_argument(
        "--efforts", default="high,medium,low,off", help="comma-separated effort variants"
    )
    parser.add_argument("--out", default=None, help="output JSON path")
    args = parser.parse_args(argv)

    source = Trajectory(args.run)
    answer = (source.run_dir / "answer.md").read_text(encoding="utf-8")
    texts = recorded_chunk_texts(source)
    events = [
        json.loads(line)
        for line in source.trajectory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sub_event = _failed_event(events, "sub")
    claim_event = _failed_event(events, "claim")
    mode = "failed-batch" if sub_event and claim_event else "sample"

    if sub_event and claim_event:
        sub_messages, sub_response, sub_max, sub_items = _sub_inputs(
            _load_prompt(source, sub_event), texts
        )
        claim_messages, claim_response, claim_max, claim_units, claim_indices = _claim_inputs(
            _load_prompt(source, claim_event)
        )
    else:
        sub_messages, sub_response, sub_max, sub_items = _sample_sub(texts)
        claim_messages, claim_response, claim_max, claim_units, claim_indices = _sample_claim(
            answer, texts
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []

    # The probe logs to a throwaway runs root: it is a measurement, not a run the
    # viewer or `--list` should see.
    with tempfile.TemporaryDirectory() as tmp:
        original_root = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = Path(tmp)

        try:
            for name in [part.strip() for part in args.efforts.split(",") if part.strip()]:
                if name not in EFFORTS:
                    print(f"unknown effort {name!r}", file=sys.stderr)
                    return 2

                spec = replace(config.MODEL_REGISTRY["sub"], reasoning=EFFORTS[name])
                trajectory = Trajectory(f"probe-sub-{stamp}-{name}")
                client = LensClient(
                    trajectory,
                    registry={**config.MODEL_REGISTRY, "sub": spec},
                )
                sub = _run_sub(
                    client, sub_messages, sub_response, sub_max, sub_items, f"probe-sub-{name}"
                )
                claim = _run_claim(
                    client,
                    claim_messages,
                    claim_response,
                    claim_max,
                    claim_units,
                    claim_indices,
                    texts,
                    f"probe-claim-{name}",
                )
                rows.append({"effort": name, "sub_extract": sub, "claim": claim})
                print(
                    f"{name:>6}  extract: {sub['records']} records, "
                    f"{sub['quote_count']} quotes ({sub['quotes_dropped']} dropped), "
                    f"{sub['tokens_out']} out ({sub['reasoning_tokens']} reasoning), "
                    f"{sub['finish_reason']}, {sub['latency_ms'] / 1000:.1f} s, "
                    f"${sub['cost_usd']:.4f}  |  claim: {claim['frames']} frames, "
                    f"{claim['coverage']:.0%} coverage, {claim['tokens_out']} out "
                    f"({claim['reasoning_tokens']} reasoning), "
                    f"{claim['finish_reason']}, {claim['latency_ms'] / 1000:.1f} s, "
                    f"${claim['cost_usd']:.4f}"
                )
        finally:
            trajectory_module.LENS_RUNS_ROOT_PATH = original_root

    out = (
        Path(args.out)
        if args.out
        else config.DATA_ROOT / "campaigns" / f"sub-reasoning-probe-{stamp}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source_run": args.run,
                "mode": mode,
                "created_at": stamp,
                "model": config.MODEL_REGISTRY["sub"].id,
                "chunk_ids": [item_id for item_id, _ in sub_items],
                "units": len(claim_units),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
