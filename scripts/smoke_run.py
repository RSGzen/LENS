"""M0 live smoke test — four real OpenRouter calls, each cost-logged.

This exercises the real transport (no fakes) end-to-end on the host:

  1. chat("root")  -> openai/gpt-5
  2. chat("sub")   -> openai/gpt-5-mini
  3. chat("dev")   -> qwen/qwen3-coder
  4. jev(...)      -> typesafe/jev-1.13

Every call is written to ``<runs_root>/<run_id>/trajectory.jsonl`` with
``usage.cost``, and its prompt/completion are content-addressed under ``blobs/``.

!!! SPENDS REAL CREDITS !!!  It is intentionally **not** run automatically.
Requires ``OPENROUTER_API_KEY`` in the environment or a gitignored ``.env``.

Inspect, then run from the repo root:

    uv run python scripts/smoke_run.py
    # or: .\\.venv\\Scripts\\python.exe scripts\\smoke_run.py

Exit code: 0 = all four calls succeeded, 1 = one or more failed, 2 = config error.
"""

from __future__ import annotations

import datetime as dt
import sys

from lens import config
from lens.clients import LensClient, LensTransportError
from lens.models import Question, Role
from lens.observability.trajectory import TraceCtx, Trajectory, Actor

SYSTEM_PROMPT = "You are a terse assistant."
SHORT_USER_PROMPT = "Reply with the single word: ok."

# (role, actor, messages) — keep prompts tiny to minimise cost.
CHAT_CALLS: list[tuple[Role, Actor, list[dict[str, str]]]] = [
    (
        "root",
        "root",
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SHORT_USER_PROMPT},
        ],
    ),
    ("sub", "sub", [{"role": "user", "content": SHORT_USER_PROMPT}]),
    ("dev", "root", [{"role": "user", "content": "What is 7 + 7? Reply with just the number."}]),
]


def _print_chat(label: str, tokens_in: int, tokens_out: int, model: str, latency_ms: int, cost: float) -> None:
    print(f"[ok]   {label:9s} model={model:32s} tokens={tokens_in}/{tokens_out} "
          f"latency={latency_ms}ms cost=${cost:.6f}")


def main() -> int:
    run_id = "smoke-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    trajectory = Trajectory(run_id)

    try:
        client = LensClient(trajectory)
    except RuntimeError as exc:  # missing OPENROUTER_API_KEY
        print(f"[config error] {exc}")
        return 2

    print(f"run_id      : {run_id}")
    print(f"runs_root   : {config.LENS_RUNS_ROOT_PATH}")
    print(f"trajectory  : {trajectory.trajectory_path}")
    print("-" * 88)

    total_cost = 0.0
    failures = 0

    for step_idx, (role, actor, messages) in enumerate(CHAT_CALLS):
        trace_ctx = TraceCtx(trace_id=run_id, step_idx=step_idx, actor=actor)
        try:
            result = client.chat(role, messages, trace_ctx=trace_ctx)
        except LensTransportError as exc:
            failures += 1
            print(f"[FAIL] {role:9s} {exc} (status={exc.status}, attempts={exc.attempts})")
            continue
        total_cost += result.cost_usd
        _print_chat(f"chat/{role}", result.tokens_in, result.tokens_out,
                    result.model_snapshot or result.model, result.latency_ms, result.cost_usd)

    # JEV probe: two questions in one request (noul + choice).
    jev_ctx = TraceCtx(trace_id=run_id, step_idx=len(CHAT_CALLS), actor="tool")
    questions = {
        "is_problem": Question(
            type="noul",
            instructions="Is this a problem statement?",
            criteria={"true": "Yes", "false": "No"},
        ),
        "bucket": Question(
            type="choice",
            instructions="Which bucket does this clause best fit?",
            criteria={
                "problem": "Problem statement",
                "objective": "Objective",
                "constraints": "Constraints / limitations",
                "requirements": "Special requirements",
            },
        ),
    }
    try:
        jev = client.jev(
            state="We need to reduce inference cost while keeping answer accuracy.",
            questions=questions,
            trace_ctx=jev_ctx,
        )
    except LensTransportError as exc:
        failures += 1
        print(f"[FAIL] jev       {exc} (status={exc.status}, attempts={exc.attempts})")
    else:
        total_cost += jev.cost_usd
        print(f"[ok]   {'jev':9s} model={jev.model:32s} tokens={jev.tokens_in} "
              f"latency={jev.latency_ms}ms cost=${jev.cost_usd:.6f}")
        for name, answer in jev.answers.items():
            print(f"         answers[{name!r}] = {answer}")

    fee = total_cost * 0.055  # OpenRouter credit-load fee (LOG-29 / R4)
    print("-" * 88)
    print(f"calls       : {len(CHAT_CALLS) + 1}  (failures: {failures})")
    print(f"usage.cost  : ${total_cost:.6f}")
    print(f"+5.5% fee   : ${fee:.6f}")
    print(f"effective   : ${total_cost + fee:.6f}")
    print(f"artifacts   : {trajectory.trajectory_path}")
    print(f"              {trajectory.blobs_dir}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
