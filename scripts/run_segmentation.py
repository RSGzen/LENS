r"""M6 runner -- one real segmentation call (query -> 4 buckets -> ``query.json``).

Wiring only: build the trajectory sink + transport, run
``segment_query`` (one real ``gpt-5-mini`` structured-JSON call), write the
buckets to ``query.json`` in the run workspace, and print the ``usage.cost`` +
dated ``model`` snapshot read back from the trajectory (the M6 verify clause,
E-04). Stdout is mirrored to a timestamped log under ``config.LENS_LOGS_ROOT_PATH``.

!!! SPENDS REAL CREDITS !!!  Not run automatically. Requires
``OPENROUTER_API_KEY`` in the environment or a gitignored ``.env``.

Run from the repo root:

    .\.venv\Scripts\python.exe scripts\run_segmentation.py "your query here"

Exit code: 0 = ok, 1 = call failed, 2 = config error.
"""

from __future__ import annotations

import datetime as dt
import json
import sys

from lens.agent.segmentation import segment_query, write_query_json
from lens.clients import LensClient, LensTransportError
from lens.observability.run_logging import setup_file_logging
from lens.observability.trajectory import TraceCtx, Trajectory
from lens.tools.retrieval import format_query

DEFAULT_QUERY = (
    "How do retrieval-augmented language models aggregate evidence across a long "
    "context, and can recursion help under a fixed token budget?"
)


def _read_chat_events(trajectory: Trajectory) -> list[dict]:
    """Every ``call_type == "chat"`` event from the run's trajectory, in order."""
    if not trajectory.trajectory_path.exists():
        return []
    lines = trajectory.trajectory_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    return [e for e in events if e.get("call_type") == "chat"]


def main() -> int:
    setup_file_logging("lens.run_segmentation", "segment_query")

    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    run_id = "seg-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    trajectory = Trajectory(run_id)

    try:
        client = LensClient(trajectory)
    except RuntimeError as exc:  # missing OPENROUTER_API_KEY
        print(f"[config error] {exc}")
        return 2

    print(f"run_id     : {run_id}")
    print(f"query      : {query!r}")
    print(f"trajectory : {trajectory.trajectory_path}")
    print("-" * 88)

    ctx = TraceCtx(trace_id=run_id, step_idx=0, actor="sub")
    try:
        buckets = segment_query(client, query, trace_ctx=ctx)
    except LensTransportError as exc:
        print(f"[FAIL] segmentation {exc} (status={exc.status}, attempts={exc.attempts})")
        return 1

    query_json_path = write_query_json(buckets, trajectory.run_dir / "query.json")

    print("buckets:")
    for name, clauses in buckets.items():
        print(f"  {name:13s} ({len(clauses)}): {clauses}")
    print(f"format_query : {format_query(buckets)!r}")
    print(f"query.json   : {query_json_path}")

    events = _read_chat_events(trajectory)
    total_cost = 0.0
    for event in events:
        total_cost += event.get("cost_usd", 0.0)
        print(f"[log] model={event.get('model_snapshot')} "
              f"provider={event.get('provider')} "
              f"tokens={event.get('tokens_in')}/{event.get('tokens_out')} "
              f"latency={event.get('latency_ms')}ms "
              f"cost=${event.get('cost_usd', 0.0):.6f}")

    fee = total_cost * 0.055  # OpenRouter credit-load fee (LOG-29 / R4)
    print("-" * 88)
    print(f"usage.cost : ${total_cost:.6f}")
    print(f"+5.5% fee  : ${fee:.6f}")
    print(f"effective  : ${total_cost + fee:.6f}")

    return 0 if events else 1


if __name__ == "__main__":
    sys.exit(main())
