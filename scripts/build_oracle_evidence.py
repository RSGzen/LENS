"""Build coverage-matched evidence sets from recorded loop runs.

The flat baseline sees one funnel call. The loop sees that plus every paper it
fetched. To ask whether a single call could do the same work given the loop's
evidence, the flat arm needs that evidence as a file: one JSON object per line,
the same shape the funnel returns.

Two sets are written per instance, and they bound the answer from both sides:

``union``
    every chunk the loop's tool payloads contained, deduplicated. An upper bound
    on what the loop had available, including papers it fetched and discarded.

``cited``
    only the chunks the loop's answer cited. A lower bound: the evidence it
    actually used, which is the most favourable set a single call could be given.

Both are derived from the **recorded** run, so the arm inherits the loop's
retrieval decisions. That is the point of a coverage-matched comparison and the
reason the union set is an oracle rather than a pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lens.agent.runtime import count_tokens  # noqa: E402
from lens.config import LENS_RUNS_ROOT_PATH  # noqa: E402
from lens.observability.citations import cited_ids, expand_short_ids  # noqa: E402

BLOBS = Path(LENS_RUNS_ROOT_PATH) / "blobs"
READ_ONLY_TOOLS = ("retrieval_funnel", "fetch_paper_chunks", "fetch_chunks")


def load_blob(ref: str) -> Any:
    """Return the decoded blob behind ``ref`` (``sha256:<hex>``)."""
    if not ref.startswith("sha256:"):
        return None
    path = BLOBS / ref.split(":", 1)[1]
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_rows(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield chunk rows from any chunk-returning tool result."""
    if isinstance(payload, dict):
        rows = payload.get("chunks") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        return

    for row in rows:
        if isinstance(row, dict) and row.get("chunk_uuid") and row.get("chunk_text"):
            yield {
                "chunk_uuid": str(row["chunk_uuid"]),
                "arxiv_id": str(row.get("arxiv_id") or ""),
                "chunk_text": str(row["chunk_text"]),
            }


def run_evidence(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (chunks in tool-payload order, tools that were dispatched)."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    tools: list[str] = []
    trajectory = run_dir / "trajectory.jsonl"

    for line in trajectory.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        name = event.get("tool")

        if event.get("action") != "tool.dispatch" or name not in READ_ONLY_TOOLS:
            continue

        tools.append(name)

        for row in chunk_rows(load_blob(str(event.get("result_ref") or ""))):
            if row["chunk_uuid"] in seen:
                continue
            seen.add(row["chunk_uuid"])
            rows.append(row)

    return rows, tools


def write_set(path: Path, rows: list[dict[str, Any]]) -> int:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return sum(count_tokens(row["chunk_text"]) + 8 for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build coverage-matched evidence sets")
    parser.add_argument("--campaign", required=True, help="the loop arm's campaign.csv")
    parser.add_argument("--out", default=str(REPO_ROOT / "experiments" / "instruments" / "oracle"))
    parser.add_argument("--arm", default="a2")
    parser.add_argument("--runs-root", default=LENS_RUNS_ROOT_PATH)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [
        r
        for r in csv.DictReader(open(args.campaign, encoding="utf-8"))
        if r["campaign_arm"] == args.arm
    ]
    if not rows:
        raise SystemExit(f"no {args.arm!r} rows in {args.campaign}")

    header = f"{'instance':<26} {'tools':>5} {'union':>7} {'u_tok':>7}"
    print(f"{header} {'cited':>6} {'c_tok':>7} {'lost':>5}")
    totals = {"union": 0, "cited": 0, "lost": 0}

    for row in rows:
        instance = row["query_id"]
        run_dir = Path(args.runs_root) / row["run_id"]

        if not (run_dir / "trajectory.jsonl").is_file():
            print(f"{instance:<26} no trajectory: {run_dir}")
            continue

        chunks, tools = run_evidence(run_dir)
        answer = (run_dir / "answer.md").read_text(encoding="utf-8")
        known = {chunk["chunk_uuid"] for chunk in chunks}
        wanted = set(cited_ids(expand_short_ids(answer, known)))
        cited = [chunk for chunk in chunks if chunk["chunk_uuid"] in wanted]

        union_tokens = write_set(out / f"{instance}.union.jsonl", chunks)
        cited_tokens = write_set(out / f"{instance}.cited.jsonl", cited)
        lost = len(wanted - known)

        totals["union"] += union_tokens
        totals["cited"] += cited_tokens
        totals["lost"] += lost
        print(
            f"{instance:<26} {len(tools):>5} {len(chunks):>7} {union_tokens:>7} "
            f"{len(cited):>6} {cited_tokens:>7} {lost:>5}"
        )

    print(f"\nwrote {len(rows)} x 2 sets to {out}")
    print(f"union tokens total {totals['union']}, cited {totals['cited']}")
    print(f"cited ids that are not chunk ids of the union: {totals['lost']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
