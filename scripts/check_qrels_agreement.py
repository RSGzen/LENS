"""LLM cross-check of the Tier-2 relevance labels.

The rubric promises this and calls it optional: an LLM judge over the **same** pool,
with agreement against the human labels reported as a percentage and a Cohen's kappa.
It never replaces a human label - if the two disagree the human label stands - but it
is the only evidence about how hard the labelling task was.

One call per query group, because the rubric is applied to a query at a time and the
model should read the query once, as the annotator did. The judge sees the same
900-character abstracts and nothing else: no PDF, no paper lookup, no knowledge of
which retriever found the row.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from lens.clients import LensClient
from lens.observability.trajectory import TraceCtx, Trajectory

SYSTEM_PROMPT = """\
You apply a fixed relevance rubric to candidate papers for one research query.

The question for every row is: would this paper help answer the query?
Judge usefulness to the question - not quality, not recency, not whether the paper
agrees with anyone. Read only the title and abstract given. If the abstract does not
support the judgement, the label is 0.

Labels are binary and untyped:
1 = relevant: the abstract reports a method, a comparison, a measurement or a
    resource that the query asks about, on a topic the query names.
0 = anything else: a different topic, the right topic but the wrong need, a passing
    mention, or an abstract you cannot judge.

Four cases that come up:
- Right area, wrong need: the query asks for a comparison and the paper reports one
  system -> 0. The query asks how something is measured and the paper proposes a
  method without evaluating it -> 0.
- A survey or position paper that covers the topic and would help answer the query
  -> 1. Relevance is usefulness, not novelty.
- A passing mention: the term appears once inside a list of related work -> 0. The
  test: could you write a sentence answering the query from this abstract alone?
- Ambiguous or truncated: -> 0.

Reply with JSON only: {"judgements": [{"doc_id": "<id>", "rel": 1}, ...]} - one entry
per row given, in any order, rel as the integer 1 or 0.
"""

USER_PROMPT = """\
Query: <<QUERY>>

Rows
<<ROWS>>
"""


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_query(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["query_id"], []).append(row)
    return groups


def user_prompt(query: str, rows: Sequence[dict[str, str]]) -> str:
    blocks = [
        f"{index}. doc_id: {row['doc_id']}\n"
        f"   title: {row['title']}\n"
        f"   abstract: {row['abstract']}"
        for index, row in enumerate(rows, start=1)
    ]
    return USER_PROMPT.replace("<<QUERY>>", query).replace("<<ROWS>>", "\n\n".join(blocks))


def parse(text: str) -> dict[str, int]:
    """Return ``{doc_id: rel}`` from the reply, tolerating prose around the JSON."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    judgements = payload.get("judgements") or []
    out: dict[str, int] = {}
    for item in judgements:
        if not isinstance(item, dict):
            continue
        doc = str(item.get("doc_id") or "").strip()
        try:
            rel = int(item.get("rel"))
        except (TypeError, ValueError):
            continue
        if doc:
            out[doc] = 1 if rel else 0
    return out


def kappa(pairs: Sequence[tuple[int, int]]) -> float:
    """Cohen's kappa for two binary raters."""
    total = len(pairs)
    if not total:
        return float("nan")
    observed = sum(1 for human, model in pairs if human == model) / total
    human_pos = sum(human for human, _ in pairs) / total
    model_pos = sum(model for _, model in pairs) / total
    expected = human_pos * model_pos + (1 - human_pos) * (1 - model_pos)
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM cross-check of the qrels labels")
    parser.add_argument("--worksheet", default=r"D:\lens-data\campaigns\qrels-worksheet.csv")
    parser.add_argument("--out", default="qrels-llm-croscheck.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    rows = load_rows(Path(args.worksheet))
    groups = group_by_query(rows)
    print(f"{len(rows)} rows over {len(groups)} queries")

    if args.dry_run:
        print("dry run: no calls")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    trajectory = Trajectory(run_id=f"qrels-llm-{stamp}")
    client = LensClient(trajectory=trajectory)

    results: list[dict[str, Any]] = []
    pairs: list[tuple[int, int]] = []

    for step, (query_id, group) in enumerate(sorted(groups.items())):
        completion = client.chat(
            "root",
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(group[0]["query"], group)},
            ],
            trace_ctx=TraceCtx(trace_id="qrels-llm", step_idx=step, actor="root"),
        )
        judged = parse(completion.text)
        matched = 0
        for row in group:
            human = 1 if row["rel"].strip() == "1" else 0
            model = judged.get(row["doc_id"])
            if model is None:
                results.append(
                    {
                        "query_id": query_id,
                        "doc_id": row["doc_id"],
                        "human": human,
                        "llm": "",
                        "agree": "",
                    }
                )
                continue
            pairs.append((human, model))
            matched += 1
            results.append(
                {
                    "query_id": query_id,
                    "doc_id": row["doc_id"],
                    "human": human,
                    "llm": model,
                    "agree": int(human == model),
                }
            )
        print(f"  {query_id:<28} {matched}/{len(group)} judged by the model")

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "doc_id", "human", "llm", "agree"])
        writer.writeheader()
        writer.writerows(results)

    agreement = sum(1 for human, model in pairs if human == model) / len(pairs) if pairs else 0.0
    print(f"\njudged {len(pairs)} of {len(rows)} rows")
    print(f"agreement {agreement:.3f}  kappa {kappa(pairs):.3f}")
    print(f"labels (human / llm): {Counter(h for h, _ in pairs)} / {Counter(m for _, m in pairs)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
