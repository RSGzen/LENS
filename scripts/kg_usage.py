r"""Audit knowledge-graph use across recorded runs (read-only).

Why this exists: the KG ablation was dropped because the graph was built a few
times and traversed zero times. This counts the whole store from the artifacts,
one row per run: build calls, graph size, whether the code ever touched the
graph, the sandbox ``kg used:`` note, KG stage-gate verdicts and how many of the
answer's citations land on a KG edge chunk.

No API key, no database, no Docker. Reads ``trajectory.jsonl``, ``answer.md``,
``manifest.json`` and the content-addressed blob store.

Run::

    .\.venv\Scripts\python.exe scripts\kg_usage.py
    .\.venv\Scripts\python.exe scripts\kg_usage.py --runs-root D:\lens-data\runs --out kg.csv
    .\.venv\Scripts\python.exe scripts\kg_usage.py --self-check
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from lens import config

FULL_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
SHORT_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")
KG_CODE = re.compile(r"kg_neighbors|kg_triples|kg_subgraph|kg_query|node_link_graph")
#: A match inside a quoted string is a search pattern, not a call: one run wrote
#: ``["kg_neighbors(", "kg_triples("]`` to grep the chunks for those tokens.
QUOTED = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'")

CSV_FIELDS = [
    "run",
    "group",
    "status",
    "builds",
    "chunks",
    "triples_raw",
    "triples_kept",
    "nodes",
    "edges",
    "kg_notes",
    "kg_code_refs",
    "kg_gate_verdicts",
    "kg_citations",
    "answer_chars",
    "query",
]


def _events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed trajectory events; skip corrupt lines (a viewer must open a
    damaged bundle, while a strict reader is for ``Trajectory``).
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event


def _blob_path(runs_root: Path, ref: Any) -> Path | None:
    if not isinstance(ref, str) or not ref:
        return None
    return runs_root / "blobs" / ref.split(":", 1)[-1]


def _load_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


def _graph_row(payload: Any) -> dict[str, Any]:
    """Pull size + provenance out of one ``build_kg`` payload."""
    row: dict[str, Any] = {
        "chunks": None,
        "triples_raw": None,
        "triples_kept": None,
        "nodes": None,
        "edges": None,
        "edge_ids": set(),
    }
    if not isinstance(payload, dict):
        return row
    summary = payload.get("summary") or {}
    stats = payload.get("stats") or {}
    graph = payload.get("graph") or {}
    edges = graph.get("edges") or graph.get("links") or []
    row.update(
        chunks=stats.get("chunks"),
        triples_raw=stats.get("triples_raw"),
        triples_kept=stats.get("triples_kept"),
        nodes=summary.get("n_nodes"),
        edges=summary.get("n_edges", len(edges)),
        edge_ids={
            e.get("chunk_uuid") for e in edges if isinstance(e, dict) and e.get("chunk_uuid")
        },
    )
    return row


def _count_kg_citations(answer: str, edge_ids: set[str]) -> int:
    """Number of distinct KG edge chunks the answer cites (full id or 8-char prefix)."""
    full = set(FULL_UUID.findall(answer))
    short = set(SHORT_ID.findall(answer))
    return sum(1 for cid in edge_ids if cid in full or cid[:8] in short)


def _audit_run(
    run_dir: Path, runs_root: Path, code_cache: dict[str, bool]
) -> dict[str, Any] | None:
    trajectory = run_dir / "trajectory.jsonl"
    if not trajectory.is_file():
        return None

    builds: list[dict[str, Any]] = []
    code_refs: set[str] = set()
    notes = 0
    kg_gates = 0

    for event in _events(trajectory):
        if event.get("action") == "tool.dispatch" and event.get("tool") == "build_kg":
            builds.append(event)
        ref = event.get("code_ref")
        if isinstance(ref, str):
            code_refs.add(ref)
        if event.get("call_type") == "note":
            extra = event.get("extra") or {}
            text = extra.get("text") if isinstance(extra, dict) else None
            if isinstance(text, str) and "kg used:" in text:
                notes += 1
            if event.get("action") == "gate.verdict" and event.get("stage") == "kg":
                kg_gates += 1

    kg_code_refs = 0
    for ref in code_refs:
        if ref not in code_cache:
            payload = _blob_path(runs_root, ref)
            text = ""
            if payload is not None and payload.is_file():
                text = payload.read_text(encoding="utf-8", errors="replace")
            code_cache[ref] = any(
                KG_CODE.search(QUOTED.sub("", line)) for line in text.splitlines()
            )
        if code_cache[ref]:
            kg_code_refs += 1

    graph = {
        "chunks": None,
        "triples_raw": None,
        "triples_kept": None,
        "nodes": None,
        "edges": None,
    }
    edge_ids: set[str] = set()
    for event in builds:
        payload = _load_json(_blob_path(runs_root, event.get("result_ref")))
        row = _graph_row(payload)
        edge_ids |= row.pop("edge_ids")
        if row.get("edges") is not None:
            graph = row

    answer_path = run_dir / "answer.md"
    answer = (
        answer_path.read_text(encoding="utf-8", errors="replace") if answer_path.is_file() else ""
    )
    manifest = _load_json(run_dir / "manifest.json") or {}
    query = manifest.get("query") if isinstance(manifest, dict) else ""

    return {
        "run": run_dir.name,
        "group": run_dir.name.split(".", 1)[0],
        "status": manifest.get("status", "") if isinstance(manifest, dict) else "",
        "builds": len(builds),
        "chunks": graph.get("chunks"),
        "triples_raw": graph.get("triples_raw"),
        "triples_kept": graph.get("triples_kept"),
        "nodes": graph.get("nodes"),
        "edges": graph.get("edges"),
        "kg_notes": notes,
        "kg_code_refs": kg_code_refs,
        "kg_gate_verdicts": kg_gates,
        "kg_citations": _count_kg_citations(answer, edge_ids),
        "answer_chars": len(answer),
        "query": (query or "")[:100],
    }


def _audit(runs_root: Path, out_path: Path) -> int:
    code_cache: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and p.name != "blobs"):
        row = _audit_run(run_dir, runs_root, code_cache)
        if row is not None:
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    groups_with_builds = Counter(row["group"] for row in rows if row["builds"])
    m11 = [row for row in rows if row["run"].startswith("m11-")]
    print(f"runs={len(rows)}")
    print(f"runs_with_build={sum(1 for row in rows if row['builds'])}")
    print(f"build_calls={sum(row['builds'] for row in rows)}")
    print(f"runs_with_traversal_note={sum(1 for row in rows if row['kg_notes'])}")
    print(f"runs_with_kg_code={sum(1 for row in rows if row['kg_code_refs'])}")
    print(f"runs_with_kg_gate={sum(1 for row in rows if row['kg_gate_verdicts'])}")
    print(f"runs_with_kg_citation={sum(1 for row in rows if row['kg_citations'])}")
    print(
        f"m11_runs={len(m11)} m11_with_build={sum(1 for r in m11 if r['builds'])} "
        f"m11_build_calls={sum(r['builds'] for r in m11)}"
    )
    print(f"groups_with_build={dict(sorted(groups_with_builds.items()))}")
    print(f"wrote {out_path}")
    return 0


def _self_check() -> int:
    """One synthetic run, asserted end to end (no corpus, no key)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        blobs = root / "blobs"
        blobs.mkdir()
        run = root / "demo.a1.q1"
        run.mkdir()

        graph = {
            "kg_id": "x",
            "graph": {
                "edges": [
                    {"chunk_uuid": "11111111-1111-1111-1111-111111111111"},
                    {"chunk_uuid": "22222222-2222-2222-2222-222222222222"},
                ]
            },
            "summary": {"n_nodes": 2, "n_edges": 2},
            "stats": {"chunks": 3, "triples_raw": 5, "triples_kept": 2},
        }
        (blobs / "abc").write_text(json.dumps(graph), encoding="utf-8")
        (blobs / "code").write_text("G = kg_neighbors(graph, entity='x')", encoding="utf-8")

        events = [
            {"action": "tool.dispatch", "tool": "build_kg", "result_ref": "sha256:abc"},
            {"code_ref": "sha256:code"},
            {"call_type": "note", "action": "note", "extra": {"text": "kg used: kg_neighbors"}},
            {"call_type": "note", "action": "gate.verdict", "stage": "kg", "verdict": "fail"},
        ]
        (run / "trajectory.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events), encoding="utf-8"
        )
        (run / "manifest.json").write_text(
            json.dumps({"status": "completed", "query": "q"}), encoding="utf-8"
        )
        (run / "answer.md").write_text(
            "11111111-1111-1111-1111-111111111111 and 22222222", encoding="utf-8"
        )

        row = _audit_run(run, root, {})
        assert row is not None
        assert row["builds"] == 1, row
        assert row["edges"] == 2 and row["nodes"] == 2, row
        assert row["chunks"] == 3 and row["triples_raw"] == 5 and row["triples_kept"] == 2, row
        assert row["kg_notes"] == 1, row
        assert row["kg_code_refs"] == 1, row
        assert row["kg_gate_verdicts"] == 1, row
        assert row["kg_citations"] == 2, row

    print("self-check ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=config.LENS_RUNS_ROOT_PATH)
    parser.add_argument(
        "--out",
        type=Path,
        default=config.DATA_ROOT / "campaigns" / "kg-usage.csv",
    )
    parser.add_argument(
        "--self-check", action="store_true", help="run the synthetic assertion and exit"
    )
    args = parser.parse_args()

    if args.self_check:
        return _self_check()
    return _audit(Path(args.runs_root), Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
