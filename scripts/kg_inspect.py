r"""Inspect a saved KG JSON and print its triples.

Reads a probe result (``kg_probe_*.json``) or a cache payload
(``D:\lens-data\cache\kg\<sha256>.json``) — both have ``{kg_id, graph, summary,
stats}`` — rebuilds the networkx graph, and prints per-type/relation counts plus the
triples sorted by score.

Run:
  .\.venv\Scripts\python.exe scripts\kg_inspect.py                 # newest probe JSON
  .\.venv\Scripts\python.exe scripts\kg_inspect.py path.json --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lens import config
from lens.tools.kg import graph_from_nodelink, kg_triples


def _newest_probe() -> Path | None:
    files = sorted(
        Path(config.LENS_LOGS_ROOT_PATH).glob("kg_probe_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def main() -> None:
    # Entity names can contain math symbols (e.g. ∞); Windows consoles are cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Inspect a saved KG")
    parser.add_argument("path", nargs="?", help="kg_probe_*.json or cache payload")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    path = Path(args.path) if args.path else _newest_probe()
    if path is None or not path.is_file():
        raise SystemExit("No KG JSON found; pass a path or run the probe first.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    graph = graph_from_nodelink(payload["graph"])
    triples = kg_triples(graph)
    triples.sort(key=lambda t: t.get("score") or 0.0, reverse=True)

    summary = payload.get("summary", {})
    print(f"file: {path}")
    print(f"kg_id: {payload.get('kg_id')}")
    print(f"stats: {payload.get('stats', {})}")
    print(f"entities by type: {summary.get('entity_types')}")
    print(f"type sources: {summary.get('type_sources')}")
    print(f"relations: {summary.get('relations')}")
    print("top entities:")
    for node in summary.get("top_entities", []):
        print(f"  {node['degree']:>4}  {node['type']:<8}  {node['entity']}")

    shown = triples if args.all else triples[: args.limit]
    print(f"triples: {len(triples)} (showing {len(shown)})")
    for triple in shown:
        score = triple.get("score") or 0.0
        print(
            f"  {score:.3f}  ({triple['head_type']}) {triple['head']} "
            f"--{triple['relation']}--> ({triple['tail_type']}) {triple['tail']} "
            f"[{triple.get('arxiv_id')} {triple.get('section_type')}]"
        )


if __name__ == "__main__":
    main()
