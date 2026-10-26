r"""M7.2 — small gold-sample tooling for K-04 (triple precision) and K-05 (entity recall).

Pure helpers + CSV/JSON I/O; ``scripts/build_gold_sample.py`` and
``scripts/score_gold.py`` are thin wrappers.

Triple precision (K-04)
-----------------------
Sample kept triples **stratified by relation** into a CSV. The annotator fills
``gold_relation_ok`` / ``gold_head_type_ok`` / ``gold_tail_type_ok`` with ``y``/``n``.
:func:`score_triples` reports precision overall and per relation, with Wilson 95% CIs.

Entity recall (K-05)
--------------------
Sample chunks into a CSV containing **text only** (plus a blank ``gold_entities``
field); the KG predictions go to a **separate JSON**, so the annotator is not shown
the model output. :func:`score_entities` matches the annotated gold entities against
the KG entities (typed + untyped recall, precision).

The gold labels themselves are human; these tools only sample and score.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from lens.kg_postprocess import is_benchmark_span as _is_benchmark_span

Triple = dict[str, Any]

TRIPLE_COLUMNS = [
    "id",
    "head",
    "head_type",
    "relation",
    "tail",
    "tail_type",
    "score",
    "chunk_uuid",
    "arxiv_id",
    "section_type",
    "gold_relation_ok",
    "gold_head_type_ok",
    "gold_tail_type_ok",
    "notes",
]
ENTITY_COLUMNS = [
    "chunk_uuid",
    "arxiv_id",
    "section_type",
    "chunk_text",
    "gold_entities",
    "notes",
]


# ------------------------------------------------------------------ sampling
def stratified_sample(triples: Sequence[Triple], n: int, seed: int = 0) -> list[Triple]:
    """Proportional-by-relation sample with at least one triple per observed relation."""
    if n <= 0 or not triples:
        return []
    rng = random.Random(seed)
    groups: dict[str, list[Triple]] = defaultdict(list)
    for triple in triples:
        groups[triple.get("relation")].append(triple)
    for group in groups.values():
        rng.shuffle(group)

    total = len(triples)
    chosen: list[Triple] = []
    for relation, group in groups.items():
        quota = max(1, round(n * len(group) / total))
        chosen.extend(group[:quota])

    rng.shuffle(chosen)
    if len(chosen) > n:
        return chosen[:n]
    if len(chosen) < n:
        chosen_ids = {id(t) for t in chosen}
        pool = [t for t in triples if id(t) not in chosen_ids]
        rng.shuffle(pool)
        chosen.extend(pool[: n - len(chosen)])
    return chosen


def sample_items(items: Sequence[Any], n: int, seed: int = 0) -> list[Any]:
    pool = list(items)
    random.Random(seed).shuffle(pool)
    return pool[: max(0, n)]


# ------------------------------------------------------------------ worksheets
def write_triples_worksheet(triples: Sequence[Triple], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIPLE_COLUMNS)
        writer.writeheader()
        for index, triple in enumerate(triples, 1):
            row = {key: triple.get(key, "") for key in TRIPLE_COLUMNS}
            row["id"] = index
            for key in ("gold_relation_ok", "gold_head_type_ok", "gold_tail_type_ok", "notes"):
                row[key] = ""
            writer.writerow(row)


def read_triples_worksheet(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_entities_worksheet(rows: Sequence[dict], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENTITY_COLUMNS)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key, "") for key in ENTITY_COLUMNS}
            record["gold_entities"] = ""
            record["notes"] = ""
            writer.writerow(record)


def read_entities_worksheet(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


_DOT_DIGITS = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")
_PUNCT_TIGHT = re.compile(r"\s+([.,;:!?\)\]\}])")
_OPEN_TIGHT = re.compile(r"([\(\[\{])\s+")


def is_benchmark_span(text: str) -> bool:
    """Benchmark-like span using the **broad** matcher (scoring leniency only)."""
    return _is_benchmark_span(text, broad=True)


def clean_surface(text: str) -> str:
    """Undo whitespace introduced by joining token lists ("GPT 3 . 5" -> "GPT 3.5")."""
    text = _DOT_DIGITS.sub(".", text or "")  # decimals / versions
    text = _PUNCT_TIGHT.sub(r"\1", text)
    text = _OPEN_TIGHT.sub(r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def format_triple_query(row: dict) -> str:
    """One-line natural-language form for manual verification / search."""
    return (
        f"Is it true that {row['head']} ({row['head_type']}) {row['relation']} "
        f"{row['tail']} ({row['tail_type']}) at arxiv paper {row['arxiv_id']}"
    )


def snippet(text: str, terms: Sequence[str], width: int = 400) -> str:
    """Return a short window around the first occurrence of any term (for annotation)."""
    text = " ".join((text or "").split())
    if not text:
        return ""
    low = text.casefold()
    position = -1
    for term in terms:
        if not term:
            continue
        position = low.find(term.casefold())
        if position >= 0:
            break
    if position < 0:
        return text[:width]

    start = max(0, position - width // 3)
    end = min(len(text), start + width)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


# ------------------------------------------------------------------ scoring
def _ok(value: Any) -> bool:
    return str(value).strip().casefold() in {"y", "yes", "1", "true", "ok"}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _type_flags(row: dict, benchmark_lenient: bool) -> tuple[bool, bool]:
    head_ok = _ok(row["gold_head_type_ok"])
    tail_ok = _ok(row["gold_tail_type_ok"])
    if benchmark_lenient:
        # A benchmark can be a Dataset or a Method; accept either for benchmark-like spans.
        if (
            not head_ok
            and is_benchmark_span(row.get("head", ""))
            and row.get("head_type") in {"Dataset", "Method"}
        ):
            head_ok = True
        if (
            not tail_ok
            and is_benchmark_span(row.get("tail", ""))
            and row.get("tail_type") in {"Dataset", "Method"}
        ):
            tail_ok = True
    return head_ok, tail_ok


def score_triples(rows: Sequence[dict], *, benchmark_lenient: bool = False) -> dict:
    """Precision of sampled triples; strict labels + a benchmark-lenient view."""
    labeled = [r for r in rows if str(r.get("gold_relation_ok", "")).strip() != ""]
    n = len(labeled)

    strict = [(r, *_type_flags(r, False)) for r in labeled]
    lenient = [(r, *_type_flags(r, True)) for r in labeled]

    relation = sum(_ok(r["gold_relation_ok"]) for r in labeled)
    head = sum(head_ok for _r, head_ok, _t in strict)
    tail = sum(tail_ok for _r, _h, tail_ok in strict)
    triple = sum(
        _ok(r["gold_relation_ok"]) and head_ok and tail_ok for r, head_ok, tail_ok in strict
    )
    head_lenient = sum(head_ok for _r, head_ok, _t in lenient)
    tail_lenient = sum(tail_ok for _r, _h, tail_ok in lenient)
    triple_lenient = sum(
        _ok(r["gold_relation_ok"]) and head_ok and tail_ok for r, head_ok, tail_ok in lenient
    )

    by_relation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in labeled:
        name = row.get("relation", "")
        by_relation[name][1] += 1
        if _ok(row["gold_relation_ok"]):
            by_relation[name][0] += 1

    return {
        "n": n,
        "benchmark_lenient": benchmark_lenient,
        "relation_precision": relation / n if n else 0.0,
        "head_type_precision": head / n if n else 0.0,
        "tail_type_precision": tail / n if n else 0.0,
        "triple_precision": triple / n if n else 0.0,
        "triple_precision_ci": wilson_ci(triple, n),
        "head_type_precision_lenient": head_lenient / n if n else 0.0,
        "tail_type_precision_lenient": tail_lenient / n if n else 0.0,
        "triple_precision_lenient": triple_lenient / n if n else 0.0,
        "triple_precision_lenient_ci": wilson_ci(triple_lenient, n),
        "by_relation": {
            name: {
                "correct": values[0],
                "n": values[1],
                "precision": values[0] / values[1] if values[1] else 0.0,
            }
            for name, values in sorted(by_relation.items())
        },
    }


def parse_gold_entities(text: str) -> list[tuple[str, str]]:
    """Parse ``Name:Type; Name:Type`` (Type optional -> ``Concept``)."""
    out: list[tuple[str, str]] = []
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, kind = part.rsplit(":", 1)
            out.append((name.strip(), kind.strip() or "Concept"))
        else:
            out.append((part, "Concept"))
    return out


def score_entities(rows: Sequence[dict], predictions: dict[str, list[dict]]) -> dict:
    """Typed/untyped entity recall (gold -> KG) and untyped precision (KG -> gold)."""
    gold_total = gold_hit = gold_hit_untyped = pred_total = pred_hit = 0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for row in rows:
        gold = {
            name.casefold(): (name, kind)
            for name, kind in parse_gold_entities(row.get("gold_entities"))
        }
        pred = {
            item["entity"].casefold(): item
            for item in predictions.get(row["chunk_uuid"], [])
            if item.get("entity")
        }
        gold_total += len(gold)
        pred_total += len(pred)

        for key, (_name, kind) in gold.items():
            by_type[kind][1] += 1
            match = pred.get(key)
            if match is not None:
                gold_hit_untyped += 1
                if str(match.get("type", "")).casefold() == kind.casefold():
                    gold_hit += 1
                    by_type[kind][0] += 1

        for key in pred:
            if key in gold:
                pred_hit += 1

    return {
        "gold_entities": gold_total,
        "predicted_entities": pred_total,
        "recall_typed": gold_hit / gold_total if gold_total else 0.0,
        "recall_untyped": gold_hit_untyped / gold_total if gold_total else 0.0,
        "precision_untyped": pred_hit / pred_total if pred_total else 0.0,
        "recall_typed_ci": wilson_ci(gold_hit, gold_total),
        "by_type": {
            kind: {
                "matched": values[0],
                "gold": values[1],
                "recall": values[0] / values[1] if values[1] else 0.0,
            }
            for kind, values in sorted(by_type.items())
        },
    }


# ------------------------------------------------------------------ helpers
def triples_from_payload(payload: dict[str, Any]) -> list[Triple]:
    """Rebuild triple dicts from a saved ``{graph: node-link}`` payload."""
    from lens.tools.kg import graph_from_nodelink, kg_triples

    return kg_triples(graph_from_nodelink(payload["graph"]))


def entities_from_payload(
    payload: dict[str, Any], context_key: str = "chunk_uuid"
) -> dict[str, list[dict]]:
    """Per-chunk KG entities (from edges), for the entity-recall predictions file."""
    from lens.tools.kg import graph_from_nodelink

    graph = graph_from_nodelink(payload["graph"])
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for u, v, data in graph.edges(data=True):
        context = data.get(context_key)
        if context is None:
            continue
        for node in (u, v):
            out[str(context)][node] = {
                "entity": graph.nodes[node].get("display", node),
                "type": graph.nodes[node].get("type", "Concept"),
                "type_source": graph.nodes[node].get("type_source", "model"),
            }
    return {key: list(nodes.values()) for key, nodes in out.items()}


def load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
