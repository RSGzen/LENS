r"""— modular KG post-process: relation-first typing, gazetteer, directionality.

This module is **pure** (no DB, no model) and runs in ``build_kg`` **before**
``filter_ontology``. Each stage is independently toggleable and removable:

    soft defaults -> gazetteer/suffix type hints -> relation-slot type inference
                  -> benchmark/task role filter -> directionality

Why relation-first
------------------
In academic text, entity **type is contextual/functional, not lexical**. The NER
model is good at spans but weak at the `Method`/`Dataset`/`Problem`/`Task`
distinction. The relation schema already encodes most of it: an
``evaluated_on`` tail is a `Dataset`, a ``measures`` tail is a `Metric`, an
``outperforms`` head is a `Method`, and so on. Inferring types from relation
slots turns a 6-way typing problem into a role-assignment problem.

Soft types
----------
Triples/nodes carry ``{head,tail}_type`` plus ``{head,tail}_type_source``
(``model`` | ``relation`` | ``gazetteer``) and ``{head,tail}_type_confidence``.
The RLM can resolve residual ambiguity; nothing is claimed to be ground truth.

Directionality
--------------
- Drop self-loops.
- For asymmetric relations, keep **one direction per unordered pair per context**
  (the higher ``score``). Different contexts may keep both directions, because
  ``A outperforms B`` in one experiment and ``B outperforms A`` in another are
  different claims.
- ``compares_to`` is symmetric: keep one canonical edge per pair/context.
- Optional cycle break for ``outperforms``/``extends``: drop the lowest-score edge
  in each simple cycle within a context.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import networkx as nx

from lens import config
from lens.kg_extract import RELATION_SCHEMA

Triple = dict[str, Any]

# Coarse priors for the type source (not a calibrated probability).
TYPE_CONFIDENCE = {"model": 0.5, "relation": 0.7, "gazetteer": 0.9}

SYMMETRIC_RELATIONS = frozenset({"compares_to"})
CYCLE_RELATIONS = frozenset({"outperforms", "extends"})


# ------------------------------------------------------------------ helpers
def _fold(text: str) -> str:
    """Case-/whitespace-insensitive entity key (display form is preserved elsewhere)."""
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split()).casefold()


def _compact(text: str) -> str:
    """Alphanumeric-only key: ``SQuAD 2.0`` / ``SQuAD2.0`` -> ``squad20``."""
    return re.sub(r"[^a-z0-9]", "", _fold(text))


def _score(triple: Triple) -> float:
    return float(triple.get("score") or 0.0)


def _context(triple: Triple, context_key: str) -> str:
    value = triple.get(context_key)
    if value is None:
        value = triple.get("arxiv_id")
    return "" if value is None else str(value)


def _set_type(triple: Triple, side: str, value: str, source: str) -> None:
    triple[f"{side}_type"] = value
    triple[f"{side}_type_source"] = source
    triple[f"{side}_type_confidence"] = TYPE_CONFIDENCE.get(source, 0.5)


def with_soft_defaults(triples: list[Triple]) -> list[Triple]:
    """Ensure every triple has soft-type fields (model is the default source)."""
    for triple in triples:
        triple.setdefault("head_type_source", "model")
        triple.setdefault("tail_type_source", "model")
        triple.setdefault("head_type_confidence", TYPE_CONFIDENCE["model"])
        triple.setdefault("tail_type_confidence", TYPE_CONFIDENCE["model"])
    return triples


# ------------------------------------------------------------------ gazetteer
def load_gazetteer(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a gazetteer JSON, keyed by both **folded** and **compact** names."""
    gazetteer_path = Path(path)
    if not gazetteer_path.is_file():
        return {}
    payload = json.loads(gazetteer_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    result: dict[str, dict[str, Any]] = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        result[_fold(name)] = entry
        result.setdefault(_compact(name), entry)
    return result


def apply_gazetteer(triples: list[Triple], gazetteer: dict[str, dict[str, Any]]) -> list[Triple]:
    """Override entity types with a corpus-built prior where the name is known."""
    if not gazetteer:
        return triples
    for triple in triples:
        for side in ("head", "tail"):
            entry = gazetteer.get(_fold(triple[side])) or gazetteer.get(_compact(triple[side]))
            if not entry or not entry.get("type"):
                continue
            if float(entry.get("confidence", 1.0)) < config.KG_GAZETTEER_MIN_CONFIDENCE:
                continue
            _set_type(triple, side, entry["type"], "gazetteer")
    return triples


# ------------------------------------------------------------------ suffix hints
SUFFIX_TYPES = {
    "dataset": "Dataset",
    "datasets": "Dataset",
    "corpus": "Dataset",
    "corpora": "Dataset",
    "benchmark": "Dataset",
    "benchmarks": "Dataset",
    "test set": "Dataset",
    "testbed": "Dataset",
    "metric": "Metric",
    "metrics": "Metric",
    "score": "Metric",
    "scores": "Metric",
    "task": "Task",
    "tasks": "Task",
    "method": "Method",
    "methods": "Method",
    "model": "Method",
    "models": "Method",
    "algorithm": "Method",
    "algorithms": "Method",
    "framework": "Method",
    "frameworks": "Method",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\.]*")


def suffix_type_hint(triples: list[Triple]) -> list[Triple]:
    """Type from a trailing head noun ("X dataset" -> Dataset; "ROUGE score" -> Metric)."""
    for triple in triples:
        for side in ("head", "tail"):
            words = _WORD_RE.findall(triple[side])
            if not words:
                continue
            two = " ".join(word.casefold() for word in words[-2:])
            one = words[-1].casefold()
            kind = SUFFIX_TYPES.get(two) or SUFFIX_TYPES.get(one)
            if kind:
                _set_type(triple, side, kind, "suffix")
    return triples


# ------------------------------------------------------------------ benchmark/task roles
# A benchmark is a Dataset that defines a Task, so it can legitimately appear as
# a Dataset (tail of evaluated_on/uses) or as a Method-like subject (head of
# measures). It must not be the *subject* of evaluated_on/outperforms, though:
# "WCEP evaluated_on BookSum" and "biomedical QA outperforms GPT-4o" are role
# errors, not direction errors. This filter removes those impossible roles.
#
# Strict matching (exact/compact or suffix) is deliberate: the broad substring
# matcher used for benchmark-lenient *scoring* would read "quality-aware" as the
# QuALITY benchmark and could drop valid method heads.
_BENCH_SUFFIX = re.compile(r"(?:qa|bench|benchmark|mc|multihop)$", re.IGNORECASE)
_BENCHMARK_NAMES = frozenset(
    {
        "qasper",
        "quality",
        "narrativeqa",
        "squad",
        "hotpotqa",
        "musique",
        "2wikimultihopqa",
        "wcep",
        "booksum",
        "academiceval",
        "qmsum",
        "longbench",
        "multifieldqa",
    }
)
_ROLE_METHOD_HEAD_RELATIONS = frozenset(
    {
        "evaluated_on",
        "outperforms",
        "extends",
        "compares_to",
        "uses",
        "addresses",
        "explores",
    }
)
_ROLE_METHOD_TAIL_RELATIONS = frozenset(
    {
        "measures",
        "outperforms",
        "extends",
        "compares_to",
        "addresses",
    }
)


def is_benchmark_span(text: str, *, broad: bool = False) -> bool:
    """Heuristic: is this span benchmark-like (Dataset or Method are both defensible)?

    ``broad=True`` keeps the old substring behaviour used by
    ``score_gold.py --benchmark-lenient`` (e.g. "quality-aware" matches QuALITY).
    The **pipeline** uses the strict default so it only fires on exact names or
    benchmark suffixes.
    """
    folded = " ".join((text or "").split()).casefold()
    compact = re.sub(r"[^a-z0-9]", "", folded)
    if compact in _BENCHMARK_NAMES:
        return True
    if _BENCH_SUFFIX.search(compact):
        return True
    if broad:
        return any(name in compact for name in _BENCHMARK_NAMES if len(name) > 4)
    return False


def is_task_like(triple: Triple, side: str) -> bool:
    """True when the span is typed ``Task`` (suffix hints already set "X task")."""
    return triple.get(f"{side}_type") == "Task"


def apply_role_filter(triples: Sequence[Triple]) -> tuple[list[Triple], int]:
    """Drop triples whose benchmark/task endpoint contradicts the relation role.

    - Method-subject relations (``evaluated_on``, ``outperforms``, ``extends``,
      ``compares_to``, ``uses``, ``addresses``, ``explores``): a benchmark/task
      **head** is impossible — the method is the subject.
    - Tail roles: a benchmark/task **tail** is impossible for ``measures``,
      ``outperforms``, ``extends``, ``compares_to``; a benchmark tail is also
      impossible for ``addresses`` (a Task tail is valid there).
    - Benchmarks are **kept** as heads of ``measures`` and as tails of
      ``evaluated_on``/``uses``/``explores`` — those are their legitimate roles.

    Metric/model *type hints* are intentionally not part of this step, so a
    metric typed ``Method`` (e.g. ``ARGUE F1 evaluated_on ...``) is not dropped.

    Returns ``(kept, dropped_count)``; runs **before** directionality so an
    invalid direction cannot win the pair and suppress the valid one.
    """
    kept: list[Triple] = []
    dropped = 0
    for triple in triples:
        relation = triple.get("relation")
        head_bad = is_benchmark_span(triple["head"]) or is_task_like(triple, "head")
        tail_bad = is_benchmark_span(triple["tail"]) or (
            relation != "addresses" and is_task_like(triple, "tail")
        )
        if relation in _ROLE_METHOD_HEAD_RELATIONS and head_bad:
            dropped += 1
            continue
        if relation in _ROLE_METHOD_TAIL_RELATIONS and tail_bad:
            dropped += 1
            continue
        kept.append(triple)
    return kept, dropped


# ------------------------------------------------------------------ relation-first typing
def infer_types_from_relations(
    triples: list[Triple],
    *,
    protect_sources: tuple[str, ...] = ("gazetteer", "suffix"),
) -> list[Triple]:
    """Infer a type from a relation slot when that slot has exactly one allowed type.

    ``f"{side}_type"`` is overwritten only for **unambiguous** slots (e.g.
    ``evaluated_on`` tail -> Dataset). Ambiguous slots (``uses`` tail =
    Method|Dataset, ``addresses`` tail = Problem|Task, ``explores`` tail =
    Task|Dataset|Problem) keep their current (gazetteer/model) type.

    **Protection:** a type whose source is in ``protect_sources`` (the gazetteer by
    default) is **not** overwritten. This matters when the RE model wrongly emits
    e.g. ``QASPER --outperforms--> QuALITY``: forcing both to Method would hide the
    error, whereas keeping them Dataset lets ``filter_ontology`` drop the edge.
    """
    for triple in triples:
        schema = RELATION_SCHEMA.get(triple.get("relation"))
        if schema is None:
            continue
        for side in ("head", "tail"):
            if triple.get(f"{side}_type_source") in protect_sources:
                continue
            allowed = schema.get(side, ())
            if len(allowed) == 1:
                _set_type(triple, side, allowed[0], "relation")
    return triples


# ------------------------------------------------------------------ directionality
def resolve_directionality(
    triples: list[Triple],
    *,
    context_key: str = "chunk_uuid",
    cycle_break: bool = True,
) -> list[Triple]:
    """Drop self-loops and resolve contradictory/symmetric edges (see module docstring)."""
    # 1) self-loops are never meaningful here.
    kept = [t for t in triples if _fold(t["head"]) != _fold(t["tail"])]

    # 2) asymmetric relations: one direction per unordered pair per context.
    keep_ids: set[int] = set()
    pairs: dict[tuple, dict[tuple, Triple]] = defaultdict(dict)
    for triple in kept:
        relation = triple.get("relation")
        if relation in SYMMETRIC_RELATIONS:
            continue
        ctx = _context(triple, context_key)
        a, b = _fold(triple["head"]), _fold(triple["tail"])
        pair_key = (relation, ctx, tuple(sorted((a, b))))
        direction = (a, b)
        current = pairs[pair_key].get(direction)
        if current is None or _score(triple) > _score(current):
            pairs[pair_key][direction] = triple

    for bucket in pairs.values():
        keep_ids.add(id(max(bucket.values(), key=_score)))

    # 3) symmetric relations: one canonical edge per pair per context.
    symmetric_best: dict[tuple, Triple] = {}
    for triple in kept:
        relation = triple.get("relation")
        if relation not in SYMMETRIC_RELATIONS:
            continue
        ctx = _context(triple, context_key)
        pair = tuple(sorted((_fold(triple["head"]), _fold(triple["tail"]))))
        key = (relation, ctx, pair)
        if key not in symmetric_best or _score(triple) > _score(symmetric_best[key]):
            symmetric_best[key] = triple
    symmetric_ids = {id(t) for t in symmetric_best.values()}

    selected = [
        t
        for t in kept
        if (t.get("relation") in SYMMETRIC_RELATIONS and id(t) in symmetric_ids)
        or (t.get("relation") not in SYMMETRIC_RELATIONS and id(t) in keep_ids)
    ]

    if cycle_break:
        selected = _break_cycles(selected, context_key)
    return selected


def _break_cycles(triples: list[Triple], context_key: str) -> list[Triple]:
    """Break simple cycles in ``outperforms``/``extends`` by dropping the weakest edge."""
    removed: set[int] = set()
    scopes = {
        (t.get("relation"), _context(t, context_key))
        for t in triples
        if t.get("relation") in CYCLE_RELATIONS
    }

    for relation, ctx in scopes:
        group = [
            t for t in triples if t.get("relation") == relation and _context(t, context_key) == ctx
        ]
        while True:
            graph = nx.DiGraph()
            for triple in group:
                if id(triple) in removed:
                    continue
                graph.add_edge(_fold(triple["head"]), _fold(triple["tail"]), triple=triple)

            cycle = next(iter(nx.simple_cycles(graph)), None)
            if not cycle:
                break

            cycle_triples = []
            for i, u in enumerate(cycle):
                v = cycle[(i + 1) % len(cycle)]
                if graph.has_edge(u, v):
                    cycle_triples.append(graph[u][v]["triple"])
            if not cycle_triples:
                break
            removed.add(id(min(cycle_triples, key=_score)))

    return [t for t in triples if id(t) not in removed]


# ------------------------------------------------------------------ pipeline
def postprocess(
    triples: list[Triple],
    *,
    gazetteer: dict[str, dict[str, Any]] | None = None,
    suffix_hints: bool = True,
    type_inference: bool = True,
    role_filter: bool = True,
    directionality: bool = True,
    cycle_break: bool = True,
    context_key: str = "chunk_uuid",
    stats: dict[str, Any] | None = None,
) -> list[Triple]:
    """Run the modular post-process.

    Order: gazetteer -> suffix hints -> relation typing -> **role filter** ->
    directionality. The role filter runs before directionality on purpose: an
    invalid direction (benchmark head) must not win the pair and suppress the
    valid one. ``stats`` (optional) receives ``role_dropped``.
    """
    out = with_soft_defaults(list(triples))
    if gazetteer:
        out = apply_gazetteer(out, gazetteer)
    if suffix_hints:
        out = suffix_type_hint(out)
    if type_inference:
        out = infer_types_from_relations(out)
    if role_filter:
        out, dropped = apply_role_filter(out)
    else:
        dropped = 0
    if stats is not None:
        stats["role_dropped"] = dropped
    if directionality:
        out = resolve_directionality(out, context_key=context_key, cycle_break=cycle_break)
    return out
