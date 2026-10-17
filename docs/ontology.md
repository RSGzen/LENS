# LENS ontology (K-01) — query-time knowledge graph

> **Status:** LENS-defined, **SciERC-aligned** compact ontology. Registry source of
> truth: `src/lens/kg_extract.py` (`ENTITY_TYPES`, `RELATION_SCHEMA`,
> `ONTOLOGY_VERSION`). This file is the human-facing mirror and the sandbox context
> file ("ontology.md", Tech Stack §4).
>
> **Scope:** the KG is built **at query time**, deterministically, over the
> **deep-matched chunks** of the reranked top-`n` papers (D-06/A-02). There is **no
> corpus-scale KG**. Extraction is host-side (GLiNER/GLiREL); the graph is handed to
> the RLM as a `networkx` variable (K-07), not injected as text.
>
> **Version:** `lens-ontology-1`. Bump `KG_ONTOLOGY_VERSION` on any change here — it
> is the re-extraction key and part of the KG cache key (K-06).

## Why this set

Labels are **zero-shot**, so the taxonomy is a precision/usefulness choice, not a
model constraint. It is aligned to **SciERC** (Luan et al., EMNLP 2018 — the
scientific-IE reference) with two adaptations:

1. **Method-like types are collapsed** into `Method`. Splitting Method / Model /
   Algorithm / Framework is semantically fuzzy and costs zero-shot precision.
2. **Claim relations are added** (`extends`, `outperforms`) that SciERC's generic
   `Compare` does not separate — LENS's "idea checking" needs them.

## Entities (6)

| Entity | Meaning | SciERC analogue |
|---|---|---|
| `Method` | algorithm / model / framework / technique (absorbs the old Algorithm/Model/Framework) | Method |
| `Dataset` | corpus / benchmark / dataset / material | Material |
| `Metric` | evaluation metric (accuracy, nDCG, F1, latency…) | Metric |
| `Task` | research task / problem setting | Task |
| `Problem` | research problem / gap (kept distinct for problem discovery) | — |
| `Concept` | catch-all for other scientific terms | OtherScientificTerm |

## Relations (8, directed, head → tail typed)

| Relation | Head | Tail | SciERC analogue |
|---|---|---|---|
| `uses` | Method | Method, Dataset | Used-For |
| `evaluated_on` | Method | Dataset | Evaluate-For |
| `measures` | Method | Metric | — |
| `compares_to` | Method | Method | Compare |
| `extends` | Method | Method | — (improves upon) |
| `outperforms` | Method | Method | — (result claim) |
| `addresses` | Method | Problem, Task | — |
| `explores` | Method | Task, Dataset, Problem | — (Fig 4.2.2) |

**Deltas from the original K-01 draft** (log as a decision): Algorithm/Model/
Framework → `Method`; `utilizes` → `uses`; `evaluates` + `evaluated_on` →
`evaluated_on`; **dropped** `proposes` (a paper is provenance, not a graph node);
**added** `Concept`, `measures`, `addresses`; kept `explores`.

## Provenance (the M7 verify clause)

Every triple carries `chunk_uuid`, `arxiv_id`, and `section_type`, so any edge
resolves back to the exact retrieved chunk — LENS citations trace
`chunk_uuid → arxiv_id + section_type`.

## Open validation

- **K-05:** measured recall on a 20-paper cs.AI sample; fallback = few-shot LLM
  extraction (strict JSON schema) if recall is low.
- **K-02 remainder:** fuzzy entity resolution / confidence merging is a deferred
  sensitivity item; M7 does normalization + exact dedup only.

## Benchmark ambiguity (17 Oct 2026)

A benchmark is a **Dataset that defines a Task** (it holds items/questions scored by metrics), so a benchmark span can legitimately be `Dataset` or `Method` depending on its role:

- as the **tail of `evaluated_on` / `uses`** → `Dataset`;
- as the **head of `measures` / `outperforms`** (the benchmark/system acting as the method) → `Method`.

**Evaluation policy:** for benchmark-like spans, accept `Dataset | Method`
(`score_gold.py --benchmark-lenient`).

**Pipeline policy:** type by **relation role**, not one label. A benchmark/task span
must not be the head of `evaluated_on` / `outperforms` (the *method* is the subject);
that is a **role/selection filter**, not an NER label change. (Follow-up, not yet
implemented.)
