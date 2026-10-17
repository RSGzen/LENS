r"""M7 — deterministic NER/RE extraction (GLiNER family), host-side.

Spec: ``MVP Build Roadmap.md`` §4 (M7); ``Thesis Drafts/Tech Stack (Draft).md``
§2 (stage 12), §3, §4i. Decisions: D-06/A-02 (query-time, deterministic, over the
reranked set), K-01 (ontology), K-05 (recall probe + fallback), K-06 (cache).

What this module is
-------------------
The **host-side extractor**: text in -> raw triples out. It is host-side because
GLiNER/GLiREL need torch, and the sandbox holds no torch/key (LOG-32). The graph
assembly, dedup, caching and the RLM-facing helpers live in ``lens.tools.kg``.

Ontology (K-01, SciERC-aligned — see ``docs/ontology.md``)
----------------------------------------------------------
Entities and relations are **zero-shot labels**, so the taxonomy is a precision /
usefulness choice, not a model constraint. The set is compact on purpose:
fine-grained method-like types hurt zero-shot precision.

- **Entities (6):** Method, Dataset, Metric, Task, Problem, Concept.
- **Relations (8):** uses, evaluated_on, measures, compares_to, extends,
  outperforms, addresses, explores — each with allowed head/tail entity types
  (:data:`RELATION_SCHEMA`).

Variant (decided by the K-05 probe)
-----------------------------------
:class:`GlinerExtractor` is written so the probe can pick between:
  * **pipeline** — GLiNER (spans) then GLiREL (relations, constrained by
    ``allowed_head``/``allowed_tail``), and
  * **joint** — a single GLiNER multi-task / "relex" checkpoint.
The losing family's dependency is removed once the probe is resolved (Roadmap §4 M7).

Why the models are loaded lazily
--------------------------------
Importing ``gliner``/``glirel`` pulls in torch (heavy; GPU init), so models are
built on first :meth:`GlinerExtractor.extract`. Unit tests inject a fake and never
load them. ``tests/test_kg.py`` is the contract (no key, zero cost).

Scaffold-and-fill (LOG-33)
--------------------------
Signatures, type hints, docstrings, the ontology registry, and config land here;
the **student fills the logic bodies** (``raise NotImplementedError``). Bodies are
described step by step so the contract is unambiguous.
"""

from __future__ import annotations

from typing import Any, Protocol

from lens import config

import re

# Re-export the version here so callers key everything off one value.
ONTOLOGY_VERSION = config.KG_ONTOLOGY_VERSION

# Entity types — the labels GLiNER is asked for. Order is cosmetic (kept stable so
# tests and logs read the same way). "Concept" is the OtherScientificTerm catch-all
# (SciERC) so useful nouns are not silently dropped.
ENTITY_TYPES: tuple[str, ...] = (
    "Method",
    "Dataset",
    "Metric",
    "Task",
    "Problem",
    "Concept",
)

# Relation schema — relation name -> allowed head / tail entity types. Drives both
# GLiREL's label constraints (allowed_head / allowed_tail) and the post-hoc
# ontology filter in ``tools/kg.filter_ontology``. A triple whose head/tail types
# are not in the lists here is off-ontology and gets dropped.
RELATION_SCHEMA: dict[str, dict[str, tuple[str, ...]]] = {
    "uses": {"head": ("Method",), "tail": ("Method", "Dataset")},
    "evaluated_on": {"head": ("Method",), "tail": ("Dataset",)},
    "measures": {"head": ("Method",), "tail": ("Metric",)},
    "compares_to": {"head": ("Method",), "tail": ("Method",)},
    "extends": {"head": ("Method",), "tail": ("Method",)},
    "outperforms": {"head": ("Method",), "tail": ("Method",)},
    "addresses": {"head": ("Method",), "tail": ("Problem", "Task")},
    "explores": {"head": ("Method",), "tail": ("Task", "Dataset", "Problem")},
}

# GLiREL label payload built from RELATION_SCHEMA (registry data). GLiREL expects
# ``{"glirel_labels": {<label>: {"allowed_head": [...], "allowed_tail": [...]}}}``;
# an empty dict means "any entity type".
GLIREL_LABELS: dict[str, dict[str, dict[str, list[str]]]] = {
    "glirel_labels": {
        relation: {
            "allowed_head": list(types["head"]),
            "allowed_tail": list(types["tail"]),
        }
        for relation, types in RELATION_SCHEMA.items()
    }
}


def relation_label_config() -> dict[str, dict[str, dict[str, list[str]]]]:
    """Return the GLiREL label payload for :data:`RELATION_SCHEMA` (registry data)."""

    return GLIREL_LABELS


class KGExtractor(Protocol):
    """Text -> raw triples. Both the pipeline and joint extractors satisfy this.

    A returned triple is a plain dict::

        {"head": str, "head_type": str, "relation": str, "tail": str,
         "tail_type": str, "score": float}

    Provenance (``chunk_uuid`` / ``arxiv_id`` / ``section_type``) is **not** the
    extractor's job — ``tools/kg.build_kg`` attaches it per chunk.
    """

    def extract(self, text: str) -> list[dict[str, Any]]: ...


class GlinerExtractor:
    """GLiNER (+ GLiREL) host extractor. Models are built on first use.

    Parameters
    ----------
    ner_model_id:
        Entity (GLiNER) checkpoint; used for the pipeline variant.
    rel_model_id:
        Relation (GLiREL) checkpoint; used for the pipeline variant.
    entity_types / relation_schema:
        Override the ontology labels (defaults are the module constants).
    threshold:
        Minimum confidence kept for a span / relation.
    ner_model / rel_model:
        Pre-built models (or a joint model passed as ``ner_model`` with
        ``rel_model=None``) — injectable for tests; ``None`` builds the real one
        lazily.
    """

    def __init__(
        self,
        ner_model_id: str = config.KG_NER_MODEL_ID,
        rel_model_id: str = config.KG_REL_MODEL_ID,
        *,
        entity_types: tuple[str, ...] = ENTITY_TYPES,
        relation_schema: dict[str, dict[str, tuple[str, ...]]] = RELATION_SCHEMA,
        threshold: float = config.KG_MIN_SCORE,
        ner_model: Any | None = None,
        rel_model: Any | None = None,
    ) -> None:
        self.ner_model_id = ner_model_id
        self.rel_model_id = rel_model_id
        self.entity_types = entity_types
        self.relation_schema = relation_schema
        self.threshold = threshold
        self._ner_model = ner_model
        self._rel_model = rel_model
        # GLiNER returns labels lower-cased (its decoder does ``label.lower()``),
        # so map back to the canonical ENTITY_TYPES casing or the ontology filter
        # (tools/kg.filter_ontology) would drop every triple.
        self._canonical_types = {t.casefold(): t for t in entity_types}

    @property
    def ner_model(self) -> Any:
        """The GLiNER model; builds the real one on first access, then caches it."""
        if self._ner_model is None:
            self._ner_model = self._load_ner()
        return self._ner_model

    @property
    def rel_model(self) -> Any:
        """The GLiREL model; builds the real one on first access, then caches it."""
        if self._rel_model is None:
            self._rel_model = self._load_rel()
        return self._rel_model

    def _load_ner(self) -> Any:
        """Build the real ``GLiNER`` (heavy import kept inside).

        Returns ``GLiNER.from_pretrained(self.ner_model_id)``. The import stays
        inside this method so importing :mod:`lens.kg_extract` never loads torch.
        """
        import torch
        from gliner import GLiNER

        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        return GLiNER.from_pretrained(self.ner_model_id, map_location=map_location)

    def _load_rel(self) -> Any:
        """Build the real ``GLiREL`` (heavy import kept inside).

        Returns a loaded ``GLiREL``. The import stays inside this method so
        importing :mod:`lens.kg_extract` never loads torch.

        Compatibility shim: ``glirel==1.2.1`` predates ``huggingface_hub>=1.0``,
        which (a) no longer passes ``proxies`` / ``resume_download`` to
        ``_from_pretrained`` and (b) removed both kwargs from ``hf_hub_download``.
        Call ``_from_pretrained`` with the required kwargs ourselves and wrap
        glirel's module-level ``hf_hub_download`` so the removed kwargs are dropped.
        """
        import torch
        import glirel.model as glirel_model
        from glirel import GLiREL

        if not getattr(glirel_model.hf_hub_download, "_lens_compat", False):
            original_download = glirel_model.hf_hub_download

            def _compat_hf_hub_download(*args, proxies=None, resume_download=None, **kwargs):
                return original_download(*args, **kwargs)

            _compat_hf_hub_download._lens_compat = True
            glirel_model.hf_hub_download = _compat_hf_hub_download

        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        return GLiREL._from_pretrained(
            model_id=self.rel_model_id,
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=False,
            token=None,
            map_location=map_location,
        )

    def map_entity_token_indices(
        self,
        token_spans: list[tuple[int, int]],
        ents_list: list[dict[str, Any]],
    ) -> list[list[Any]]:
        """Map GLiNER **char** spans to GLiREL **inclusive word-token** spans.

        ``token_spans`` are the ``(char_start, char_end)`` offsets of the tokens
        (from GLiREL's own regex), aligned 1:1 with the token list. For each entity
        return the token range whose char span overlaps its ``[start, end)`` span:
        ``[start_tok, end_tok, canonical_type, surface_text]`` — ``end_tok`` is
        **inclusive** (GLiREL's ``ner`` convention). Entities that overlap no token
        are dropped. Overlap (not equality) is used so an entity that begins or ends
        mid-token is still captured.
        """
        mapping: list[list[Any]] = []
        for ent in ents_list:
            cs, ce = ent["start"], ent["end"]  # GLiNER char span [cs, ce)
            start_tok: int | None = None
            end_tok: int | None = None
            for i, (ts, te) in enumerate(token_spans):
                if te <= cs:            # token ends before the entity starts
                    continue
                if ts >= ce:            # token starts after the entity ends
                    break
                if start_tok is None:
                    start_tok = i
                end_tok = i             # keep the last overlapping token

            if start_tok is None:
                continue

            canonical = self._canonical_types.get(ent["label"].casefold(), ent["label"])
            mapping.append([start_tok, end_tok, canonical, ent["text"]])

        return mapping
                    

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Return raw triples for one chunk of ``text``.

        Return ``[]`` for blank text (do **not** call the models).

        **Pipeline variant (GLiNER NER -> GLiREL RE).** The two models use
        **different index spaces**, so a mapping is required — GLiNER returns
        **character** offsets, GLiREL wants **word-token** indices (end-inclusive).

          1. ``ents = self.ner_model.predict_entities(text, list(self.entity_types),
             threshold=self.threshold)`` — each dict has ``start`` / ``end`` =
             **character** offsets (``{start, end, text, label, score}``).
          2. Tokenise ``text`` with **GLiREL's own scheme** —
             ``re.finditer(r"\\w+(?:[-_]\\w+)*|\\S", text)`` — and map each entity's
             char span ``[start, end)`` to **inclusive** word-token indices. This is
             **not** the same index space as GLiNER's char offsets; feeding char
             offsets straight into ``ner`` silently misattaches entities. (Equally
             valid: pass your **own** token list to GLiREL and index into that —
             but then both sides must use the one tokenisation.)
          3. ``rels = self.rel_model.predict_relations(tokens, <relation labels>,
             threshold=self.threshold, ner=ner, top_k=...)`` where
             ``ner = [[start_tok, end_tok, type, text], ...]``. The relation labels
             are the **canonical** relation names (:meth:`relation_label_config`);
             ``predict_relations`` reads only ``ner[i][0]`` / ``ner[i][1]`` — the
             ``type`` slot is ignored by the RE model.
          4. Ontology constraints: ``predict_relations`` does **not** honour
             ``allowed_head`` / ``allowed_tail``. This extractor emits **typed**
             triples and ``tools/kg.filter_ontology`` enforces the schema
             downstream. (To filter here instead, call
             ``glirel.modules.utils.constrain_relations_by_entity_type(ents, ...)``
             with entity shims exposing ``.start`` / ``.end`` / ``.label_`` in
             end-**exclusive** token space.)
          5. Map each relation to the triple dict. Caveats on GLiREL's output:
             ``head_pos`` / ``tail_pos`` are end-**exclusive** (it adds ``+1``),
             and ``head_text`` / ``tail_text`` are **token lists**, not strings —
             join them. Set ``head_type`` / ``tail_type`` from the **GLiNER**
             labels, not from GLiREL. ``relation`` = label, ``score`` = GLiREL score.

        **Joint variant.** A single GLiNER multi-task / "relex" checkpoint returns
        entities **and** relations in one call (``gliner.multitask``
        relation-extraction head) — adapt its output to the same triple dict. This
        arm needs **no** char->token mapping, which is one reason the K-05 probe
        compares it against the pipeline.
        """
        # Blank guard (docstring contract): do not load/run the models.
        if not text or not text.strip():
            return []

        ents = self.ner_model.predict_entities(
            text, list(self.entity_types), threshold=self.threshold
        )
        if len(ents) < 2:
            return []  # GLiREL needs at least two entities to relate

        # Tokenise with GLiREL's own scheme so our token indices line up with its.
        pattern = r"\w+(?:[-_]\w+)*|\S"
        tokens: list[str] = []
        token_spans: list[tuple[int, int]] = []
        for match in re.finditer(pattern, text):
            tokens.append(match.group())
            token_spans.append((match.start(), match.end()))

        # GLiNER char spans -> GLiREL inclusive token spans: [start, end, type, text].
        ner = self.map_entity_token_indices(token_spans, ents)
        if len(ner) < 2:
            return []

        relation_labels = list(self.relation_schema.keys())
        rels = self.rel_model.predict_relations(
            tokens,                     # positional 'text' arg accepts a token list
            relation_labels,            # list[str] of canonical relation names
            threshold=self.threshold,
            ner=ner,                    # [[start_tok, end_tok, type, text]], end INCLUSIVE
            top_k=-1,                   # return all
        )

        # GLiREL returns end-EXCLUSIVE positions; map back to the entity type via
        # the inclusive (start, end) key we built above. filter_ontology enforces
        # the allowed_head/allowed_tail constraints downstream.
        span_to_type = {
            (start_tok, end_tok): ent_type
            for start_tok, end_tok, ent_type, _ in ner
        }

        triples: list[dict[str, Any]] = []
        for rel in rels:
            head_start, head_end = rel["head_pos"]
            tail_start, tail_end = rel["tail_pos"]
            head_type = span_to_type.get((head_start, head_end - 1))
            tail_type = span_to_type.get((tail_start, tail_end - 1))
            if head_type is None or tail_type is None:
                continue
            triples.append(
                {
                    "head": " ".join(rel["head_text"]),
                    "head_type": head_type,
                    "relation": rel["label"],
                    "tail": " ".join(rel["tail_text"]),
                    "tail_type": tail_type,
                    "score": float(rel["score"]),
                }
            )

        return triples
