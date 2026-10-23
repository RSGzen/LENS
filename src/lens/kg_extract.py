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

Implementation status
---------------------
The pipeline extractor (GLiNER NER -> GLiREL RE) is implemented; the joint
variant remains a K-05 probe arm. The losing family is removed after the probe.
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


# Canonical entity type -> the **label string** asked of GLiNER. Zero-shot NER is
# sensitive to label phrasing, so descriptive labels are a cheap K-05 lever. The
# extractor maps the (lower-cased) returned label back to the canonical type, so
# ``filter_ontology`` still sees ``ENTITY_TYPES`` / ``RELATION_SCHEMA``.
ENTITY_LABEL_SETS: dict[str, dict[str, str]] = {
    "abstract": {t: t for t in ENTITY_TYPES},
    "descriptive": {
        "Method": "machine learning method, model, or algorithm",
        "Dataset": "dataset, corpus, or benchmark",
        "Metric": "evaluation metric or measure",
        "Task": "research task or problem setting",
        "Problem": "research problem, limitation, or gap",
        "Concept": "technical concept, term, or technique",
    },
    # SciERC-aligned reference arm (Material ~= Dataset; OtherScientificTerm ~= Concept).
    "scierc": {
        "Method": "Method",
        "Dataset": "Material",
        "Metric": "Metric",
        "Task": "Task",
        "Problem": "Problem",
        "Concept": "OtherScientificTerm",
    },
}


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
        Override the ontology (defaults are the module constants).
    entity_labels:
        Canonical type -> GLiNER label string. Defaults to
        ``{t: t for t in entity_types}`` (the ``abstract`` set); see
        :data:`ENTITY_LABEL_SETS`.
    threshold:
        Minimum confidence kept for a span / relation.
    max_tokens / overlap:
        Extraction window size (GLiNER truncates its input — the medium checkpoint
        at 384 tokens) and the trailing overlap carried into the next window.
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
        entity_labels: dict[str, str] | None = None,
        threshold: float = config.KG_MIN_SCORE,
        max_tokens: int = config.KG_EXTRACT_MAX_TOKENS,
        overlap: int = config.KG_EXTRACT_WINDOW_OVERLAP,
        ner_model: Any | None = None,
        rel_model: Any | None = None,
    ) -> None:
        self.ner_model_id = ner_model_id
        self.rel_model_id = rel_model_id
        self.entity_types = entity_types
        self.relation_schema = relation_schema
        self.entity_labels = entity_labels or {t: t for t in entity_types}
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.overlap = overlap
        self._ner_model = ner_model
        self._rel_model = rel_model
        # GLiNER returns labels lower-cased (its decoder does ``label.lower()``).
        # Map the returned label -> canonical type; fall back to the canonical
        # type's own casing so a mis-cased label still resolves.
        self._canonical_types = {t.casefold(): t for t in entity_types}
        self._label_to_type = {
            label.casefold(): canonical
            for canonical, label in self.entity_labels.items()
        }

    @property
    def ner_model(self) -> Any:
        """The GLiNER model; builds the real one on first access, then caches it."""
        if self._ner_model is None:
            self._ner_model = self._load_ner()
        return self._ner_model

    @property
    def fingerprint(self) -> str:
        """Stable id of the extractor configuration (cache-key component).

        Covers the checkpoint ids, the label set, the threshold, and the window
        parameters, so a label/model/threshold/window change cannot reuse a stale
        cached graph.
        """
        labels = "|".join(f"{k}={v}" for k, v in sorted(self.entity_labels.items()))
        return (
            f"gliner:{self.ner_model_id}:{self.rel_model_id}:"
            f"{labels}:{self.threshold}:{self.max_tokens}:{self.overlap}"
        )

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

            folded = ent["label"].casefold()
            canonical = self._label_to_type.get(
                folded, self._canonical_types.get(folded, ent["label"])
            )
            mapping.append([start_tok, end_tok, canonical, ent["text"]])

        return mapping
                    

    # ------------------------------------------------------------ windowing
    def _tokenizer(self) -> Any:
        """Return the NER tokenizer (GLiNER keeps it on its data processor)."""
        processor = getattr(self.ner_model, "data_processor", None)
        tokenizer = getattr(processor, "transformer_tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        return getattr(self.ner_model, "tokenizer", None)

    def _token_budget(self, tokenizer: Any) -> int:
        """Window budget; leave room for the tokenizer's special tokens."""
        return max(1, self.max_tokens - 2) if tokenizer is not None else self.max_tokens

    def _count_tokens(self, tokenizer: Any, text: str) -> int:
        """Approximate token count with the NER tokenizer (words if unavailable)."""
        tokenize = getattr(tokenizer, "tokenize", None)
        if tokenize is not None:
            return len(tokenize(text))
        return len(text.split())

    def _split_units(self, tokenizer: Any, text: str) -> list[tuple[str, int]]:
        """Split into sentence units (word units if a sentence exceeds the window)."""
        budget = self._token_budget(tokenizer)
        units: list[tuple[str, int]] = []
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            if not sentence:
                continue
            n = self._count_tokens(tokenizer, sentence)
            if n <= budget:
                units.append((sentence, n))
                continue
            # One sentence longer than the window: emit word units so the window
            # packer can apply the overlap at word granularity.
            for word in sentence.split():
                units.append((word, self._count_tokens(tokenizer, word)))
        return units

    def _text_windows(self, text: str) -> list[str]:
        """Pack ``text`` into windows <= ``max_tokens`` with a trailing overlap.

        GLiNER truncates its input (the medium checkpoint at 384 tokens) while our
        chunks are <= 512 nomic tokens, so extraction must be windowed or the tail
        of long chunks is never seen.
        """
        tokenizer = self._tokenizer()
        budget = self._token_budget(tokenizer)
        units = self._split_units(tokenizer, text)
        if not units:
            return []

        windows: list[str] = []
        cur: list[tuple[str, int]] = []
        cur_n = 0

        for unit, n in units:
            if cur and cur_n + n > budget:
                windows.append(" ".join(u for u, _ in cur))
                # Carry a small trailing overlap into the next window.
                overlap_units: list[tuple[str, int]] = []
                overlap_n = 0
                for u, k in reversed(cur):
                    if overlap_n >= self.overlap:
                        break
                    overlap_units.insert(0, (u, k))
                    overlap_n += k
                if overlap_n + n > budget:
                    overlap_units, overlap_n = [], 0
                cur, cur_n = overlap_units, overlap_n
            cur.append((unit, n))
            cur_n += n

        if cur:
            windows.append(" ".join(u for u, _ in cur))
        return windows

    def extract(self, text: str) -> list[dict[str, Any]]:
        r"""Return raw triples for one chunk of ``text``.

        Return ``[]`` for blank text (do **not** call the models). The text is
        split into <= ``max_tokens`` windows (see :meth:`_text_windows`) because
        GLiNER truncates its input; each window is extracted independently and the
        triples are concatenated (``build_kg`` dedups).

        Per window, the **pipeline variant (GLiNER NER -> GLiREL RE)** works as
        follows. The two models use **different index spaces**, so a mapping is
        required — GLiNER returns **character** offsets, GLiREL wants
        **word-token** indices (end-inclusive).

          1. ``ents = self.ner_model.predict_entities(text,
             list(self.entity_labels.values()), threshold=self.threshold)`` — each
             dict has ``start`` / ``end`` = **character** offsets.
          2. Tokenise with **GLiREL's own scheme** —
             ``re.finditer(r"\w+(?:[-_]\w+)*|\S", text)`` — and map each entity's
             char span ``[start, end)`` to **inclusive** word-token indices.
          3. ``rels = self.rel_model.predict_relations(tokens, <relation labels>,
             threshold=self.threshold, ner=ner, top_k=...)``.
          4. Ontology constraints are enforced downstream by
             ``tools/kg.filter_ontology``; this extractor emits typed triples.
          5. Map GLiREL's output (end-**exclusive** positions; token-list texts)
             back to typed triple dicts.

        **Joint variant.** A single GLiNER multi-task / "relex" checkpoint returns
        entities **and** relations in one call — adapt its output to the same
        triple dict. This arm needs no char->token mapping.
        """
        if not text or not text.strip():
            return []

        triples: list[dict[str, Any]] = []
        for window in self._text_windows(text):
            triples.extend(self._extract_window(window))
        return triples

    def _extract_window(self, text: str) -> list[dict[str, Any]]:
        """Run the GLiNER -> GLiREL pipeline on one window."""
        ents = self.ner_model.predict_entities(
            text, list(self.entity_labels.values()), threshold=self.threshold
        )
        if len(ents) < 2:
            return []  # GLiREL needs at least two entities to relate

        pattern = r"\w+(?:[-_]\w+)*|\S"
        tokens: list[str] = []
        token_spans: list[tuple[int, int]] = []
        for match in re.finditer(pattern, text):
            tokens.append(match.group())
            token_spans.append((match.start(), match.end()))

        ner = self.map_entity_token_indices(token_spans, ents)
        if len(ner) < 2:
            return []

        relation_labels = list(self.relation_schema.keys())
        rels = self.rel_model.predict_relations(
            tokens,
            relation_labels,
            threshold=self.threshold,
            ner=ner,
            top_k=-1,
        )

        span_to_entity = {
            (start_tok, end_tok): (ent_type, surface)
            for start_tok, end_tok, ent_type, surface in ner
        }

        triples: list[dict[str, Any]] = []
        for rel in rels:
            head_start, head_end = rel["head_pos"]
            tail_start, tail_end = rel["tail_pos"]
            head = span_to_entity.get((head_start, head_end - 1))
            tail = span_to_entity.get((tail_start, tail_end - 1))
            if head is None or tail is None:
                continue
            # Use GLiNER's **surface** span, not GLiREL's token list: the latter
            # splits punctuation ("GPT-3.5" -> ["GPT-3", ".", "5"]) and joining it
            # with spaces corrupts decimals/versions.
            triples.append(
                {
                    "head": head[1],
                    "head_type": head[0],
                    "relation": rel["label"],
                    "tail": tail[1],
                    "tail_type": tail[0],
                    "score": float(rel["score"]),
                }
            )

        return triples
