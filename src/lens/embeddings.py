"""Host embedding wrapper — nomic-embed-text-v1.5 (Matryoshka 768 -> 512, halfvec).

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §2 (stage 7) + §3 (the
``halfvec(512)`` columns); ``MVP Build Roadmap.md`` §4 (M3).

One shared host-side embedder (M3 documents; M5 reuses it for queries). It is
deliberately thin: ``sentence-transformers`` does the work; this module only
pins the model id, the Matryoshka width, and nomic's task-prefix convention.

Why the model is loaded lazily
------------------------------
Importing ``sentence-transformers`` pulls in torch (heavy; GPU init), so the
model is built on first use. Unit tests inject a fake and never load it.
``tests/test_embeddings.py`` is the contract (no key, zero cost).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from lens import config


class Embedder:
    """nomic-embed-text-v1.5 wrapper producing 512-dim, L2-normalized vectors.

    Parameters
    ----------
    model_id:
        Hugging Face id of the sentence-transformers model.
    dim:
        Matryoshka truncation width — must equal :data:`config.EMBEDDING_DIM`.
    model:
        An already-built encoder exposing ``.encode(texts, ...)``. Injectable
        for tests; when ``None`` the real model is loaded on first use.
    """

    def __init__(
        self,
        model_id: str = config.EMBEDDING_MODEL_ID,
        dim: int = config.EMBEDDING_DIM,
        precision: str = config.EMBEDDING_PRECISION,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.dim = dim
        self.precision = precision
        self._model = model

    @property
    def model(self) -> Any:
        """The encoder; builds the real one on first access, then caches it."""
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> Any:
        """Build the real ``SentenceTransformer`` (heavy import kept inside).

        Fill point. Contract:
          - ``from sentence_transformers import SentenceTransformer`` (inside this
            method, so importing :mod:`lens.embeddings` never loads torch);
          - return ``SentenceTransformer(self.model_id, trust_remote_code=True)``
            **without** ``truncate_dim`` — the model emits the full 768 dims and
            :meth:`encode_documents` performs the Matryoshka slice.

        ``trust_remote_code=True`` is required: nomic ships custom modelling code.

        Scoping note: nomic's *reference* recipe layer-norms over the full 768
        dims **before** slicing (``layer_norm -> slice -> normalize``). That fix is
        tracked **separately** so the dimension/precision change lands uncounfounded
        — see the roadmap "layer_norm recipe" item. :meth:`encode_documents` is
        slice-only for now (same semantics as ST ``truncate_dim``); do not silently
        add the layer-norm.
        """
        from sentence_transformers import SentenceTransformer

        # Load Nomic embed model
        model = SentenceTransformer(model_name_or_path=self.model_id,
                                    trust_remote_code=True)
    
        return model

    def encode_documents(self, texts: Sequence[str], task_prefix: str, batch_size: int) -> np.ndarray:
        """Embed document texts -> ``np.ndarray``, shape ``(len(texts), dim)``.

        Fill point. Contract:
          - prefix every text with :data:`config.EMBEDDING_DOC_PREFIX`; nomic is
            prefix-conditioned, and a missing/incorrect prefix silently degrades
            retrieval, so the prefix is applied here and nowhere else;
          - encode the prefixed texts (the model emits the full 768 dims), then
            Matryoshka-slice to ``self.dim``: ``encoded[..., : self.dim]``;
          - L2-normalize the sliced vectors (required for the cosine/HNSW index);
          - return a float numpy array (``np.asarray(...)``).

        The nomic reference ``layer_norm`` step is a **separate** scoped item; this
        method is slice-only for now (see the ``_load`` scoping note).
        """

        # Prepend the task prefix variable to each text
        formatted_texts = [f"{task_prefix}{text}" for text in texts]

        embeddings = self.model.encode(
            formatted_texts,
            batch_size=batch_size,
            truncate_dim=self.dim,
            precision=self.precision,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embeddings