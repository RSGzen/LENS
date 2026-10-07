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


def _matryoshka(full_vec: np.ndarray, dim: int) -> np.ndarray:
    """Nomic's Matryoshka transform: layer-norm the full vector, slice, L2-normalize.

    Mirrors the reference recipe for ``nomic-embed-text-v1.5``
    (``F.layer_norm(full) -> [:, :dim] -> F.normalize``). The checkpoint's ST
    pipeline is only ``Transformer -> mean Pooling`` (no ``Normalize``/``LayerNorm``
    module), so this step must be applied explicitly. Pure: numpy in, numpy out —
    no model, no I/O — and directly unit-tested. ``eps=1e-5`` matches
    ``torch.nn.functional.layer_norm``'s default.
    """
    vec = np.asarray(full_vec, dtype=np.float32)
    mean = vec.mean(axis=-1, keepdims=True)
    var = vec.var(axis=-1, keepdims=True)
    vec = (vec - mean) / np.sqrt(var + 1e-5)   # F.layer_norm, no affine
    vec = vec[..., :dim]                        # Matryoshka truncation
    norms = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.clip(norms, 1e-12, None)    # F.normalize (p=2)


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
        *,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.dim = dim
        self._model = model

    @property
    def model(self) -> Any:
        """The encoder; builds the real one on first access, then caches it."""
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> Any:
        """Build the real ``SentenceTransformer`` (heavy import kept inside).

        Returns ``SentenceTransformer(self.model_id, trust_remote_code=True)``
        **without** ``truncate_dim`` — the model emits the full 768 dims, and
        :func:`_matryoshka` (called by :meth:`encode_documents`) applies nomic's
        reference truncation. The import stays inside this method so importing
        :mod:`lens.embeddings` never loads torch.

        ``trust_remote_code=True`` is required: nomic ships custom modelling code.
        """
        from sentence_transformers import SentenceTransformer

        # Load Nomic embed model
        model = SentenceTransformer(model_name_or_path=self.model_id,
                                    trust_remote_code=True)
    
        return model

    def encode_documents(self, texts: Sequence[str], task_prefix: str, batch_size: int) -> np.ndarray:
        """Embed ``texts`` -> ``np.ndarray`` of shape ``(len(texts), dim)``.

        Each text is prefixed with ``task_prefix`` (nomic is prefix-conditioned:
        ``search_document:`` for indexing, ``search_query:`` for queries; a missing
        prefix silently degrades retrieval), encoded at the full 768 dims, then
        passed through :func:`_matryoshka` (``layer_norm -> slice -> L2-normalize``),
        matching nomic's reference recipe.
        """

        # Prepend the task prefix variable to each text
        formatted_texts = [f"{task_prefix}{text}" for text in texts]

        full_embeddings = self.model.encode(
            formatted_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
        )

        return _matryoshka(full_embeddings, self.dim)