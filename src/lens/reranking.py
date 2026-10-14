"""Host reranker wrapper — bge-reranker-v2-m3 cross-encoder (Stage A + Stage B).

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §2 (stages 10-11), §3 (M5 tuning),
§7 (reranker placement); ``MVP Build Roadmap.md`` §4 (M5).

One shared host-side cross-encoder drives **both** funnel stages: Stage A ranks
the ``k`` candidate paper abstracts down to ``n``, Stage B ranks the ``C``
candidate chunks down to ``c``. It is deliberately thin — ``sentence-transformers``
does the work; this module only pins the model id and the batch size.

Why the model is loaded lazily
------------------------------
Importing ``sentence-transformers`` pulls in torch (heavy; GPU init), so the
model is built on first use. Unit tests inject a fake cross-encoder and never
load it. ``tests/test_reranking.py`` is the contract (no key, zero cost).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lens import config


class Reranker:
    """bge-reranker-v2-m3 cross-encoder producing one relevance score per text.

    Parameters
    ----------
    model_id:
        Hugging Face id of the sentence-transformers **cross-encoder**.
    model:
        An already-built cross-encoder exposing ``.predict(pairs, batch_size=...)``.
        Injectable for tests; when ``None`` the real model is loaded on first use.
    """

    def __init__(
        self,
        model_id: str = config.RERANKER_MODEL_ID,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self._model = model

    @property
    def model(self) -> Any:
        """The cross-encoder; builds the real one on first access, then caches it."""
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> Any:
        """Build the real ``CrossEncoder`` (heavy import kept inside).

        Returns ``CrossEncoder(self.model_id)``. The import stays inside this
        method so importing :mod:`lens.reranking` never loads torch.
        """
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_id)

    def score(
        self,
        query: str,
        texts: Sequence[str],
        batch_size: int = config.RERANK_BATCH_SIZE,
    ) -> list[float]:
        """Return one relevance score per text (higher = more relevant).

        The cross-encoder scores ``(query, text)`` **pairs**, so build
        ``pairs = [(query, t) for t in texts]`` and call
        ``self.model.predict(pairs, batch_size=batch_size)``; return the scores as
        a plain ``list[float]`` in the **same order** as ``texts``. A blank
        ``texts`` -> ``[]`` (do **not** call the model).
        """
        raise NotImplementedError(
            "M5 scaffold: implement Reranker.score — build (query, text) pairs, "
            "call self.model.predict(pairs, batch_size=batch_size), return a "
            "list[float] in the input order ([] for empty texts)."
        )
