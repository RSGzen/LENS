r"""Contract tests for the host embedder (no network, no model, zero cost).

A ``FakeModel`` stands in for ``SentenceTransformer``, so these tests never load
torch. They fail with ``NotImplementedError`` until the fill points are done.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from lens import config
from lens.embeddings import Embedder


class FakeModel:
    """Records every ``encode`` call; returns a fixed, correctly-shaped array."""

    def __init__(self, dim: int = config.EMBEDDING_DIM) -> None:
        self.dim = dim
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs) -> np.ndarray:
        texts = list(texts)
        self.calls.append({"texts": texts, "kwargs": kwargs})
        return np.ones((len(texts), self.dim), dtype=np.float32)


class EmbedderTest(unittest.TestCase):
    def test_encode_documents_prepends_doc_prefix(self) -> None:
        model = FakeModel()
        emb = Embedder(dim=config.EMBEDDING_DIM, model=model)

        emb.encode_documents(["alpha", "beta"])

        self.assertEqual(
            model.calls[0]["texts"],
            [config.EMBEDDING_DOC_PREFIX + "alpha", config.EMBEDDING_DOC_PREFIX + "beta"],
        )

    def test_encode_documents_normalizes_and_shapes(self) -> None:
        model = FakeModel()
        emb = Embedder(dim=config.EMBEDDING_DIM, model=model)

        out = emb.encode_documents(["alpha", "beta", "gamma"])

        self.assertTrue(model.calls[0]["kwargs"].get("normalize_embeddings"))
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (3, config.EMBEDDING_DIM))

    def test_model_is_loaded_lazily_and_cached(self) -> None:
        emb = Embedder(dim=config.EMBEDDING_DIM)
        loads: list[int] = []

        def fake_load():
            loads.append(1)
            return FakeModel()

        emb._load = fake_load  # type: ignore[method-assign]

        self.assertIs(emb.model, emb.model)  # second access reuses the cache
        self.assertEqual(len(loads), 1)


if __name__ == "__main__":
    unittest.main()
