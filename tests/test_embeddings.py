r"""Contract tests for the host embedder (no network, no model, zero cost).

A ``FakeModel`` stands in for ``SentenceTransformer``, so these tests never load
torch or the model. ``_matryoshka`` is pure numpy and is tested directly.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from lens import config
from lens.embeddings import Embedder, _matryoshka


class FakeModel:
    """Records every ``encode`` call; returns a deterministic non-constant array.

    Default width is the model's native **768**: the truncation to ``dim`` is done
    by :func:`_matryoshka`, not the model.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs) -> np.ndarray:
        texts = list(texts)
        self.calls.append({"texts": texts, "kwargs": kwargs})
        rng = np.random.default_rng(0)
        return rng.random((len(texts), self.dim)).astype(np.float32)


class MatryoshkaTest(unittest.TestCase):
    """Pure-helper contract: layer-norm -> slice -> L2-normalize (numpy only)."""

    def test_mean_center_slice_and_unit_norm(self) -> None:
        vec = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        out = _matryoshka(vec, dim=3)

        self.assertEqual(out.shape, (1, 3))
        centered = vec - vec.mean(axis=1, keepdims=True)
        expected = centered[:, :3] / np.linalg.norm(centered[:, :3], axis=1, keepdims=True)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_single_vector(self) -> None:
        out = _matryoshka(np.arange(8, dtype=np.float32), dim=4)

        self.assertEqual(out.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=5)


class EmbedderTest(unittest.TestCase):
    def test_encode_documents_prepends_doc_prefix(self) -> None:
        model = FakeModel()
        emb = Embedder(dim=config.EMBEDDING_DIM, model=model)

        emb.encode_documents(["alpha", "beta"], config.EMBEDDING_DOC_PREFIX, batch_size=2)

        self.assertEqual(
            model.calls[0]["texts"],
            [config.EMBEDDING_DOC_PREFIX + "alpha", config.EMBEDDING_DOC_PREFIX + "beta"],
        )

    def test_encode_documents_applies_matryoshka_and_normalizes(self) -> None:
        model = FakeModel(dim=768)
        emb = Embedder(dim=config.EMBEDDING_DIM, model=model)

        out = emb.encode_documents(["alpha", "beta", "gamma"], config.EMBEDDING_DOC_PREFIX, batch_size=2)

        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (3, config.EMBEDDING_DIM))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
        # Truncation/normalization are done by _matryoshka, not inside encode().
        self.assertNotIn("truncate_dim", model.calls[0]["kwargs"])
        self.assertNotIn("normalize_embeddings", model.calls[0]["kwargs"])

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
