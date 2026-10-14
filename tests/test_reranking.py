r"""Contract tests for the M5 cross-encoder wrapper (no network, no model, zero cost).

A ``FakeCrossEncoder`` stands in for ``sentence-transformers.CrossEncoder``, so
these tests never load torch or the model. These FAIL until the student fills the
logic bodies (scaffold-and-fill — expected for an M5 scaffold module).

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from lens import config
from lens.reranking import Reranker


class FakeCrossEncoder:
    """Records every ``predict`` call; returns a fixed score array."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[dict] = []

    def predict(self, pairs, batch_size: int | None = None):
        self.calls.append({"pairs": list(pairs), "batch_size": batch_size})
        return np.asarray(self._scores, dtype=np.float32)


class RerankerTest(unittest.TestCase):
    def test_score_builds_query_text_pairs_in_order(self) -> None:
        model = FakeCrossEncoder([0.9, 0.1, 0.5])
        rr = Reranker(model=model)

        scores = rr.score("why x", ["a", "b", "c"], batch_size=16)

        self.assertEqual(scores, [0.9, 0.1, 0.5])
        self.assertIsInstance(scores[0], float)
        self.assertEqual(
            model.calls[0]["pairs"],
            [("why x", "a"), ("why x", "b"), ("why x", "c")],
        )
        self.assertEqual(model.calls[0]["batch_size"], 16)

    def test_empty_texts_skips_the_model(self) -> None:
        model = FakeCrossEncoder([])
        rr = Reranker(model=model)

        self.assertEqual(rr.score("q", []), [])
        self.assertEqual(model.calls, [])

    def test_model_is_loaded_lazily_and_cached(self) -> None:
        rr = Reranker()
        loads: list[int] = []

        def fake_load():
            loads.append(1)
            return FakeCrossEncoder([1.0])

        rr._load = fake_load  # type: ignore[method-assign]

        self.assertIs(rr.model, rr.model)  # second access reuses the cache
        self.assertEqual(len(loads), 1)


if __name__ == "__main__":
    unittest.main()
