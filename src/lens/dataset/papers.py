"""M3 — load the manifest papers with abstract embeddings.

Spec: ``MVP Build Roadmap.md`` §4 (M3); ``Thesis Drafts/Tech Stack (Draft).md``
§3 (``papers`` columns) and §4f (the M1 manifest this consumes).

Pipeline
--------
``manifest.json`` (166,704 papers, M1)
  -> :func:`load_papers` — embed each abstract via :class:`lens.embeddings.Embedder`
     and upsert one ``papers`` row (metadata + ``abstract_embedding`` +
     ``embedding_version``).

The **full** manifest is loaded — there is no paper subset (advisor decision,
6 Oct 2026). M3 verify clause: ``rows == manifest["count"]`` and no NULL
``abstract_embedding``. ``tests/test_papers.py`` is the contract — the DB test
skips when the container is down (no API key, zero cost).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from lens import config
from lens.embeddings import Embedder


def load_papers(
    conn: psycopg.Connection,
    entries: Sequence[Mapping[str, Any]],
    embedder: Embedder,
    *,
    embedding_version: str = config.EMBEDDING_VERSION,
) -> int:
    """Embed each abstract and upsert one ``papers`` row per entry.

    ``conn`` is an admin connection (this writes). Fill point. Contract:

      1. ``vectors = embedder.encode_documents([e["abstract"] for e in entries])``
         -> ``np.ndarray`` of shape ``(len(entries), config.EMBEDDING_DIM)`` (512).
      2. ``register_vector(conn)`` once — pgvector's psycopg3 adapter registers
         ``vector``, ``bit``, ``halfvec``, and ``sparsevec``. The column is
         ``halfvec(512)``, whose dumper accepts **only** ``pgvector.HalfVector``
         (unlike ``vector``, which also accepts a raw ``np.ndarray``), so wrap each
         row: ``HalfVector(vectors[i])``.
      3. Upsert keyed on the primary key, so re-running the load is idempotent::

             INSERT INTO papers (arxiv_id, title, authors, categories, arxiv_doi,
                                 journal_ref, current_ver, update_date, abstract,
                                 abstract_embedding, embedding_version)
             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
             ON CONFLICT (arxiv_id) DO UPDATE SET
                 title = EXCLUDED.title, ..., abstract_embedding = EXCLUDED.abstract_embedding
             -- (set every non-key column from EXCLUDED)

         Manifest -> column mapping: ``version`` -> ``current_ver``;
         ``categories`` is a ``list`` (``TEXT[]``); ``arxiv_doi`` / ``journal_ref``
         may be ``None``; ``update_date`` is an ISO ``YYYY-MM-DD`` string (Postgres
         casts it to ``DATE``). Zip each entry with its row of ``vectors``.
      4. Do **not** commit here — the caller owns the transaction.
      5. Return ``len(entries)``.

    The caller passes ``manifest["papers"]``; on the real corpus that is all
    166,704 papers, so batch/commit in the runner rather than one giant
    transaction (Roadmap §10).
    """
    raise NotImplementedError("M3 fill point: embed + upsert papers")
