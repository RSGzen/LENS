"""Does a lexical channel change which papers reach the reranker?

A diagnostic, not part of the pipeline: the lexical list is a sequential scan with
``to_tsvector`` computed on the fly (no index, no DDL), fused with the dense list by
reciprocal rank, and both lists are handed to the real ``rerank_papers``. The
comparison is "same reranker, two candidate lists", so a difference can only come
from the candidate sets.

It was written to decide whether to build hybrid retrieval, and the answer was no:
every known target is already the first, second or third dense hit, and fusion moved
nothing into the top-20 that was not already there. The script is kept because that
negative result is a measurement a reader may want to repeat, and because the same
run answers a different question first — how many papers in the corpus state the
phrase at all.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Sequence

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from lens.db.connect import connect  # noqa: E402
from lens.embeddings import Embedder  # noqa: E402
from lens.reranking import Reranker  # noqa: E402
from lens.tools.retrieval import rerank_papers, shallow_match  # noqa: E402

TSV = "to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,''))"
RRF_K = 60

#: query -> arXiv id the query should surface
TESTS: list[tuple[str, str, str]] = [
    (
        "aggregation",
        "Across the retrieved papers, what methods do recursive language model systems use "
        "to aggregate information that exceeds a single context window, and how do they "
        "differ in where the aggregation happens?",
        "2603.15653",
    ),
    ("rlm", "recursive language model long context", "2603.15653"),
    ("oolong", "OOLONG long context reasoning aggregation benchmark", "2511.02817"),
    ("rcc", "recurrent context compression expanding the context window", "2406.06110"),
    ("corpusqa", "corpus-level analysis benchmark ten million tokens", "2601.14952"),
    ("dcp", "decoupled context processing context augmented language model", "2210.05758"),
]


def lexical_query(text: str) -> str | None:
    terms = re.findall(r"[a-z0-9]{3,}", text.lower())

    return " | ".join(terms) if terms else None


def lexical_match(cur: Any, text: str, k: int) -> list[str]:
    q = lexical_query(text)

    if q is None:
        return []

    try:
        cur.execute(
            "SELECT arxiv_id FROM ("
            f"  SELECT arxiv_id, {TSV} AS tsv FROM papers WHERE abstract IS NOT NULL"
            ") s, to_tsquery('english', %s) AS q "
            "WHERE s.tsv @@ q ORDER BY ts_rank_cd(s.tsv, q) DESC LIMIT %s",
            (q, k),
        )
    except Exception as exc:  # noqa: BLE001 - prototype
        print(f"    lexical query failed: {type(exc).__name__}: {exc}")
        return []

    return [row[0] for row in cur.fetchall()]


def rrf(lists: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    first: dict[str, int] = {}

    for list_index, ids in enumerate(lists):
        for rank, doc in enumerate(ids, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            first.setdefault(doc, list_index * 100000 + rank)

    return sorted(scores, key=lambda doc: (-scores[doc], first[doc]))


def position(ids: Sequence[str], target: str) -> str:
    return f"rank {ids.index(target) + 1}" if target in ids else "absent"


def main() -> int:
    embedder = Embedder()
    reranker = Reranker()
    phrase_ids: list[str] = []

    with connect(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT arxiv_id FROM papers WHERE "
                f"{TSV} @@ phraseto_tsquery('english','recursive language model')"
            )
            phrase_ids = [row[0] for row in cur.fetchall()]
            print(
                f"corpus papers whose title/abstract states 'recursive language model': "
                f"{len(phrase_ids)}\n"
            )

            for name, text, target in TESTS:
                print(f"== {name}: {text[:70]}")
                dense = shallow_match(conn, text, embedder, k=200)
                lexical = lexical_match(cur, text, k=200)
                fused = rrf([dense, lexical])[:200]

                base = [row["arxiv_id"] for row in rerank_papers(conn, dense, text, reranker, n=20)]
                hybrid = [
                    row["arxiv_id"] for row in rerank_papers(conn, fused, text, reranker, n=20)
                ]
                conn.rollback()

                print(
                    f"   target {target}: dense {position(dense, target)} | "
                    f"lexical {position(lexical, target)} | fused {position(fused, target)}"
                )
                print(
                    f"   after rerank  dense-candidates {position(base, target)} | "
                    f"hybrid-candidates {position(hybrid, target)}"
                )
                rlm_base = len(set(base) & set(phrase_ids))
                rlm_hybrid = len(set(hybrid) & set(phrase_ids))
                print(f"   RLM papers in the top-20: dense {rlm_base} | hybrid {rlm_hybrid}")
                print(f"   overlap of the two top-20s: {len(set(base) & set(hybrid))}/20")
                print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
