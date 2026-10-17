r"""Build a corpus-only dataset/metric gazetteer from cue patterns.

Reads a deterministic sample of chunks (Experiment / Result / Methodology), extracts
candidate Dataset and Metric names from **corpus cue patterns**, and writes a JSON
prior consumed by ``lens.kg_postprocess.apply_gazetteer``.

No external list, no LLM, no model. Deterministic; stdout is mirrored to a timestamped
log under ``config.LENS_LOGS_ROOT_PATH``.

Run:
  .\.venv\Scripts\python.exe scripts\build_gazetteer.py --limit 20000 --min-count 3
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from lens import config
from lens.db.connect import connect
from lens.observability.run_logging import setup_file_logging

SECTION_TYPES = ("Experiment", "Result/Discussion", "Methodology")

# A capitalised phrase of 1-5 words (keeps hyphenated / dotted names).
_NAME = r"[A-Z][A-Za-z0-9\-\+\.]*(?:\s+[A-Z][A-Za-z0-9\-\+\.]*){0,4}"

DATASET_CUES = [
    re.compile(
        rf"(?:evaluat(?:e|ed|ion)|test(?:ed|ing)?|benchmark(?:ed|ing)?|"
        rf"experiment(?:s|ed|ing)?)\s+(?:on|using|with)\s+(?:the\s+)?({_NAME})"
    ),
    re.compile(rf"({_NAME})\s+(?:dataset|corpus|benchmark)\b"),
]
METRIC_RE = re.compile(
    r"\b(F1(?:-score)?|EM|Exact Match|BLEU(?:-\d)?|ROUGE(?:-[Ll])?|"
    r"nDCG(?:@\d+)?|MRR|MAP|accuracy|precision|recall)\b"
)

STOPWORDS = {
    "we", "the", "our", "this", "that", "in", "on", "for", "of", "and", "to",
    "table", "figure", "section", "results", "result", "experiments", "experiment",
    "model", "models", "method", "methods", "data", "dataset", "benchmark",
    "training", "test", "testing", "baseline", "baselines", "approach", "task",
    "tasks", "performance", "paper", "work", "study", "analysis", "setting",
}


def _clean(name: str) -> str:
    name = " ".join(name.split()).strip(" .,:;()[]")
    return name


def _keep(name: str) -> bool:
    if len(name) < 2 or len(name) > 80:
        return False
    if name.casefold() in STOPWORDS:
        return False
    if not any(ch.isalpha() for ch in name):
        return False
    return True


def _confidence(count: int) -> float:
    if count >= 10:
        return 0.9
    if count >= 5:
        return 0.8
    return 0.7


def _sample_chunks(conn, limit: int) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_text, section_type FROM chunks "
            "WHERE section_type = ANY(%s) "
            "ORDER BY chunk_uuid LIMIT %s",
            (list(SECTION_TYPES), limit),
        )
        return cur.fetchall()


def main() -> None:
    setup_file_logging("lens.build_gazetteer", "build_gazetteer")

    parser = argparse.ArgumentParser(description="Corpus-only KG gazetteer builder")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--out", default=config.KG_GAZETTEER_PATH)
    args = parser.parse_args()

    print(f"gazetteer: sample={args.limit} chunks from {SECTION_TYPES}")

    t0 = time.perf_counter()
    counts: Counter[tuple[str, str]] = Counter()
    displays: dict[str, str] = {}

    with connect(readonly=True) as conn:
        rows = _sample_chunks(conn, args.limit)

    for chunk_text, _section in rows:
        for pattern in DATASET_CUES:
            for match in pattern.finditer(chunk_text or ""):
                name = _clean(match.group(1))
                if _keep(name):
                    key = name.casefold()
                    counts[(key, "Dataset")] += 1
                    displays.setdefault(key, name)
        for match in METRIC_RE.finditer(chunk_text or ""):
            name = _clean(match.group(1))
            if _keep(name):
                key = name.casefold()
                counts[(key, "Metric")] += 1
                displays.setdefault(key, name)

    entries: dict[str, dict] = {}
    for (key, kind), count in counts.items():
        if count < args.min_count:
            continue
        # Keep the most frequent kind for a name seen under both cues.
        current = entries.get(key)
        if current and current["count"] >= count:
            continue
        entries[key] = {
            "name": displays.get(key, key),
            "type": kind,
            "count": count,
            "confidence": _confidence(count),
            "source": "corpus_pattern",
        }

    payload = {
        "version": 1,
        "source": "corpus_patterns",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample": {"chunks": len(rows), "section_types": list(SECTION_TYPES)},
        "min_count": args.min_count,
        "entries": entries,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    by_type = Counter(entry["type"] for entry in entries.values())
    print(
        f"gazetteer: {len(entries)} entries ({dict(by_type)}) "
        f"from {len(rows)} chunks in {elapsed:.1f}s "
        f"({len(rows) / elapsed:.1f} chunks/s)"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
