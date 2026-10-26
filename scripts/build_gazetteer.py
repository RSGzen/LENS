r"""Build a corpus-only dataset/metric gazetteer from cue patterns.

Streams a deterministic sample of chunks (Experiment / Result / Methodology), extracts
candidate Dataset/Metric names from **corpus cue patterns**, and writes a JSON prior
consumed by ``lens.kg_postprocess.apply_gazetteer``. No external list, no LLM, no model.

Coverage levers (all corpus-only):
- evaluation contexts now capture **lists** ("evaluated on X, Y and Z");
- suffix cues ("X dataset/corpus/benchmark/test set");
- a wider metric vocabulary.

Run:
  .\.venv\Scripts\python.exe scripts\build_gazetteer.py --limit 500000 --min-count 3
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

DEFAULT_SECTION_TYPES = ("Experiment", "Result/Discussion", "Methodology")

# A capitalised phrase of 1-5 words (keeps hyphenated / dotted names).
_NAME = r"[A-Z][A-Za-z0-9\-\+\.]*(?:\s+[A-Z][A-Za-z0-9\-\+\.]*){0,4}"
_NAME_RE = re.compile(_NAME)

# "evaluated (the model) on X, Y and Z" -> capture the tail after the cue.
EVAL_CONTEXT = re.compile(
    r"(?:evaluat\w*|test\w*|benchmark\w*|experiment\w*)[^.;\n]{0,60}?"
    r"\s+(?:on|using|with)\s+(?:the\s+)?([^.;\n]{2,180})",
    re.IGNORECASE,
)
SUFFIX_CUE = re.compile(rf"({_NAME})\s+(?:dataset|corpus|benchmark|test set|testbed)\b")
USE_SUFFIX = re.compile(
    rf"(?:use|using|adopt|adopting)\s+(?:the\s+)?({_NAME})\s+(?:dataset|corpus|benchmark)\b",
    re.IGNORECASE,
)
METRIC_RE = re.compile(
    r"\b(F1(?:-score)?|EM|Exact Match|BLEU(?:-\d)?|ROUGE(?:-[Ll])?|METEOR|"
    r"BERTScore|nDCG(?:@\d+)?|MRR|MAP|AUC|AUROC|RMSE|MAE|accuracy|precision|"
    r"recall|pass@\d+|perplexity)\b"
)

STOPWORDS = {
    "we",
    "the",
    "our",
    "this",
    "that",
    "in",
    "on",
    "for",
    "of",
    "and",
    "to",
    "table",
    "figure",
    "section",
    "results",
    "result",
    "experiments",
    "experiment",
    "model",
    "models",
    "method",
    "methods",
    "data",
    "dataset",
    "benchmark",
    "training",
    "test",
    "testing",
    "baseline",
    "baselines",
    "approach",
    "task",
    "tasks",
    "performance",
    "paper",
    "work",
    "study",
    "analysis",
    "setting",
}


def _clean(name: str) -> str:
    name = " ".join(name.split()).strip(" .,:;()[]")
    return name


def _keep(name: str) -> bool:
    if len(name) < 2 or len(name) > 80:
        return False
    if name.casefold() in STOPWORDS:
        return False
    return any(ch.isalpha() for ch in name)


def _confidence(count: int) -> float:
    if count >= 10:
        return 0.9
    if count >= 5:
        return 0.8
    return 0.7


def _names(text: str) -> tuple[list[str], list[str], list[str]]:
    """Return (eval-context candidates, suffix-backed candidates, metric candidates)."""
    eval_datasets: list[str] = []
    suffix_datasets: list[str] = []
    metrics: list[str] = []

    for match in EVAL_CONTEXT.finditer(text):
        # The tail may be a list: "SQuAD, HotpotQA and MuSiQue".
        eval_datasets.extend(_NAME_RE.findall(match.group(1)))
    for pattern in (SUFFIX_CUE, USE_SUFFIX):
        suffix_datasets.extend(match.group(1) for match in pattern.finditer(text))
    metrics.extend(match.group(1) for match in METRIC_RE.finditer(text))

    def _clean_all(names):
        return [n for n in (_clean(x) for x in names) if _keep(n)]

    return (_clean_all(eval_datasets), _clean_all(suffix_datasets), _clean_all(metrics))


def _iter_chunks(conn, limit: int, section_types: tuple[str, ...], batch_size: int):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_text FROM chunks WHERE section_type = ANY(%s) "
            "ORDER BY chunk_uuid LIMIT %s",
            (list(section_types), limit),
        )
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for (text,) in rows:
                yield text or ""


def main() -> None:
    setup_file_logging("lens.build_gazetteer", "build_gazetteer")

    parser = argparse.ArgumentParser(description="Corpus-only KG gazetteer builder")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--sections",
        default=",".join(DEFAULT_SECTION_TYPES),
        help="comma-separated section_type filter",
    )
    parser.add_argument("--out", default=config.KG_GAZETTEER_PATH)
    args = parser.parse_args()
    section_types = tuple(s.strip() for s in args.sections.split(",") if s.strip())

    print(f"gazetteer: sample={args.limit} chunks from {section_types}")

    t0 = time.perf_counter()
    eval_counts: Counter[tuple[str, str]] = Counter()
    suffix_counts: Counter[tuple[str, str]] = Counter()
    metric_counts: Counter[tuple[str, str]] = Counter()
    displays: dict[str, str] = {}
    seen_chunks = 0

    with connect(readonly=True) as conn:
        for text in _iter_chunks(conn, args.limit, section_types, args.batch_size):
            seen_chunks += 1
            eval_datasets, suffix_datasets, metrics = _names(text)
            for name in eval_datasets:
                key = name.casefold()
                eval_counts[(key, "Dataset")] += 1
                displays.setdefault(key, name)
            for name in suffix_datasets:
                key = name.casefold()
                suffix_counts[(key, "Dataset")] += 1
                displays.setdefault(key, name)
            for name in metrics:
                key = name.casefold()
                metric_counts[(key, "Metric")] += 1
                displays.setdefault(key, name)

    entries: dict[str, dict] = {}
    dataset_keys = {key for key, _kind in eval_counts} | {key for key, _kind in suffix_counts}
    for key in dataset_keys:
        suffix = suffix_counts[(key, "Dataset")]
        eval_ = eval_counts[(key, "Dataset")]
        total = suffix + eval_
        if total < args.min_count:
            continue
        if suffix >= 5:
            confidence = 0.9
        elif suffix >= 1:
            confidence = 0.8
        elif eval_ >= 10:
            confidence = 0.7
        else:
            confidence = 0.6
        entries[key] = {
            "name": displays.get(key, key),
            "type": "Dataset",
            "count": total,
            "suffix_count": suffix,
            "eval_count": eval_,
            "confidence": confidence,
            "source": "corpus_pattern",
        }

    for (key, _kind), count in metric_counts.items():
        if count < args.min_count:
            continue
        current = entries.get(key)
        if current and current["count"] >= count:
            continue
        entries[key] = {
            "name": displays.get(key, key),
            "type": "Metric",
            "count": count,
            "confidence": _confidence(count),
            "source": "corpus_pattern",
        }

    payload = {
        "version": 1,
        "source": "corpus_patterns",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample": {"chunks": seen_chunks, "section_types": list(section_types)},
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
        f"from {seen_chunks} chunks in {elapsed:.1f}s "
        f"({seen_chunks / elapsed:.1f} chunks/s)"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
