"""M3 runner — load the full manifest into ``papers``.

Wiring only: read the M1 manifest, embed every abstract, upsert. All stdout —
including the per-chunk timing lines printed by ``dataset/papers.py`` — is
mirrored to a timestamped log file under ``config.LENS_LOGS_ROOT_PATH`` for
thesis evidence.

Prereqs: the ``lens-pg`` container is running with the schema applied
(``python scripts/setup_db.py``); the M1 manifest exists under the manifests root.
For the fastest bulk load, build the HNSW indexes **after** the load
(``python scripts/build_indexes.py``) — see ``Tech Stack (Draft).md`` §3.

Run:  .\\.venv\\Scripts\\python.exe scripts\\load_papers.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from lens import config
from lens.dataset.manifest import load_manifest
from lens.dataset.papers import load_papers
from lens.db.connect import connect
from lens.embeddings import Embedder


class _StreamLogger:
    """File-like object that forwards complete lines to a :mod:`logging` logger.

    Assigned to ``sys.stdout`` / ``sys.stderr`` so that **every** ``print()`` in
    the load path (including the per-chunk lines from ``dataset/papers.py``) is
    captured in the log file with a timestamp.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self._logger.log(self._level, line)
        return len(message)

    def flush(self) -> None:
        if self._buffer:
            self._logger.log(self._level, self._buffer)
            self._buffer = ""


def _setup_logging() -> logging.Logger:
    """Configure a console + file logger and mirror stdout/stderr into it."""
    log_dir = Path(config.LENS_LOGS_ROOT_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"load_papers_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("lens.load_papers")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Grab the real stdout before we replace sys.stdout below.
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Mirror print()/tracebacks from the load path into the logger.
    sys.stdout = _StreamLogger(logger, logging.INFO)
    sys.stderr = _StreamLogger(logger, logging.ERROR)

    logger.info("logging to %s", log_path)
    return logger


def main() -> None:
    _setup_logging()

    manifest_path = Path(config.LENS_MANIFESTS_ROOT_PATH) / config.MANIFEST_FILENAME
    manifest = load_manifest(manifest_path)

    embedder = Embedder()

    process_chunk_size = config.PROCESS_CHUNK_SIZE
    embedding_batch_size = config.EMBEDDING_ABSTRACT_BATCH_SIZE

    start = time.perf_counter()
    with connect(autocommit=True) as conn:
        count = load_papers(conn=conn,
                            entries=manifest["papers"],
                            process_chunk_size=process_chunk_size,
                            encode_batch_size=embedding_batch_size,
                            embedder=embedder)
    elapsed_s = time.perf_counter() - start

    rate = count / elapsed_s if elapsed_s > 0 else 0.0
    print(
        f"loaded {count} papers in {elapsed_s:.1f}s "
        f"({rate:.1f} papers/s; embedding_version={config.EMBEDDING_VERSION})"
    )


if __name__ == "__main__":
    main()
