"""Shared console + file logger for long-running bulk jobs (M3/M4 runners).

Wiring only, no domain logic. Long load jobs print one progress line per batch
(see ``dataset/papers.py`` and ``dataset/chunks.py``); the runner needs those
lines persisted to a timestamped log file under ``config.LENS_LOGS_ROOT_PATH``
for thesis evidence (Roadmap §4 M3/M4, §8 evidence column).

:func:`setup_file_logging` installs a :class:`logging.Logger` with a console and
a file handler, then replaces ``sys.stdout`` / ``sys.stderr`` with a
:class:`StreamLogger`, so **every** ``print()`` in the load path (including the
per-batch timing lines) is captured with an ``asctime`` timestamp.

Used by ``scripts/load_papers.py`` and ``scripts/load_chunks.py``.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from lens import config


class StreamLogger:
    """File-like object that forwards complete lines to a :mod:`logging` logger.

    Assigned to ``sys.stdout`` / ``sys.stderr`` so that ``print()`` calls in the
    load path are captured in the log file with a timestamp.
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


def setup_file_logging(name: str, prefix: str) -> logging.Logger:
    """Configure a console + file logger and mirror stdout/stderr into it.

    Parameters
    ----------
    name:
        Logger name (e.g. ``"lens.load_chunks"``).
    prefix:
        Filename prefix; the file lands at
        ``<LENS_LOGS_ROOT_PATH>/<prefix>_<YYYYmmdd_HHMMSS>.log``.

    Returns
    -------
    logging.Logger
        The configured logger; per-batch ``print()`` lines flow into it via the
        installed :class:`StreamLogger`.
    """
    log_dir = Path(config.LENS_LOGS_ROOT_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger(name)
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
    sys.stdout = StreamLogger(logger, logging.INFO)
    sys.stderr = StreamLogger(logger, logging.ERROR)

    logger.info("logging to %s", log_path)
    return logger
