"""Write the freeze record: everything a reported number depends on.

A campaign is only comparable to itself months later if the code, the models, the
container and the corpus are named. This script collects them into ``freeze.json``
and can re-check an existing file, so "the system was frozen" is a command rather
than a claim.

Required fields (``--check`` fails on a missing or empty one):

    commit, branch, models, jev, caps, embedding, python, platform, prompt, run_mode

Optional blocks are recorded when available and listed as ``null`` when not: the
container digest needs Docker, the corpus counts need the database, the instrument
hashes need the instrument directory. A ``null`` is a fact ("not measured here"),
which is why it is not the same as an empty string.

``git_dirty`` is reported and, by default, fails the check: a freeze on an
uncommitted tree does not describe the code that ran.

Usage::

    python scripts/freeze_manifest.py                     # write freeze.json
    python scripts/freeze_manifest.py --out freeze.json
    python scripts/freeze_manifest.py --check             # verify an existing file
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

from lens import config

FREEZE_FILENAME = "freeze.json"
REQUIRED = (
    "commit",
    "branch",
    "models",
    "jev",
    "caps",
    "embedding",
    "python",
    "platform",
    "prompt",
    "run_mode",
)
INSTRUMENT_SUFFIXES = (".jsonl",)


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str | None:
    """Return trimmed stdout, or ``None`` when the command is unavailable."""
    try:
        finished = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return finished.stdout.strip() if finished.returncode == 0 else None


def git_value(*args: str) -> str | None:
    """Return one git value from the repository root, or ``None``."""
    return _run(["git", *args], cwd=config.REPO_ROOT)


def image_digest(image: str | None = None) -> str | None:
    """Return the local sandbox image id (a tag is not reproducible)."""
    return _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image or config.SANDBOX_IMAGE]
    )


def package_version(name: str) -> str | None:
    """Return an installed distribution's version, or ``None`` when absent."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def instrument_hashes(directory: Path | None) -> dict[str, str]:
    """Return ``{filename: sha256}`` for the instrument files present."""
    import hashlib

    if directory is None or not directory.is_dir():
        return {}

    hashes: dict[str, str] = {}

    for path in sorted(directory.iterdir()):
        if path.suffix in INSTRUMENT_SUFFIXES and path.is_file():
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    return hashes


def corpus_counts() -> dict[str, Any] | None:
    """Return row counts from the database, or ``None`` when it is unreachable.

    Best effort on purpose: the freeze must be writable on a machine without the
    corpus (the demo laptop), and a missing count is recorded as ``null``.
    """
    try:
        from lens.db.connect import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM papers")
            papers = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM chunks")
            chunks = cur.fetchone()[0]

        return {"papers": papers, "chunks": chunks}
    except Exception:  # noqa: BLE001 - a freeze without Docker or the DB is still a freeze
        return None


def build_freeze(instruments: Path | None = None, *, with_db: bool = False) -> dict[str, Any]:
    """Collect the freeze record. Never raises on a missing optional input."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "models": {role: spec.id for role, spec in config.MODEL_REGISTRY.items()}
        | {"jev": config.JEV_MODEL_ID},
        "jev": config.JEV_MODEL_ID,
        "caps": {
            "iterations": config.MAX_ITERATIONS,
            "tokens": config.MAX_TOKENS_PER_QUERY,
            "cost_usd": config.MAX_COST_USD_PER_QUERY,
            "context_tokens": config.CONTEXT_TOKEN_LIMIT,
            "exec_block_s": config.MAX_EXEC_BLOCK_S,
        },
        "embedding": config.EMBEDDING_VERSION,
        "reranker": config.RERANKER_MODEL_ID,
        "retrieval": {
            "k": config.RETRIEVAL_K,
            "n": config.RETRIEVAL_N,
            "C": config.RETRIEVAL_C,
            "c": config.RETRIEVAL_C_TOP,
            "ef_search": config.HNSW_EF_SEARCH,
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "prompt": config.PROMPT_VARIANT,
        "run_mode": config.RUN_MODE,
        "sub_calls": config.SUB_CALLS_ENABLED,
        "kg_enabled": config.KG_ENABLED,
        "gate_enabled": config.GATE_ENABLED,
        "stage_thresholds": dict(config.STAGE_THRESHOLDS),
        "sandbox_image": config.SANDBOX_IMAGE,
        "sandbox_digest": image_digest(),
        "uv": _run(["uv", "--version"]),
        "certifi": package_version("certifi"),
        "corpus": corpus_counts() if with_db else None,
        "instruments": instrument_hashes(instruments),
    }


def check_freeze(record: dict[str, Any], *, allow_dirty: bool = False) -> list[str]:
    """Return the problems that make this record unusable as a freeze."""
    problems: list[str] = []

    for key in REQUIRED:
        value = record.get(key)

        if value is None or value == "" or value == {} or value == []:
            problems.append(f"{key} is missing or empty")

    if record.get("git_dirty") and not allow_dirty:
        problems.append("the working tree is dirty (commit or pass --allow-dirty)")

    commit = record.get("commit")

    if isinstance(commit, str) and len(commit) != 40:
        problems.append(f"commit {commit!r} is not a full 40-character hash")

    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write or verify the freeze record.")
    parser.add_argument("--out", default=str(config.REPO_ROOT / FREEZE_FILENAME))
    parser.add_argument(
        "--check", nargs="?", const="", default=None, help="verify this file (default: --out)"
    )
    parser.add_argument("--instruments", default=None, help="directory of instrument JSONL files")
    parser.add_argument("--with-db", action="store_true", help="record corpus row counts")
    parser.add_argument("--allow-dirty", action="store_true", help="accept an uncommitted tree")
    args = parser.parse_args(argv)

    if args.check is not None:
        path = Path(args.check or args.out)

        if not path.is_file():
            print(f"{path}: no freeze record", file=sys.stderr)
            return 2

        record = json.loads(path.read_text(encoding="utf-8"))
        problems = check_freeze(record, allow_dirty=args.allow_dirty)

        for problem in problems:
            print(f"  - {problem}")

        print(f"{path}: {'ok' if not problems else 'NOT a valid freeze'}")

        return 0 if not problems else 1

    record = build_freeze(
        Path(args.instruments) if args.instruments else None,
        with_db=args.with_db,
    )
    Path(args.out).write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    problems = check_freeze(record, allow_dirty=args.allow_dirty)
    print(f"wrote {args.out}")

    for problem in problems:
        print(f"  warning: {problem}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
