"""One-off: make every stored blob byte-exact with its filename (M9 S5, LOG-116).

Why this is needed
------------------
``Trajectory.store_blob`` hashed ``text.encode("utf-8")`` but wrote it with
``Path.write_text``, which translates ``\\n`` -> ``\\r\\n`` on Windows. The filename
was therefore **not** the hash of the file, and bundle verification (correctly)
rejected the blobs: "blob … does not match its name".

The writer is fixed (bytes in, bytes out). This script repairs the blobs written
before that fix: if a blob's name equals the sha256 of its **LF-normalised** bytes,
the file is rewritten with those bytes. Anything else is reported and left alone —
a blob that matches neither form is genuinely corrupt, and guessing would defeat
the tamper-evidence the store exists for.

Idempotent: a second run reports ``ok`` for every blob.

Usage::  python scripts/repair_blobs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from lens import config


def repair(blobs_dir: Path, *, dry_run: bool) -> dict[str, int]:
    """Return counts of ``{ok, repaired, corrupt}`` for ``blobs_dir``."""
    counts = {"ok": 0, "repaired": 0, "corrupt": 0}

    if not blobs_dir.is_dir():
        return counts

    for path in sorted(blobs_dir.iterdir()):
        if not path.is_file():
            continue

        raw = path.read_bytes()

        if hashlib.sha256(raw).hexdigest() == path.name:
            counts["ok"] += 1
            continue

        normalised = raw.replace(b"\r\n", b"\n")

        if hashlib.sha256(normalised).hexdigest() != path.name:
            counts["corrupt"] += 1
            print(f"CORRUPT {path.name}: neither the bytes nor their LF form hash to the name")
            continue

        counts["repaired"] += 1

        if not dry_run:
            path.write_bytes(normalised)

    return counts


def main(argv: list[str] | None = None) -> int:
    """Repair the shared blob store and report what was found."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--blobs", default=str(Path(config.LENS_RUNS_ROOT_PATH) / "blobs"))
    args = parser.parse_args(argv)

    counts = repair(Path(args.blobs), dry_run=args.dry_run)

    print(
        f"blobs {args.blobs}: {counts['ok']} ok, {counts['repaired']} repaired, "
        f"{counts['corrupt']} corrupt" + (" (dry run)" if args.dry_run else "")
    )

    return 1 if counts["corrupt"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
