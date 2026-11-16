"""Import ``.lens`` bundles into a runs root, without loading the model stack.

``python -m lens.run --import`` imports the runtime, which imports torch, the
transformers stack and the retrieval encoders - several gigabytes of wheels and
a slow start for an operation that only copies verified files. ``runstore`` is
standard-library only, so this script works on a machine that plays recordings
and never runs a model:

    python scripts/import_bundles.py D:\\lens-data\\bundles

Takes one or more paths: a bundle directory, a directory of bundles, or both.
An already-imported run is skipped rather than failed, so the command is safe to
repeat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lens.agent.runstore import BUNDLE_EXT, import_run, list_runs  # noqa: E402


def collect_bundles(paths: Sequence[Path]) -> list[Path]:
    """Return every ``.lens`` directory named by ``paths``, in a stable order."""
    found: list[Path] = []

    for path in paths:
        if path.suffix == BUNDLE_EXT and path.is_dir():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(p for p in path.iterdir() if p.is_dir() and p.suffix == BUNDLE_EXT))

    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import .lens bundles without the model stack")
    parser.add_argument("paths", nargs="+", help="a bundle directory, or a directory of bundles")
    parser.add_argument("--runs-root", default=None, help="override LENS_RUNS_ROOT for this import")
    args = parser.parse_args(argv)

    bundles = collect_bundles([Path(p) for p in args.paths])

    if not bundles:
        print(f"no {BUNDLE_EXT} directories under {', '.join(args.paths)}")
        return 2

    imported = skipped = failed = 0

    for bundle in bundles:
        try:
            manifest = import_run(bundle, path=args.runs_root)
        except FileExistsError:
            skipped += 1
            print(f"skipped  {bundle.name}: already in the runs root")
        except (ValueError, OSError) as exc:
            failed += 1
            print(f"FAILED   {bundle.name}: {exc}")
        else:
            imported += 1
            cost = (manifest.totals or {}).get("cost_usd", 0.0)
            print(f"imported {manifest.run_id} ({manifest.status}, ${cost:.4f})")

    print(f"\n{imported} imported, {skipped} skipped, {failed} failed")
    print(f"runs root now holds {len(list_runs(path=args.runs_root))} run(s)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
