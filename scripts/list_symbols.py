#!/usr/bin/env python
"""List the public surface of ``src/lens`` as ``path:symbol`` lines.

The module/class index in the engineering document (S17) is hand-maintained, so it
can drift from the code without anyone noticing. This script prints the surface the
code actually has, in the same ``path:symbol`` form the index uses, so the two can
be diffed in one command::

    python scripts/list_symbols.py
    python scripts/list_symbols.py --package tools

It lists **top-level classes and functions** (the things the index names), not
methods: the index calls out a method only when its behaviour is surprising, and a
list of 400 methods would not be diffable by eye. Nothing here imports the package —
the file is parsed with :mod:`ast`, so no model, no GPU, no database.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "lens"


def surface(path: Path, root: Path) -> list[str]:
    """Return ``package/module.py:symbol`` for every top-level class and function."""
    module = path.resolve().relative_to(root.parent).with_suffix("")
    dotted = ".".join(module.parts)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    symbols = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]

    return [f"{dotted}:{name}" for name in symbols]


def main(argv: list[str] | None = None) -> int:
    """Print the surface, optionally filtered to one package."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", help="only modules under this package, e.g. tools")
    args = parser.parse_args(argv)

    root = Path(args.package) if args.package else None
    files = sorted(PACKAGE_ROOT.rglob("*.py"))

    if root is not None:
        files = [path for path in files if root in path.relative_to(PACKAGE_ROOT).parents]

    if not files:
        print(f"no modules under {PACKAGE_ROOT} matching {args.package!r}", file=sys.stderr)
        return 2

    for path in files:
        for line in surface(path, PACKAGE_ROOT):
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
