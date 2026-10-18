"""M8 live sandbox verification (requires Docker Desktop).

Checks, in order:
  1. build/start ``lens-repl`` via compose;
  2. ``/health`` answers;
  3. namespace persistence (``x = 1`` then ``print(x + 1)`` -> ``2``);
  4. a timeout returns an error without killing the runner;
  5. the workspace mount is writable;
  6. no ``OPENROUTER_API_KEY`` / ``LENS_DB_PASSWORD`` is present inside the
     container (``docker compose exec`` or ``/exec`` + ``os.environ``).

This is an integration script, not part of the no-key unit suite.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Run the checks and return a process exit code."""
    raise NotImplementedError("M8 scaffold: implement verify_sandbox.main")


if __name__ == "__main__":
    raise SystemExit(main())
