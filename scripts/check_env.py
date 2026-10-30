#!/usr/bin/env python
"""Check the environment before running a pipeline script.

Reports, in one pass, where ``.env`` was found and whether the settings a command
needs are present. Secrets are printed as a fingerprint only (first eight and last
four characters), never in full.

    python scripts/check_env.py                     # everything
    python scripts/check_env.py --need api_key,db   # just what a live run needs
    python scripts/check_env.py --need metadata     # just the corpus build

Exit status is 0 when every requested setting is present, 2 otherwise, so it works
as a preflight in a shell chain or a CI step.
"""

from __future__ import annotations

from lens.preflight import main

if __name__ == "__main__":
    raise SystemExit(main())
