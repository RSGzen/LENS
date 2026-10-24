"""M8 live sandbox verification (requires Docker Desktop).

Checks, in dependency order — the first two are what the thesis claims, so they
run before anything convenient:

  0. **host RPC reachable from inside the container** — a block calls
     ``rlm_api.budget()`` (or ``GET $LENS_HOST_RPC_URL/health``). If this fails on
     ``sandbox-net: internal: true``, apply the compose fallback (fixed subnet +
     bridge-gateway URL), do **not** drop ``internal``: no-egress is a claim
     (Tech Stack 4b), not a nicety.

     > **The image has no shell and no ``curl``** (LOG-32 deletes ``/bin/sh``), so
     > probe it with Python before the runner exists::
     >
     >     docker compose run --rm --entrypoint python lens-repl -c ^
     >       "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:8765/health').status)"
     >
     > with any listener on the host (e.g. ``python -m http.server 8765``). Run
     > this **before** filling ``runner.py``: it is the one failure that blocks
     > everything downstream, and it needs no harness code.
  1. **general egress blocked** — a connect to ``1.1.1.1:443`` (or any host) must
     fail from inside the sandbox. Record the failure: absence of egress is
     evidence, not an assumption.
  2. build/start ``lens-repl`` via compose; ``/health`` answers **from inside the
     container** (``docker compose exec -T lens-repl python /app/exec_client.py
     health``). It cannot be reached over the published port: an ``internal``
     network gets no DNAT rule, so Docker silently drops the mapping (LOG-107);
   3. namespace persistence (``x = 1`` then ``print(x + 1)`` -> ``2``);
  4. a timeout returns an error without killing the runner;
  5. the workspace mount is writable **and** the run artifacts are not visible
     (``trajectory.jsonl`` / ``answer.md`` must be absent from ``/workspace``);
  6. no ``OPENROUTER_API_KEY`` / ``LENS_DB_PASSWORD`` is present inside the
     container (``/exec`` + ``os.environ``);
  7. the image contains ``runner.py``, ``rlm_api.py`` and ``kg_query.py`` and the
     RLM can ``import kg_query``.

This is an integration script, not part of the no-key unit suite.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Run the checks and return a process exit code."""
    raise NotImplementedError("M8 scaffold: implement verify_sandbox.main")


if __name__ == "__main__":
    raise SystemExit(main())
