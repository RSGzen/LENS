"""M8 live sandbox verification (requires Docker Desktop).

Checks, in dependency order - the first two are what the thesis claims, so they
run before anything convenient. The transport is the real one (``exec_client``
inside the container), so a pass here means the path ``Runtime`` uses works, not
a parallel debug path:

  0. **host RPC reachable from inside the container** - a block reads
     ``$LENS_HOST_RPC_URL/health``. If this fails on ``sandbox-net: internal:
     true``, apply the compose fallback (fixed subnet + bridge-gateway URL), do
     **not** drop ``internal``: no-egress is a claim (Tech Stack 4b).
  1. **general egress blocked** - a connect to ``1.1.1.1:443`` must fail. Record
     the failure: absence of egress is evidence, not an assumption.
  2. build/start ``lens-repl`` **and** ``lens-relay`` via compose; health answers
     from **inside** the container. There is no published port to try: an
     ``internal`` network gets no DNAT rule, so Docker silently drops the mapping
     (LOG-107/114).
  3. namespace persistence (``x = 41`` then ``print(x + 1)`` -> ``42``);
  4. a timeout returns a structured error without killing the runner;
  5. the workspace mount is writable **and** the run artifacts are not visible
     (``trajectory.jsonl`` / ``answer.md`` must be absent from ``/workspace``);
  6. no ``OPENROUTER_API_KEY`` / ``LENS_DB_*`` secret is present inside the
     container;
  7. the image ships ``runner.py`` / ``rlm_api.py`` / ``kg_query.py`` and the RLM
     can ``import kg_query``.

The exec cap is lowered with ``--exec-cap`` (default 5 s) so the timeout check
costs seconds instead of the production 120 s; ``SandboxSession`` forwards the
same number to the runner, which is why the mechanism is what gets verified.

Usage::  python scripts/verify_sandbox.py [--exec-cap 5]

This is an integration script, not part of the no-key unit suite. It spends no
API money: the host RPC server is a stub that answers ``/health`` only.
"""

from __future__ import annotations

import argparse
import tempfile

from pathlib import Path
from typing import Any, Mapping

from lens import config
from lens.agent.sandbox_session import SandboxSession
from lens.agent.tool_server import ToolServer


class _StubRuntime:
    """Minimal ``Runtime`` surface: ``ToolServer`` only needs ``handle_rpc``."""

    def handle_rpc(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Answer every RPC, so only reachability is under test here."""
        return {"ok": True, "result": {"stub": endpoint}}


def _check(name: str, ok: bool, detail: str) -> bool:
    """Print one PASS/FAIL line and return ``ok``."""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main(argv: list[str] | None = None) -> int:
    """Run the checks and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exec-cap", type=float, default=5.0, help="per-block cap in seconds")
    args = parser.parse_args(argv)

    token = "verify-token"
    server = ToolServer(runtime=_StubRuntime(), token=token)
    server.start()

    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "work"
        workspace.mkdir(parents=True, exist_ok=True)

        sandbox = SandboxSession(
            run_id="verify-sandbox",
            workspace=workspace,
            token=token,
            exec_timeout_s=args.exec_cap,
        )
        # The relay's target is only known after start(); the container's URL is
        # the constant relay address, so only the port needs setting.
        sandbox.host_rpc_port = server.port

        try:
            sandbox.start()

            reach = sandbox.exec(
                "import os, urllib.request as u\n"
                "print(u.urlopen(os.environ['LENS_HOST_RPC_URL'] + '/health', timeout=10).status)"
            )
            results.append(
                _check("host RPC from container", reach.ok and "200" in reach.stdout,
                       f"stdout={reach.stdout.strip()!r} error={reach.error}")
            )

            # Absence of egress is evidence: a successful connect is the failure.
            egress = sandbox.exec(
                "import socket\nsocket.create_connection(('1.1.1.1', 443), timeout=5)\n"
                "print('CONNECTED')"
            )
            results.append(
                _check("egress blocked", not egress.ok or "CONNECTED" not in egress.stdout,
                       f"ok={egress.ok} error={(egress.error or {}).get('type')}")
            )

            results.append(_check("/health via exec_client", sandbox.health(), "in-container health"))

            sandbox.exec("x = 41")
            persist = sandbox.exec("print(x + 1)")
            results.append(
                _check("namespace persistence", persist.stdout.strip() == "42",
                       f"stdout={persist.stdout.strip()!r}")
            )

            slow = sandbox.exec(f"import time; time.sleep({args.exec_cap * 4})")
            results.append(
                _check("timeout is structured", not slow.ok and "imeout" in str(slow.error),
                       f"ok={slow.ok} error={(slow.error or {}).get('type')}")
            )

            write = sandbox.exec(
                "from pathlib import Path\n"
                "Path('/workspace/probe.txt').write_text('ok')\n"
                "hidden = {'trajectory.jsonl', 'answer.md', 'tasks.json', 'manifest.json'}\n"
                "print(sorted(p.name for p in Path('/workspace').iterdir() if p.name in hidden))"
            )
            results.append(
                _check(
                    "workspace writable, artifacts hidden",
                    write.ok and (workspace / "probe.txt").is_file() and write.stdout.strip() == "[]",
                    f"leaked={write.stdout.strip()} error={write.error}",
                )
            )

            secrets = sandbox.exec(
                "import os\n"
                "bad = [k for k in ('OPENROUTER_API_KEY', 'LENS_DB_PASSWORD', "
                "'LENS_DB_RO_PASSWORD') if k in os.environ]\n"
                "print(bad)"
            )
            results.append(
                _check("no secrets in the container", secrets.stdout.strip() == "[]",
                       f"stdout={secrets.stdout.strip()!r}")
            )

            modules = sandbox.exec(
                "import kg_query, rlm_api, runner\n"
                "print(sorted(__import__('os').listdir('/app')))"
            )
            results.append(
                _check("image modules", modules.ok and "kg_query" in modules.stdout,
                       f"stdout={modules.stdout.strip()[:140]} error={modules.error}")
            )
        finally:
            sandbox.stop()
            server.stop()

    print("")
    passed = sum(1 for result in results if result)
    print(f"{passed}/{len(results)} checks passed")
    print("image    " + config.SANDBOX_IMAGE)

    return 0 if passed == len(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
