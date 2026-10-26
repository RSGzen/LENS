"""Single-destination TCP relay: the only way out of the no-egress sandbox.

Why this exists
---------------
The sandbox network is ``internal: true`` - the sandbox must have **no route off
its bridge**. But the RLM's code calls back to the host
RPC server on every ``tool()`` / ``sub_call()`` / ``emit()``, so *some* path from
the container to the host is required.

Measured on Docker Desktop / Windows : there is no such path
from an internal network. ``extra_hosts: host.docker.internal:host-gateway`` maps
to the WSL2 VM's own gateway (172.x.0.1) and is ``ENETUNREACH`` from the bridge;
without that override the internal network has no DNS at all (``gaierror``); the
host's real address (192.168.65.254) is off-bridge and therefore blocked. On a
normal bridge the same probe returns ``200`` - so the transport is fine and the
``internal`` flag is exactly what blocks it.

Dropping ``internal`` would open general egress and weaken the claim, so instead
this relay is the sandbox's only peer: it listens on the internal network and
forwards to **one** hard-coded host:port. A block cannot use it as a general
proxy - the destination is not part of the protocol, and nothing here resolves or
accepts a target from the client.

Topology (docker-compose.yaml)::

    lens-repl --(sandbox-net, internal)--> lens-relay --(rpc-net)--> host RPC
        no egress                            forwards one port

The relay never sees the RPC token (it forwards bytes) and holds no key.

Env: ``LENS_RELAY_LISTEN_PORT`` (default 8090), ``LENS_RELAY_TARGET_HOST``
(default ``host.docker.internal``, resolved by Docker Desktop's DNS on a normal
network), ``LENS_RELAY_TARGET_PORT`` (the host RPC server's port).
"""

from __future__ import annotations

import os
import select
import socket
import socketserver
import sys

LISTEN_PORT = int(os.environ.get("LENS_RELAY_LISTEN_PORT", "8090"))
TARGET_HOST = os.environ.get("LENS_RELAY_TARGET_HOST", "host.docker.internal")
TARGET_PORT = int(os.environ.get("LENS_RELAY_TARGET_PORT", "8765"))

# One chunk at a time, both directions; long enough for a slow cross-encoder tool
# call, short enough that a dead peer cannot pin a relay thread for ever.
IDLE_TIMEOUT_S = float(os.environ.get("LENS_RELAY_IDLE_TIMEOUT_S", "300"))


class _Handler(socketserver.BaseRequestHandler):
    """Tunnel one client connection to the fixed target and back."""

    def handle(self) -> None:
        try:
            upstream = socket.create_connection((TARGET_HOST, TARGET_PORT), timeout=10)
        except OSError as exc:
            print(f"relay: target unreachable ({exc})", file=sys.stderr, flush=True)
            return

        peers = {self.request: upstream, upstream: self.request}

        try:
            while True:
                readable, _, _ = select.select(list(peers), [], [], IDLE_TIMEOUT_S)

                if not readable:
                    return  # idle: drop, never wedge a thread

                for source in readable:
                    data = source.recv(65536)

                    if not data:
                        return  # one side closed; close both

                    peers[source].sendall(data)
        except OSError:
            pass
        finally:
            upstream.close()


class _Server(socketserver.ThreadingTCPServer):
    """Threaded so one long tool call cannot block the next RPC."""

    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    """Serve until killed; one JSON-free line on stdout per start."""
    server = _Server(("0.0.0.0", LISTEN_PORT), _Handler)
    print(
        f"relay: listening on 0.0.0.0:{LISTEN_PORT} -> {TARGET_HOST}:{TARGET_PORT}",
        flush=True,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
