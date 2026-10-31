# Rebuilding the LENS containers

Run this after changing `src/lens/sandbox/Dockerfile` or
`src/lens/sandbox/requirements.txt`, or when the base image / host toolchain has
moved.

**`lens-pgdata` is a named volume holding the loaded corpus and indexes.**
`docker compose down -v` deletes it and the / loads have to run again.
Never pass `-v` here.

## 1. Preconditions

```powershell
wsl --update                        # Docker Desktop 4.93 needs a current WSL VM
docker version --format "{{.Server.Version}}"
cd D:\lens; docker compose config -q   # no output = OK
```

## 2. Rebuild the sandbox image

```powershell
cd D:\lens
docker compose build lens-repl
```

The Dockerfile updates pip to 26.2.2 from a hash-pinned wheel **before** it
installs anything: pip < 26.2.2 trusts its vendored certifi regardless of the
system store, so it would keep trusting the revoked eMudhra intermediates
(CVE-2026-111943) on every download. The hash in the URL is what makes that
first fetch safe.

## 3. Bring the DB up (data preserved)

```powershell
docker compose up -d lens-pg
docker inspect --format "{{.State.Health.Status}}" lens-pg   # healthy
```

The `volume "lens-pgdata" already exists but was not created by Docker Compose`
warning is expected — the volume predates compose. Data is intact.

## 4. Verify the image

The image has no shell and no `curl`, so probe it with `--entrypoint python`.

```powershell
docker run --rm --entrypoint python lens-repl:0.1.0 -c "import sys, pip; print(sys.version.split()[0], pip.__version__)"
# expect: 3.12.x 26.2.2

docker run --rm --entrypoint python lens-repl:0.1.0 -c "import httpx, numpy, pandas, networkx; print('deps ok')"
```

## 5. Record provenance

```powershell
docker inspect --format "{{index .RepoDigests 0}}" lens-repl:0.1.0
uv --version
.\.venv\Scripts\python.exe -c "import certifi; print(certifi.__version__)"
```

Image digest, uv version and certifi version are run inputs, not trivia: a
rebuilt image with a different pip changes the artifact set. Put them in the run
manifest (`manifest.json`) so a recorded run can be traced to an exact image.

The same three numbers are printable in one command, which is what a fresh machine
needs before anything else is trusted:

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py --need trust
```

It reports the interpreter's `certifi`, whether pip's **vendored** copy is present,
the Windows root store and the msys64 bundle — the per-layer split that Windows
Update does not keep aligned.

## 6. Sandbox contract check

```powershell
.\.venv\Scripts\python.exe scripts\verify_sandbox.py
```

Checks host-RPC reachability, that general egress is **blocked**, namespace
persistence, the workspace mount, and that no API key or DB password is
visible inside the container. Re-run after every rebuild.

## Not done here (deliberate)

- **Base image is not pinned by digest yet** — `FROM python:3.12-slim` still
  floats. TODO owns it: pin `python:3.12-slim@sha256:...` and record the
  digest in the log.
- **The image's `ca-certificates` is not refreshed.** The sandbox has no egress
  (`sandbox-net: internal: true`) and reaches the host over plain HTTP, so it
  makes no TLS call at runtime. Revisit if the sandbox is ever given outbound
  TLS (e.g. web search), then add `apt-get update && apt-get upgrade
  ca-certificates` before the shell is deleted, or rebuild the base.
- **`requirements.txt` is not hash-locked** (`--require-hashes`). The pip pin
  closes the CA-trust gap; a lock file would close the supply-chain gap.
