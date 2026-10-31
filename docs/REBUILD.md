# Rebuilding the LENS containers

Run this after changing `src/lens/sandbox/Dockerfile`,
`src/lens/sandbox/requirements.txt` (then regenerate `requirements.lock` first), or
when the base image / host toolchain has moved.

**`lens-pgdata` is a named volume holding the loaded corpus and indexes.**
`docker compose down -v` deletes it and the loads have to run again.
Never pass `-v` here, and never delete the volume's `postgres` directory.

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

## Reproducibility of the image itself

- **Base image pinned by digest** — `FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159`
  (Debian slim, pulled 31 Oct 2026). A tag is republished in place, so the tag alone
  was never an identity: two builds of one commit could compile against different
  base layers. Bumping it is a deliberate edit plus a rebuild plus
  `verify_sandbox.py`.
- **Dependencies installed from a hashed lock** — `src/lens/sandbox/requirements.lock`
  is generated with hashes and installed with `--require-hashes`, so a swapped or
  re-uploaded wheel fails the build instead of being installed.
  `requirements.txt` stays the readable spec. Regenerate after changing it:

  ```powershell
  uv pip compile src\lens\sandbox\requirements.txt --generate-hashes `
      --python-platform x86_64-manylinux_2_28 --python-version 3.12 `
      -o src\lens\sandbox\requirements.lock
  ```

  The platform flag matters: resolving on Windows would lock `win_amd64` wheels and
  the Linux build would fail.

## Not done here (deliberate)

- **The image's `ca-certificates` is not refreshed.** The sandbox has no egress
  (`sandbox-net: internal: true`) and reaches the host over plain HTTP, so it
  makes no TLS call at runtime. Revisit if the sandbox is ever given outbound
  TLS (e.g. web search), then add `apt-get update && apt-get upgrade
  ca-certificates` before the shell is deleted, or rebuild the base.
- **The `requirements.lock` hashes are per-platform.** Regenerating it for another
  architecture (e.g. arm64) needs the matching `--python-platform`, otherwise the
  install fails on a hash mismatch — which is the correct failure, not a silent one.
