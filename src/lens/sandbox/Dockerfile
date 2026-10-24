# lens-repl: passive Python executor for the LENS RLM (M8).
#
# Security posture (LOG-32): non-root, no API key, no DB driver, no runtime
# installs, read-only root filesystem (compose), workspace-only RW mount.
#
# kg_query.py is the pure KG-traversal module from ``src/lens/kg_query.py``
# (stdlib + networkx only) — copied here so the RLM imports the same code the
# host re-exports (K-07). It is the only project file in the image besides the
# runner and the façade.
# TODO(E-04): after the first successful build, pin the base image by digest
# (python:3.12-slim@sha256:...) and record it in the thesis log.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Update pip before it fetches anything: pip < 26.2.2 trusts its vendored certifi
# whatever the system store says, so it keeps trusting the revoked eMudhra
# intermediates (CVE-2026-111943). Hash-pinned direct URL, so this one download
# needs no trust in the CA chain at all.
RUN python -m pip install --no-cache-dir --no-deps \
    "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"

COPY lens/sandbox/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY lens/sandbox/runner.py lens/sandbox/rlm_api.py lens/sandbox/exec_client.py lens/kg_query.py ./

RUN useradd --create-home --uid 10001 lens \
    && mkdir -p /workspace \
    && chown -R lens:lens /app /workspace

# Best-effort shell removal: the RLM writes Python, not shell commands. Keep this
# as the last shell-using layer — subsequent RUN instructions would need /bin/sh.
RUN rm -f /bin/sh /bin/bash /usr/bin/bash

USER lens

EXPOSE 8080

CMD ["python", "runner.py"]
