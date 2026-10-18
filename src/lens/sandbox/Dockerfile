# lens-repl: passive Python executor for the LENS RLM (M8).
#
# Security posture (LOG-32): non-root, no API key, no DB driver, no runtime
# installs, read-only root filesystem (compose), workspace-only RW mount.
# TODO(E-04): after the first successful build, pin the base image by digest
# (python:3.12-slim@sha256:...) and record it in the thesis log.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY runner.py rlm_api.py ./

RUN useradd --create-home --uid 10001 lens \
    && mkdir -p /workspace \
    && chown -R lens:lens /app /workspace

# Best-effort shell removal: the RLM writes Python, not shell commands. Keep this
# as the last shell-using layer — subsequent RUN instructions would need /bin/sh.
RUN rm -f /bin/sh /bin/bash /usr/bin/bash

USER lens

EXPOSE 8080

CMD ["python", "runner.py"]
