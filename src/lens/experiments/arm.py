"""Arm fingerprint: the config knobs that define one experimental condition.

Why a *fingerprint* and not an ``apply_arm()``
---------------------------------------------
``lens.config`` is import-time constants: caps, the retrieval funnel, the KG and
gate flags are read once when the process starts, and ``SandboxSession`` forwards
some of them into the container's environment. An in-process ``apply_arm()``
would therefore change the Python side but leave the *container* and the
``Budget`` object on the old values - the classic stale-config bug, invisible
until two arms produce identical numbers.

So **an arm is a process environment**: ``scripts/run_campaign.py`` spawns one
``python -m lens.run`` per (arm, query) with that arm's variables, and this module
records what was in force, so a run can be attributed months later. Reading the
values here (at call time, from ``config``) means the fingerprint cannot drift
from the run it describes.

Contract tests: ``tests/test_arm.py`` (no key, no DB, no Docker).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from lens import config
from lens.tools.retrieval import hnsw_ef_for


def _document_identity() -> dict[str, Any] | None:
    """Return the evidence file's path and content hash, or ``None``.

    A boolean cannot tell two document instances apart in the manifest, and the
    file *is* the evidence: the hash is what makes the instance reproducible.
    """
    if not config.DOCUMENTS_PATH:
        return None

    source = Path(config.DOCUMENTS_PATH).expanduser()

    try:
        digest: str | None = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError:
        digest = None

    return {"path": str(source), "sha256": digest}


def arm_fingerprint(*, recorded: bool, replay: bool) -> dict[str, Any]:
    """Return the run's arm as plain JSON (recorded in ``manifest.json``).

    ``recorded`` and ``replay`` are passed in because they are runtime arguments,
    not config constants. Read the config here (not at import) so a test can patch
    it, exactly like ``runstore.runs_root()``.
    """
    return {
        "retrieval": {
            "k": config.RETRIEVAL_K,
            "n": config.RETRIEVAL_N,
            "C": config.RETRIEVAL_C,
            "c": config.RETRIEVAL_C_TOP,
            # The query path uses ``hnsw_ef_for(k)``, not the base constant: a
            # k swept above the 100-default raises ef_search with it, so recording
            # the constant would describe a setting the run never used.
            "ef_search": hnsw_ef_for(config.RETRIEVAL_K),
            "reranker": config.RERANKER_MODEL_ID,
        },
        "kg": {
            "enabled": config.KG_ENABLED,
            "postprocess": config.KG_POSTPROCESS_ENABLED,
            "verify": config.KG_VERIFY_ENABLED,
            "verify_budget": config.KG_VERIFY_BUDGET,
            "role_filter": config.KG_ROLE_FILTER,
            "ontology_version": config.KG_ONTOLOGY_VERSION,
            "min_score": config.KG_MIN_SCORE,
        },
        "gate": {
            "enabled": config.GATE_ENABLED,
            "thresholds": dict(config.STAGE_THRESHOLDS),
        },
        "mode": {
            "recorded": recorded,
            "replay": replay,
            "allow_sql": config.ALLOW_SQL,
            "resume": config.RESUME,
            # The arm *is* the environment: a run that changed its harness must
            # record which harness it changed to, or two arms look identical.
            "run": config.RUN_MODE,
            "evidence": config.EVIDENCE_SOURCE,
            "treatment": config.TREATMENT,
            "prompt": config.PROMPT_VARIANT,
            "sub_calls": config.SUB_CALLS_ENABLED,
            "segmentation": config.SEGMENTATION,
            "raw_query": config.RAW_QUERY,
            "documents": _document_identity(),
            "document_evidence": config.DOCUMENT_EVIDENCE,
        },
        # The prompt-audit switches. A variant that changes one of these must be
        # distinguishable from the baseline in the manifest, not only in the operator's
        # memory.
        "prompt": {
            "variant": config.PROMPT_VARIANT,
            "context_files": config.CONTEXT_FILES,
            "kg": config.KG_PROMPT,
            "extract_hint": config.EXTRACT_HINT,
            "retrieval_rule": config.RETRIEVAL_RULE,
        },
        "models": {role: spec.id for role, spec in config.MODEL_REGISTRY.items()}
        | {"jev": config.JEV_MODEL_ID},
        # The reasoning object is part of the arm: a sub role at low and one at high
        # are different environments even when the model id is identical.
        "reasoning": {
            role: dict(spec.reasoning or {}) for role, spec in config.MODEL_REGISTRY.items()
        },
        # Concurrency is part of the arm: a run at three workers and one at eight
        # are different schedules, and the wall-clock comparison must say which.
        "workers": {
            "sub_tasks": config.SUB_TASK_WORKERS,
            "claim_extract": config.CLAIM_WORKERS,
        },
    }


__all__ = ["arm_fingerprint"]
