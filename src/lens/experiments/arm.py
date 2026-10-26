"""Arm fingerprint: the config knobs that define one experimental condition (M9).

Spec: ``MVP Build Roadmap.md`` M9 item 3 ("experiment-runner scaffolding - arm
config, scoring hooks") and M11 item 1 ("freeze the system", E-04).

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

from typing import Any

from lens import config


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
            "ef_search": config.HNSW_EF_SEARCH,
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
        },
        "models": {role: spec.id for role, spec in config.MODEL_REGISTRY.items()}
        | {"jev": config.JEV_MODEL_ID},
    }


__all__ = ["arm_fingerprint"]
