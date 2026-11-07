"""Campaign matrix: validate an arm file and expand it into launch jobs.

Pure functions only - no subprocess, no Docker, no files written - so the matrix
can be validated for free (``--dry-run``) and tested without a key. The runner in
``scripts/run_campaign.py`` is the thin I/O shell around this module.

Contract tests: ``tests/test_campaign.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Name chunks become directory names and manifest run ids.
_SAFE = set("abcdefghijklmnopqrstuvwxyz0123456789-_.")


def _env(raw: Any, label: str) -> dict[str, str]:
    """Return one env block as strings; raise when it is not an object.

    ``docker compose`` refuses a non-string value, so a campaign that carries an
    int would fail at launch rather than at validation. One place converts and one
    place checks, so an arm and a query cannot disagree about the shape.
    """
    if raw is None:
        return {}

    if not isinstance(raw, dict):
        raise ValueError(f"{label} env must be an object")

    return {str(key): str(value) for key, value in raw.items()}


def load_campaign(path: str | Path) -> dict[str, Any]:
    """Load and validate a campaign file; raise ``ValueError`` on a bad shape.

    Validation is strict on purpose: a duplicated query id or arm name would
    collapse two cells into one run id (one directory, one manifest), and the
    resulting CSV would look complete while testing nothing.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("campaign must be a JSON object")

    for key in ("name", "arms", "queries"):
        if key not in payload:
            raise ValueError(f"campaign is missing {key!r}")

    if not payload["arms"] or not payload["queries"]:
        raise ValueError("campaign needs at least one arm and one query")

    if not set(payload["name"]) <= _SAFE:
        raise ValueError(f"bad campaign name {payload['name']!r} (use a-z 0-9 - _ .)")

    for field, items, key in (
        ("arm", payload["arms"], "name"),
        ("query", payload["queries"], "id"),
    ):
        seen: set[str] = set()

        for item in items:
            ident = item.get(key) if isinstance(item, dict) else None

            if not ident or not set(ident) <= _SAFE:
                raise ValueError(f"bad {field} id {ident!r} (use a-z 0-9 - _ .)")

            if ident in seen:
                raise ValueError(f"duplicate {field} id {ident!r}")

            seen.add(ident)

            if isinstance(item, dict):
                _env(item.get("env"), f"{field} {ident!r}")

    return payload


def build_jobs(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one job per (arm, query, repeat) - the launch matrix.

    ``repeats`` defaults to 1: n >= 3 belongs to the arms, not to a smoke that
    only proves the plumbing. Run ids are deterministic, so re-running a campaign
    fails loudly on an existing directory instead of silently mixing two runs.
    """
    repeats = int(campaign.get("repeats") or 1)
    jobs: list[dict[str, Any]] = []

    for arm in campaign["arms"]:
        arm_env = _env(arm.get("env"), f"arm {arm['name']!r}")

        for query in campaign["queries"]:
            # A query may carry its own env: one document per instance cannot be
            # expressed by the arm, which is shared by every job in the campaign.
            env = {**arm_env, **_env(query.get("env"), f"query {query['id']!r}")}

            for repeat in range(repeats):
                suffix = f".r{repeat + 1}" if repeats > 1 else ""
                jobs.append(
                    {
                        "run_id": (f"{campaign['name']}.{arm['name']}.{query['id']}{suffix}"),
                        "arm": arm["name"],
                        "query_id": query["id"],
                        "query": query["text"],
                        "expect": query.get("expect"),
                        "env": dict(env),
                    }
                )

    return jobs


__all__ = ["build_jobs", "load_campaign"]
