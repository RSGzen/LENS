"""Central configuration: paths, endpoints, guardrail constants, model registry.

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §2a (model routing / pinning) and
``MVP Build Roadmap.md`` §3 (paths) / §7 (guardrails).

Rules
-----
- **Paths live here** (they are not secrets) so no module scatters hard-coded
  values that are hard to debug.
- **Secrets do NOT live here.** The OpenRouter API key is read from the OS
  environment at call time by :func:`require_api_key`; it is never stored,
  logged, or committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from lens.models import Role

# A gitignored `.env` may mirror the OS env var (MVP Roadmap §3).
load_dotenv()

# --------------------------------------------------------------------- paths
# One source of truth. The env override keeps the same code portable across
# machines; the default is the local runtime root (outside Google Drive).
LENS_RUNS_ROOT_PATH = os.environ.get("LENS_RUNS_ROOT", r"D:\lens-data\runs")
BLOBS_DIR_NAME = "blobs"
TRAJECTORY_FILENAME = "trajectory.jsonl"

# ----------------------------------------------------------------- endpoints
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"

# ---------------------------------------------------------------- guardrails
# R-05: 120 s per execution block, max 3 retries.
DEFAULT_TIMEOUT_S = 120
MAX_RETRIES = 3


# ------------------------------------------------------------- model registry
@dataclass(frozen=True)
class ModelSpec:
    """A frozen description of one model role (Tech Stack §2a).

    Callers ask for a *role* ("root"/"sub"/"dev"); they never pass a raw model
    ID. This is where the exact ID, capability flags, and provider pinning live.
    """

    id: str
    supports_reasoning: bool
    reasoning_effort: str | None
    supports_tools: bool
    supports_response_format: bool
    # OpenRouter provider pinning (recorded runs): allow_fallbacks=False + these.
    provider_only: tuple[str, ...] = ()
    provider_order: tuple[str, ...] = ()
    # Append ":exacto" when tool-calling reliability is required.
    exacto: bool = False


MODEL_REGISTRY: dict[Role, ModelSpec] = {
    "root": ModelSpec(
        id="openai/gpt-5",
        supports_reasoning=True,
        reasoning_effort="medium",
        supports_tools=True,
        supports_response_format=True,
        # TODO(student): fill provider_only / provider_order / exacto from a live
        # OpenRouter provider lookup before the first *recorded* run (LOG-29).
    ),
    "sub": ModelSpec(
        id="openai/gpt-5-mini",
        supports_reasoning=True,
        reasoning_effort="low",
        supports_tools=True,
        supports_response_format=True,
        # TODO(student): provider pinning as above.
    ),
    "dev": ModelSpec(
        id="qwen/qwen3-coder",
        supports_reasoning=False,
        reasoning_effort=None,
        supports_tools=True,
        supports_response_format=False,
        # TODO(student): qwen3-coder is multi-host -> provider pinning is required.
    ),
}


def require_api_key() -> str:
    """Return ``OPENROUTER_API_KEY`` or fail loudly.

    The key is read from the OS environment only. It is never written to a file,
    returned in logs, or included in a trajectory event.
    """

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it in the OS environment (or a "
            "gitignored .env) outside the repo — see MVP Build Roadmap §3."
        )
    return key
