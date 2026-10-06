"""
Central configuration: paths, endpoints, guardrail constants, model registry.

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

load_dotenv()

# --------------------------------------------------------------------- paths
# One source of truth. The env override keeps the same code portable across
# machines; the default is the local runtime root (outside Google Drive).
LENS_RUNS_ROOT_PATH = os.environ.get("LENS_RUNS_ROOT", r"D:\lens-data\runs")
BLOBS_DIR_NAME = "blobs"
TRAJECTORY_FILENAME = "trajectory.jsonl"

# --------------------------------------------------------------- dataset (M1)
# Source corpus locations and manifest-build knobs. Same env-override pattern as the runs root.

TARGET_CATEGORY = "cs.AI"

MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_FILENAME = "manifest.checkpoint.json"

# Filter on the arXiv id prefix (YYMM), NOT update_date: the first two digits are
# the submission year. PAPER_YEAR_RANGE is the set of valid two-digit prefixes, so
# old-style ids ("acc-phys/9411001") are excluded with no int() cast (no crash).
# START/END are ints, used only for the manifest's provenance ``year_range``.
PAPER_YEAR_RANGE = {"15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"}
PAPER_START_YEAR = 2015
PAPER_END_YEAR = 2026

LENS_MANIFESTS_ROOT_PATH = os.environ.get("LENS_MANIFESTS_ROOT", r"D:\lens-data\manifests")
ARXIV_METADATA_JSON_PATH = os.environ.get(
    "LENS_ARXIV_METADATA",
    r"D:\Thesis Dataset\Dataset\kaggle_arxiv-metadata-oai-snapshot.json",
)
GROBID_XML_DIR_PATH = os.environ.get("LENS_GROBID_XML_DIR", r"D:\Thesis Dataset\cs_AI_Extracted_XML")
# Source lines between checkpoint writes during the scan.
MANIFEST_CHECKPOINT_INTERVAL = 3000

# -------------------------------------------------------------- database (M2)
# Local pgvector container (MVP Roadmap §2.1/§3). These are DEV defaults only;
# override any of them via the environment. A real password is never committed
# (AGENTS §3) — the defaults match the `docker run` flags documented in
# `db/schema.py`.
LENS_DB_HOST = os.environ.get("LENS_DB_HOST", "localhost")
LENS_DB_PORT = int(os.environ.get("LENS_DB_PORT", "5432"))
LENS_DB_NAME = os.environ.get("LENS_DB_NAME", "lens")
LENS_DB_USER = os.environ.get("LENS_DB_USER", "lens")
LENS_DB_PASSWORD = os.environ.get("LENS_DB_PASSWORD")

# Read-only role the host tool layer uses on the RLM's behalf: SELECT only,
# never write. The sandbox itself holds no DB credentials (LOG-32).
LENS_DB_RO_USER = os.environ.get("LENS_DB_RO_USER", "lens_ro")
LENS_DB_RO_PASSWORD = os.environ.get("LENS_DB_RO_PASSWORD")

# Container / image / volume used by `scripts/setup_db.py` docs and M2 evidence.
LENS_DB_CONTAINER_NAME = os.environ.get("LENS_DB_CONTAINER", "lens-pg")
LENS_DB_IMAGE = "pgvector/pgvector:pg16"
LENS_DB_VOLUME_PATH = os.environ.get("LENS_DB_VOLUME", r"D:\lens-data\postgres")

# Embedding width + pgvector storage type: nomic-embed-text-v1.5, Matryoshka
# 768 -> 512, stored as `halfvec` (fp16). Note 2*512+8 == 4*256+8 bytes/row —
# the same storage as the retired fp32 vector(256) but with 512 dims; `halfvec`
# is native + HNSW-indexable with no quantization calibration (Tech Stack §3).
EMBEDDING_DIM = int(os.environ.get("LENS_EMBEDDING_DIM", "512"))
# pgvector column type + HNSW op class must match what the embedder produces.
EMBEDDING_PGVECTOR_TYPE = os.environ.get("LENS_EMBEDDING_TYPE", "halfvec")
EMBEDDING_INDEX_OPS = os.environ.get("LENS_EMBEDDING_INDEX_OPS", "halfvec_cosine_ops")

# ------------------------------------------------- papers + embeddings (M3)
# No paper subset: the full manifest is loaded (advisor decision, 6 Oct 2026).

# nomic-embed-text-v1.5, Matryoshka-truncated 768 -> 512, stored halfvec
# (Tech Stack §2 stage 7). EMBEDDING_VERSION is copied into every row so a
# re-index is detectable (T-08); bump it on any model/width/precision/prefix change.
EMBEDDING_MODEL_ID = os.environ.get(
    "LENS_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"
)
EMBEDDING_VERSION = os.environ.get("LENS_EMBEDDING_VERSION", "nomic-embed-text-v1.5@512-halfvec")
# nomic is prefix-conditioned: documents and queries use different task tokens.
# A missing prefix does not error — it silently degrades matching, so it lives here.
EMBEDDING_DOC_PREFIX = "search_document: "
EMBEDDING_QUERY_PREFIX = "search_query: "

# ----------------------------------------------------------------- endpoints
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
# Decision layer is single-provider and NOT part of MODEL_REGISTRY (LOG-29).
JEV_MODEL_ID = "typesafe/jev-1.13"

# ---------------------------------------------------------------- guardrails
# R-05: 120 s per execution block, max 3 retries.
DEFAULT_TIMEOUT_S = 120
MAX_RETRIES = 3

# ------------------------------------------------------------- model registry
@dataclass(frozen=True)
class ModelSpec:
    """A frozen description of one model role.

    Callers ask for a *role* ("root"/"sub"/"dev"); they never pass a raw model ID.
    This is where the exact id, capability flags, sampling controls, and provider
    pinning live (Tech Stack §2a).
    """

    id: str
    supports_reasoning: bool
    reasoning_effort: str | None
    supports_tools: bool
    supports_response_format: bool
    # OpenRouter provider pinning (recorded runs): allow_fallbacks=False + these.
    provider_only: tuple[str, ...] = ()
    provider_order: tuple[str, ...] = ()
    allow_fallbacks: bool = False
    # Append ":exacto" when tool-calling reliability is required.
    exacto: bool = False
    # Sampling controls. None = omit the key (provider default applies). Only
    # meaningful for non-reasoning models; GPT-5 reasoning roles use reasoning_effort.
    temperature: float | None = None
    # Determinism is NOT guaranteed on all models/providers (OpenRouter docs).
    seed: int | None = None


MODEL_REGISTRY: dict[Role, ModelSpec] = {
    "root": ModelSpec(
        id="openai/gpt-5",
        supports_reasoning=True,
        reasoning_effort="medium",
        supports_tools=True,
        supports_response_format=True,
        provider_only=("openai",),
        provider_order=("openai",),
    ),
    "sub": ModelSpec(
        id="openai/gpt-5-mini",
        supports_reasoning=True,
        reasoning_effort="low",
        supports_tools=True,
        supports_response_format=True,
        provider_only=("openai",),
        provider_order=("openai",),
    ),
    "dev": ModelSpec(
        id="qwen/qwen3-coder",
        supports_reasoning=False,
        reasoning_effort=None,
        supports_tools=True,
        supports_response_format=False,
        provider_only=("google-vertex",),
        provider_order=("google-vertex",),
        temperature=0.1,
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
