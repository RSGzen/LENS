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

# Quiet third-party chatter so run logs stay readable: Hugging Face Hub /
# transformers / tqdm progress bars and one-off warnings would otherwise be
# mirrored at ERROR level by `observability/run_logging.py`. setdefault -> the
# caller can still override any of these in the environment.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

# ------------------------------------------------------------------ paths
# One source of truth. The env override keeps the same code portable across
# machines; the default is the local runtime root (outside Google Drive).
LENS_RUNS_ROOT_PATH = os.environ.get("LENS_RUNS_ROOT", r"D:\lens-data\runs")
BLOBS_DIR_NAME = "blobs"
TRAJECTORY_FILENAME = "trajectory.jsonl"
# Long-running dev-run logs (e.g. the M3 bulk load) kept for thesis evidence.
LENS_LOGS_ROOT_PATH = os.environ.get("LENS_LOGS_ROOT", r"D:\lens-data\logs")

# ------------------------------------------------------------------ dataset (M1)
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

# ------------------------------------------------------------------ database (M2)
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
# Abstracts embedded per model forward pass during the M3 load.
EMBEDDING_ABSTRACT_BATCH_SIZE = 64

# pgvector column type + HNSW op class must match what the embedder produces.
EMBEDDING_PGVECTOR_TYPE = os.environ.get("LENS_EMBEDDING_TYPE", "halfvec")
EMBEDDING_INDEX_OPS = os.environ.get("LENS_EMBEDDING_INDEX_OPS", "halfvec_cosine_ops")

# ------------------------------------------------------------------ papers + embeddings (M3)
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

# Number of chunks (entries) to extract, encode, re-format and insert into the DB table
PROCESS_CHUNK_SIZE = 3500

# ------------------------------------------------------------------ chunks (M4)
# Full-corpus structural chunking (Roadmap §4 M4; Tech Stack §2 stages 5-6).
# Threshold is the frozen C-02/N-04 value: 512 nomic (BERT WordPiece) tokens.
# The splitter (a stdlib paragraph-preserving packer, C-01 amendment) keeps a
# whole GROBID section as one chunk when it is under the threshold, and only
# splits sections that exceed it.
CHUNK_TOKEN_THRESHOLD = int(os.environ.get("LENS_CHUNK_TOKEN_THRESHOLD", "512"))
# Overlap between consecutive chunks within a section (0 = structural, no overlap).
CHUNK_OVERLAP_TOKENS = int(os.environ.get("LENS_CHUNK_OVERLAP_TOKENS", "0"))
# Chunks per model forward pass during the M4 load (GPU batch; separate from the
# paper batch below because one paper yields many chunks).
EMBEDDING_CHUNK_BATCH_SIZE = int(os.environ.get("LENS_EMBEDDING_CHUNK_BATCH", "120"))
# Papers parsed+chunked+embedded per DB batch. ~500 papers ~= 11k chunks ~= 25 MB
# in flight, ~334 batches over the corpus — bounds memory and gives the runner a
# per-batch checkpoint/resume granularity for a multi-day load.
PAPERS_PER_BATCH = int(os.environ.get("LENS_PAPERS_PER_BATCH", "240"))
# Optional dev/test cap: process only the first N manifest papers (0 = all).
# Used for the timed smoke run before the full-corpus load (e.g. 100).
PAPERS_LIMIT = int(os.environ.get("LENS_PAPERS_LIMIT", "0"))
# Resume a partially-loaded run: skip manifest papers whose ``arxiv_id`` is already
# in ``chunks`` (batches commit per batch, so completed papers are durable). The
# load is idempotent either way, but this avoids re-embedding finished papers.
# Set LENS_RESUME=0 to force a full pass.
RESUME = os.environ.get("LENS_RESUME", "1").strip().lower() not in {"0", "false", "no", ""}

# ------------------------------------------------------------------ retrieval (M5)
# The frozen R-01 funnel (Tech Stack §2 stages 10-11, §3, §4i):
#   shallow_match k candidates -> Stage-A rerank -> n papers
#   deep_match    C chunk candidates (filtered) -> Stage-B rerank -> c chunks
# `k`/`C` are the candidate-recall knobs; the cross-encoder cost is linear in
# them (one pass per candidate). These are the guardrail caps (Roadmap §7).
RETRIEVAL_K = int(os.environ.get("LENS_RETRIEVAL_K", "100"))         # Stage A candidates
RETRIEVAL_N = int(os.environ.get("LENS_RETRIEVAL_N", "20"))          # papers after Stage A
RETRIEVAL_C = int(os.environ.get("LENS_RETRIEVAL_C", "100"))         # Stage B candidates
RETRIEVAL_C_TOP = int(os.environ.get("LENS_RETRIEVAL_C_TOP", "100")) # chunks after Stage B
# The KG scope is the deep-match candidate set (C); the answer context is top-c.
# Keep C >= C_TOP so the reranked context is a subset of the KG's chunk set.
if RETRIEVAL_C < RETRIEVAL_C_TOP:
    raise ValueError("RETRIEVAL_C must be >= RETRIEVAL_C_TOP")

# HNSW query-time recall/latency knob. `m` / `ef_construction` are build-time
# (see scripts/build_indexes.py); `ef_search` is set per query. Record all three
# per run (E-04).
HNSW_EF_SEARCH = int(os.environ.get("LENS_HNSW_EF_SEARCH", "40"))

# Cross-encoder reranker, used at BOTH stages (Stage A = papers, Stage B = chunks;
# Tech Stack §7 "both levels").
RERANKER_MODEL_ID = os.environ.get("LENS_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_BATCH_SIZE = int(os.environ.get("LENS_RERANK_BATCH", "32"))

# ------------------------------------------------------------------ knowledge graph (M7)
# Deterministic query-time NER/RE over the **deep-matched chunks** (D-06/A-02;
# Tech Stack §2 stage 12). NER (GLiNER) + RE (GLiREL) run **host-side** (torch/GPU);
# the resulting graph is handed to the RLM as a networkx namespace variable (K-07),
# never dumped into root context. The subgraph is cached by (paper set + ontology
# version + extractor identity) — K-06.
LENS_CACHE_ROOT_PATH = os.environ.get("LENS_CACHE_ROOT", r"D:\lens-data\cache")
KG_CACHE_DIR_NAME = "kg"
# Bump on ANY change to the entity/relation schema: it is the KG re-extraction key
# (mirrors EMBEDDING_VERSION / T-08). Also the cache-key component (K-06).
KG_ONTOLOGY_VERSION = os.environ.get("LENS_KG_ONTOLOGY_VERSION", "lens-ontology-1")
# Both extractor families are installed for the M7 K-05 recall probe; the losing
# family is removed once the probe decides (Roadmap §4 M7).
KG_NER_MODEL_ID = os.environ.get("LENS_KG_NER_MODEL", "urchade/gliner_medium-v2.1")
KG_REL_MODEL_ID = os.environ.get("LENS_KG_REL_MODEL", "jackboyla/glirel-large-v0")
# Minimum span / relation confidence kept. Kept in one place so the probe can sweep it.
KG_MIN_SCORE = float(os.environ.get("LENS_KG_MIN_SCORE", "0.5"))

# Extraction window. GLiNER truncates its input (384 tokens on the medium
# checkpoint) while chunks are <= 512 nomic tokens, so extraction is windowed to
# avoid dropping long-chunk tails. The overlap is carried into the next window.
KG_EXTRACT_MAX_TOKENS = int(os.environ.get("LENS_KG_EXTRACT_MAX_TOKENS", "384"))
KG_EXTRACT_WINDOW_OVERLAP = int(os.environ.get("LENS_KG_EXTRACT_OVERLAP", "32"))

# ------------------------------------------------------------------ KG post-process (M7.1)
# Modular post-process applied **before** filter_ontology: relation-slot type
# inference, a corpus-built gazetteer, and directionality resolution. Types are
# soft (``type`` + ``type_source`` + ``type_confidence``). Every stage is
# toggleable/removable; see LOG-68 / LOG-69.
KG_POSTPROCESS_ENABLED = os.environ.get("LENS_KG_POSTPROCESS", "1").strip() != "0"
KG_TYPE_INFERENCE = os.environ.get("LENS_KG_TYPE_INFERENCE", "1").strip() != "0"
KG_GAZETTEER_ENABLED = os.environ.get("LENS_KG_GAZETTEER", "1").strip() != "0"
KG_GAZETTEER_PATH = os.environ.get(
    "LENS_KG_GAZETTEER_PATH", r"D:\lens-data\cache\kg\gazetteer_corpus.json"
)
# Only trust gazetteer entries above this confidence (count-based: 0.7=<5,
# 0.8=5-9, 0.9>=10). Prevents low-count false positives (e.g. "spaCy").
KG_GAZETTEER_MIN_CONFIDENCE = float(
    os.environ.get("LENS_KG_GAZETTEER_MIN_CONF", "0.9")
)
KG_DIRECTIONALITY_ENABLED = os.environ.get("LENS_KG_DIRECTIONALITY", "1").strip() != "0"
KG_CYCLE_BREAK = os.environ.get("LENS_KG_CYCLE_BREAK", "1").strip() != "0"

# ------------------------------------------------------------------ KG quality (M7.3 / Phase 1.5)
# The KG is an ablation arm: KG_ENABLED=False removes it from the RLM path.
KG_ENABLED = os.environ.get("LENS_KG_ENABLED", "1").strip() != "0"
KG_SPAN_FILTER = os.environ.get("LENS_KG_SPAN_FILTER", "1").strip() != "0"
KG_COREF = os.environ.get("LENS_KG_COREF", "1").strip() != "0"
KG_VERIFY_ENABLED = os.environ.get("LENS_KG_VERIFY", "1").strip() != "0"
# Budgeted JEV direction verification (ambiguous asymmetric triples only).
KG_VERIFY_BUDGET = int(os.environ.get("LENS_KG_VERIFY_BUDGET", "40"))
KG_VERIFY_MARGIN = float(os.environ.get("LENS_KG_VERIFY_MARGIN", "0.05"))
KG_VERIFY_HIGH = float(os.environ.get("LENS_KG_VERIFY_HIGH", "0.7"))
KG_VERIFY_LOW = float(os.environ.get("LENS_KG_VERIFY_LOW", "0.4"))
KG_VERIFY_MAX_STATE_CHARS = int(os.environ.get("LENS_KG_VERIFY_MAX_STATE", "1200"))
# Suffix type hints ("X dataset" -> Dataset, "ROUGE score" -> Metric) applied
# before relation-slot typing and protected like gazetteer hints.
KG_SUFFIX_HINTS = os.environ.get("LENS_KG_SUFFIX_HINTS", "1").strip() != "0"
# Benchmark/task role filter (after relation typing, before directionality):
# drop benchmark/task heads of method-subject relations, keep benchmark heads of
# ``measures`` and benchmark tails of ``evaluated_on``/``uses``/``explores``.
KG_ROLE_FILTER = os.environ.get("LENS_KG_ROLE_FILTER", "1").strip() != "0"

# Maximum number of entities ranked by degree (in + out) in the KG index summary.
# A context-budget knob, not a scientific value (the summary is what the RLM sees).
KG_SUMMARY_TOP_ENTITIES = int(os.environ.get("LENS_KG_SUMMARY_TOP", "20"))

# ------------------------------------------------------------------ endpoints
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
# Decision layer is single-provider and NOT part of MODEL_REGISTRY (LOG-29).
JEV_MODEL_ID = "typesafe/jev-1.13"

# ------------------------------------------------------------------ guardrails
# R-05: 120 s per execution block, max 3 retries.
DEFAULT_TIMEOUT_S = 120
MAX_RETRIES = 3

# ------------------------------------------------------------------ harness (M8)
# Central RLM caps (Tech Stack §4a / §6, R-05). The orchestrator enforces these;
# the sandbox can only read them through ``rlm_api.budget()``.
MAX_ITERATIONS = int(os.environ.get("LENS_MAX_ITERATIONS", "10"))
MAX_TOKENS_PER_QUERY = int(os.environ.get("LENS_MAX_TOKENS", "200000"))
MAX_COST_USD_PER_QUERY = float(os.environ.get("LENS_MAX_COST_USD", "0.50"))
MAX_EXEC_BLOCK_S = float(os.environ.get("LENS_MAX_EXEC_BLOCK_S", "120"))
MAX_BACKTRACKS = int(os.environ.get("LENS_MAX_BACKTRACKS", "3"))

# Context assembly. Compaction triggers when the estimated root context exceeds
# ``CONTEXT_COMPACT_RATIO * CONTEXT_TOKEN_LIMIT``.
CONTEXT_TOKEN_LIMIT = int(os.environ.get("LENS_CONTEXT_TOKEN_LIMIT", "60000"))
CONTEXT_COMPACT_RATIO = float(os.environ.get("LENS_CONTEXT_COMPACT_RATIO", "0.75"))
TOOL_RESULT_SUMMARY_CHARS = int(os.environ.get("LENS_TOOL_SUMMARY_CHARS", "2000"))

# Run mode. ``LENS_RECORDED=1`` disables dev-only escape hatches (T3 SQL) so a
# recorded run cannot bypass the graded funnel.
LENS_RECORDED = os.environ.get("LENS_RECORDED", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "",
}

# The root's final answer is explicit, so the loop never has to guess.
FINAL_ANSWER_MARKER = "FINAL_ANSWER:"

# ------------------------------------------------------------------ sandbox (M8)
# One ``lens-repl`` container per query session (LOG-31/LOG-32).
SANDBOX_IMAGE = os.environ.get("LENS_SANDBOX_IMAGE", "lens-repl:0.1.0")
SANDBOX_SERVICE = os.environ.get("LENS_SANDBOX_SERVICE", "lens-repl")
SANDBOX_COMPOSE_FILE = os.environ.get(
    "LENS_SANDBOX_COMPOSE_FILE", "docker-compose.yaml"
)
SANDBOX_WORKSPACE_PATH = os.environ.get("LENS_SANDBOX_WORKSPACE", "/workspace")
# Host-side port mapped to the runner's /exec server (compose publishes 127.0.0.1).
SANDBOX_PORT = int(os.environ.get("LENS_SANDBOX_PORT", "8080"))
SANDBOX_CPUS = os.environ.get("LENS_SANDBOX_CPUS", "2")
SANDBOX_MEMORY = os.environ.get("LENS_SANDBOX_MEMORY", "2g")
SANDBOX_PIDS_LIMIT = int(os.environ.get("LENS_SANDBOX_PIDS_LIMIT", "256"))
SANDBOX_STDOUT_MAX_CHARS = int(os.environ.get("LENS_SANDBOX_STDOUT_MAX_CHARS", "20000"))

# Host RPC server the sandbox calls back into. It binds all interfaces because
# Docker Desktop reaches it via ``host.docker.internal``; a per-run token is the
# auth boundary (the API key never enters the container).
HOST_RPC_BIND = os.environ.get("LENS_HOST_RPC_BIND", "0.0.0.0")
HOST_RPC_PORT = int(os.environ.get("LENS_HOST_RPC_PORT", "8765"))
HOST_RPC_TOKEN_ENV = "LENS_RPC_TOKEN"

# ------------------------------------------------------------------ gated T3 (M8)
# Dev-only read-only SQL escape hatch. Off by default; forced off in recorded runs.
ALLOW_SQL = os.environ.get("LENS_ALLOW_SQL", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "",
}
SQL_ROW_CAP = int(os.environ.get("LENS_SQL_ROW_CAP", "1000"))
SQL_TIMEOUT_MS = int(os.environ.get("LENS_SQL_TIMEOUT_MS", "5000"))

# ------------------------------------------------------------------ model registry
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
