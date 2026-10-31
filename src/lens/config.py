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
from dataclasses import dataclass, replace
from pathlib import Path

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
# **Path portability.** Code is installed read-only; runtime data lives in
# ONE writable root. Nothing below is an absolute literal, so the same checkout
# runs on a machine with no `D:` drive.
#
#   PACKAGE_DIR / REPO_ROOT   files that ship with the code (compose file, docs)
#   DATA_ROOT                 runs / logs / cache / manifests   (env: LENS_DATA_ROOT)
#   LENS_ARXIV_METADATA / LENS_GROBID_XML_DIR   external corpus inputs, NO default
#
# The local developer keeps the old layout by putting `LENS_DATA_ROOT=D:\lens-data`
# in the gitignored `.env` (see `.env.example`) — the default never has to be a
# particular machine's path.
PACKAGE_DIR = Path(__file__).resolve().parent  # .../src/lens
REPO_ROOT = PACKAGE_DIR.parents[1]  # .../lens (the git checkout)


def _data_root() -> Path:
    """Resolve the writable data root: env override, else the platform dir.

    No third-party dependency: ``%LOCALAPPDATA%`` on Windows, ``$XDG_DATA_HOME``
    on Linux, else ``~/.local/share`` (macOS users normally set
    ``LENS_DATA_ROOT``). One anchor, everything else is relative to it.
    """
    override = (os.environ.get("LENS_DATA_ROOT") or "").strip()
    if override:
        return Path(override).expanduser()

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"

    return root / "lens"


DATA_ROOT = _data_root()

LENS_RUNS_ROOT_PATH = Path(os.environ.get("LENS_RUNS_ROOT") or DATA_ROOT / "runs")
BLOBS_DIR_NAME = "blobs"
TRAJECTORY_FILENAME = "trajectory.jsonl"
# Long-running dev-run logs kept for evidence.
LENS_LOGS_ROOT_PATH = Path(os.environ.get("LENS_LOGS_ROOT") or DATA_ROOT / "logs")

# ------------------------------------------------------------------ dataset
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

LENS_MANIFESTS_ROOT_PATH = Path(os.environ.get("LENS_MANIFESTS_ROOT") or DATA_ROOT / "manifests")
# **Corpus inputs: no default on purpose.** They are machine-specific, huge, and
# cannot be shipped; the harness itself runs without them, so only the //
# loaders resolve them — via :func:`require_path`, which fails early with the env
# var name instead of a ``TypeError`` deep inside a loader.
ARXIV_METADATA_JSON_PATH: str | None = os.environ.get("LENS_ARXIV_METADATA") or None
GROBID_XML_DIR_PATH: str | None = os.environ.get("LENS_GROBID_XML_DIR") or None
# Source lines between checkpoint writes during the scan.
MANIFEST_CHECKPOINT_INTERVAL = 3000


def require_path(env_name: str, value: str | None) -> Path:
    """Resolve a corpus path or exit with an actionable message.

    Only the // loaders call this — the harness itself must run on a machine
    that has no corpus at all, and validating at *use* (not at import) keeps
    ``--help`` working there too. Failing here names the env var; failing later
    would be a ``TypeError`` inside a loader, hours into a run.
    """
    if not value:
        raise SystemExit(
            f"{env_name} is not set. Copy .env.example to .env (gitignored) and point "
            f"it at the corpus, or export {env_name}=<path>."
        )

    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"{env_name} does not exist: {path}")

    return path


# ------------------------------------------------------------------ database
# Local pgvector container. These are DEV defaults only;
# override any of them via the environment. A real password is never committed
# — the defaults match the `docker run` flags documented in
# `db/schema.py`.
LENS_DB_HOST = os.environ.get("LENS_DB_HOST", "localhost")
LENS_DB_PORT = int(os.environ.get("LENS_DB_PORT", "5432"))
LENS_DB_NAME = os.environ.get("LENS_DB_NAME", "lens")
LENS_DB_USER = os.environ.get("LENS_DB_USER", "lens")
LENS_DB_PASSWORD = os.environ.get("LENS_DB_PASSWORD")

# Read-only role the host tool layer uses on the RLM's behalf: SELECT only,
# never write. The sandbox itself holds no DB credentials.
LENS_DB_RO_USER = os.environ.get("LENS_DB_RO_USER", "lens_ro")
LENS_DB_RO_PASSWORD = os.environ.get("LENS_DB_RO_PASSWORD")

# Container / image used by `scripts/setup_db.py` docs and evidence.
# PGDATA itself is the Docker **named volume** `lens-pgdata` (never a host path):
# a Windows bind mount made index I/O ~100x slower and was migrated away on 11 Oct.
LENS_DB_CONTAINER_NAME = os.environ.get("LENS_DB_CONTAINER", "lens-pg")
LENS_DB_IMAGE = "pgvector/pgvector:pg16"
LENS_DB_VOLUME_NAME = os.environ.get("LENS_DB_VOLUME_NAME", "lens-pgdata")

# Embedding width + pgvector storage type: nomic-embed-text-v1.5, Matryoshka
# 768 -> 512, stored as `halfvec` (fp16). Note 2*512+8 == 4*256+8 bytes/row —
# the same storage as the retired fp32 vector(256) but with 512 dims; `halfvec`
# is native + HNSW-indexable with no quantization calibration.
EMBEDDING_DIM = int(os.environ.get("LENS_EMBEDDING_DIM", "512"))
# Abstracts embedded per model forward pass during the load.
EMBEDDING_ABSTRACT_BATCH_SIZE = 64

# pgvector column type + HNSW op class must match what the embedder produces.
EMBEDDING_PGVECTOR_TYPE = os.environ.get("LENS_EMBEDDING_TYPE", "halfvec")
EMBEDDING_INDEX_OPS = os.environ.get("LENS_EMBEDDING_INDEX_OPS", "halfvec_cosine_ops")

# ------------------------------------------------------------------ papers + embeddings
# No paper subset: the full manifest is loaded.

# nomic-embed-text-v1.5, Matryoshka-truncated 768 -> 512, stored halfvec.
# EMBEDDING_VERSION is copied into every row so a re-index is detectable; bump
# it on any model, width, precision or prefix change.
EMBEDDING_MODEL_ID = os.environ.get("LENS_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBEDDING_VERSION = os.environ.get("LENS_EMBEDDING_VERSION", "nomic-embed-text-v1.5@512-halfvec")
# nomic is prefix-conditioned: documents and queries use different task tokens.
# A missing prefix does not error — it silently degrades matching, so it lives here.
EMBEDDING_DOC_PREFIX = "search_document: "
EMBEDDING_QUERY_PREFIX = "search_query: "

# Number of chunks (entries) to extract, encode, re-format and insert into the DB table
PROCESS_CHUNK_SIZE = 3500

# ------------------------------------------------------------------ chunks
# Full-corpus structural chunking.
# Threshold is the frozen /N-04 value: 512 nomic (BERT WordPiece) tokens.
# The splitter keeps a
# whole GROBID section as one chunk when it is under the threshold, and only
# splits sections that exceed it.
CHUNK_TOKEN_THRESHOLD = int(os.environ.get("LENS_CHUNK_TOKEN_THRESHOLD", "512"))
# Overlap between consecutive chunks within a section (0 = structural, no overlap).
CHUNK_OVERLAP_TOKENS = int(os.environ.get("LENS_CHUNK_OVERLAP_TOKENS", "0"))
# Chunks per model forward pass during the load (GPU batch; separate from the
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

# ------------------------------------------------------------------ retrieval
# The frozen funnel :
#   shallow_match k candidates -> Stage-A rerank -> n papers
#   deep_match    C chunk candidates (filtered) -> Stage-B rerank -> c chunks
# `k`/`C` are the candidate-recall knobs; the cross-encoder cost is linear in
# them (one pass per candidate). These are the guardrail caps.
RETRIEVAL_K = int(os.environ.get("LENS_RETRIEVAL_K", "100"))  # Stage A candidates
RETRIEVAL_N = int(os.environ.get("LENS_RETRIEVAL_N", "20"))  # papers after Stage A
RETRIEVAL_C = int(os.environ.get("LENS_RETRIEVAL_C", "100"))  # Stage B candidates
RETRIEVAL_C_TOP = int(os.environ.get("LENS_RETRIEVAL_C_TOP", "100"))  # chunks after Stage B
# The KG scope is the deep-match candidate set (C); the answer context is top-c.
# Keep C >= C_TOP so the reranked context is a subset of the KG's chunk set.
if RETRIEVAL_C < RETRIEVAL_C_TOP:
    raise ValueError("RETRIEVAL_C must be >= RETRIEVAL_C_TOP")

# HNSW query-time recall/latency knob. `m` / `ef_construction` are build-time
# (see scripts/build_indexes.py); `ef_search` is set per query. Record all three
# per run.
HNSW_EF_SEARCH = int(os.environ.get("LENS_HNSW_EF_SEARCH", "40"))

# Cross-encoder reranker, used at BOTH stages (Stage A = papers, Stage B =
# chunks) so one notion of relevance ranks both.
RERANKER_MODEL_ID = os.environ.get("LENS_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_BATCH_SIZE = int(os.environ.get("LENS_RERANK_BATCH", "32"))

# ------------------------------------------------------------------ knowledge graph
# Deterministic query-time NER/RE over the **deep-matched chunks**, not over the
# whole corpus. NER (GLiNER) + RE (GLiREL) run **host-side** (torch/GPU); the
# resulting graph is handed to the RLM as a networkx namespace variable and is
# never dumped into root context. The subgraph is cached by paper set, ontology
# version and extractor identity, so unchanged inputs cost nothing to re-open.
LENS_CACHE_ROOT_PATH = Path(os.environ.get("LENS_CACHE_ROOT") or DATA_ROOT / "cache")
KG_CACHE_DIR_NAME = "kg"
# Bump on ANY change to the entity/relation schema: it is both the KG
# re-extraction key and part of the cache key.
KG_ONTOLOGY_VERSION = os.environ.get("LENS_KG_ONTOLOGY_VERSION", "lens-ontology-1")
# Both extractor families are installed for the recall probe; the losing
# family is removed once the probe decides.
KG_NER_MODEL_ID = os.environ.get("LENS_KG_NER_MODEL", "urchade/gliner_medium-v2.1")
KG_REL_MODEL_ID = os.environ.get("LENS_KG_REL_MODEL", "jackboyla/glirel-large-v0")
# Minimum span / relation confidence kept. Kept in one place so the probe can sweep it.
KG_MIN_SCORE = float(os.environ.get("LENS_KG_MIN_SCORE", "0.5"))

# Extraction window. GLiNER truncates its input (384 tokens on the medium
# checkpoint) while chunks are <= 512 nomic tokens, so extraction is windowed to
# avoid dropping long-chunk tails. The overlap is carried into the next window.
KG_EXTRACT_MAX_TOKENS = int(os.environ.get("LENS_KG_EXTRACT_MAX_TOKENS", "384"))
KG_EXTRACT_WINDOW_OVERLAP = int(os.environ.get("LENS_KG_EXTRACT_OVERLAP", "32"))

# ------------------------------------------------------------------ KG post-process
# Modular post-process applied **before** filter_ontology: relation-slot type
# inference, a corpus-built gazetteer, and directionality resolution. Types are
# soft (``type`` + ``type_source`` + ``type_confidence``). Every stage is
# toggleable/removable; see /.
KG_POSTPROCESS_ENABLED = os.environ.get("LENS_KG_POSTPROCESS", "1").strip() != "0"
KG_TYPE_INFERENCE = os.environ.get("LENS_KG_TYPE_INFERENCE", "1").strip() != "0"
KG_GAZETTEER_ENABLED = os.environ.get("LENS_KG_GAZETTEER", "1").strip() != "0"
KG_GAZETTEER_PATH = Path(
    os.environ.get("LENS_KG_GAZETTEER_PATH")
    or LENS_CACHE_ROOT_PATH / "kg" / "gazetteer_corpus.json"
)
# Only trust gazetteer entries above this confidence (count-based: 0.7=<5,
# 0.8=5-9, 0.9>=10). Prevents low-count false positives (e.g. "spaCy").
KG_GAZETTEER_MIN_CONFIDENCE = float(os.environ.get("LENS_KG_GAZETTEER_MIN_CONF", "0.9"))
KG_DIRECTIONALITY_ENABLED = os.environ.get("LENS_KG_DIRECTIONALITY", "1").strip() != "0"
KG_CYCLE_BREAK = os.environ.get("LENS_KG_CYCLE_BREAK", "1").strip() != "0"

# ------------------------------------------------------------------ KG quality
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
# Decision layer is single-provider and NOT part of MODEL_REGISTRY.
JEV_MODEL_ID = "typesafe/jev-1.13"

# ------------------------------------------------------------------ guardrails
# : 120 s per execution block, max 3 retries.
DEFAULT_TIMEOUT_S = 120
MAX_RETRIES = 3

# ------------------------------------------------------------------ harness
# Central RLM caps. The orchestrator enforces these;
# the sandbox can only read them through ``rlm_api.budget()``.
#
# The caps are ORDERED so that a run ends because the model finished, not because
# a counter ran out: FINAL_ANSWER -> iterations -> tokens -> cost. The token cap
# is DERIVED below from the turn and context caps rather than being an
# independent number, so the three can never fight each other by accident (an
# independent 200k token cap used to bind at ~4 turns while the 10-turn cap
# never fired).
#
# Model windows (OpenRouter model list, 18 Oct 2026): root ``openai/gpt-5``
# 400,000 ctx / 128,000 max out, $1.25/1M in + $10/1M out; sub ``gpt-5-mini``
# 400,000; dev ``qwen/qwen3-coder`` 262,144 / 65,536, $0.30 + $1.00. The
# **narrowest** role (262k) bounds CONTEXT_TOKEN_LIMIT, not the root.
MAX_ITERATIONS = int(os.environ.get("LENS_MAX_ITERATIONS", "8"))
MAX_EXEC_BLOCK_S = float(os.environ.get("LENS_MAX_EXEC_BLOCK_S", "120"))
MAX_BACKTRACKS = int(os.environ.get("LENS_MAX_BACKTRACKS", "3"))
# Per-query COST cap. Raised 0.50 -> 1.00 on 26 Oct 2026 because the
# measured Tier-1 anchor spent **$0.436 in 8 iterations = 87 % of $0.50**: at that
# ceiling the cap was one turn from becoming the *reason* a `gpt-5` run ended, so a
# model-quality result would have been reported as an artefact of a number we
# picked. $1.00 is the maximum that means anything at the current token cap
# (300k tokens x ~$3.1/M blended ~= $0.93), so the binding cap is now
# **iterations** or **tokens** - both of which are defensible results. Do not raise
# tokens or iterations to "use" the extra headroom: each turn grows the context and
# that is what costs money. artifacts were recorded at $0.50; state that
# difference when reporting (the cap changes the stop-reason distribution, not the
# pipeline).
MAX_COST_USD_PER_QUERY = float(os.environ.get("LENS_MAX_COST_USD", "1.00"))

# Context assembly. Compaction triggers when the estimated root context exceeds
# ``CONTEXT_COMPACT_RATIO * CONTEXT_TOKEN_LIMIT`` (= 30,000 tokens by default).
# 40k is ~10% of gpt-5's window and ~15% of qwen3-coder's. A turn needs far less:
# the retrieved evidence stays a sandbox variable, so the context only carries a
# printed digest (~1.5k tokens), the tool manifest, query.json and budget.md.
CONTEXT_TOKEN_LIMIT = int(os.environ.get("LENS_CONTEXT_TOKEN_LIMIT", "40000"))
CONTEXT_COMPACT_RATIO = float(os.environ.get("LENS_CONTEXT_COMPACT_RATIO", "0.75"))
# Wrap-up nudge (see ``agent.context.budget_warning``): at this fraction of any
# cap the orchestrator asks the root to land the plane, so hitting a cap produces
# a labelled partial answer instead of a truncation mid-thought.
CONTEXT_WARN_RATIO = float(os.environ.get("LENS_CONTEXT_WARN_RATIO", "0.80"))
# Output is the expensive half (gpt-5: $10/1M out vs $1.25/1M in), so each root
# call gets a hard output cap. Long summarisation belongs in ``sub_call`` (mini).
ROOT_MAX_OUTPUT_TOKENS = int(os.environ.get("LENS_ROOT_MAX_OUTPUT_TOKENS", "8000"))
# Context-side observation cap (chars). The runner caps captured stdout too
# (``SANDBOX_STDOUT_MAX_CHARS``); this is the second, tighter cut applied before
# the observation becomes a message.
TOOL_RESULT_SUMMARY_CHARS = int(os.environ.get("LENS_TOOL_SUMMARY_CHARS", "3000"))

# Cumulative token cap, DERIVED (not tuned): iterations x compacted-context
# target x 1.25 overhead (system prompt + tool manifest + final answer).
# Default 8 x 30,000 x 1.25 = 300,000. With the cost cap at $1.00 this is
# the cap most likely to bind on a measured gpt-5 run (the anchor used 141k tokens
# and $0.436), which is a defensible stop reason; a measured estimate, not a result
# - reports the observed distribution.
MAX_TOKENS_PER_QUERY = int(
    os.environ.get(
        "LENS_MAX_TOKENS",
        str(int(MAX_ITERATIONS * CONTEXT_TOKEN_LIMIT * CONTEXT_COMPACT_RATIO * 1.25)),
    )
)

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

# ------------------------------------------------------- experimental arms
# Which harness the process runs. ``rlm`` is the system under test; ``naive`` is
# the flat-prompt baseline (one chat call over the same retrieved chunks, no
# loop, no tools); ``document`` runs the RLM over a supplied evidence file
# instead of the corpus, so the retrieval funnel is not used at all.
RUN_MODE = os.environ.get("LENS_MODE", "rlm").strip().lower()
RUN_MODES = ("rlm", "naive", "document")

# Which system prompt the root sees. ``full`` is the system under test. The
# variants differ **only** in the instruction block: the task text stays
# byte-identical, or the comparison would measure prompt crafting. A mode that is
# not ``rlm`` implies its own prompt, so the two cannot disagree by accident.
PROMPT_VARIANT = (
    os.environ.get("LENS_PROMPT_VARIANT", RUN_MODE if RUN_MODE in ("naive", "document") else "full")
    .strip()
    .lower()
)
PROMPT_VARIANTS = ("full", "naive", "document")

# ``0`` removes the ``/sub_call`` route instead of merely not mentioning it: an
# arm that must not recurse has to be unable to, otherwise it measures prompt
# obedience.
SUB_CALLS_ENABLED = os.environ.get("LENS_SUB_CALLS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Evidence file for ``document`` mode (JSONL: one object per record). Copied
# into the run workspace so the sandbox reads it as a file; the records never
# pass through the prompt. No default: a document run without one is a mistake.
DOCUMENTS_PATH: str | None = os.environ.get("LENS_DOCUMENTS") or None


def _check_run_mode() -> None:
    """Fail at import on an unknown mode or variant, naming the accepted set."""
    if RUN_MODE not in RUN_MODES:
        raise ValueError(f"bad LENS_MODE {RUN_MODE!r} (expected one of {RUN_MODES})")

    if PROMPT_VARIANT not in PROMPT_VARIANTS:
        raise ValueError(f"bad LENS_PROMPT_VARIANT {PROMPT_VARIANT!r} (one of {PROMPT_VARIANTS})")

    if RUN_MODE in ("naive", "document") and PROMPT_VARIANT != RUN_MODE:
        raise ValueError(
            f"LENS_MODE={RUN_MODE} needs LENS_PROMPT_VARIANT={RUN_MODE}, not {PROMPT_VARIANT!r}"
        )

    if RUN_MODE == "document" and not DOCUMENTS_PATH:
        raise ValueError("LENS_MODE=document needs LENS_DOCUMENTS=<path.jsonl>")


_check_run_mode()

# ------------------------------------------------------------------ sandbox
# One ``lens-repl`` container per query session.
SANDBOX_IMAGE = os.environ.get("LENS_SANDBOX_IMAGE", "lens-repl:0.1.0")
SANDBOX_SERVICE = os.environ.get("LENS_SANDBOX_SERVICE", "lens-repl")
# The sandbox's only peer. An ``internal: true`` bridge has no route to the host,
# so the container cannot use ``host.docker.internal``; it dials this relay,
# which forwards one port to the host RPC server. Same image, different
# entrypoint (``/app/relay.py``).
SANDBOX_RELAY_SERVICE = os.environ.get("LENS_RELAY_SERVICE", "lens-relay")
SANDBOX_RELAY_URL = os.environ.get(
    "LENS_RELAY_URL",
    f"http://{os.environ.get('LENS_RELAY_SERVICE', 'lens-relay')}:"
    f"{os.environ.get('LENS_RELAY_PORT', '8090')}",
)
SANDBOX_COMPOSE_FILE = Path(
    os.environ.get("LENS_SANDBOX_COMPOSE_FILE") or REPO_ROOT / "docker-compose.yaml"
)
SANDBOX_WORKSPACE_PATH = os.environ.get("LENS_SANDBOX_WORKSPACE", "/workspace")
# Host-side port mapped to the runner's /exec server (compose publishes 127.0.0.1).
SANDBOX_PORT = int(os.environ.get("LENS_SANDBOX_PORT", "8080"))
SANDBOX_CPUS = os.environ.get("LENS_SANDBOX_CPUS", "2")
SANDBOX_MEMORY = os.environ.get("LENS_SANDBOX_MEMORY", "2g")
SANDBOX_PIDS_LIMIT = int(os.environ.get("LENS_SANDBOX_PIDS_LIMIT", "256"))
# Raw stdout captured per exec block. This is the first cut on observation size;
# ``TOOL_RESULT_SUMMARY_CHARS`` is the second (before it becomes a message). The
# compose file defaults the same variable to the same number, so a container
# started by hand captures no more than a harness run: at 20,000 chars (~5k
# tokens) a single turn could eat a sixth of the context.
SANDBOX_STDOUT_MAX_CHARS = int(os.environ.get("LENS_SANDBOX_STDOUT_MAX_CHARS", "6000"))

# Host RPC server the sandbox calls back into. It binds all interfaces because the
# *relay* connects to it over ``rpc-net``; a per-run token is the auth
# boundary (the API key never enters the container, and neither does the sandbox
# ever see this address — it dials ``lens-relay``).
HOST_RPC_BIND = os.environ.get("LENS_HOST_RPC_BIND", "0.0.0.0")
HOST_RPC_PORT = int(os.environ.get("LENS_HOST_RPC_PORT", "8765"))
HOST_RPC_TOKEN_ENV = "LENS_RPC_TOKEN"

# ------------------------------------------------------------------ gated T3
# Dev-only read-only SQL escape hatch. Off by default; forced off in recorded runs.
ALLOW_SQL = os.environ.get("LENS_ALLOW_SQL", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "",
}
SQL_ROW_CAP = int(os.environ.get("LENS_SQL_ROW_CAP", "1000"))
SQL_TIMEOUT_MS = int(os.environ.get("LENS_SQL_TIMEOUT_MS", "5000"))

# ------------------------------------------------------------------ gating
# The decision-layer gates : the per-stage gate after each
# stage-completing tool, and the final groundedness check before an answer is
# accepted. Turning this off is the **decision-layer-OFF arm** of the
# ablation — no JEV call and **no `gate` key** in the RPC envelope: an arm with no
# gate must look like it has no gate, not like a gate that always passes.
GATE_ENABLED = os.environ.get("LENS_GATE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _parse_stage_thresholds(raw: str) -> dict[str, float]:
    """Parse ``"retrieval=0.72,kg=0.65"`` into ``{stage: threshold}``.

    Strict on purpose: a malformed entry is a calibration mistake, and a typo that
    silently removes a threshold would make the gate look calibrated when it is not.
    """
    thresholds: dict[str, float] = {}

    for item in raw.split(","):
        item = item.strip()

        if not item:
            continue

        stage, separator, value = item.partition("=")

        if not separator or not stage.strip():
            raise ValueError(f"bad LENS_STAGE_THRESHOLDS entry: {item!r} (expected stage=float)")

        thresholds[stage.strip()] = float(value)

    return thresholds


# Calibrated per-stage acceptance thresholds.
# **Empty until a stage has a reliability curve** : do not invent numbers — an
# empty map simply skips tier 1 and lets the JEV `noul` check decide, which is the
# honest state before calibration. A single threshold can only *accept*: without a
# lower τ_fail, "below τ_pass" means "ask tier 2", never "fail".
# Override without a code edit: LENS_STAGE_THRESHOLDS="retrieval=0.72,kg=0.65".
STAGE_THRESHOLDS: dict[str, float] = _parse_stage_thresholds(
    os.environ.get("LENS_STAGE_THRESHOLDS", "")
)


# ------------------------------------------------------------------ model registry
@dataclass(frozen=True)
class ModelSpec:
    """A frozen description of one model role.

    Callers ask for a *role* ("root"/"sub"/"dev"); they never pass a raw model ID.
    This is where the exact id, capability flags, sampling controls, and provider
    pinning live.
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


def _model_override(spec: ModelSpec, raw: str | None) -> ModelSpec:
    """Return ``spec`` with its id replaced by ``raw`` (a dev/ablation override).

    Why this exists: a plumbing run does not need the Tier-1 instrument, and
    ``MODEL_REGISTRY`` used to be the only way to say which model a role uses, so
    every harness debug iteration cost ``gpt-5`` prices. ``LENS_ROOT_MODEL`` /
    ``LENS_SUB_MODEL`` / ``LENS_DEV_MODEL`` override the **id only**; the id is
    recorded per call, so a recorded run stays attributable.

    Provider pins are vendor **slugs**: an override from a different vendor drops
    them rather than pinning the wrong one (``allow_fallbacks`` stays ``False``).
    The price columns above stay documentation — ``Budget`` books the real
    ``usage.cost``, so an override cannot silently under-count.
    """
    if not raw:
        return spec

    vendor = raw.split("/", 1)[0]

    if vendor in spec.provider_only:
        return replace(spec, id=raw)

    return replace(spec, id=raw, provider_only=(), provider_order=())


MODEL_REGISTRY: dict[Role, ModelSpec] = {
    "root": _model_override(
        ModelSpec(
            id="openai/gpt-5",
            supports_reasoning=True,
            reasoning_effort="medium",
            supports_tools=True,
            supports_response_format=True,
            provider_only=("openai",),
            provider_order=("openai",),
        ),
        os.environ.get("LENS_ROOT_MODEL"),
    ),
    "sub": _model_override(
        ModelSpec(
            id="openai/gpt-5-mini",
            supports_reasoning=True,
            reasoning_effort="low",
            supports_tools=True,
            supports_response_format=True,
            provider_only=("openai",),
            provider_order=("openai",),
        ),
        os.environ.get("LENS_SUB_MODEL"),
    ),
    "dev": _model_override(
        ModelSpec(
            id="qwen/qwen3-coder",
            supports_reasoning=False,
            reasoning_effort=None,
            supports_tools=True,
            supports_response_format=False,
            provider_only=("google-vertex",),
            provider_order=("google-vertex",),
            temperature=0.1,
        ),
        os.environ.get("LENS_DEV_MODEL"),
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
            "OPENROUTER_API_KEY is not set. Put it in the OS environment or in a "
            "git-ignored .env file at the repo root."
        )
    return key
