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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
# Measured, not inherited. A capacity sweep over ten corpus queries (k, n, C and
# C_TOP) found two things: the shipped C == C_TOP == 100 made the stage-B reranker a
# pure reorder, and coverage of the selected papers tracks C_TOP rather than C --
# at n=50, raising C from 600 to 1500 left 18 papers with no chunk at all, because
# 100 surviving chunks spread over about 35 papers. Raising C to 400 with C_TOP at
# 100 takes coverage from 18.7 to 19.3 of 20 papers and drops the worst-case starved
# papers from 5 to 2, for 174 kB instead of 152 kB per bundle. Depth past ~400 buys
# nothing: C=600 starves the same two papers and costs 40% more time.
RETRIEVAL_K = int(os.environ.get("LENS_RETRIEVAL_K", "200"))  # Stage A candidates
RETRIEVAL_N = int(os.environ.get("LENS_RETRIEVAL_N", "20"))  # papers after Stage A
RETRIEVAL_C = int(os.environ.get("LENS_RETRIEVAL_C", "400"))  # Stage B candidates
RETRIEVAL_C_TOP = int(os.environ.get("LENS_RETRIEVAL_C_TOP", "100"))  # chunks after Stage B
# The KG scope is the deep-match candidate set (C); the answer context is top-c.
# Keep C >= C_TOP so the reranked context is a subset of the KG's chunk set.
if RETRIEVAL_C < RETRIEVAL_C_TOP:
    raise ValueError("RETRIEVAL_C must be >= RETRIEVAL_C_TOP")

#: Chunks returned per paper by ``fetch_paper_chunks`` - the T2 path for a paper
#: the query *names* but the funnel did not reach. Small on purpose: it is a way
#: to read a named paper, not a second retrieval, and eight chunks per paper is
#: about a tenth of the funnel's payload.
FETCH_PAPER_CHUNK_LIMIT = int(os.environ.get("LENS_FETCH_PAPER_CHUNK_LIMIT", "8"))

# HNSW query-time recall/latency knob. `m` / `ef_construction` are build-time
# (see scripts/build_indexes.py); `ef_search` is set per query. Record all three
# per run.
#
# 40 is pgvector's own default and it is far too small here: an ANN scan returns at
# most `ef_search` candidates, so asking for the top 100 with ef_search 100 explores
# exactly as many nodes as it returns. Measured against an exact scan on this corpus,
# ef_search 40 recovered 40 % of the true top-100 and ef_search 100 recovered 73 % --
# the 4th-closest paper to a recorded query was missing from the candidate set, so no
# reranker could recover it. 400 restores 95 % and 1000 restores 97 %. The cost is
# milliseconds: the whole stage-A search was 1.2 ms at ef 40 against a 30 s tool
# budget, so recall is bought far more cheaply than anything downstream.
HNSW_EF_SEARCH = int(os.environ.get("LENS_HNSW_EF_SEARCH", "400"))

#: Candidate list as a multiple of the requested LIMIT. A fixed ef_search is right
#: for the default k and wrong for a `k` swept in W2a, so the query path takes the
#: larger of the two: ef_search must exceed the LIMIT, not equal it.
HNSW_EF_MULTIPLIER = int(os.environ.get("LENS_HNSW_EF_MULTIPLIER", "4"))

#: pgvector accepts ef_search in 1..1000 and raises above that. The multiplier is
#: therefore clamped rather than trusted: at k=400, four times the LIMIT is already
#: outside the range, and a sweep would fail on its largest point instead of running.
HNSW_EF_MAX = 1000

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
# State the direction verifier sends to JEV: the **chunk** the triple came from,
# capped here. 2,400 chars covers a 512-token chunk, which is the unit the
# extractor read; a one-sentence state hid the hedge in the next sentence.
KG_VERIFY_MAX_STATE_CHARS = int(os.environ.get("LENS_KG_VERIFY_MAX_STATE", "2400"))
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
# Model windows (OpenRouter model list, 6 Nov 2026): root and sub
# ``deepseek/deepseek-v4.1-flash`` 1,048,576 ctx / 393,216 max out,
# $0.051/1M in + $0.567/1M out (provider ``wafer/us``, pinned); dev
# ``qwen/qwen3-coder`` 262,144 / 65,536, $0.30 + $1.00. The **narrowest** role
# (262k) bounds how much context any role can hold, not the root's.
MAX_ITERATIONS = int(os.environ.get("LENS_MAX_ITERATIONS", "10"))
#: Sub-task calls one query may make. A guard rather than a budget: each call is a
#: few seconds of the sub model and ~$0.001, but a bad loop inside one code block
#: could otherwise spend the whole per-query cost cap in a single turn. Exceeding it
#: returns an error to the sandbox and the run continues -- it is not a stop reason.
MAX_SUB_TASKS = int(os.environ.get("LENS_MAX_SUB_TASKS", "40"))
#: Sub-task batches one RPC may run at once. A 24-id call is six batches of four,
#: and the host waits on that call as a unit, so the pool covers a full call.
SUB_TASK_WORKERS = max(1, int(os.environ.get("LENS_SUB_TASK_WORKERS", "8")))
#: Claim-extraction batches one answer may run at once. The batches are independent
#: and the pass is sequential otherwise; four covers a 24-unit answer at six units
#: per batch in one round.
CLAIM_WORKERS = max(1, int(os.environ.get("LENS_CLAIM_WORKERS", "4")))
MAX_EXEC_BLOCK_S = float(os.environ.get("LENS_MAX_EXEC_BLOCK_S", "120"))
MAX_BACKTRACKS = int(os.environ.get("LENS_MAX_BACKTRACKS", "3"))
# Per-query COST cap. Raised 0.50 -> 1.00 on 26 Oct 2026 because the
# measured Tier-1 anchor spent **$0.436 in 8 iterations = 87 % of $0.50**: at that
# ceiling the cap was one turn from becoming the *reason* a run ended, so a
# model-quality result would have been reported as an artefact of a number we
# picked. The figure was measured on the previous root model and is **not**
# re-derived by the model swap of 6 Nov 2026: the cap is kept so recorded runs
# stay comparable, and a new value needs a measured unit-cost cell on the new rig.
# Earlier artifacts were recorded at $0.50 and later ones at $1.00; state that
# difference when reporting, because the cap changes the stop-reason distribution,
# not the pipeline.
MAX_COST_USD_PER_QUERY = float(os.environ.get("LENS_MAX_COST_USD", "0.50"))

# OpenRouter charges 5.5 % to load credit, so a reported "$/query" has two numbers:
# the raw sum of ``usage.cost`` and that sum times this fee. Accounting never
# applies it (``Budget`` books the real per-call cost); it exists for reporting,
# which is why it belongs here and not in three modules.
CREDIT_FEE = 1.055

# Blob-store format the writer uses. **v1** wrote blobs in text mode, so on Windows
# "\n" became "\r\n" on disk while the hash had been taken over the LF bytes: the
# filename stopped being the hash of the file. **v2** writes and reads the exact
# bytes. The version is recorded in each new run's manifest, so an artifact states
# which rule produced it; a v1 store is migrated once by ``scripts/repair_blobs.py``
# rather than by a reader that silently rewrites newlines.
BLOB_FORMAT_VERSION = 2

# A ``running`` run is presumed dead when nothing has been appended to its log for
# this long. ``--list``/``--resume`` sweep stale runs to ``interrupted``; a run
# inside the window is left alone because it may be live in another terminal, and
# two processes appending to one trajectory is worse than a late sweep. Five
# minutes is well above the 120 s exec cap and a full turn on this hardware.
RUN_HEARTBEAT_STALE_S = float(os.environ.get("LENS_RUN_HEARTBEAT_STALE_S", "300"))

# Inline evidence budget for the flat (naive RAG) arm: how many tokens of chunk text
# are pasted into one prompt. It is not a context ceiling and the looping arms never
# use it -- those keep evidence in a sandbox variable and print a digest.
FLAT_EVIDENCE_TOKEN_LIMIT = int(os.environ.get("LENS_FLAT_EVIDENCE_TOKENS", "40000"))

# Inline budget for the flat **document** arm. It is separate because the 40,000
# corpus figure is a frozen baseline: raising it would change what the corpus
# baseline sees. The document arm needs the whole file, so the default holds a
# 128k-token evidence file plus output margin.
FLAT_DOCUMENT_TOKEN_LIMIT = int(os.environ.get("LENS_FLAT_DOCUMENT_TOKENS", "140000"))

# Compaction trigger, derived from the **budget that remains**, not from the model's
# window. The window was never the binding constraint: with 10 iterations and about
# 2,500 tokens added per turn the largest prompt a run can reach is ~40,000 tokens
# against a 400,000 window, so a window-relative trigger would never fire -- and if
# it did, one such prompt would spend most of a query's token budget on one turn.
# The trigger is a quarter of what is left **when the loop starts**, fixed for the
# run; the floor stops it compacting below a useful history. A compaction evicts the
# oldest messages down to ``CONTEXT_COMPACT_TARGET_RATIO`` of the trigger, and the
# eviction point never moves back, so the context's prefix stays cacheable until
# the next compaction instead of changing on every turn.
CONTEXT_TRIGGER_RATIO_OF_REMAINING = float(os.environ.get("LENS_CONTEXT_TRIGGER_RATIO", "0.25"))
CONTEXT_TRIGGER_FLOOR = int(os.environ.get("LENS_CONTEXT_TRIGGER_FLOOR", "12000"))
CONTEXT_COMPACT_TARGET_RATIO = float(os.environ.get("LENS_CONTEXT_COMPACT_TARGET", "0.5"))
# Wrap-up nudge (see ``agent.context.budget_warning``): at this fraction of any
# cap the orchestrator asks the root to land the plane, so hitting a cap produces
# a labelled partial answer instead of a truncation mid-thought.
CONTEXT_WARN_RATIO = float(os.environ.get("LENS_CONTEXT_WARN_RATIO", "0.80"))
# Output is the expensive half (the shipped model bills $0.567/1M out against
# $0.051/1M in), so each root call gets a hard output cap. Long summarisation
# belongs in a sub-task.
ROOT_MAX_OUTPUT_TOKENS = int(os.environ.get("LENS_ROOT_MAX_OUTPUT_TOKENS", "8000"))
# Context-side observation cap (chars). The runner caps captured stdout too
# (``SANDBOX_STDOUT_MAX_CHARS``); this is the second, tighter cut applied before
# the observation becomes a message.
TOOL_RESULT_SUMMARY_CHARS = int(os.environ.get("LENS_TOOL_SUMMARY_CHARS", "4000"))

# Cumulative token cap, DERIVED from a **measurement** rather than from a chosen
# context ceiling: iterations x measured tokens per turn x 1.25 overhead.
#   tokens per turn = 18,562 (p90 of 576 recorded root calls on rig v18, W5b2)
#   10 x 18,562 x 1.25 = 232,025 for the root's own turns
# plus an allowance for sub-task calls, which spend this same budget. The measured
# 16k-cap replies averaged about 9,000 output tokens; 15 calls x 10,000 = 150,000
# leaves margin for the extracts a comparison actually makes. A 75,000 allowance
# bound on a real 10-turn run at 275,192 tokens, so the stop reason was an
# artefact of the allowance rather than of the work.
#
# The 14,000 that stood here was measured before `fetch_paper_chunks` existed. On
# rig v18 a run reads whole papers, so the per-turn figure rose 33 % (14,000 ->
# 18,562) and sub-task output rose 39 % (43k -> 60k median), which pushed the
# median run to 89 % of the old 325,000 and left one run with no answer at all.
# Re-deriving with the same formula gives 382,025; the value is a backstop, not a
# budget - a run at the cap bills about $0.08 against a $0.50 cost cap.
CONTEXT_TURN_TOKENS_ESTIMATE = int(os.environ.get("LENS_CONTEXT_TURN_TOKENS", "18562"))
SUB_TASK_TOKENS_ESTIMATE = int(os.environ.get("LENS_SUB_TASK_ALLOWANCE", "150000"))
MAX_TOKENS_PER_QUERY = int(
    os.environ.get(
        "LENS_MAX_TOKENS",
        str(int(MAX_ITERATIONS * CONTEXT_TURN_TOKENS_ESTIMATE * 1.25) + SUB_TASK_TOKENS_ESTIMATE),
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
# Two independent choices decide a run: **where the evidence comes from** and
# **how it is processed**. ``LENS_MODE`` keeps its three historical values for
# existing arms; ``LENS_EVIDENCE`` names the source explicitly, which is what
# makes the flat-document arm (``naive`` + ``document``) expressible.
#
#   corpus   + rlm   -> the system under test (retrieval funnel + loop)
#   corpus   + naive -> the flat-prompt baseline (one call, same retrieved chunks)
#   document + rlm   -> the RLM over a supplied evidence file (no retrieval)
#   document + naive -> the flat-prompt baseline over that same file
RUN_MODE = os.environ.get("LENS_MODE", "rlm").strip().lower()
RUN_MODES = ("rlm", "naive", "document")

EVIDENCE_SOURCE = (
    (os.environ.get("LENS_EVIDENCE") or ("document" if RUN_MODE == "document" else "corpus"))
    .strip()
    .lower()
)
EVIDENCE_SOURCES = ("corpus", "document")

# ``document`` is the historical spelling of "document evidence, RLM loop".
TREATMENT = "naive" if RUN_MODE == "naive" else "rlm"
TREATMENTS = ("rlm", "naive")

# Which system prompt the root sees. ``full`` is the system under test. The
# variants differ **only** in the instruction block: the task text stays
# byte-identical, or the comparison would measure prompt crafting. The default
# follows the treatment and the evidence source, so a mode cannot silently
# advertise tools it was not given.
_DEFAULT_VARIANT = (
    "naive" if TREATMENT == "naive" else ("document" if EVIDENCE_SOURCE == "document" else "full")
)
PROMPT_VARIANT = os.environ.get("LENS_PROMPT_VARIANT", _DEFAULT_VARIANT).strip().lower()
PROMPT_VARIANTS = ("full", "thin", "naive", "document")

# Who splits the query into the four buckets, and whether the root also sees the
# raw wording. ``host`` is the shipped path: one sub-model call writes
# ``query.json`` and the root receives the buckets. ``root`` is the ablation:
# the host does not segment, the raw query is in the prompt, and the root writes
# the bucket dict itself. ``RAW_QUERY`` adds the raw wording beside the host
# buckets, so a clause the segmenter dropped is still recoverable.
SEGMENTATION = os.environ.get("LENS_SEGMENTATION", "host").strip().lower()
SEGMENTATIONS = ("host", "root")
RAW_QUERY = os.environ.get("LENS_RAW_QUERY", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# The prompt-experiment switches. Each is recorded in the arm, so a run that
# changed one is not mistaken for the baseline. They default to the shipped
# behaviour; a variant is one environment variable, not a code edit.
CONTEXT_FILES = os.environ.get("LENS_CONTEXT_FILES", "full").strip().lower()
KG_PROMPT = os.environ.get("LENS_KG_PROMPT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
EXTRACT_HINT = os.environ.get("LENS_EXTRACT_HINT", "claims").strip().lower()
RETRIEVAL_RULE = os.environ.get("LENS_RETRIEVAL_RULE", "refine").strip().lower()

if CONTEXT_FILES not in {"full", "none"}:
    raise ValueError(f"bad LENS_CONTEXT_FILES {CONTEXT_FILES!r} (full or none)")

if EXTRACT_HINT not in {"claims", "quotes"}:
    raise ValueError(f"bad LENS_EXTRACT_HINT {EXTRACT_HINT!r} (claims or quotes)")

if RETRIEVAL_RULE not in {"once", "refine"}:
    raise ValueError(f"bad LENS_RETRIEVAL_RULE {RETRIEVAL_RULE!r} (once or refine)")

# ``0`` removes the ``/sub_task`` route instead of merely not mentioning it: an
# arm that must not recurse has to be unable to, otherwise it measures prompt
# obedience.
SUB_CALLS_ENABLED = os.environ.get("LENS_SUB_CALLS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Evidence file for document evidence (JSONL: one object per record). Copied
# into the run workspace so the sandbox reads it as a file; the records never
# pass through the prompt in the RLM arm. No default: a document run without one
# is a mistake.
DOCUMENTS_PATH: str | None = os.environ.get("LENS_DOCUMENTS") or None

# Run the claim pass and the final gate against the document records. Off by
# default: the mechanism arm is measured on its answer, and a gate over record
# ids buys it nothing. On for the needle guardrail, where the cited record can
# be checked mechanically.
DOCUMENT_EVIDENCE = os.environ.get("LENS_DOCUMENT_EVIDENCE", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _resolved_modes(run_mode: str, evidence: str) -> tuple[str, str]:
    """Return ``(evidence_source, treatment)`` for one run-mode spelling."""
    if run_mode == "document":
        return "document", "rlm"

    if run_mode == "naive":
        return evidence, "naive"

    return evidence, "rlm"


def _check_run_mode() -> None:
    """Fail at import on an unknown mode or variant, naming the accepted set."""
    if RUN_MODE not in RUN_MODES:
        raise ValueError(f"bad LENS_MODE {RUN_MODE!r} (expected one of {RUN_MODES})")

    if EVIDENCE_SOURCE not in EVIDENCE_SOURCES:
        raise ValueError(
            f"bad LENS_EVIDENCE {EVIDENCE_SOURCE!r} (expected one of {EVIDENCE_SOURCES})"
        )

    evidence, treatment = _resolved_modes(RUN_MODE, EVIDENCE_SOURCE)

    if SEGMENTATION not in SEGMENTATIONS:
        raise ValueError(
            f"bad LENS_SEGMENTATION {SEGMENTATION!r} (expected one of {SEGMENTATIONS})"
        )

    if SEGMENTATION == "root" and (evidence != "corpus" or treatment != "rlm"):
        raise ValueError("LENS_SEGMENTATION=root needs corpus evidence with the loop")

    if RUN_MODE == "document" and EVIDENCE_SOURCE != "document":
        raise ValueError("LENS_MODE=document is document evidence; drop LENS_EVIDENCE=corpus")

    if PROMPT_VARIANT not in PROMPT_VARIANTS:
        raise ValueError(f"bad LENS_PROMPT_VARIANT {PROMPT_VARIANT!r} (one of {PROMPT_VARIANTS})")

    if treatment == "naive" and PROMPT_VARIANT != "naive":
        raise ValueError(
            f"a naive treatment needs LENS_PROMPT_VARIANT=naive, not {PROMPT_VARIANT!r}"
        )

    if evidence == "document" and treatment == "rlm" and PROMPT_VARIANT != "document":
        raise ValueError(
            f"document evidence with the loop needs LENS_PROMPT_VARIANT=document, "
            f"not {PROMPT_VARIANT!r}"
        )

    if evidence == "corpus" and treatment == "rlm" and PROMPT_VARIANT not in ("full", "thin"):
        raise ValueError(
            f"corpus evidence with the loop needs LENS_PROMPT_VARIANT=full or thin, "
            f"not {PROMPT_VARIANT!r}"
        )

    if evidence == "document" and not DOCUMENTS_PATH:
        raise ValueError("document evidence needs LENS_DOCUMENTS=<path.jsonl>")


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
    #: The OpenRouter ``reasoning`` object sent with every call for this role, or
    #: ``None`` to send none. It is one object because the endpoint takes one:
    #: ``{"effort": "high"}``, ``{"max_tokens": N}`` or ``{"enabled": true}``.
    #: Changing it per turn changes the request prefix and invalidates the prompt
    #: cache, so it is a property of the role, not of the turn.
    reasoning: Mapping[str, Any] | None
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

    Provider pins are dropped by an override from a different model family rather
    than pinning a provider that cannot serve it (``allow_fallbacks`` stays
    ``False``). The price columns above stay documentation — ``Budget`` books the
    real ``usage.cost``, so an override cannot silently under-count.
    """
    if not raw:
        return spec

    vendor = raw.split("/", 1)[0]

    # Compare the model's namespace, not the provider slug: the pin names a
    # *serving provider* ("wafer/us") while the id names a *model vendor*
    # ("deepseek"), so those two strings never match. What has to be preserved is
    # "an override inside the same family keeps the pin"; anything else would pin
    # the wrong provider.
    if vendor == spec.id.split("/", 1)[0]:
        return replace(spec, id=raw)

    return replace(spec, id=raw, provider_only=(), provider_order=())


#: Reasoning effort for the ``sub`` role. The sub model serves structured calls
#: (segmentation, typed sub-tasks, claim extraction); at ``high`` a four-chunk
#: extract batch spent 12-15k of its 16k cap on reasoning and one batch per run
#: returned empty. The probe on the recorded failed batch: high = 16,000 tokens /
#: 0 records; low = 8,862 tokens / 4 records / 12 quotes; off = 1,376 tokens but
#: dropped 2 quotes and lost claim coverage. Low is the trade-off; the root stays
#: high. ``off`` is kept for probes, not as a default.
SUB_EFFORT = os.environ.get("LENS_SUB_EFFORT", "low").strip().lower()

if SUB_EFFORT not in {"high", "medium", "low", "off"}:
    raise ValueError(f"bad LENS_SUB_EFFORT {SUB_EFFORT!r} (high, medium, low or off)")

SUB_REASONING = (
    {"enabled": False} if SUB_EFFORT == "off" else {"enabled": True, "effort": SUB_EFFORT}
)

MODEL_REGISTRY: dict[Role, ModelSpec] = {
    "root": _model_override(
        ModelSpec(
            id="deepseek/deepseek-v4.1-flash",
            supports_reasoning=True,
            # One effort for the whole role. The endpoint takes `effort` or a token
            # budget, not both; DeepSeek is an effort-level model, so a per-turn
            # dial would only be a coarser level and would break the prefix cache.
            reasoning={"enabled": True, "effort": "high"},
            supports_tools=True,
            supports_response_format=True,
            # The same model id is served by providers whose input price differs
            # several times over, so the provider is part of the recording.
            provider_only=("wafer/us",),
            provider_order=("wafer/us",),
        ),
        os.environ.get("LENS_ROOT_MODEL"),
    ),
    "sub": _model_override(
        ModelSpec(
            id="deepseek/deepseek-v4.1-flash",
            supports_reasoning=True,
            # The sub model serves structured extraction, not open reasoning. High
            # effort is paid for in the 16k output cap: one four-chunk batch per run
            # spent the cap on thinking and returned no records. Low keeps 4 records
            # and 12 quotes on the same batch. Root stays high.
            reasoning=SUB_REASONING,
            supports_tools=True,
            supports_response_format=True,
            provider_only=("wafer/us",),
            provider_order=("wafer/us",),
        ),
        os.environ.get("LENS_SUB_MODEL"),
    ),
    "dev": _model_override(
        ModelSpec(
            id="qwen/qwen3-coder",
            supports_reasoning=False,
            reasoning=None,
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
