# LENS — Architecture & Module Interactions

> **Purpose:** how the harness modules and classes interact: the layer boundaries,
> the one-way dependencies, and the contracts that keep a run auditable.
>
> **Status:** covers the built and live system; the loop has run end to end against
> the real corpus. Section 1 is the layer map, section 2 the module graph (generated
> from the package), section 3 the hand-curated class map (with the reason the
> generated one is sparse), section 4 the runtime flows a static tool cannot show,
> and section 5 the caveats that cost someone an afternoon if they are forgotten.

---

## 1. Layered overview

Two runtimes, **one-way imports**:

- **Host** (Windows-native Python 3.12) owns the loop, the API key, the DB
  credentials, and the logs.
- **Sandbox** executes the RLM's generated Python only. It holds
  **no key and no DB driver**; every external effect is an RPC back to the host
  through `lens-relay`.

| Module | Responsibility | Imports |
|---|---|---|
| `src/lens/models.py` | Shared types: `Role`, `CallResult`, `JevResult`, `Question`. No project imports (bottom of the graph). | stdlib + `pydantic` |
| `src/lens/config.py` | Paths, endpoints, guardrails, caps, DB settings, `MODEL_REGISTRY`, `JEV_MODEL_ID`, `require_api_key()`, `require_path()`. | `models` |
| `src/lens/preflight.py` | `require_environment(need=…)`: `api_key` / `db` / `corpus` / `metadata` / `trust`, one report, secrets masked. | `config`, `dataset.paths` |
| `src/lens/observability/trajectory.py` | The injected sink: `Trajectory` (content-addressed blobs + one-schema JSONL) and `TraceCtx`. | `config` |
| `src/lens/observability/replay.py` · `citations.py` · `run_logging.py` | Fold the log into turns; resolve citations into `sources.md`; mirror `print()` into a log file. | `trajectory`, `context`, `config` |
| `src/lens/clients.py` | Transport: `LensClient.chat()` / `.jev()` — auth, retries, provider pinning, `usage`/cost, one event per call. Stateless. | `config`, `models`, `trajectory` |
| `src/lens/chunking.py` | TEI → sections → chunks: hardened parse, `normalize_head`, `classify_section`, the 512-token packer. | `config` |
| `src/lens/dataset/manifest.py` · `papers.py` · `chunks.py` · `fetch.py` · `paths.py` | Manifest builder (single-pass cs.AI filter), the two loaders, corpus download/convert, the path chain. | `config`, `chunking`, `embeddings` |
| `src/lens/db/connect.py` · `schema.py` | Connection factory (`connect(readonly=…)` → psycopg, admin / `lens_ro`) and the idempotent pgvector migration. | `config` |
| `src/lens/embeddings.py` · `reranking.py` | `nomic-embed-text-v1.5` (768 → 512, `halfvec`) and the `bge-reranker-v2-m3` cross-encoder; both lazy. | `config` |
| `src/lens/tools/retrieval.py` | The T1 funnel and T2 fetches: `shallow_match`, `rerank_papers`, `deep_match`, `rerank_chunks`, `fetch_papers`, `fetch_chunks`. | `config`, DB driver, `embeddings`, `reranking` |
| `src/lens/tools/kg.py` | `build_kg`: extraction, post-process, verifier hook, ontology filter, graph + summary, cache. | `config`, `kg_*`, `retrieval` |
| `src/lens/kg_extract.py` · `kg_postprocess.py` · `kg_verify.py` · `kg_query.py` · `kg_gold.py` | GLiNER/GLiREL extraction, the ordered quality layer, the three-tier direction verifier, pure traversal, the gold worksheet. | `config`, `models` |
| `src/lens/tools/evaluate.py` · `sql.py` | The three-tier stage gate and the final claim gate; the gated T3 `query(sql)`. | `clients`, DB driver, `trajectory` |
| `src/lens/agent/runtime.py` | Orchestrator: owns the loop, context assembly and compaction, sub-calls, budget, ledger, RPC dispatch, the gates, finalisation. | `clients`, `tools`, `trajectory`, `context`, `experiments.arm` |
| `src/lens/agent/{context,budget,ledger,tool_registry,tool_server,sandbox_session,segmentation}.py` | Prompt and workspace files, caps, `tasks.json`, tool manifest/dispatch, host RPC, Docker lifecycle, the segmentation call. | `config`, `clients`, `trajectory` |
| `src/lens/agent/runstore.py` | `manifest.json` (identity/index), `resume.json` (cache), `list_runs`, `sweep_interrupted`, `fork_run`, bundle export/import (with the export-time secret scan). | `config` (lazy) |
| `src/lens/sandbox/{runner,rlm_api,relay,exec_client}.py` | Passive persistent executor, façade (`tool()` / `sub_call()` / `budget()` / `emit()` / `note()`), the single-destination relay, the in-container exec client. | sandbox only (**no** project imports) |
| `src/lens/experiments/{arm,campaign,polarity}.py` | Arm fingerprint (written once, into `manifest.json`), campaign matrix (validated, expanded to subprocess jobs), the four polarity probes. | `config`, `models` |
| `src/lens/ui/{data,app}.py` | Read-only data layer (no Streamlit) and the Streamlit pages: recorded playback, live query via subprocess. | `runstore`, `replay`, `citations` |
| `src/lens/run.py` | Entry point: `python -m lens.run "<query>"`, `--list/--show/--resume/--fork/--export/--import`; the only module that builds the object graph. | `agent.runtime`, `runstore`, `tools`, `embeddings`, `reranking`, `kg_extract`, `connect` |

**Import direction (no cycles):** `models` ← `config` ← {`trajectory`, `clients`,
`chunking`, `dataset.*`, `db.*`, `embeddings`, `reranking`, `kg_*`}
← `agent.{budget,ledger,context,tool_registry,tool_server,sandbox_session,segmentation}`
← `agent.runtime` ← `run`. The sandbox quartet (`runner`, `rlm_api`, `relay`,
`exec_client`) and `kg_query` are the exception by design: they import **nothing**
from `lens`, because the image ships them without the package.

---

## 2. Module dependency graph (auto-generated)

Source: `docs/packages_lens.mmd` (pyreverse, regenerated 31 Oct 2026). Edges are
file-level imports. The graph **is** complete and correct: **55 analysed modules,
141 imports**, covering `agent/`, `sandbox/`, `tools/`, `experiments/`, `ui/` and
`run.py`.

```mermaid
classDiagram
  class lens {
  }
  class agent {
  }
  class budget {
  }
  class context {
  }
  class ledger {
  }
  class runstore {
  }
  class runtime {
  }
  class sandbox_session {
  }
  class segmentation {
  }
  class tool_registry {
  }
  class tool_server {
  }
  class chunking {
  }
  class clients {
  }
  class config {
  }
  class dataset {
  }
  class chunks {
  }
  class fetch {
  }
  class manifest {
  }
  class papers {
  }
  class paths {
  }
  class db {
  }
  class connect {
  }
  class schema {
  }
  class embeddings {
  }
  class experiments {
  }
  class arm {
  }
  class campaign {
  }
  class polarity {
  }
  class kg_extract {
  }
  class kg_gold {
  }
  class kg_postprocess {
  }
  class kg_query {
  }
  class kg_verify {
  }
  class models {
  }
  class observability {
  }
  class citations {
  }
  class replay {
  }
  class run_logging {
  }
  class trajectory {
  }
  class preflight {
  }
  class reranking {
  }
  class run {
  }
  class sandbox {
  }
  class exec_client {
  }
  class relay {
  }
  class rlm_api {
  }
  class runner {
  }
  class tools {
  }
  class evaluate {
  }
  class kg {
  }
  class retrieval {
  }
  class sql {
  }
  class ui {
  }
  class app {
  }
  class data {
  }
  agent --> segmentation
  budget --> lens
  budget --> config
  budget --> models
  context --> lens
  context --> config
  runstore --> lens
  runstore --> config
  runtime --> lens
  runtime --> budget
  runtime --> context
  runtime --> ledger
  runtime --> runstore
  runtime --> sandbox_session
  runtime --> segmentation
  runtime --> tool_registry
  runtime --> clients
  runtime --> config
  runtime --> arm
  runtime --> citations
  runtime --> trajectory
  runtime --> evaluate
  runtime --> retrieval
  sandbox_session --> lens
  sandbox_session --> config
  segmentation --> clients
  segmentation --> trajectory
  tool_registry --> lens
  tool_registry --> config
  tool_registry --> trajectory
  tool_server --> lens
  tool_server --> config
  clients --> lens
  clients --> config
  clients --> models
  clients --> trajectory
  config --> models
  dataset --> manifest
  dataset --> papers
  chunks --> lens
  chunks --> chunking
  chunks --> config
  chunks --> embeddings
  fetch --> manifest
  fetch --> paths
  manifest --> lens
  manifest --> config
  papers --> lens
  papers --> config
  papers --> embeddings
  paths --> lens
  paths --> config
  db --> connect
  db --> schema
  connect --> lens
  connect --> config
  schema --> lens
  schema --> config
  embeddings --> lens
  embeddings --> config
  arm --> lens
  arm --> config
  polarity --> models
  kg_extract --> lens
  kg_extract --> config
  kg_gold --> kg_postprocess
  kg_gold --> kg
  kg_postprocess --> lens
  kg_postprocess --> config
  kg_postprocess --> kg_extract
  kg_verify --> lens
  kg_verify --> config
  kg_verify --> models
  observability --> trajectory
  citations --> lens
  citations --> config
  citations --> replay
  citations --> trajectory
  replay --> lens
  replay --> context
  replay --> config
  replay --> trajectory
  run_logging --> lens
  run_logging --> config
  trajectory --> config
  preflight --> lens
  preflight --> config
  preflight --> paths
  reranking --> lens
  reranking --> config
  run --> lens
  run --> agent
  run --> runstore
  run --> runtime
  run --> sandbox_session
  run --> tool_registry
  run --> tool_server
  run --> clients
  run --> config
  run --> connect
  run --> embeddings
  run --> kg_extract
  run --> kg_verify
  run --> trajectory
  run --> preflight
  run --> reranking
  run --> kg
  run --> retrieval
  run --> sql
  runner --> rlm_api
  tools --> retrieval
  evaluate --> lens
  evaluate --> clients
  evaluate --> config
  evaluate --> models
  evaluate --> citations
  evaluate --> trajectory
  kg --> lens
  kg --> config
  kg --> kg_extract
  kg --> kg_postprocess
  kg --> kg_query
  kg --> kg_verify
  kg --> retrieval
  retrieval --> lens
  retrieval --> config
  retrieval --> embeddings
  retrieval --> reranking
  sql --> lens
  sql --> config
  app --> agent
  app --> runstore
  app --> replay
  app --> ui
  app --> data
  data --> lens
  data --> runstore
  data --> config
  data --> citations
  data --> replay
  data --> trajectory
```

> Read it for the **invariant**, not the detail: edges point *down* the layers
> (`run → agent → tools → config → models`) and never back up. The `lens` nodes are
> package `__init__` re-exports, and `observability → trajectory` is the same
> pattern.
>
> Three edges deserve a note. `runtime → arm` is the arm fingerprint
> (`experiments/arm.py`). `polarity → models` is the probe's `Question`/`JevResult`
> types, which is all it needs. And `run → {connect, embeddings, kg_extract,
> reranking, sql}` is the CLI's wiring: `run.py` is the only module allowed to touch
> every layer, because that is where the object graph is built.

---

## 3. Class map

> **Hand-authored, and here is why.** `pyreverse` cannot see how this harness is
> wired. It links an attribute only when it can *infer a value* — a constructed
> default (`Budget()`, `BudgetLimits()`, `Task()`) or an attribute-level annotation.
> LENS injects its collaborators through **keyword-only** parameters
> (`*, client: LensClient` … then `self.client = client`), which pyreverse ignores.
> Measured on the real package: **8 relationships across 52 classes**, two of them
> false positives (`JevResult --o DirectionVerifier`, inferred from a default-arg
> construction, and `Trajectory --o LensClient`, whose direction reads backwards).
> The raw output is kept as `docs/classes_lens.mmd`; the diagrams below are the ones
> to read.

### 3.1 Harness core — who owns whom

```mermaid
classDiagram
    class Runtime {
        run() RunResult
        handle_rpc(endpoint, payload) dict
        trace_ctx(actor) TraceCtx
        _run_loop() tuple
    }
    class Budget {
        can_continue() bool
        record_chat(result) None
        record_jev(result) None
        snapshot() dict
    }
    class BudgetLimits {
        max_iterations
        max_tokens
        max_cost_usd
    }
    class TaskLedger {
        create(requirement) Task
        update(task_id, status) Task
        snapshot() list
    }
    class Task {
        task_id
        status
        evidence_refs
    }
    class ToolRegistry {
        register(name, func) None
        manifest() str
        dispatch(name, args) ToolCallResult
    }
    class ToolSpec {
        name
        signature
        description
    }
    class SandboxSession {
        start() None
        exec(code) ExecResult
        health() bool
    }
    class Trajectory {
        store_blob(text) str
        load_blob(ref) str
        log_event(...) None
        totals() dict
        rebuild() RebuiltState
    }
    class TraceCtx {
        trace_id
        step_idx
        actor
    }
    class LensClient {
        chat(role, messages) CallResult
        jev(state, questions) JevResult
    }
    class DirectionVerifier {
        verify(triples, chunk_texts) tuple
    }
    class ModelSpec {
        id
        provider_only
    }
    class ToolServer {
        handle_request(method, path, payload, token)
    }

    Runtime *-- Budget : budget
    Runtime *-- TaskLedger : ledger
    Runtime *-- ToolRegistry : registry
    Runtime *-- SandboxSession : sandbox
    Runtime *-- Trajectory : trajectory
    Runtime *-- LensClient : client
    Budget *-- BudgetLimits : limits
    TaskLedger *-- Task : tasks
    ToolRegistry o-- ToolSpec : tools
    ToolServer ..> Runtime : handle_rpc
    SandboxSession ..> ToolServer : relay target port
    LensClient ..> Trajectory : logs every call
    LensClient ..> ModelSpec : role to pinned model
    Trajectory ..> TraceCtx : per-event context
    DirectionVerifier ..> LensClient : jev_call (built in run.py)
```

**Reading it.** `Runtime` is the only class that composes the others; everything it
needs arrives through the constructor, which is what makes `tests/test_runtime.py`
runnable with fakes and **no** Docker, DB or key. `DirectionVerifier` is the one
collaborator that is *not* owned by the runtime: `run.py:build_runtime` constructs
it with a `client.jev` closure and hands it to `build_kg`, which is why the arrow is
dashed and points at the client rather than at the runtime.

The other dashed edges are the ones a reader usually gets wrong:

- `ToolServer ..> Runtime` — the host RPC server holds a runtime and calls
  `handle_rpc`; the runtime never imports the server.
- `SandboxSession ..> ToolServer` — a *port number*, not an import: the session
  forwards the port the server bound to `lens-relay`, which is the sandbox's only
  route to the host.

### 3.2 Sandbox pair (a separate process, no project imports)

```mermaid
classDiagram
    class Executor {
        namespace
        run(code) dict
    }
    class RLMError {
        status
        error
    }
    class ToolServer

    Executor ..> ToolServer : JSON over lens-relay (not an import)
    Executor ..> RLMError : relays a raised façade error
```

The executor is **passive**: it runs a block in one persistent namespace and never
calls a model or the DB. `rlm_api` (the façade the generated code imports) raises
`RLMError`; the executor relays that as a structured envelope, so a failed *block*
is a 200 with `ok: false` while a failed *transport* is not.

Two things cross into that namespace that the diagram cannot show: the host tools,
bound as plain functions by `context.tool_bindings`, and — when the KG is on — the
traversal helpers, bound by `context.kg_bindings`, which also emits one
`rlm_api.note("kg used: …")` on the first call. That note is the ablation's use
signal.

### 3.3 Artifact types (the dataclasses that cross module boundaries)

```mermaid
classDiagram
    class Runtime {
    }
    class Trajectory {
    }
    class RunManifest {
        run_id
        status
        totals
        caps
        extra
    }
    class RunResult {
        answer
        cost_usd
        stop_reason
    }
    class Turn {
        turn
        completion
        observation
        stderr_full
        tools
    }
    class ToolStep {
        name
        args
        result_ref
        gate
    }
    class StageVerdict {
        verdict
        confidence
        tier
    }
    class CitationReport {
        citations
        unresolved
    }
    class Citation {
        token
        arxiv_id
        resolved
    }
    class RebuiltState {
        messages
        budget
        emits
        last_seq
    }
    class ExecResult {
        ok
        stdout
        stderr
        error
    }

    Runtime ..> RunResult : run()
    Runtime ..> RebuiltState : resume fold
    Trajectory ..> RebuiltState : rebuild()
    Trajectory ..> ExecResult : exec event
    Turn *-- ToolStep : tools
    ToolStep ..> StageVerdict : gate
    CitationReport *-- Citation : citations
    RunManifest ..> RunResult : totals written at finalize
```

Each of these is a **serialisation boundary**: `RunManifest` is `manifest.json`,
`Turn`/`ToolStep` are the viewer's timeline, `CitationReport` is `sources.md`,
`ExecResult` is the `/exec` envelope. `Runtime` and `Trajectory` appear here as
endpoints only; their ownership is in 3.1.

### 3.4 Raw pyreverse output (reference only)

`docs/classes_lens.mmd` — 52 classes, regenerated 31 Oct 2026. The complete
relationship set it could infer:

```
Runtime --> Budget : budget
Budget --> BudgetLimits : limits
TaskLedger --> Task : tasks
Runtime --> TaskLedger : ledger
RunView --> RunManifest : manifest
RunView --> CitationReport : citations
JevResult --o DirectionVerifier : jev_call     <- false positive
Trajectory --o LensClient : trajectory         <- direction reads backwards
```

The two `RunView` edges are real but trivial: the UI's view object is built from a
manifest and a citation report, and pyreverse sees those because they arrive through
annotated dataclass fields rather than constructor injection.

Do not paste the whole file into this section: it duplicates the artifact and the
two drift apart. Read it from `docs/classes_lens.mmd` when you need attribute-level
detail.

---

## 4. Runtime interactions (not visible in import graphs)

The important flows are **runtime**, not imports — this is the part a static tool
cannot produce.

### 4.1 Request path (live)

```mermaid
flowchart TD
    run["run.py"] --> runtime["agent/runtime.py<br/>orchestrator"]
    runtime --> clients["clients.py<br/>transport"]
    clients --> ornet(("OpenRouter API"))
    runtime --> seg["agent/segmentation.py<br/>one structured-JSON chat call"]
    seg --> qjson["work/query.json"]
    runtime --> tools["tools/*<br/>funnel · fetch · KG · gate"]
    tools --> db[("PostgreSQL + pgvector")]
    runtime --> traj["observability/trajectory.py"]
    clients --> traj
    runtime --> relay["lens-relay<br/>(one port, no egress)"]
    relay --> sbx["sandbox runner.py<br/>passive executor"]
```

The gate and the direction verifier both run **host-side** and both call the decision
layer through `clients.jev`, so the sandbox never sees either.

### 4.2 Host ↔ sandbox loop

The host pulls every turn; the sandbox is a **passive** executor. The DB and the
LLM are reached **only** from the host — the sandbox never holds the key or a DB
driver.

```mermaid
sequenceDiagram
    participant R as agent/runtime.py (host)
    participant S as sandbox runner.py
    participant API as OpenRouter (host)
    participant DB as tools → Postgres
    R->>API: chat(root context)
    API-->>R: generated code
    R->>S: POST code
    S->>R: rlm_api.tool("deep_match", …)
    R->>DB: query (lens_ro, read-only)
    DB-->>R: rows
    R-->>S: compact result
    S-->>R: stdout / return value
    R->>R: append observation, budget check, then loop or finish
```

> `db/connect.py` is used **only** by the host side of this loop (the tool layer),
> never imported by the sandbox.

### 4.3 The harness, as built

The harness is **built and live**: every module is filled and reviewed, and the loop
has executed end-to-end against the real corpus. Suite: **692 tests — 691 pass,
1 skip** (no key, zero cost); `ruff check` and `ruff format --check` clean.

Invariants fixed in the rework: the sandbox mounts **`<run_dir>/work`** only (run
artifacts live one level up and cannot be rewritten by generated code); caps are
**derived and ordered** (8 iterations · 40k context · 300k tokens derived from them ·
**$1.00** per query, raised from $0.50 on 26 Oct 2026 after the anchor spent 87 % of
it · 3 backtracks · 80 % wrap-up nudge · root output 8k · 120 s per exec block); a
reported `$/query` comes from `Trajectory.totals()`, not `Budget`; `emit()` returns a
container path; the KG traversal helpers ship in the image as `kg_query` **and** are
bound as plain functions in the namespace, so a model does not need an import to walk
the graph.

#### 4.3.1 — first live runs (25 Oct 2026)

| Step | Result |
|---|---|
| sandbox boundary | `verify_sandbox.py` **8/8** (host RPC 200 through the relay, egress blocked, timeout, artifacts hidden, no secrets) |
| plumbing (mini) | 8 iterations, 43–52 tool calls, ledger written, 46/46 citations resolved |
| crash-resume | killed mid-loop → `--list` sweeps to `interrupted` → `--resume` finishes; history a byte-identical prefix |
| portable bundle | export 1.13 MB / 138 blobs → import → replay **identical**, 0 `missing_refs`; a one-byte flip is rejected |
| gpt-5 E2E | 13/13 citations, `$0.436`, gate rejects unprovable global negatives (correctly) |
| five-query smoke | 5/5 pass `check_run`, 47/47 citations, `$0.2568` |
| artifacts | `answer.md` + `sources.md` + `report.md` per run |

**The RPC path was replaced by a relay.** `host.docker.internal` is **unreachable**
from an `internal: true` bridge on Docker Desktop/Windows (measured: `host-gateway` →
the WSL2 VM → `ENETUNREACH`; no DNS without it; the host's real address is
off-bridge). The sandbox therefore dials **`lens-relay`**, a container on
`sandbox-net` + `rpc-net` that forwards one hard-coded `host:port`. No port is
published (an internal network drops the mapping). Egress stays blocked and is still
probed after the relay exists. The contract that follows:

| Component | Responsibility | Promise |
|---|---|---|
| `agent/runtime.py` | Owns the loop, RPC handlers, final/partial answer. | Host-pulled; every turn logged; stops at caps. |
| `agent/context.py` | System prompt, tool manifest, budget, workspace files, compaction, tool **and KG** bindings. | Root context stays bounded; large data stays in the sandbox. |
| `agent/budget.py` | Iterations, tokens, cost, exec seconds, backtracks. | `budget()` is read-only; the orchestrator alone enforces caps. |
| `agent/ledger.py` | Atomic `tasks.json`. | The backtracking source; crash-recoverable. |
| `agent/segmentation.py` | The one structured-JSON call; writes `work/query.json`. | One writer; never re-segmented on resume. |
| `agent/tool_registry.py` | Manifest + dispatch. | Names/signatures/docstrings only; args hashed; full result blobbed. |
| `agent/tool_server.py` | Host RPC endpoints. | The sandbox reaches only `/tool`, `/sub_call`, `/budget`, `/emit`, `/note`; per-run token. |
| `agent/sandbox_session.py` | Compose lifecycle + `/exec` client. | One container per run (plus the relay); mounts `<run_dir>/work` only; forwards the bound port as `LENS_RELAY_TARGET_PORT`; strips secrets from the child env. |
| `sandbox/runner.py` | Passive persistent executor. | Variables survive steps; never calls an LLM; 120 s timeout. |
| `sandbox/rlm_api.py` | Façade. | No key/DB; depth-1 only; no recursion. |
| `kg_query.py` | Pure KG traversal (`kg_neighbors` / `kg_triples` / `kg_subgraph`). | Copied into the image; `tools/kg.py` re-exports it (one source of truth); bound as functions, first call noted. |
| `tools/evaluate.py` | Three-tier stage gate + claim-level final gate. | Verdict + failing criterion; logged; fail-soft to `uncertain`. |
| `tools/sql.py` | Gated T3. | Read-only; off by default; forced off in recorded runs **and** in any mode without a corpus. |
| `agent/runstore.py` | `manifest.json` (identity/index), `resume.json` (cache), `list_runs`, `sweep_interrupted`, `fork_run`, bundle export/import. | Tolerant readers; atomic writes; artifacts in `run_dir`, never `work/`; an export refuses a bundle carrying a secret. |
| `run.py` | CLI wiring. | `python -m lens.run "<query>"`; `--list` / `--show` / `--resume` / `--fork` / `--export` / `--import` / `--no-replay`. |

**Protocol pinned by the scaffold**

- Final answer = explicit **`FINAL_ANSWER:`** sentinel, recognised only **outside** a
  fenced block (a sentinel printed from Python is code, not an answer).
- Host and runner use Python **stdlib HTTP**; the sandbox calls the host through
  **`lens-relay`**, never `host.docker.internal`.
- `docker-compose.yaml` declares `lens-pg`, `lens-repl` **and** `lens-relay`; the DB
  is on its own network, `sandbox-net` is `internal: true`, and only the relay sits
  on `rpc-net` (its trusted side).
- The sandbox holds **no OpenRouter key and no DB driver**, and the docker CLI child
  that starts it no longer inherits the key either.
- Fill order (historical, all done): budget → ledger → context → registry → tool
  server → `rlm_api` → `runner` → `sandbox_session` → `runtime` → `run.py` →
  evaluate/sql → **runstore**.

**One-time migration:** stop/remove the old manual `lens-pg` container before
compose starts; the `lens-pgdata` volume preserves data. Verify the host boundary in
order — host RPC reachable from the container **through the relay** → general egress
**blocked** → `/exec` reachable from the host via `docker compose exec`. Do **not**
drop `internal`; the no-egress property is a claim, and the fixed-subnet fallback
from the early notes **does not work** on Docker Desktop.

---

### 4.4 Session persistence

**The log is the session.** Every turn appends a root `chat` event (whose
`prompt_ref` is the exact request body), an `exec` event (`code_ref` +
`stdout_ref`, plus `stderr_ref` when the block failed) and a `budget` snapshot, so the
root message list, the counters and the namespace map are **folded** from
`trajectory.jsonl` (`Trajectory.rebuild() -> RebuiltState`) rather than duplicated
into a checkpoint. That is the whole design: a session cannot drift from its own
record.

| File | Role | Authoritative? |
|---|---|---|
| `manifest.json` | identity + status + index (id, query, parent, models, caps, totals, provenance, artifacts) | yes (which runs exist / what state) |
| `trajectory.jsonl` | the event log | yes (state) |
| `resume.json` | cache: emit map (`var -> ref`) + workspace hashes + `namespace_complete` | no — rebuildable |
| `tasks.json` / `work/` / `answer.md` | ledger / sandbox workspace / answer | yes |

Resume = **rebuild**, never restore a container: new sandbox → context files
rewritten into `work/` → `query.json` read from disk (re-segmenting would spend a
call and could silently change the retrieval) → variables restored from **emit
blobs only** → one `run.resume` note appended → continue at `last_seq`. A variable
that was never emitted sets `namespace_complete: false` and the root is told to
re-derive it — a labelled degraded resume. A corrupt `query.json` is not an empty
one: the resume says so in the note.

Recorded replay serves a tool result from `(tool, args_hash) -> result_ref` when
present, tagged `"replayed": true`, so a replay is deterministic **and** offline.
A fork is provenance (new run, `parent_run_id`, **fresh** log); resume is
continuation. `--list` / `--show` build no runtime — they are what the UI calls.

Tests: `tests/test_runstore.py`, plus `rebuild` in `test_trajectory.py` and
resume/replay in `test_runtime.py` / `test_run_wiring.py`.

### 4.5 Demo playback + portable bundle

A full run needs a GPU (embeddings, reranker, GLiNER), the DB, Docker and a key, so a
live demo is not reproducible on an examiner's laptop — and a video cannot be
interrogated. The recording can: `observability/replay.py` folds the log into
**one `Turn` per `step_idx`** (completion, code, `stdout_full`, the **observation
the model saw**, `tools[]` with args/latency/gate, `emits`, `budget`,
`missing_refs`), and `replay_summary()` supplies the header numbers.

`stdout_full` is the **observation**, not the runner's raw capture: the trajectory
records what the model saw, and the viewer labels it that way.

| Mode | What it is | Needs |
|---|---|---|
| **A. scrubbed playback** (the demo) | viewer over a recording; `Step`/`Play` move a cursor | Python + Streamlit |
| **B. deterministic harness replay** | re-runs the loop serving recorded tool results | Docker (no key/GPU) |
| **C. live run** | the pipeline | GPU, DB, key |

Blobs live in **one global deduplicated store** (`<runs_root>/blobs/`), so copying
a run directory alone leaves every ref dangling. `--export` writes
`<run_id>.lens/` with the artifacts, `work/` (minus `emits/`) and **only the
referenced blobs** plus `bundle.json` (sizes, `trajectory_sha256`, missing refs);
every referenced blob is scanned for the configured secrets first, and a hit refuses
the export. `--import` re-hashes every blob and refuses on mismatch. `args_ref` on
tool events is what makes a call renderable (`args_hash` alone cannot be shown).

Tests: `tests/test_replay.py`; bundle contracts in `test_runstore.py`.

---

## 5. Notes and caveats

- The generated package graph is **complete** (55 modules); the class map is curated
  by hand for the reason in §3, and the raw pyreverse output is kept as
  `docs/classes_lens.mmd`. Regenerating either file is one command from the repo
  root — `uv run --with pylint --no-project pyreverse -o mmd -d docs -p lens src/lens`
  — after which `packages_lens.mmd` is pasted into §2 and `classes_lens.mmd` is left
  as reference.
- Static graphs show **imports and classes**, not the host↔sandbox RPC, the DB access
  path, or the per-run data flow; §4 covers those.
- One-way import direction is the invariant to keep; a future
  [`import-linter`](https://import-linter.readthedocs.io/) contract could enforce it
  in tests (not added yet).
- This file is **not** generated, but §2 is a paste of a generated artifact, so it can
  go stale — that is what happened between 18 Oct and 26 Oct. When a milestone lands,
  re-run the command above and re-paste §2; the hand-authored blocks change only when
  the ownership graph changes.
