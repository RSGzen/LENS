# LENS

Literature Evidence & Narrative Synthesis — an agentic systematic-literature-review
harness over a GROBID-structured arXiv cs.AI corpus.

> **Status (20 Oct 2026):** M0-M7 logic built; **M8 harness scaffolded and
> reworked** (LOG-91/LOG-92), **M8.5 session persistence** and **M8.5+ demo
> playback + portable bundle** scaffolded (LOG-93/LOG-95). All harness logic
> bodies are student-owned (scaffold-and-fill, LOG-33). Suite: **325 tests — 235
> pass, 90 contract tests fail with `NotImplementedError`** until filled.
> Fill progress: **`budget` + `ledger` + `trajectory` done** (3/15, LOG-96/97/98).

> **M8 invariants that are easy to break:** the sandbox workspace is
> `<run_dir>/work` (the run artifacts — `trajectory.jsonl`, `tasks.json`,
> `manifest.json`, `resume.json`, `answer.md` — live one level up and are *never*
> mounted); the caps are **derived and ordered** (8 iterations · 40k context ·
> 300k tokens · $0.50, with an 80 % wrap-up nudge); a reported `$/query` comes
> from `Trajectory.totals()`, not from `Budget`; `emit()` returns a **container**
> path; the KG traversal helpers ship into the image as `kg_query`.
>
> **M8.5 invariants:** **the log is the session** — root messages, budget
> counters and the emit map are folded from `trajectory.jsonl`
> (`Trajectory.rebuild`), never duplicated; `manifest.json` is identity/index,
> `resume.json` is a cache; resume **rebuilds** (fresh container, variables from
> emit blobs only) and appends `run.resume`; a fork starts a **fresh** log;
> `--list` / `--show` build no runtime.
>
> **M8.5+ invariants:** the demo is **mode A — scrubbed playback** of a recorded
> run (no GPU/DB/key), never presented as a live run; `replay_turns()` groups by
> `step_idx` and distinguishes the **observation the model saw** from the full
> stdout; blobs live in one global deduplicated store, so a portable run is a
> `<run_id>.lens` **bundle** (`--export`/`--import`, only referenced blobs,
> hash-verified). `args_ref` must exist in the log **before** a run is recorded
> for the demo.

## Layout

```text
src/lens/
  config.py · models.py · clients.py · embeddings.py · reranking.py · kg_*.py
  dataset/        M1 manifest · M3 papers · M4 chunks
  db/             M2 connect · schema
  observability/  trajectory · run_logging
  tools/          retrieval · kg · evaluate · sql
  agent/          segmentation (M6) · runtime/context/budget/ledger/tool_* (M8)
  sandbox/        runner · rlm_api · Dockerfile · requirements.txt (M8)
  run.py          CLI entry (M8/M9)
tests/            no-key unit/contract tests
tests/DESTRUCTIVE_db_tests/   WIPES papers/chunks — run only on purpose
scripts/          live/setup/probe scripts
docs/             ontology · ARCHITECTURE
```

## Setup

- Python **3.12** via `uv`; venv at `.venv`.
- Runtime data outside the repo: `D:\lens-data` (runs, manifests, cache).
- Secrets: `OPENROUTER_API_KEY` in the OS environment (or a gitignored `.env`).
  Never commit keys.
- DB: `pgvector/pgvector:pg16` on host port **5433** (native PG17 owns 5432);
  named volume `lens-pgdata`.
- Sandbox: `lens-repl` image built from `src/lens/sandbox/Dockerfile`.

```powershell
uv sync
docker compose up -d lens-pg
```

> **One-time migration:** stop/remove the old manual `lens-pg` container before
> compose starts; the `lens-pgdata` volume preserves data.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

M8 scaffold fill order (run the matching test file after each):

1. `agent/budget.py` — `tests.test_budget`
2. `agent/ledger.py` — `tests.test_ledger`
3. `agent/context.py` — `tests.test_context`
4. `agent/tool_registry.py` — `tests.test_tool_registry`
5. `agent/tool_server.py` — `tests.test_tool_server`
6. `sandbox/rlm_api.py` — `tests.test_rlm_api`
7. `sandbox/runner.py` — `tests.test_runner`
8. `agent/sandbox_session.py` — `tests.test_sandbox_session`
9. `agent/runtime.py` — `tests.test_runtime`
10. `run.py` — live smoke
11. `tools/evaluate.py` / `tools/sql.py` — `tests.test_evaluate` / `tests.test_sql`

Then:

```powershell
python scripts/verify_sandbox.py   # health, persistence, timeout, mount, no key
python -m lens.run "<query>"       # one live query (M8/M9 verify)
```

## Contracts

- Host-driven, sandbox-executed: `agent/runtime.py` owns the loop; `sandbox/runner.py`
  is a **passive persistent executor**; `sandbox/rlm_api.py` is the façade.
- Sandbox holds **no API key and no DB driver**; every effect is an RPC to the host.
- One `lens-repl` container per session; workspace-only RW mount.
- Depth 1: `sub_call` is a plain GPT-5-mini call; **no recursion**.
- Final answer = explicit `FINAL_ANSWER:` sentinel.
- T3 `query(sql)` is dev-only: off by default, forced off when recorded.
- Trajectory: `runs/<run_id>/trajectory.jsonl` + content-addressed blobs.

Full host/sandbox contracts: `docs/ARCHITECTURE.md` and
`G:\My Drive\SyncOpenCode\AURA\Thesis Drafts\Tech Stack (Draft).md` §4a/§4m.
