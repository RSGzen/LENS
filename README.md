# LENS

Literature Evidence & Narrative Synthesis — an agentic systematic-literature-review
harness over a GROBID-structured arXiv cs.AI corpus.

> **Status (26 Oct 2026):** – built ( = harness E2E, 5-query smoke, Tier-1
> anchor, portable bundles). ** UI filled** : `ui/data.py` and
> `ui/app.py` are implemented and tested offline (`ui_app` self-skips without the
> extra). Review rounds 1–2 applied : local display names, reasoning
> first then collapsed code then the full message, and the pages are **Recorded
> Runs** / **Live Query**. Suite: **487 tests — 486 pass, 1 skip**.

> ** invariants that are easy to break:** the sandbox workspace is
> `<run_dir>/work` (the run artifacts — `trajectory.jsonl`, `tasks.json`,
> `manifest.json`, `resume.json`, `answer.md` — live one level up and are *never*
> mounted); the caps are **derived and ordered** (8 iterations · 40k context ·
> 300k tokens · **$1.00**, raised from $0.50 on 26 Oct 2026 —, and
> recordings still carry $0.50 in their manifest, with an 80 % wrap-up nudge); a
> reported `$/query` comes from `Trajectory.totals()`, not from `Budget`;
> `emit()` returns a **container** path; the KG traversal helpers ship into the
> image as `kg_query`.
>
> ** invariants:** **the log is the session** — root messages, budget
> counters and the emit map are folded from `trajectory.jsonl`
> (`Trajectory.rebuild`), never duplicated; `manifest.json` is identity/index,
> `resume.json` is a cache; resume **rebuilds** (fresh container, variables from
> emit blobs only) and appends `run.resume`; a fork starts a **fresh** log;
> `--list` / `--show` build no runtime.
>
> **+ invariants:** the demo is **mode A — scrubbed playback** of a recorded
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
  ui/             data (read-only) · app (Streamlit) (M10)
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

 scaffold fill order (run the matching test file after each):

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
`G:\My Drive\SyncOpenCode\AURA\Drafts\ /.

## UI

Two pages over the same artifacts ( ):

- **Recorded Runs** — read-only recorded playback: run list → turn timeline →
  meters → answer + resolved citations. Needs **no GPU, no DB, no key**: it is mode
  A (scrubbed playback of a recording), and the page says so.
- **Live Query** — the live path: shells out to `python -m lens.run` and streams
  stdout. Needs the workstation stack (GPU, DB, Docker, key).

Runs can be renamed locally (`aliases.json` beside the blob store — a viewer label,
never run identity).

```powershell
uv sync --extra ui
streamlit run src/lens/ui/app.py

# demo laptop: point at an imported bundle instead
$env:LENS_RUNS_ROOT='D:\lens-data\runs-import'; streamlit run src/lens/ui/app.py
```

`src/lens/ui/data.py` is the read-only data layer (no Streamlit import, tested by
`tests.test_ui_data`); `src/lens/ui/app.py` is the page, filled in units U1–U9 in
the roadmap's order. The UI adds **no pipeline logic** : every number comes
from `manifest.json`, `trajectory.jsonl`, `answer.md` and the blobs.
