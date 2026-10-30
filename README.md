![LENS](docs/images/banner.svg)

LENS is an agentic pipeline for systematic literature reviews over a
GROBID-structured arXiv corpus. It takes a research question, retrieves evidence
in two stages, optionally builds a knowledge graph over the retrieved chunks, and
runs a recursive language model loop inside a Docker sandbox that writes and
executes Python against that evidence. The output is a synthesised answer in
which every claim carries a `chunk_uuid` that resolves back to the retrieved text.

```text
      question
         |
   segmentation (one JSON call -> objectives / problems / constraints / requirements)
         |
   shallow_match        HNSW over abstracts            k = 100 candidates
   rerank_papers        cross-encoder                  n = 20 papers
   deep_match           HNSW over chunks in those n    C = 100 candidates
   rerank_chunks        cross-encoder                  c = 100 chunks (+ paper rows)
         |
   build_kg             GLiNER + GLiREL over those chunks -> networkx variable
         |
   RLM loop             root model writes Python, sandbox runs it,
                        host tools answer over RPC, observation comes back
         |
   answer.md + sources.md + trajectory.jsonl + manifest.json
```

## Contents

- [What it does](#what-it-does)
- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Recorded runs](#recorded-runs)
- [The viewer](#the-viewer)
- [Screenshots](#screenshots)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Safety properties](#safety-properties)
- [Limitations](#limitations)
- [Credits](#credits)
- [License](#license)

## What it does

Most retrieval systems answer a query. LENS is built around a *review*: the unit
of work is a question that needs several sources compared, and the artefact is an
answer whose evidence can be audited afterwards.

A run is recorded as it happens. Every model call, tool call, code block and
budget snapshot is appended to `trajectory.jsonl`, and every payload is stored
content-addressed, so the run can be replayed turn by turn later without a GPU, a
database, or an API key. That makes the system inspectable after the fact rather
than only while it runs.

The retrieval funnel and the recursion are both bounded on purpose: the question
under test is whether orchestration over structured chunks beats a flat prompt at
the same token budget, not whether a bigger context window helps.

## Features

- Two-stage retrieval with a hard cap at every step: `k=100` candidates,
  `n=20` papers, `C=100` chunk candidates, `c=100` chunks into the answer path.
- A cross-encoder at both stages, so abstracts and chunks are ranked by the same
  notion of relevance.
- A query-time knowledge graph over the retrieved chunks only. Entities and
  relations are extracted with GLiNER and GLiREL, post-processed (span filter,
  conservative coreference, relation-slot typing, directionality), and handed to
  the model as a `networkx` variable instead of being pasted into the context.
- A Docker sandbox with no network, no API key and no database driver. The model's
  Python runs there; every effect it has is an RPC to the host.
- Per-query caps enforced in one place: 8 iterations, 300,000 tokens, 1.00 USD,
  120 seconds per code block, 3 retries.
- An optional decision layer (JEV) that gates stage output and the final answer.
  It is switched off in one configuration flag for the ablation arm.
- Every run is replayable: manifest, JSONL trajectory, content-addressed blobs, and
  a portable `.lens` bundle whose blobs are re-hashed on import.
- A Streamlit viewer with two pages: recorded playback of a finished run, and a
  live query box. The playback page needs no GPU, database or key.

## How it works

### Retrieval

`tools/retrieval.py` implements the funnel as four tools the model can call. Each
stage narrows the candidate set, and the chunk cap is global rather than per paper,
so a single long paper cannot fill the answer context.

### The loop

`agent/runtime.py` owns the loop. The root model receives a system prompt, the
query, the segmentation buckets and a manifest of available tools. It replies with
Python. The code is sent to `lens-repl`, a container whose only job is to execute
blocks and keep the namespace between them. When the code calls a tool, the call
travels from the container through a relay to the host, where the tool runs against
the corpus; the result is summarised and returned. The model then sees a compact
observation and writes the next block.

The loop ends at a live budget cap, at an explicit `FINAL_ANSWER:` sentinel, or at
a groundedness rejection that persists to the iteration cap. In every case the run
is finalised with a labelled stop reason, so a partial answer is reported as a
partial answer.

### The sandbox boundary

The container has `internal: true` networking, a read-only root filesystem, and no
credentials. It reaches the host through `lens-relay`, a process whose only job is
to forward one hard-coded port. The workspace it mounts is `<run_dir>/work`, so the
run artifacts, which live one directory up, are not writable from inside the
sandbox: a generated block cannot rewrite its own evidence.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker Desktop (the corpus database and the sandbox run in containers)
- An OpenRouter API key (the models are called through OpenRouter)
- The arXiv cs.AI corpus: a metadata JSON dump and one GROBID TEI XML file per paper
- A CUDA GPU is needed for the embedding, reranking and NER models. The viewer does
  not need one.

## Installation

```powershell
git clone https://github.com/RSGzen/LENS.git
cd LENS
uv sync
```

Copy `.env.example` to `.env` and fill in the key and the database passwords. The
corpus paths have sensible defaults (`<data_root>/corpus`), so set them only to
reuse a corpus that already exists elsewhere.

```text
OPENROUTER_API_KEY=...
LENS_DB_PASSWORD=...
LENS_DB_RO_PASSWORD=...
```

Check the environment before anything else runs. Every pipeline script runs the same
check itself, so this is also a way to see what it will report:

```powershell
python scripts/check_env.py                       # everything
python scripts/check_env.py --need api_key,db     # just what a live run needs
```

It reports where `.env` was found and whether each requirement is present, with the
API key reduced to a fingerprint. The exit status is 2 when something is missing, so
it also works as a preflight in a shell chain or a CI step.

Runtime data (runs, logs, cache, manifests, and by default the corpus) goes to
`LENS_DATA_ROOT`, which defaults to the platform data directory.

### Build or reuse a corpus

The corpus is not redistributed here: arXiv's terms forbid serving third-party full
text, and most papers carry the non-exclusive licence rather than a permissive one.
What the repository ships is the recipe.

Download the metadata dump from
[Kaggle: Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)
(`arXiv-metadata-oai-snapshot.json`, one JSON object per line, roughly 4 GB), then
either put it at `<data_root>/corpus/metadata.jsonl` or point
`LENS_ARXIV_METADATA` at it. Start GROBID once, and let the script do the rest:

```powershell
docker run -d --restart unless-stopped --init -m 12g `
    -e JAVA_OPTS=-Xmx7G -p 8070:8070 lfoppiano/grobid:0.8.0

python scripts/fetch_corpus.py status                       # what is already on disk
python scripts/fetch_corpus.py fetch --limit 100 --dry-run  # count, download nothing
python scripts/fetch_corpus.py fetch                        # download, convert, burn
python scripts/fetch_corpus.py manifest                     # metadata + TEI -> manifest.json
```

`fetch` streams the dump, keeps `cs.AI` papers in the id-year window
(`--years`, default 15-26), downloads each PDF from the public arXiv mirror,
converts it with GROBID, writes `<arxiv_id><version>.tei.xml`, and deletes the PDF.
It is safe to re-run: a paper with a TEI file is skipped, and a PDF without one is
converted instead of downloaded again. Permanent failures go to
`<pdf_dir>/failed_papers.txt`, and a retry rewrites that file with what is still
failing:

```powershell
python scripts/fetch_corpus.py retry
```

If you already have PDFs, `python scripts/fetch_corpus.py grobid --pdf-dir <dir>`
converts them without touching the network. Names must be
`<arxiv_id><version>.pdf`: that stem is the join key between the manifest, the TEI
directory and every chunk row.

Then load the corpus, and create the schema (the schema step refuses to run while
the corpus directory is empty, so that a wrong path is caught before any loading):

```powershell
docker compose up -d lens-pg
python scripts/setup_db.py             # schema; checks the corpus first
python scripts/load_papers.py          # embed every manifest abstract -> papers
python scripts/load_chunks.py          # chunk + embed every paper -> chunks
python scripts/build_indexes.py        # HNSW indexes, after the bulk load
```

## Usage

Run one review:

```powershell
python -m lens.run "How is long-context aggregation evaluated on OOLONG?"
```

The command prints the answer, then the run id, cost, iteration count and stop
reason, then the artifact paths.

Inspect a finished run without touching the database:

```powershell
python -m lens.run --list
python -m lens.run --show <run_id>
python scripts/check_run.py <run_id>        # asserts the run's contract
python scripts/run_report.py <run_id>       # writes a static Markdown report
```

Resume an interrupted run, or start a child run that records its parent:

```powershell
python -m lens.run --resume <run_id>
python -m lens.run --fork <run_id> --run-id ui-20261026-1200 "the same question"
```

Move a run to another machine as a single directory:

```powershell
python -m lens.run --export <run_id> --destination D:\demo
python -m lens.run --import D:\demo\<run_id>.lens
```

## Recorded runs

Each run lives in `runs/<run_id>/`:

| File | Contents |
|---|---|
| `manifest.json` | identity, status, models, caps, totals, provenance |
| `trajectory.jsonl` | every chat, tool, code, emit and budget event, append-only |
| `tasks.json` | the ledger the model writes while it works |
| `answer.md` | the answer, in the model's own words |
| `sources.md` | the citation trace, generated from the recorded payloads |
| `resume.json` | cache: emitted variables and workspace hashes |
| `work/` | the sandbox workspace for this run |
| `<run_id>.lens` | optional portable bundle (export) |

Blobs are shared across runs in `runs/blobs/`, named by the SHA-256 of their
contents. A bundle contains only the blobs its own events reference, and importing
one re-hashes every blob against its filename before installing anything, so an
edited or truncated bundle is refused rather than rendered.

## The viewer

```powershell
uv sync --extra ui
streamlit run src/lens/ui/app.py
```

To view recordings on a machine with no corpus, no database and no key, point the
viewer at a directory of imported runs:

```powershell
$env:LENS_RUNS_ROOT='D:\lens-data\runs-import'
streamlit run src/lens/ui/app.py
```

The two pages are deliberately different in what they can do:

- **Recorded Runs** replays a finished run from its own trajectory: run list, turn
  timeline with the code and tool calls, an evidence panel that fills as playback
  advances, and the answer with its resolved citations. It has no live controls at
  all, and every view is captioned as a recording.
- **Live Query** runs the pipeline for real and streams the CLI output into the
  page. It needs the GPU, database, Docker and key.

Runs can be renamed in the viewer. The name is stored in `aliases.json` next to the
run store and never enters the run's own artifacts, so a label cannot change what
the run recorded.

## Screenshots

Screenshots come from one recorded run (`m9-s3-plumbing-6`, imported as a bundle),
so the pages show a recording rather than a live run.

**The Recorded Runs page: the run list, the recorded-playback caption, and the run
header with its cost against the cap the run was created with.**

![Recorded Runs page](docs/images/recorded-runs.png)

**One turn of the trajectory: the model's reasoning, the code it wrote in a
collapsed expander, and the tool calls with their latency.**

![A turn in the timeline](docs/images/turn.png)

**The full payload of a `rerank_chunks` call, opened on demand, showing the chunk
ids the answer cites.**

![Full tool payload](docs/images/payload.png)

**The answer, followed by the citation table. Every cited chunk that a recorded
payload explains is marked `OK`; unresolved ids would be listed here too.**

![Answer and citations](docs/images/answer.png)

**Playback controls and the meters: prompt tokens per turn, cost against the
recorded cap, and the budget counters for the selected turn.**

![Controls and meters](docs/images/meters.png)

**The Live Query page. It runs the pipeline for real and streams the CLI output
into the page.**

![Live Query page](docs/images/live-query.png)

## Project layout

```text
src/lens/
  config.py          settings, path resolution, model registry, caps
  models.py          typed request/response objects
  clients.py         OpenRouter transport (chat + decision layer)
  embeddings.py      nomic-embed-text wrapper
  reranking.py       bge-reranker cross-encoder wrapper
  chunking.py        GROBID TEI -> section-typed chunks
  kg_*.py            knowledge-graph extraction, post-processing, querying
  dataset/           manifest, papers, chunks (corpus build)
  db/                connection layer and pgvector schema
  tools/             retrieval funnel, knowledge graph, gated SQL, evaluation
  agent/             loop, budget, ledger, tools, sandbox session, segmentation
  sandbox/           container image, executor, RPC façade, relay
  observability/     trajectory, replay, citation resolution
  experiments/       arm fingerprints and campaign matrices
  ui/                Streamlit viewer and its read-only data layer
  run.py             command line entry point
scripts/             corpus build and loads, probes, gates and reports
tests/               no-key unit and contract tests
docs/                architecture, ontology, image rebuild notes
```

## Tests

```powershell
uv run ruff check .
uv run ruff format --check .
python -m unittest discover -s tests
```

The suite needs no API key, no database and no Docker. Tests that touch the
database are excluded from discovery and live in `tests/DESTRUCTIVE_db_tests/`;
they drop the `papers` and `chunks` tables, so run them only against a throwaway
database.

Two checks need the stack running:

```powershell
python scripts/verify_sandbox.py   # container boundary: health, egress, persistence, mount
python scripts/check_run.py <run_id>
python scripts/check_env.py        # secrets, database credentials, corpus location
```

## Safety properties

These are enforced by tests rather than by convention:

- The sandbox holds no API key and no database driver, and general egress from it
  fails.
- The run artifacts are not visible inside the sandbox.
- The per-query caps are enforced in one place and reported by name.
- The reported cost is read from the event log, not from the in-memory counter.
- The answer's citations are resolved against recorded payloads; an id that no
  payload explains is listed as unresolved rather than counted as supported.

## Limitations

- The corpus is arXiv cs.AI with GROBID XML. Papers without parseable TEI are
  skipped, and scanned PDFs are out of scope.
- The retrieval funnel is linear in `k`, so a larger candidate set costs
  proportionally more time.
- The decision-layer thresholds ship empty, which means its first tier is skipped
  until a calibration curve exists. The gate then reports `uncertain` and the
  orchestrator decides.
- The viewer is single-user and runs one live query at a time.
- Entity resolution in the knowledge graph is exact-string only. Fuzzy matching is
  not implemented.

## Credits

LENS stands on these projects:

- [GROBID](https://github.com/kermitt2/grobid) for structured full text
- [pgvector](https://github.com/pgvector/pgvector) for vector search in Postgres
- [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) for
  embeddings
- [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) for
  cross-encoder ranking
- [GLiNER](https://github.com/urchade/GLiNER) and
  [GLiREL](https://github.com/jackboyla/GLiREL) for entity and relation extraction
- [Streamlit](https://streamlit.io/) for the viewer
- [OpenRouter](https://openrouter.ai/) for model access

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
