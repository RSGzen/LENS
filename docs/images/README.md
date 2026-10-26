# Screenshots

The README references the six images below. Capture them from a **recording**, so
no API key, query text or machine path is visible in the frame.

Capture settings:

- Window width 1400 px or wider. Below roughly 1200 px the viewer stacks its
  columns, which loses the turn-card layout these shots exist to show.
- The viewer's default dark theme, browser at 100 % zoom, no browser chrome in the
  frame.
- PNG, cropped to the application only.
- Run the viewer against an imported bundle, so the page is read-only:
  `$env:LENS_RUNS_ROOT='D:\lens-data\runs-import'; streamlit run src/lens/ui/app.py`
- Use the run `m9-s3-plumbing-6` (it has nine turns, 43 tool calls and 46 resolved
  citations).

| File | What it must show |
|---|---|
| `01-recorded-runs.png` | The Recorded Runs page at rest: the run table with a display name and status, the "Recorded playback" caption, and the run header with status, turns, cost and the recorded cap. |
| `02-turn.png` | One turn card: the reasoning paragraph, the `Code Written (N lines)` expander closed, one tool row showing `name(args)`, latency in seconds, and the gate badge. The `Full Model Message` expander closed below it. |
| `03-payload.png` | The same turn with `Full Payload` ticked on a `rerank_chunks` call, so the JSON is visible with `chunk_uuid` and `arxiv_id` fields, plus the download button. This is the shot that shows the evidence is inspectable. |
| `04-answer.png` | The Answer section: the answer text and the citation table with the `OK` column filled, so a reader can see that citations resolve to recorded payloads. |
| `05-meters.png` | The meters block and the top of the evidence panel: the prompt-token chart, the cost bar against the recorded cap, and the count of hydrated chunks. |
| `06-live-query.png` | The Live Query page with a query typed in and the CLI output streaming into the status box, or the finished run summary line. |

After adding the files, check that every path in the README's Screenshots table
matches a file name here.
