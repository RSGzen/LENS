"""LENS UI (M10) — a thin Streamlit viewer over recorded runs.

Spec: ``MVP Build Roadmap.md`` §4 (M10a/M10b); contract: ``Tech Stack (Draft).md`` §4p.

The UI adds **no pipeline logic** (T-06). Every number it shows is read from
``runs/<run_id>/`` through :mod:`lens.ui.data`: the manifest, the trajectory folded
by :func:`lens.observability.replay.replay_turns`, ``answer.md``, and the
content-addressed blobs behind each tool result.

Two pages, deliberately:

- **Runs** — read-only: run list → recorded playback. It has **no live-run
  button at all**, because mode A is a *recording*, not a run (§4n.1). The banner
  says so on every view.
- **Run** — the live path: shell out to ``python -m lens.run`` and stream its
  stdout. It needs the workstation stack (GPU, DB, Docker, key); on a laptop it
  fails at the press, which is the honest state rather than a disabled mystery.

Units are kept in the roadmap's order (U1 list · U2 header · U3 controls + tick ·
U4 timeline · U5 turn card · U6 meters · U7 evidence · U8 answer · U9 live) so each
one can be reviewed on its own.

Run it::

    uv sync --extra ui
    streamlit run src/lens/ui/app.py
    # the demo laptop (no GPU, no DB, no key) points at an imported bundle:
    LENS_RUNS_ROOT=D:\\lens-data\\runs-import streamlit run src/lens/ui/app.py

Editing this file hot-reloads on the next interaction. Editing ``lens/ui/data.py``
does **not**: the app imports it once, so restart the server (a stale module reads
as "module has no attribute ..."). Display names live in ``<runs_root>/aliases.json``
— local to the viewer, never inside a run directory or a bundle.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from lens.agent import runstore
from lens.observability.replay import ToolStep, Turn
from lens.ui import data

#: Playback cadence: one turn per fragment tick (§4n.1). ``st.fragment`` needs
#: Streamlit >= 1.37 — a full-page rerun per tick would make ``Play`` unusable.
PLAY_SECONDS = 1.5

#: ``Show full payload`` renders the JSON only when toggled: a collapsed
#: ``st.expander`` still builds its DOM, and a ``build_kg`` node-link is the
#: largest blob in a run.
PAYLOAD_PREVIEW_CHARS = 20_000

#: How much of a turn's *reasoning* (the completion minus its code fences) is
#: shown outside the expander. The full message stays one click away.
REASONING_CHARS = 700

#: A fenced block inside a model completion: ```lang ... ```
_FENCE_RE = re.compile(r"```[^\n]*\n.*?(?:```|$)", re.DOTALL)

#: ``st.session_state`` contract. Every unit reads and writes these keys and
#: nothing else, so a page can be filled (or replaced) without touching another.
STATE: dict[str, Any] = {
    "page": "Runs",      # "Runs" | "Run"
    "run_id": None,      # the selected run; mirrored in ``st.query_params["run"]``
    "cursor": 0,         # the last turn shown (playback position)
    "playing": False,    # Play is running (drives ``st.fragment(run_every=)``)
}

RUN_STATUSES: list[str] = ["all", "completed", "partial", "failed", "interrupted", "running"]


# --------------------------------------------------------------------- glue
def _init_state() -> None:
    """Seed every state key once, so no unit has to guard for a missing key."""
    for key, value in STATE.items():
        st.session_state.setdefault(key, value)

    # A run can arrive from a URL (the examiner's bookmark) or from the list.
    linked = st.query_params.get("run")

    if linked and st.session_state["run_id"] is None and data.is_run_id(linked):
        st.session_state["run_id"] = linked


def _open(run_id: str) -> None:
    """Select ``run_id`` for playback and restart the cursor at turn 0."""
    st.session_state["run_id"] = run_id
    st.session_state["cursor"] = 0
    st.session_state["playing"] = False
    st.query_params["run"] = run_id


@st.cache_data(show_spinner=False)
def _load_cached(run_id: str, stamp: tuple[int, float]) -> data.RunView | None:
    """Cache ``load_run`` under a cheap invalidation key.

    ``stamp`` is the trajectory's ``(size, mtime)`` and is **only** a cache key: a
    live or resumed run rewrites the log, and Play re-reads every 1.5 s, so
    without it the page would re-fold the log on every tick.
    """
    return data.load_run(run_id)


def _load(run_id: str | None) -> data.RunView | None:
    """Load the selected run, or ``None`` when it is not a run (see ``data``)."""
    run_dir = data.run_dir(run_id or "")

    if run_dir is None:
        return None

    log = run_dir / "trajectory.jsonl"
    stamp = (log.stat().st_size, log.stat().st_mtime) if log.is_file() else (0, 0.0)

    return _load_cached(str(run_id), stamp)


def _sidebar() -> None:
    """Title, page switch and the runs root (the one thing that changes a demo)."""
    with st.sidebar:
        st.title("LENS")
        st.caption("Literature Evidence & Narrative Synthesis")

        pages = ["Runs", "Run"]
        st.session_state["page"] = st.radio(
            "Page", pages, index=pages.index(st.session_state["page"])
        )

        if st.session_state["run_id"]:
            st.caption(f"selected: `{st.session_state['run_id']}`")

        counts = Counter(row["status"] for row in data.run_rows())

        if counts:
            st.caption(" · ".join(f"{status} {n}" for status, n in sorted(counts.items())))

        st.caption(f"runs root: `{runstore.runs_root()}`")
        st.divider()
        st.caption(
            "Read-only view of recorded artifacts. No pipeline logic (T-06)."
        )


def render_recording_banner() -> None:
    """State, on every recorded view, that this is a **recording** (§4n.1, mode A).

    Claim discipline: "live runs were performed on the GPU workstation; this
    session is replayed from its content-addressed trajectory" is a stronger
    statement than a video, because the examiner can open the turn, read the
    ``chunk_uuid``, and find it in the answer.
    """
    st.caption(
        "**Recorded playback (mode A).** Live runs were performed on the GPU "
        "workstation; this session is replayed from its content-addressed "
        "trajectory. No model is called and no corpus is queried."
    )


# ------------------------------------------------------------------- pages
def main() -> None:
    """Route between the two pages. One page, one concern: no live button on Runs."""
    st.set_page_config(page_title="LENS", page_icon="🔎", layout="wide")
    _init_state()
    _sidebar()

    if st.session_state["page"] == "Run":
        page_run()
    else:
        page_runs()


def page_runs() -> None:
    """Runs page (U1): list, select, then recorded playback. No live controls."""
    st.subheader("Runs")

    status = st.selectbox("Status", RUN_STATUSES, index=0, key="status_filter")
    rows = data.run_rows(None if status == "all" else status)

    if not rows:
        st.caption(f"No runs with status `{status}` under `{runstore.runs_root()}`.")
        render_import()
        return

    aliases = data.load_aliases()
    st.dataframe(
        [
            {
                "name": row["name"] or "—",
                "run_id": row["run_id"],
                "status": row["status"],
                "updated": _when(row["updated_at"]),
                "query": row["query"],
                "cost $": row["cost_usd"],
                "turns": row["iterations"],
            }
            for row in rows
        ],
        hide_index=True,
        column_config={
            "cost $": st.column_config.NumberColumn(format="%.4f"),
            "query": st.column_config.TextColumn("query"),
        },
    )

    ids = [row["run_id"] for row in rows]
    current = st.session_state.get("run_id")
    options = ["— select a run —", *ids]

    # The picker owns its widget state, so it is re-pointed at the selected run
    # whenever that changed elsewhere (URL, a rename, the live page). Without this
    # the dropdown drifts out of sync with the playback cursor.
    if st.session_state.get("picker_sync") != current:
        st.session_state["run_picker"] = current if current in ids else options[0]
        st.session_state["picker_sync"] = current

    picked = st.selectbox(
        "Open run",
        options,
        key="run_picker",
        format_func=lambda value: _run_label(value, aliases, options[0]),
    )

    if picked != options[0] and picked != current:
        _open(picked)

    view = _load(st.session_state.get("run_id"))

    if view is None:
        st.caption("Pick a run to open its recorded playback.")
        render_import()
        return

    render_recording_banner()
    render_header(view)

    # The playback block is the fragment body: a tick then re-renders only this
    # block, not the whole page, and ``run_every`` is None while paused.
    playing = st.session_state["playing"]
    st.fragment(
        render_playback, run_every=PLAY_SECONDS if playing else None
    )(view)

    render_answer(view)
    render_import()


def page_run() -> None:
    """Live page (U9): a query box that shells out to the CLI, streaming stdout."""
    st.subheader("Run a live query")
    st.caption(
        "Needs the workstation stack: GPU (embeddings, reranker, GLiNER), the "
        "Postgres corpus, Docker and `OPENROUTER_API_KEY`. The **Runs** page needs "
        "none of these. One run at a time — the app blocks while the CLI works "
        "(the terminal's Ctrl-C ends the app and leaves the run `interrupted`, "
        "which is resumable with `--resume`)."
    )

    query = st.text_area(
        "Query",
        height=110,
        placeholder="e.g. How is long-context aggregation evaluated on OOLONG?",
    )
    run_id = st.text_input("Run id", value=_next_run_id())

    if st.button("Run", type="primary", disabled=not query.strip() or not run_id.strip()):
        _stream_run(query.strip(), run_id.strip())


# -------------------------------------------------------- playback fragment
def render_playback(view: data.RunView) -> None:
    """The cursor-dependent block (U3–U7), rendered inside one fragment."""
    _advance(view)
    render_controls(view)
    render_meters(view)
    render_timeline(view)
    render_evidence(view)


def _advance(view: data.RunView) -> None:
    """Move the playback cursor one turn when ``Play`` is on (U3)."""
    if not st.session_state["playing"]:
        return

    last = max(len(view.turns) - 1, 0)

    if st.session_state["cursor"] >= last:
        st.session_state["playing"] = False
        st.rerun(scope="app")  # recreate the fragment without a timer
        return

    st.session_state["cursor"] += 1


def render_controls(view: data.RunView) -> None:
    """Playback controls (U3): ``◀ Step`` · ``Step ▶`` · ``Play`` · ``Jump`` · export.

    ``Play``/``Pause`` toggle a state flag that decides the fragment's
    ``run_every``, so they need a **full** rerun to take effect; step and jump only
    move the cursor and are handled inside the fragment.
    """
    n = len(view.turns)
    cursor = st.session_state["cursor"]

    if n == 0:
        return

    step_back, step_on, play, jump, export = st.columns([1, 1, 1, 2, 1])

    if step_back.button("◀ Step", disabled=cursor == 0):
        st.session_state["cursor"] = cursor - 1
        st.session_state["playing"] = False

    if step_on.button("Step ▶", disabled=cursor >= n - 1):
        st.session_state["cursor"] = cursor + 1
        st.session_state["playing"] = False

    playing = st.session_state["playing"]

    if play.button(
        "⏸ Pause" if playing else "▶ Play", type="primary"
    ):
        st.session_state["playing"] = not playing
        st.rerun(scope="app")  # run_every is fixed when the fragment is created

    # The step handlers above mutate session state *during* this run, so the
    # local taken at the top is stale here. A jump widget pointed at a stale turn
    # is exactly the reported bug: dropdown says turn 7, the playback is on 5.
    cursor = st.session_state["cursor"]

    if st.session_state.get("jump_sync") != cursor:
        st.session_state["jump_turn"] = cursor
        st.session_state["jump_sync"] = cursor

    jumped = jump.selectbox(
        "Jump to turn",
        list(range(n)),
        key="jump_turn",
        format_func=lambda turn: f"turn {turn}",
        label_visibility="collapsed",
    )

    if jumped != cursor:
        st.session_state["cursor"] = jumped
        st.session_state["playing"] = False

    with export.popover("Export"):
        destination = st.text_input(
            "Destination directory", value=str(_default_export_dir()), key="export_dest"
        )

        if st.button("Export .lens bundle"):
            try:
                path = runstore.export_run(view.run_id, destination)
            except (OSError, ValueError, KeyError) as exc:
                st.error(f"export failed: {exc}")
            else:
                st.success(f"wrote `{path}`")

    # Re-read the cursor: a jump or a step changed it *inside* this function, and
    # a progress bar drawn from the stale local would disagree with the dropdown
    # (and with the chart, which reads session state).
    cursor = st.session_state["cursor"]
    st.progress(
        (cursor + 1) / n,
        text=f"turn {cursor + 1} of {n}" + (" · playing" if playing else ""),
    )


def render_header(view: data.RunView) -> None:
    """Run identity, the local display name, and the headline numbers (U2)."""
    manifest = view.manifest
    cap = view.cap_usd
    aliases = data.load_aliases()
    name = data.display_name(view.run_id, aliases)

    title, rename = st.columns([6, 1])
    title.markdown(f"### {name}")

    if name != view.run_id:
        # The id stays visible: it is the key the artifacts, the bundle and the
        # campaign row all use, and a label must never hide it.
        title.caption(f"run id `{view.run_id}`")

    with rename.popover("Rename"):
        typed = st.text_input(
            "Display name",
            value=aliases.get(view.run_id, ""),
            max_chars=data.ALIAS_MAX_CHARS,
            key=f"alias::{view.run_id}",
            placeholder="e.g. anchor — gpt-5, aggregation",
        )

        if st.button("Save name"):
            try:
                data.set_alias(view.run_id, typed)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun(scope="app")

        st.caption(
            "Local only: stored in `aliases.json` beside the blob store, so the run "
            "id, the manifest and any exported bundle are untouched. Clear the field "
            "to go back to the id."
        )

    st.markdown(f"**{manifest.query or '(no query recorded)'}**")

    columns = st.columns(5)
    columns[0].metric("status", manifest.status)
    columns[1].metric("turns", len(view.turns))
    columns[2].metric("cost (chat + gate)", f"${view.cost_usd:.4f}")
    columns[3].metric("+ 5.5 % fee", f"${view.cost_with_fee:.4f}")
    columns[4].metric("cap", f"${cap:.2f}" if cap is not None else "not recorded")

    if cap:
        st.progress(
            min(view.cost_usd / cap, 1.0),
            text=f"${view.cost_usd:.4f} of the recorded ${cap:.2f} cap "
            f"({100 * view.cost_usd / cap:.1f} %)",
        )
    else:
        st.caption(
            f"${view.cost_usd:.4f} spent — this manifest records no cost cap, so no "
            "utilisation is claimed against today's default."
        )

    if manifest.status in {"failed", "interrupted", "partial"}:
        st.warning(
            f"status `{manifest.status}` — the run did not finish cleanly; the "
            "timeline below is what was recorded."
        )

    if view.missing_refs:
        st.error(
            f"{view.missing_refs} blob(s) could not be resolved — this bundle is "
            "damaged or incomplete. The affected turns say so individually."
        )

    provenance = manifest.created_by or {}
    details = {
        "models": manifest.models,
        "iterations": manifest.iterations,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "parent_run_id": manifest.parent_run_id,
        "provenance": provenance,
        "caps": manifest.caps,
    }

    with st.expander("Provenance and caps (E-04)"):
        st.json(json.loads(json.dumps(details, default=str)), expanded=False)
        arm = (manifest.extra or {}).get("arm")

        if arm:
            st.markdown("**arm** — the knobs this run was created with")
            st.json(json.loads(json.dumps(arm, default=str)), expanded=False)


def render_meters(view: data.RunView) -> None:
    """The meters (U6) — the chart is prompt size, the panel below is evidence."""
    summary = view.summary
    cursor = st.session_state["cursor"]

    columns = st.columns(5)
    columns[0].metric("tokens in (chat)", f"{summary['tokens_in']:,}")
    columns[1].metric("tokens out", f"{summary['tokens_out']:,}")
    columns[2].metric("gate tokens", f"{summary['jev_tokens_in']:,}")
    columns[3].metric("tool calls", summary["tool_calls"])
    columns[4].metric("gates pass/fail", f"{summary['gates_passed']} / {summary['gates_failed']}")

    series = view.tokens_in_series[: cursor + 1]

    if series:
        st.line_chart(
            {"tokens_in": series},
            x_label="turn",
            y_label="prompt tokens",
            height=220,
        )
        st.caption(
            "Prompt tokens per turn — the size of the context the model re-read. "
            "This chart rises with the conversation; the claim under test is "
            "evidence retrieved per token against a naive RAG baseline, not a flat "
            "context. Gate (JEV) tokens are excluded on purpose: they are not the "
            "growing context."
        )

    budget = view.turns[cursor].budget if view.turns else None

    if budget:
        used = budget.get("used") or {}
        limits = budget.get("limits") or {}
        st.caption(
            f"Budget at turn {cursor}:  Iterations {used.get('iterations', 0)}"
            f"/{limits.get('iterations', 0)} ·  Tokens "
            f"{used.get('tokens_in', 0) + used.get('tokens_out', 0):,}"
            f"/{limits.get('tokens', 0):,} ·  Cost ${used.get('cost_usd', 0.0):.4f}"
            f"/{limits.get('cost_usd', 0.0):.2f}"
            + (f" ·  Stop: {budget['stop_reason']}" if budget.get("stop_reason") else "")
        )


def render_timeline(view: data.RunView) -> None:
    """The turn timeline (U4): every turn up to the cursor."""
    cursor = st.session_state["cursor"]
    turns = view.turns

    if not turns:
        st.info("This run has no turns in its trajectory (it stopped before the loop).")
        return

    st.caption(f"showing turn 0–{cursor} of {len(turns) - 1}")

    for turn in turns[: cursor + 1]:
        render_turn(view.run_id, turn)


def render_turn(run_id: str, turn: Turn) -> None:
    """One turn card (U5): completion, code, tools, and what the model saw."""
    with st.container(border=True):
        head = st.columns([5, 1, 1, 1])

        head[0].markdown(f"**Turn {turn.turn}** · `{turn.model or 'unknown'}`")
        head[1].caption(f"{turn.tokens_in or 0} in")
        head[2].caption(f"{turn.tokens_out or 0} out")
        head[3].caption(f"${(turn.cost_usd or 0.0) + (turn.jev_cost_usd or 0.0):.6f}")

        if turn.ok is False:
            st.error(
                "**block failed** — "
                + str((turn.error or {}).get("type", "error"))
                + ": "
                + str((turn.error or {}).get("message", ""))[:300]
            )
        elif turn.ok is None and turn.code:
            st.warning("**interrupted** — this turn has code but no result in the log.")

        reasoning = _reasoning(turn.completion)

        if reasoning:
            st.markdown(_clip(reasoning, REASONING_CHARS))

        if turn.completion:
            with st.expander("Full Model Message (Reasoning + Code Written)"):
                st.markdown(turn.completion)

        if turn.code:
            st.code(turn.code, language="python")

        for slot, tool in enumerate(turn.tools):
            render_tool(run_id, tool, f"{turn.turn}-{slot}")

        if turn.observation:
            st.markdown("**Observation — what the model actually saw**")
            st.code(turn.observation, language="text")

        chips = []

        if turn.emits:
            chips.append("emits: " + ", ".join(f"`{name}`" for name in turn.emits))

        if turn.jev_cost_usd:
            chips.append(
                f"gate: ${turn.jev_cost_usd:.6f} ({turn.jev_tokens_in or 0} tokens in)"
            )

        if turn.latency_ms:
            chips.append(_duration(turn.latency_ms))

        if chips:
            st.caption(" · ".join(chips))

        if turn.stderr_full:
            with st.expander("stderr"):
                st.code(turn.stderr_full, language="text")

        if turn.missing_refs:
            st.warning(
                f"{len(turn.missing_refs)} missing refs in this turn: "
                + ", ".join(f"`{ref}`" for ref in turn.missing_refs[:3])
            )


def render_tool(run_id: str, tool: ToolStep, slot: str) -> None:
    """One tool call: name(args) · latency · gate · replayed · full payload (U5).

    ``slot`` is ``"<turn>-<index>"``: a key that is unique per call. A content
    hash is not enough — two calls can share a result (a replay serves the same
    blob), and a duplicate widget key is a hard Streamlit error.
    """
    columns = st.columns([3, 1, 1, 1, 2])
    columns[0].markdown(
        f"`{tool.name}({_args_preview(tool.args)})`" if tool.args else f"`{tool.name}`"
    )
    columns[1].caption(_duration(tool.latency_ms))
    columns[2].caption(_gate_badge(tool.gate))
    columns[3].caption("Replayed" if tool.replayed else "")

    toggle_key = f"payload::{run_id}::{slot}"
    show = columns[4].checkbox("Full Payload", key=toggle_key, help="Show the full result")

    if tool.error:
        st.error(f"tool error: {tool.error.get('type')}: {tool.error.get('message')}")

    if tool.result_summary:
        with st.expander("Result Summary (Return Context)"):
            st.code(tool.result_summary, language="text")

    if not show or not tool.result_ref:
        return

    text = data.tool_payload(run_id, tool.result_ref)

    if text is None:
        st.warning("Payload blob missing from the bundle")
        return

    st.download_button(
        "Download Full Payload",
        data=text,
        file_name=f"{run_id}-{tool.name}-{slot}.json",
        key=f"download::{toggle_key}",
    )

    if len(text) > PAYLOAD_PREVIEW_CHARS:
        st.caption(
            f"Payload is {len(text):,} chars — previewing the first "
            f"{PAYLOAD_PREVIEW_CHARS:,}; the button above has all of it."
        )
        text = text[:PAYLOAD_PREVIEW_CHARS]

    try:
        st.json(json.loads(text), expanded=False)
    except ValueError:
        st.code(text, language="text")


def render_evidence(view: data.RunView) -> None:
    """The growing evidence panel (U7) — the counterweight to the context chart."""
    cursor = st.session_state["cursor"]
    rows = data.evidence_upto(view, cursor)
    newest = {row.get("chunk_uuid") for row in view.evidence.get(cursor, [])}

    st.markdown(f"### Evidence hydrated so far — {len(rows)} chunk(s)")

    if not rows:
        st.caption(
            "No hydrating tool call yet. `shallow_match` and `deep_match` return "
            "ids only; `rerank_chunks` / `fetch_chunks` are what put chunk text in "
            "front of the model (an id list is a candidate, not evidence)."
        )
        return

    st.dataframe(
        [
            {
                "new": "●" if row.get("chunk_uuid") in newest else "",
                "chunk": str(row.get("chunk_uuid"))[:8],
                "paper": row.get("arxiv_id"),
                "section": row.get("section_type"),
                "tokens": row.get("token_count"),
            }
            for row in rows
        ],
        hide_index=True,
    )


def render_answer(view: data.RunView) -> None:
    """The answer and its citation trace (U8) — the model's words, then the join."""
    st.markdown("## Answer")

    if view.answer.strip():
        st.markdown(view.answer)
    else:
        st.caption(
            "No `answer.md` — the run stopped before an outside-a-fence "
            "`FINAL_ANSWER:` sentinel."
        )

    report = view.citations
    counts = report.counts

    if report.citations:
        st.markdown(
            f"**Citations — {counts['cited']} cited · {counts['resolved']} resolved · "
            f"{counts['unresolved']} unresolved**"
        )
        st.dataframe(
            [
                {
                    "cited id": citation.token if len(citation.token) <= 20 else citation.token[:8],
                    "kind": citation.kind,
                    "paper": citation.arxiv_id,
                    "section": citation.section_type,
                    "resolved in turn": citation.turn,
                    "status": "OK" if citation.resolved else "unresolved",
                }
                for citation in report.citations
            ],
            hide_index=True,
        )

    if report.unresolved:
        st.warning(
            f"{len(report.unresolved)} cited id(s) no recorded payload explains — "
            "listed, not counted as supported: "
            + ", ".join(f"`{token}`" for token in report.unresolved)
        )

    run_dir = data.run_dir(view.run_id)

    if run_dir is None:
        return

    downloads = st.columns(2)

    if view.has_sources:
        downloads[0].download_button(
            "sources.md (citation trace)",
            data=(run_dir / "sources.md").read_text(encoding="utf-8", errors="replace"),
            file_name=f"{view.run_id}-sources.md",
            key="download_sources",
        )

    if view.has_report:
        downloads[1].download_button(
            "report.md (static rendering)",
            data=(run_dir / "report.md").read_text(encoding="utf-8", errors="replace"),
            file_name=f"{view.run_id}-report.md",
            key="download_report",
        )


def render_import() -> None:
    """Install a portable ``<run_id>.lens`` bundle (U1, roadmap M10b.7).

    Path input rather than a file uploader: the bundle is a **directory**, and
    ``verify_bundle`` re-hashes every blob before anything is installed, so a
    tampered bundle is refused rather than rendered.
    """
    with st.expander("Import a recorded bundle (.lens)"):
        st.caption(
            "Verification happens first: every blob is re-hashed against its "
            "filename, so an edited or truncated bundle installs nothing."
        )
        path = st.text_input(
            "Bundle directory",
            placeholder=r"D:\lens-data\demo\m9-s3-plumbing-6.lens",
            key="import_path",
        )

        if not st.button("Import", disabled=not path.strip()):
            return

        report = runstore.verify_bundle(path.strip())

        if not report["ok"]:
            st.error("verification failed:\n" + "\n".join(f"- {e}" for e in report["errors"]))
            return

        try:
            manifest = runstore.import_run(path.strip())
        except (OSError, ValueError, FileExistsError) as exc:
            st.error(f"import failed: {exc}")
            return

        st.success(
            f"imported `{manifest.run_id}` — {report['blobs']} blobs, "
            f"{report['bytes'] / 1e6:.1f} MB"
        )
        st.rerun(scope="app")


# --------------------------------------------------------------- live (U9)
def _stream_run(query: str, run_id: str) -> None:
    """Run the CLI in a subprocess and stream its stdout into the page.

    The UI process therefore never imports torch, never opens a DB, and cannot be
    taken down by the run: the command is exactly the one ``check_run.py``
    validates, and ``--run-id`` is supplied up front so the finished run opens
    deterministically.
    """
    command = [sys.executable, "-m", "lens.run", query, "--run-id", run_id]

    with st.status(f"running `{run_id}`", expanded=True) as status:
        st.code(" ".join(command), language="powershell")
        log = st.empty()
        lines: list[str] = []

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            status.update(label="could not start the CLI", state="error")
            st.error(str(exc))
            return

        if process.stdout is None:  # cannot happen with PIPE, but never assume
            status.update(label="no stdout pipe", state="error")
            return

        for line in process.stdout:
            lines.append(line.rstrip())
            log.code("\n".join(lines[-30:]), language="text")

        code = process.wait()

        if code == 0:
            status.update(label=f"finished — {run_id}", state="complete")
        else:
            status.update(label=f"CLI exited {code} — see the log above", state="error")

    if code == 0 and data.run_dir(run_id) is not None:
        _open(run_id)
        st.session_state["page"] = "Runs"
        st.rerun(scope="app")


# ---------------------------------------------------------------- helpers
def _next_run_id() -> str:
    """Return a sortable default run id (``ui-YYYYmmdd-HHMMSS``, UTC)."""
    return f"ui-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"


def _when(timestamp: str | None) -> str:
    """Render an ISO timestamp readably: ``2026-10-25 14:49``."""
    text = (timestamp or "").strip()

    return f"{text[:10]} {text[11:16]}" if len(text) >= 16 else text


def _duration(latency_ms: int | None) -> str:
    """Render a latency in the unit a human reads it in: ``412 ms`` / ``19.1 s``.

    Milliseconds stop being legible where they matter most: a ``build_kg`` call
    is tens of seconds, and "19143 ms" is a number to decode rather than read.
    """
    if not latency_ms:
        return "—"

    if latency_ms < 1000:
        return f"{latency_ms} ms"

    return f"{latency_ms / 1000:.1f} s"


def _reasoning(completion: str | None) -> str:
    """Return the prose part of a completion: the fenced blocks removed.

    The executed block is rendered by ``st.code`` right below, so keeping the fence
    in the visible summary would show the same code twice — and it is the code, not
    the sentence before it, that Streamlit renders with a horizontal scrollbar.
    """
    if not completion:
        return ""

    text = _FENCE_RE.sub("", completion)

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _clip(text: str, limit: int) -> str:
    """Truncate ``text`` at ``limit`` characters, saying that it was truncated."""
    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + " … (truncated — open the full message below)"


def _run_label(value: str, aliases: dict[str, str], empty: str) -> str:
    """Label a run picker entry: the local name when set, always with its id."""
    if value == empty:
        return value

    name = aliases.get(value)

    return f"{name} — {value}" if name else value


def _default_export_dir() -> Path:
    """Export next to the runs root by default (never inside a run directory)."""
    return runstore.runs_root().parent / "demo"


def _args_preview(args: dict[str, Any] | None, limit: int = 3) -> str:
    """Render the first few tool arguments compactly: ``buckets={...}, k=100``."""
    if not args:
        return ""

    parts: list[str] = []

    for key, value in list(args.items())[:limit]:
        text = value if isinstance(value, str) else json.dumps(value, default=str)

        if len(text) > 24:
            text = text[:24] + "…"

        parts.append(f"{key}={text}")

    hidden = len(args) - len(parts)

    return ", ".join(parts) + (f", +{hidden}" if hidden > 0 else "")


def _gate_badge(gate: dict[str, Any] | None) -> str:
    """Render a stable four-key verdict as a badge (never invent a score)."""
    if not gate:
        return ""

    verdict = str(gate.get("verdict") or "—")
    icon = {"pass": "✅", "fail": "❌"}.get(verdict, "❔")
    tier = gate.get("tier")

    return f"{icon} {verdict}" + (f" · tier {tier}" if tier else "")


if __name__ == "__main__":
    main()
