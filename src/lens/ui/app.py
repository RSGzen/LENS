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

Scaffold status (LOG-33)
------------------------
The glue is filled: page routing, the ``st.session_state`` contract, the cache
key, the recording banner. Every body that *decides what to show* is a marked
``TODO U1–U9`` in the roadmap's order; until a unit is filled it renders a
placeholder, so the shell runs from the first commit and fills one unit at a
time.

Run it::

    uv sync --extra ui
    streamlit run src/lens/ui/app.py
    # the demo laptop (no GPU, no DB, no key) points at an imported bundle:
    LENS_RUNS_ROOT=D:\\lens-data\\runs-import streamlit run src/lens/ui/app.py
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from lens.agent import runstore
from lens.ui import data

#: Playback cadence: one turn per fragment tick (§4n.1). ``st.fragment`` needs
#: Streamlit >= 1.37 — a full-page rerun per tick would make ``Play`` unusable.
PLAY_SECONDS = 1.5

#: ``Show full payload`` renders the JSON only when toggled: a collapsed
#: ``st.expander`` still builds its DOM, and a ``build_kg`` node-link is the
#: largest blob in a run.
PAYLOAD_PREVIEW_CHARS = 20_000

#: ``st.session_state`` contract. Every unit reads and writes these keys and
#: nothing else, so a page can be filled (or replaced) without touching another.
STATE: dict[str, Any] = {
    "page": "Runs",      # "Runs" | "Run"
    "run_id": None,      # the selected run; mirrored in ``st.query_params["run"]``
    "cursor": 0,         # the last turn shown (playback position)
    "playing": False,    # Play is running (drives ``st.fragment(run_every=)``)
    "payload": (),       # tool ``result_ref``s whose full payload is open
}

RUN_STATUSES: list[str] = [
    "all",
    "completed",
    "partial",
    "failed",
    "interrupted",
    "running",
]


# --------------------------------------------------------------------- glue
def _init_state() -> None:
    """Seed every state key once, so no unit has to guard for a missing key."""
    for key, value in STATE.items():
        st.session_state.setdefault(key, value)

    # A run can arrive from a URL (the examiner's bookmark) or from the list.
    linked = st.query_params.get("run")

    if linked and st.session_state["run_id"] is None and data.is_run_id(linked):
        st.session_state["run_id"] = linked


def _placeholder(unit: str, what: str) -> None:
    """Render a fill-order marker instead of crashing the shell (LOG-33)."""
    st.info(f"**{unit}: not filled yet.** {what}")


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

        st.caption(f"runs root: `{runstore.runs_root()}`")
        st.caption("Read-only view of recorded artifacts. No pipeline logic (T-06).")


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
    """Runs page (M10a.2 + M10b): list, select, replay. No live controls.

    TODO U1 — the session list and its selection:

      * ``rows = data.run_rows(None if status == "all" else status)`` and render it
        (``st.dataframe`` is enough); a failed/partial run is a labelled row, never
        hidden.
      * Selecting a row sets ``st.session_state["run_id"]``, mirrors it into
        ``st.query_params["run"]``, resets ``cursor`` to 0 and clears ``playing``.
      * Below the list: an **import** control (roadmap M10b.7) — a path input for
        ``<run_id>.lens`` plus a button calling ``runstore.import_run(path)``;
        show the ``ValueError`` when verification refuses a tampered bundle.

    The composition below is the roadmap's order; each unit is filled separately.
    """
    _placeholder(
        "U1",
        "session list from `data.run_rows(status)`, a status filter, and an import "
        "button (`runstore.import_run`) — selecting a run sets `run_id`, "
        "`st.query_params['run']`, `cursor=0`, `playing=False`.",
    )

    view = _load(st.session_state.get("run_id"))

    if view is None:
        st.caption("Pick a run above to open its recorded playback.")
        return

    render_recording_banner()
    render_header(view)
    render_controls(view)
    render_meters(view)
    render_timeline(view)
    render_evidence(view)
    render_answer(view)


def page_run() -> None:
    """Live path (M10a.1): a query box that shells out to the CLI.

    TODO U9 — one run, streamed:

      * ``run_id = f"ui-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"`` and pass it
        up front as ``--run-id``, so the finished run opens deterministically.
      * ``subprocess.Popen([sys.executable, "-m", "lens.run", query, "--run-id",
        run_id], stdout=PIPE, stderr=STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1)`` and write each line into a
        ``st.status(..., expanded=True)``. Streamlit blocks the script for the
        duration (one user, one run — say so on the page); Ctrl-C in the terminal
        ends the app and leaves the run ``interrupted``, which is resumable.
      * On exit, set ``run_id`` + ``page="Runs"`` and let the recorded view take
        over. State plainly that this page needs the GPU/DB/Docker/key stack: the
        viewer does not.
    """
    _placeholder(
        "U9",
        "live query box: `python -m lens.run <query> --run-id ui-<ts>` via "
        "`subprocess.Popen`, stdout streamed into `st.status`; needs GPU + DB + "
        "Docker + key (unlike the Runs page).",
    )


# ------------------------------------------------------------- run view units
def render_header(view: data.RunView) -> None:
    """Run identity and the headline numbers (roadmap M10b.6).

    TODO U2 — ``manifest`` field by field: status, query, ``iterations`` vs the
    recorded cap, ``models``, ``created_by`` (commit/image/digest provenance),
    ``manifest.extra["arm"]`` when present, and ``missing_refs`` (0 expected; a
    damaged bundle says so here rather than at the first turn).

    Cost is ``view.cost_usd`` (chat + gate, matching ``totals()``) shown **against
    ``view.cap_usd``** — and when ``cap_usd is None`` the label is "cap not
    recorded", never today's default (LOG-120: M9 runs were recorded under $0.50).
    """
    _placeholder("U2", "run identity + headline numbers, cost vs the *recorded* cap.")


def render_controls(view: data.RunView) -> None:
    """Playback controls (roadmap M10b.5).

    TODO U3 — ``◀ Step`` / ``Step ▶`` / ``Play`` / ``Jump to turn`` /
    ``Show full payload`` / ``Export bundle``, all clamping ``cursor`` to
    ``[0, len(view.turns) - 1]``.

    Play must not rerun the page. Define the tick as a fragment::

        st.fragment(run_every=PLAY_SECONDS if st.session_state["playing"] else None)(play_tick)()

    ``play_tick`` advances one turn and clears ``playing`` at the last turn.
    Export: ``runstore.export_run(view.run_id, destination)`` behind a path input.
    """
    _placeholder(
        "U3",
        "◀ / ▶ / Play / Jump / Show full payload / Export bundle, stepping "
        "`cursor`; Play via `st.fragment(run_every=PLAY_SECONDS)`.",
    )


def play_tick() -> None:
    """Advance the playback cursor one turn (the fragment body for U3).

    TODO U3 — increment ``st.session_state["cursor"]`` (stopping at the last turn)
    and clear ``playing`` at the end. Keep it to that: the page re-renders from
    the cursor, so the tick owns no layout.
    """
    _placeholder("U3", "one playback tick: cursor += 1, stop at the last turn.")


def render_timeline(view: data.RunView) -> None:
    """The turn timeline (roadmap M10b.4): every turn up to the cursor.

    TODO U4 — iterate ``view.turns[: cursor + 1]`` in order and call
    :func:`render_turn` for each. Turns *after* the cursor stay hidden — that is
    the whole point of stepping — but the turn count is shown, so the viewer knows
    how much is left.
    """
    _placeholder("U4", "loop over `turns[:cursor + 1]` and render each one.")


def render_turn(turn: Any) -> None:
    """One turn card (roadmap M10b.4). ``turn`` is a ``replay.Turn``.

    TODO U5 — per turn:

      * ``completion`` (markdown) and ``code`` (``st.code``, ``python``);
      * a tool table: ``name(args)`` · latency · gate badge · replayed;
        ``Show full payload`` opens ``data.tool_payload(view.run_id, tool.result_ref)``
        (a toggle, not a bare expander — see ``PAYLOAD_PREVIEW_CHARS``);
      * ``observation`` — **what the model saw**, never ``stdout_full`` (the two
        must not be conflated, §4n.1);
      * emits as chips, ``gate`` verdict, ``error``, ``ok is False`` → "block
        failed", ``ok is None`` → "interrupted" (never shown as success), and
        ``missing_refs``.
    """
    _placeholder(
        "U5",
        "turn card: completion, code, tool table (args/latency/gate/replayed), "
        "observation, emits, gate verdict, error/interrupted, missing refs.",
    )


def render_meters(view: data.RunView) -> None:
    """The meters (roadmap M10b.6) — *evidence grows, context stays flat*.

    TODO U6 — ``view.tokens_in_series`` as the context-growth chart
    (``st.line_chart({"tokens_in": series})``), cumulative cost vs ``cap_usd``
    (``st.progress``), ``iterations`` vs the recorded cap, and the gate/tool
    counts from ``view.summary``.

    Caption the chart with what it *is*: ``tokens_in`` is the prompt size per turn
    and it **does** rise; the claim under test is evidence-per-token against a
    naive RAG baseline, so do not caption it "flat" (R4/R5).
    """
    _placeholder("U6", "`tokens_in` chart, cost vs cap progress, iterations, gate counts.")


def render_evidence(view: data.RunView) -> None:
    """The growing evidence panel (roadmap gate: "the evidence panel grows").

    TODO U7 — ``data.evidence_upto(view, st.session_state["cursor"])`` and show
    what entered the context: ``chunk_uuid`` · ``arxiv_id`` · ``section_type`` ·
    token count, newest turn highlighted. This is the visual counterweight to the
    context chart, and it is why the two are read from different sources.
    """
    _placeholder("U7", "hydrated chunk rows up to the cursor, newest turn highlighted.")


def render_answer(view: data.RunView) -> None:
    """The answer and its citation trace (M10a.1, M9 item 2).

    TODO U8 — ``answer.md`` verbatim (the model's own words), then
    ``view.citations.citations`` as a table with ``arxiv_id`` / ``section_type`` /
    turn / resolved. ``view.citations.unresolved`` is **listed**, never counted as
    supported; when ``has_sources`` is true, offer ``sources.md`` for download.
    """
    _placeholder("U8", "answer.md verbatim + resolved-citation table, unresolved listed.")


if __name__ == "__main__":
    main()
