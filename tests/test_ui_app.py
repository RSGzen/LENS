r"""Smoke test for the M10 Streamlit page (skipped when the ``ui`` extra is absent).

``tests/test_ui_data.py`` covers the read path. This covers the *page*: it must
start, render a recorded run, step the cursor, and open a payload — with zero
exceptions. A syntax error or a renamed field in ``app.py`` would otherwise be
found only when the demo is opened in front of an examiner.

Streamlit is an **extra** (``uv sync --extra ui``), so the whole module skips
when it is not installed: the no-key suite must stay green on a core install.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_ui_app -v
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest

from pathlib import Path

import lens.config as config_module
import lens.observability.trajectory as trajectory_module
from lens.agent import runstore
from lens.observability.trajectory import TraceCtx, Trajectory
from lens.ui import data

# Streamlit is an optional extra: import it only when present so the module can
# self-skip on a core install (the no-key suite must not need the UI stack).
if importlib.util.find_spec("streamlit") is not None:
    from streamlit.testing.v1 import AppTest

    HAS_STREAMLIT = True
else:
    HAS_STREAMLIT = False

APP = Path(__file__).resolve().parents[1] / "src" / "lens" / "ui" / "app.py"
CHUNK = "11111111-1111-5111-8111-111111111111"


@unittest.skipUnless(HAS_STREAMLIT, "streamlit is an optional extra (uv sync --extra ui)")
class UiAppTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_config = config_module.LENS_RUNS_ROOT_PATH
        self._orig_trajectory = trajectory_module.LENS_RUNS_ROOT_PATH
        config_module.LENS_RUNS_ROOT_PATH = self.root
        trajectory_module.LENS_RUNS_ROOT_PATH = self.root
        self._make_run("run-ui")

    def tearDown(self) -> None:
        config_module.LENS_RUNS_ROOT_PATH = self._orig_config
        trajectory_module.LENS_RUNS_ROOT_PATH = self._orig_trajectory
        self._tmp.cleanup()

    def _make_run(self, run_id: str) -> None:
        """Two turns, one hydrating tool call, an answer that cites it."""
        traj = Trajectory(run_id)

        for step in (0, 1):
            traj.log_event(
                trace_ctx=TraceCtx(run_id, step, "root"),
                call_type="chat",
                action="chat.completion",
                model="openai/gpt-5-mini",
                provider="openai",
                prompt_ref=traj.store_blob(json.dumps({"messages": []})),
                completion_ref=traj.store_blob(
                    f"I will look at turn {step} first.\n\n```python\n"
                    f"rows = shallow_match(buckets=buckets)\n```"
                ),
                tokens_in=100 * (step + 1),
                tokens_out=20,
                cost_usd=0.01,
                latency_ms=900,
            )
            traj.log_event(
                trace_ctx=TraceCtx(run_id, step, "tool"),
                call_type="tool",
                action="tool.dispatch",
                tool="rerank_chunks",
                args_hash=f"hash-{step}",
                args_ref=traj.store_blob(json.dumps({"chunk_ids": ["c1"], "k": 100})),
                result_ref=traj.store_blob(
                    json.dumps({"chunks": [{"chunk_uuid": CHUNK, "arxiv_id": "2511.02817"}], "papers": []})
                ),
                result_summary="rerank_chunks: 1 row",
                latency_ms=120,
                extra={"gate": {"verdict": "pass", "tier": 1, "confidence": 0.9, "failing_criterion": None}},
            )

        runstore.write_manifest(
            self.root / run_id,
            runstore.RunManifest(
                run_id=run_id,
                query="what is evaluated on trec_coarse?",
                status="completed",
                created_at="2026-10-26T09:00:00+00:00",
                updated_at="2026-10-26T09:05:00+00:00",
                caps={"cost_usd": 1.0, "iterations": 8},
                totals={"cost_usd": 0.02},
                iterations=2,
            ),
        )
        (self.root / run_id / "answer.md").write_text(f"answer citing {CHUNK}\n", encoding="utf-8")

    def _app(self) -> "AppTest":
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.session_state["run_id"] = "run-ui"
        at.run()
        return at

    def test_page_renders_a_recorded_run(self) -> None:
        at = self._app()

        self.assertEqual([str(e.value) for e in at.exception], [])
        labels = {metric.label: metric.value for metric in at.metric}
        self.assertEqual(labels["status"], "completed")
        self.assertEqual(labels["turns"], "2")
        self.assertEqual(labels["cap"], "$1.00")       # from the manifest, not config
        self.assertTrue(any("Recorded playback" in c.value for c in at.caption))

    def test_step_advances_the_cursor(self) -> None:
        at = self._app()
        step = [button for button in at.button if button.label.endswith("Step ▶")][0]

        step.click().run()

        self.assertEqual(at.session_state["cursor"], 1)
        self.assertEqual([str(e.value) for e in at.exception], [])
        self.assertTrue(any("turn 0–1 of 1" in c.value for c in at.caption))

    def test_jump_and_progress_stay_in_sync(self) -> None:
        at = self._app()
        jump = at.selectbox(key="jump_turn")

        jump.select(1).run()

        self.assertEqual(at.session_state["cursor"], 1)
        self.assertEqual(at.selectbox(key="jump_turn").value, 1)
        self.assertIn("turn 2 of 2", [bar.text for bar in at.progress])

        at.run()  # the controls re-render at the new cursor before we click again
        step_back = [button for button in at.button if button.label.startswith("◀")][0]
        step_back.click().run()

        self.assertEqual(at.session_state["cursor"], 0)
        self.assertEqual(at.selectbox(key="jump_turn").value, 0)
        self.assertIn("turn 1 of 2", [bar.text for bar in at.progress])

    def test_reasoning_is_shown_outside_the_expander(self) -> None:
        at = self._app()

        visible = [block.value.strip() for block in at.markdown]

        # The prose is its own block (what the reviewer reads), and the fenced
        # block appears only inside the collapsed full message.
        self.assertIn("I will look at turn 0 first.", visible)
        self.assertTrue(any("```python" in value for value in visible))

    def test_payload_toggle_renders_the_blob(self) -> None:
        at = self._app()

        self.assertEqual(len(at.checkbox), 1)  # one tool call in the turn 0 card
        at.checkbox[0].check().run()

        self.assertEqual([str(e.value) for e in at.exception], [])
        self.assertTrue(any(button.label == "Download Full Payload" for button in at.download_button))

    def test_rename_from_the_page(self) -> None:
        at = self._app()

        at.text_input(key="alias::run-ui").set_value("Anchor run").run()
        save = [button for button in at.button if button.label == "Save name"][0]
        save.click().run()

        self.assertEqual(data.load_aliases(), {"run-ui": "Anchor run"})
        self.assertIn("### Anchor run", [block.value for block in at.markdown])
        self.assertEqual([str(e.value) for e in at.exception], [])

        # The run id stays on screen underneath the label: a name must never hide
        # the key the artifacts are filed under.
        self.assertTrue(any("run-ui" in c.value for c in at.caption))

    def test_live_page_needs_no_pipeline_import(self) -> None:
        at = self._app()
        at.session_state["page"] = "Run"
        at.run()

        self.assertEqual([str(e.value) for e in at.exception], [])
        self.assertTrue(any(button.label == "Run" for button in at.button))


if __name__ == "__main__":
    unittest.main()
