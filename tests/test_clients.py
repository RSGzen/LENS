r"""Contract tests for the transport (no API key, no network, zero cost).

A ``FakeSession`` replaces ``requests.Session``, so these tests never leave the
machine.

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import lens.observability.trajectory as trajectory_module
from lens.clients import LensClient, LensTransportError
from lens.config import ModelSpec
from lens.models import Question
from lens.observability.trajectory import TraceCtx, Trajectory


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
        self.headers: dict = {}

    def json(self) -> dict:
        return self._payload


class FakeSession:
    """Minimal stand-in for ``requests.Session``.

    Records every ``post`` call and returns queued responses in order.
    """

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url: str, *, headers: dict, data: str, timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "data": data, "timeout": timeout})
        return self._responses.pop(0)


def chat_payload(model: str = "openai/gpt-5-20260101", provider: str = "OpenAI") -> dict:
    return {
        "id": "gen-1",
        "model": model,
        "provider": provider,
        "choices": [
            {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.00042},
    }


def jev_payload() -> dict:
    return {
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {"bucket": {"type": "choice", "choice": "objective"}},
        "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.00002},
    }


class LensClientTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.runs_root = Path(self._tmp.name)
        self._orig = trajectory_module.LENS_RUNS_ROOT_PATH
        trajectory_module.LENS_RUNS_ROOT_PATH = str(self.runs_root)
        self.traj = Trajectory("run-client")
        self.ctx = TraceCtx(trace_id="t1", step_idx=0, actor="root")

    def tearDown(self) -> None:
        trajectory_module.LENS_RUNS_ROOT_PATH = self._orig
        self._tmp.cleanup()

    # ---------------------------------------------------------------- helpers
    def _events(self) -> list[dict]:
        raw = self.traj.trajectory_path.read_text(encoding="utf-8")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def _read_blob(self, ref: str) -> str:
        return (self.traj.blobs_dir / ref.split(":", 1)[1]).read_text(encoding="utf-8")

    def _client(self, session: FakeSession, **kwargs) -> LensClient:
        return LensClient(self.traj, api_key="test-key", session=session, sleeper=lambda _s: None, **kwargs)

    # ------------------------------------------------------------------ chat
    def test_chat_returns_result_fields(self) -> None:
        session = FakeSession([FakeResponse(200, chat_payload())])
        result = self._client(session).chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        self.assertEqual(result.text, "hi")
        self.assertEqual(result.model, "openai/gpt-5")
        self.assertEqual(result.model_snapshot, "openai/gpt-5-20260101")
        self.assertEqual(result.provider, "OpenAI")
        self.assertEqual((result.tokens_in, result.tokens_out), (12, 3))
        self.assertAlmostEqual(result.cost_usd, 0.00042)

    def test_chat_logs_one_event_with_refs(self) -> None:
        session = FakeSession([FakeResponse(200, chat_payload())])
        self._client(session).chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        events = self._events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["call_type"], "chat")
        self.assertTrue(event["prompt_ref"].startswith("sha256:"))
        self.assertTrue(event["completion_ref"].startswith("sha256:"))
        self.assertEqual(event["tokens_in"], 12)
        self.assertAlmostEqual(event["cost_usd"], 0.00042)
        self.assertEqual(self._read_blob(event["completion_ref"]), "hi")

    def test_chat_pins_provider_from_root_spec(self) -> None:
        session = FakeSession([FakeResponse(200, chat_payload())])
        self._client(session).chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        body = json.loads(session.calls[0]["data"])
        self.assertIs(body["provider"]["allow_fallbacks"], False)
        self.assertEqual(body["provider"]["only"], ["openai"])

    def test_chat_omits_provider_when_no_pins(self) -> None:
        spec = ModelSpec(
            id="openai/gpt-5",
            supports_reasoning=True,
            reasoning_effort=None,
            supports_tools=True,
            supports_response_format=True,
        )
        session = FakeSession([FakeResponse(200, chat_payload())])
        client = self._client(session, registry={"root": spec})
        client.chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        body = json.loads(session.calls[0]["data"])
        self.assertNotIn("provider", body)

    def test_chat_omits_temperature_when_none(self) -> None:
        session = FakeSession([FakeResponse(200, chat_payload())])
        self._client(session).chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        body = json.loads(session.calls[0]["data"])
        self.assertNotIn("temperature", body)

    def test_chat_includes_temperature_when_set(self) -> None:
        spec = ModelSpec(
            id="qwen/qwen3-coder",
            supports_reasoning=False,
            reasoning_effort=None,
            supports_tools=True,
            supports_response_format=False,
            temperature=0.3,
        )
        session = FakeSession([FakeResponse(200, chat_payload())])
        client = self._client(session, registry={"root": spec})
        client.chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        body = json.loads(session.calls[0]["data"])
        self.assertEqual(body["temperature"], 0.3)

    def test_chat_omits_seed_when_none(self) -> None:
        session = FakeSession([FakeResponse(200, chat_payload())])
        self._client(session).chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        body = json.loads(session.calls[0]["data"])
        self.assertNotIn("seed", body)

    def test_chat_includes_seed_when_set(self) -> None:
        spec = ModelSpec(
            id="qwen/qwen3-coder",
            supports_reasoning=False,
            reasoning_effort=None,
            supports_tools=True,
            supports_response_format=False,
            seed=42,
        )
        session = FakeSession([FakeResponse(200, chat_payload())])
        client = self._client(session, registry={"root": spec})
        client.chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        body = json.loads(session.calls[0]["data"])
        self.assertEqual(body["seed"], 42)

    def test_chat_retries_then_succeeds(self) -> None:
        session = FakeSession([FakeResponse(429, {"error": "rate"}), FakeResponse(200, chat_payload())])
        result = self._client(session).chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        self.assertEqual(result.text, "hi")
        self.assertEqual(len(session.calls), 2)
        self.assertTrue(any("error" in e for e in self._events()))

    def test_chat_raises_after_exhausting_retries(self) -> None:
        session = FakeSession([FakeResponse(500, {"error": "boom"})] * 3)
        client = self._client(session, max_retries=2)
        with self.assertRaises(LensTransportError):
            client.chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        self.assertEqual(len(session.calls), 3)

    def test_api_key_never_written_to_blobs_or_events(self) -> None:
        session = FakeSession([FakeResponse(200, chat_payload())])
        self._client(session).chat("root", [{"role": "user", "content": "hi"}], trace_ctx=self.ctx)
        blob_text = "".join(p.read_text(encoding="utf-8") for p in self.traj.blobs_dir.iterdir())
        self.assertNotIn("test-key", blob_text)
        self.assertNotIn("test-key", self.traj.trajectory_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------- jev
    def test_jev_returns_result_and_logs(self) -> None:
        session = FakeSession([FakeResponse(200, jev_payload())])
        questions = {"bucket": Question(type="choice", instructions="Which?", criteria={"a": "A", "b": "B"})}
        result = self._client(session).jev("some clause", questions, trace_ctx=self.ctx)
        self.assertEqual(result.answers["bucket"]["choice"], "objective")
        self.assertEqual(result.model, "typesafe/jev-1.13-20260917")
        self.assertEqual(result.tokens_in, 476)
        self.assertAlmostEqual(result.cost_usd, 0.00002)
        events = self._events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["call_type"], "jev")

    def test_jev_request_body_shape(self) -> None:
        session = FakeSession([FakeResponse(200, jev_payload())])
        questions = {"bucket": Question(type="choice", instructions="Which?", criteria={"a": "A", "b": "B"})}
        self._client(session).jev("some clause", questions, trace_ctx=self.ctx)
        body = json.loads(session.calls[0]["data"])
        self.assertEqual(body["model"], "typesafe/jev-1.13")
        self.assertEqual(body["state"], "some clause")
        self.assertEqual(body["questions"]["bucket"]["type"], "choice")


if __name__ == "__main__":
    unittest.main()
