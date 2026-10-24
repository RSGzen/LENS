r"""Contract tests for the sandbox façade (fake host; no network, no key).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_rlm_api -v
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from lens.sandbox import rlm_api


class RlmApiTestCase(unittest.TestCase):
    def test_tool_sends_name_and_args(self) -> None:
        with patch.object(rlm_api, "_post", return_value={"ok": True, "result": {"x": 1}}) as post:
            result = rlm_api.tool("echo", text="hi")
        self.assertEqual(result, {"x": 1})
        post.assert_called_once_with("/tool", {"name": "echo", "args": {"text": "hi"}})

    def test_sub_call_sends_prompt_and_context(self) -> None:
        with patch.object(rlm_api, "_post", return_value={"ok": True, "result": "sub"}) as post:
            result = rlm_api.sub_call("summarise", context="ctx")
        self.assertEqual(result, "sub")
        post.assert_called_once_with("/sub_call", {"prompt": "summarise", "context": "ctx"})

    def test_budget_returns_snapshot(self) -> None:
        snapshot = {"left": {"iterations": 3}}
        with patch.object(rlm_api, "_post", return_value={"ok": True, "result": snapshot}) as post:
            self.assertEqual(rlm_api.budget(), snapshot)
        post.assert_called_once_with("/budget", {})

    def test_emit_and_note(self) -> None:
        with patch.object(rlm_api, "_post", return_value={"ok": True, "result": {"ref": "sha256:x"}}) as post:
            self.assertEqual(rlm_api.emit({"a": 1}, label="evidence")["ref"], "sha256:x")
            post.assert_called_once_with("/emit", {"value": {"a": 1}, "label": "evidence"})
        with patch.object(rlm_api, "_post", return_value={"ok": True, "result": None}) as post:
            self.assertIsNone(rlm_api.note("hello"))
            post.assert_called_once_with("/note", {"text": "hello"})

    def test_error_is_propagated(self) -> None:
        with patch.object(rlm_api, "_post", side_effect=rlm_api.RLMError("bad", status=500)):
            with self.assertRaises(rlm_api.RLMError):
                rlm_api.tool("echo", text="hi")


class RlmApiTransportTestCase(unittest.TestCase):
    """The glue the five contracts patch out: verb, URL, header, body.

    ``_post`` is mocked in every contract above, so a wrong verb or a missing
    bearer header passes all five. This is that coverage.
    """

    def setUp(self) -> None:
        self._env = patch.dict(
            "os.environ",
            {"LENS_HOST_RPC_URL": "http://host.test:8765", "LENS_RPC_TOKEN": "tok-123"},
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def _response(self, body: str, status: int = 200):
        from unittest.mock import MagicMock

        response = MagicMock()
        response.text = body
        response.status_code = status
        return response

    def test_tool_posts_json_with_bearer(self) -> None:
        with patch("httpx.request", return_value=self._response('{"ok": true, "result": 1}')) as req:
            self.assertEqual(rlm_api.tool("echo", text="hi"), 1)

        args, kwargs = req.call_args
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], "http://host.test:8765/tool")
        self.assertEqual(kwargs["json"], {"name": "echo", "args": {"text": "hi"}})
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer tok-123"})

    def test_budget_is_a_get_without_a_body(self) -> None:
        # A GET carrying a body leaves those bytes in the keep-alive socket.
        with patch("httpx.request", return_value=self._response('{"ok": true, "result": {}}')) as req:
            self.assertEqual(rlm_api.budget(), {})

        args, kwargs = req.call_args
        self.assertEqual(args[0], "GET")
        self.assertIsNone(kwargs["json"])

    def test_ok_false_raises_with_status_and_error(self) -> None:
        body = '{"ok": false, "error": {"type": "Unauthorized", "message": "bad token"}}'
        with patch("httpx.request", return_value=self._response(body, status=401)):
            with self.assertRaises(rlm_api.RLMError) as caught:
                rlm_api.tool("echo", text="hi")

        self.assertEqual(caught.exception.status, 401)
        self.assertEqual(caught.exception.error["type"], "Unauthorized")

    def test_non_json_body_raises_rlm_error(self) -> None:
        with patch("httpx.request", return_value=self._response("<html>500</html>", status=500)):
            with self.assertRaises(rlm_api.RLMError):
                rlm_api.budget()

    def test_transport_failure_raises_rlm_error(self) -> None:
        import httpx

        with patch("httpx.request", side_effect=httpx.ConnectError("refused")):
            with self.assertRaises(rlm_api.RLMError):
                rlm_api.budget()

    def test_missing_token_fails_before_the_request(self) -> None:
        with patch.dict("os.environ", {"LENS_RPC_TOKEN": ""}):
            with patch("httpx.request") as req:
                with self.assertRaises(rlm_api.RLMError):
                    rlm_api.budget()
        req.assert_not_called()

    def test_facade_never_imports_host_code(self) -> None:
        # The image ships this module alone; a host import is a ModuleNotFoundError
        # inside lens-repl, i.e. the façade never loads.
        source = (rlm_api.__file__ or "")
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        self.assertNotIn("from lens.", text)


if __name__ == "__main__":
    unittest.main()
