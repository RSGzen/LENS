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


if __name__ == "__main__":
    unittest.main()
