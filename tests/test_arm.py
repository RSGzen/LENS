r"""Contract tests for the arm fingerprint (M9) - no key, no DB, no Docker.

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_arm -v
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from lens import config
from lens.experiments.arm import arm_fingerprint


class ArmFingerprintTestCase(unittest.TestCase):
    """The fingerprint must describe the run, not yesterday's defaults."""

    def test_shape_is_json_and_complete(self) -> None:
        arm = arm_fingerprint(recorded=True, replay=False)

        # It is written into manifest.json, so it must serialise unchanged.
        self.assertEqual(json.loads(json.dumps(arm)), arm)
        self.assertEqual(set(arm), {"retrieval", "kg", "gate", "mode", "models"})
        self.assertEqual(arm["retrieval"]["k"], config.RETRIEVAL_K)
        self.assertEqual(arm["retrieval"]["c"], config.RETRIEVAL_C_TOP)
        self.assertEqual(arm["mode"]["recorded"], True)
        self.assertEqual(arm["mode"]["replay"], False)

    def test_reads_config_at_call_time(self) -> None:
        # Patched config must show up: the fingerprint describes the run it is
        # attached to, so a stale copy would be worse than no copy.
        with patch.object(config, "RETRIEVAL_K", 50), patch.object(config, "KG_ENABLED", False):
            arm = arm_fingerprint(recorded=False, replay=True)

        self.assertEqual(arm["retrieval"]["k"], 50)
        self.assertEqual(arm["kg"]["enabled"], False)

    def test_records_model_ids_not_specs(self) -> None:
        arm = arm_fingerprint(recorded=True, replay=True)

        self.assertEqual(arm["models"]["root"], config.MODEL_REGISTRY["root"].id)
        self.assertEqual(arm["models"]["jev"], config.JEV_MODEL_ID)


class ModelOverrideTestCase(unittest.TestCase):
    """``LENS_ROOT_MODEL`` changes the id without breaking provider pinning (M9)."""

    def test_override_same_vendor_keeps_pins(self) -> None:
        spec = config.MODEL_REGISTRY["root"]
        overridden = config._model_override(spec, "openai/gpt-5-mini")

        self.assertEqual(overridden.id, "openai/gpt-5-mini")
        self.assertEqual(overridden.provider_only, spec.provider_only)

    def test_override_other_vendor_drops_pins(self) -> None:
        # A pin is a vendor slug: keeping ("openai",) on a qwen id would pin the
        # wrong vendor and fail at the transport with a confusing empty response.
        overridden = config._model_override(config.MODEL_REGISTRY["root"], "qwen/qwen3-coder")

        self.assertEqual(overridden.id, "qwen/qwen3-coder")
        self.assertEqual(overridden.provider_only, ())
        self.assertEqual(overridden.provider_order, ())
        self.assertFalse(overridden.allow_fallbacks)

    def test_no_override_is_a_noop(self) -> None:
        spec = config.MODEL_REGISTRY["sub"]

        self.assertIs(config._model_override(spec, None), spec)
        self.assertIs(config._model_override(spec, ""), spec)


if __name__ == "__main__":
    unittest.main()
