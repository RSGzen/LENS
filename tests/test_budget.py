r"""Contract tests for the budget (no key, no network, zero cost).

Run:  .\.venv\Scripts\python.exe -m unittest tests.test_budget -v
"""

from __future__ import annotations

import json
import unittest

from lens.agent.budget import Budget, BudgetLimits
from lens.models import CallResult, JevResult


def make_result(*, tokens_in: int = 10, tokens_out: int = 5, cost: float = 0.01) -> CallResult:
    return CallResult(
        text="ok",
        message={"role": "assistant", "content": "ok"},
        model="openai/gpt-5",
        provider="openai",
        model_snapshot=None,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=1,
        cost_usd=cost,
    )


def make_jev(*, tokens_in: int = 10, cost: float = 0.001) -> JevResult:
    return JevResult(
        answers={},
        model="typesafe/jev-1.13",
        provider="TypeSafe",
        tokens_in=tokens_in,
        cost_usd=cost,
        latency_ms=1,
    )


class BudgetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = BudgetLimits(
            max_iterations=2,
            max_tokens=100,
            max_cost_usd=1.0,
            max_exec_block_s=10.0,
            max_backtracks=1,
        )
        self.budget = Budget(limits=self.limits)

    def test_initial_snapshot_is_zeroed(self) -> None:
        snap = self.budget.snapshot()
        self.assertEqual(snap["used"]["iterations"], 0)
        self.assertEqual(snap["used"]["tokens_in"], 0)
        self.assertEqual(snap["used"]["cost_usd"], 0.0)
        self.assertEqual(snap["left"]["iterations"], 2)
        self.assertEqual(snap["left"]["tokens"], 100)
        self.assertIsNone(snap["stop_reason"])

    def test_record_chat_accumulates_usage(self) -> None:
        self.budget.record_chat(make_result(tokens_in=20, tokens_out=7, cost=0.02))
        self.budget.record_chat(make_result(tokens_in=3, tokens_out=2, cost=0.01))
        snap = self.budget.snapshot()
        self.assertEqual(snap["used"]["tokens_in"], 23)
        self.assertEqual(snap["used"]["tokens_out"], 9)
        self.assertAlmostEqual(snap["used"]["cost_usd"], 0.03)

    def test_record_jev_counts_toward_the_cost_cap(self) -> None:
        # JEV bills on input only; leaving it out would understate spend.
        self.budget.record_jev(make_jev(tokens_in=40, cost=0.02))
        snap = self.budget.snapshot()
        self.assertEqual(snap["used"]["tokens_in"], 40)
        self.assertEqual(snap["used"]["tokens_out"], 0)
        self.assertAlmostEqual(snap["used"]["cost_usd"], 0.02)

    def test_iteration_cap_stops_the_loop(self) -> None:
        self.budget.record_iteration()
        self.assertTrue(self.budget.can_continue())
        self.budget.record_iteration()
        self.assertFalse(self.budget.can_continue())
        self.assertEqual(self.budget.stop_reason(), "iterations")

    def test_token_cap_stops_the_loop(self) -> None:
        self.budget.record_chat(make_result(tokens_in=90, tokens_out=10, cost=0.0))
        self.assertFalse(self.budget.can_continue())
        self.assertEqual(self.budget.stop_reason(), "tokens")

    def test_cost_cap_stops_the_loop(self) -> None:
        self.budget.record_chat(make_result(tokens_in=1, tokens_out=1, cost=1.0))
        self.assertFalse(self.budget.can_continue())
        self.assertEqual(self.budget.stop_reason(), "cost")

    def test_backtrack_cap(self) -> None:
        self.budget.record_backtrack()
        self.assertFalse(self.budget.can_continue())
        self.assertEqual(self.budget.stop_reason(), "backtracks")

    def test_exec_seconds_never_stops_the_loop(self) -> None:
        # The 120 s cap is per block (the runner enforces it), so cumulative
        # sandbox time must not end a run early with budget left.
        for _ in range(8):
            self.budget.record_exec(25.0)  # 200 s total, well past 120 s
        self.assertTrue(self.budget.can_continue())
        self.assertIsNone(self.budget.stop_reason())
        assert self.budget.snapshot()["used"]["exec_seconds"] == 200.0

    def test_left_is_clamped_and_has_no_exec_budget(self) -> None:
        # Over-run every cap: `left` may never go negative, and exec time has no
        # `left` entry because it is not a cumulative budget.
        self.budget.record_iteration()
        self.budget.record_iteration()
        self.budget.record_chat(make_result(tokens_in=500, tokens_out=500, cost=5.0))
        snap = self.budget.snapshot()
        self.assertEqual(set(snap["left"]), {"iterations", "tokens", "cost_usd", "backtracks"})
        self.assertNotIn("exec_block_s", snap["left"])
        for value in snap["left"].values():
            self.assertGreaterEqual(value, 0)

    def test_snapshot_is_json_serializable(self) -> None:
        json.dumps(self.budget.snapshot())


if __name__ == "__main__":
    unittest.main()
