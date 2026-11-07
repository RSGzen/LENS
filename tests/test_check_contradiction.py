"""Tests for the contradiction negative-control helpers.

The script's model call is not exercised here; the perturbation rule is the part
that decides what the control is allowed to change.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


check_contradiction = _load("check_contradiction")


class PerturbNumberTests(unittest.TestCase):
    def test_a_decimal_is_replaced_and_reported(self) -> None:
        text, original, replacement = check_contradiction.perturb_number(
            "RLM reaches 56.0 accuracy on OOLONG."
        )

        self.assertIn("73.0", text)
        self.assertNotIn("56.0", text)
        self.assertEqual(original, 56.0)
        self.assertEqual(replacement, 73.0)

    def test_an_integer_is_replaced_and_reported(self) -> None:
        text, original, replacement = check_contradiction.perturb_number(
            "The window is K = 64 rounds."
        )

        self.assertIn("81", text)
        self.assertEqual(original, 64.0)
        self.assertEqual(replacement, 81.0)

    def test_a_section_label_is_not_a_measurement(self) -> None:
        self.assertIsNone(check_contradiction.perturb_number("Objective 2 is below."))

    def test_an_arxiv_id_is_not_perturbed(self) -> None:
        self.assertIsNone(check_contradiction.perturb_number("2512.24601"))

    def test_a_citation_is_stripped_before_the_number_is_chosen(self) -> None:
        text = "The system uses 3 rounds (a861854f-81de-5011-bfc5-2a4ff6ee159e)."
        clean = check_contradiction.claim_text(text)
        perturbed, original, _ = check_contradiction.perturb_number(clean)

        self.assertNotIn("a861854f", clean)
        self.assertIn("20", perturbed)
        self.assertEqual(original, 3.0)

    def test_text_without_a_number_is_none(self) -> None:
        self.assertIsNone(check_contradiction.perturb_number("no numbers here"))


if __name__ == "__main__":
    unittest.main()
