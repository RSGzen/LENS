"""Tests for the compound bucket-coverage scorer (no key, no DB)."""

from __future__ import annotations

import unittest

from lens.experiments.coverage_score import auto_verdict, checklist_rows


def item(**fields):
    return {"bucket": "constraints", "item": "x", "check": "mechanical", **fields}


class AutoVerdictTests(unittest.TestCase):
    def test_a_gold_claim_that_is_stated_passes(self) -> None:
        verdict = auto_verdict(
            item(check="gold", claim_indexes=[0]),
            answer="There are 7 papers.",
            gold_claims=[{"value": "7"}],
            cited_arxiv_ids=[],
            claim_units=[],
        )

        self.assertEqual(verdict[0], "pass")

    def test_a_gold_claim_that_is_missing_fails(self) -> None:
        verdict = auto_verdict(
            item(check="gold", claim_indexes=[0]),
            answer="There are nine papers.",
            gold_claims=[{"value": "7"}],
            cited_arxiv_ids=[],
            claim_units=[],
        )

        self.assertEqual(verdict[0], "fail")

    def test_a_year_range_passes_and_fails(self) -> None:
        rule = item(rule="year_from_arxiv_id(cited) in [2020, 2023]")

        self.assertEqual(
            auto_verdict(
                rule, answer="", gold_claims=[], cited_arxiv_ids=["2101.00001"], claim_units=[]
            )[0],
            "pass",
        )
        self.assertEqual(
            auto_verdict(
                rule, answer="", gold_claims=[], cited_arxiv_ids=["2401.00001"], claim_units=[]
            )[0],
            "fail",
        )

    def test_no_cited_id_is_vacuously_in_range(self) -> None:
        verdict = auto_verdict(
            item(rule="year_from_arxiv_id(cited) in [2020, 2023]"),
            answer="None.",
            gold_claims=[],
            cited_arxiv_ids=[],
            claim_units=[],
        )

        self.assertEqual(verdict[0], "pass")

    def test_a_named_paper_set(self) -> None:
        rule = item(rule="all cited arXiv ids are in {2512.24601, 2603.15653}")

        self.assertEqual(
            auto_verdict(
                rule, answer="", gold_claims=[], cited_arxiv_ids=["2512.24601"], claim_units=[]
            )[0],
            "pass",
        )
        self.assertEqual(
            auto_verdict(
                rule, answer="", gold_claims=[], cited_arxiv_ids=["2505.00023"], claim_units=[]
            )[0],
            "fail",
        )

    def test_every_claim_unit_has_an_id(self) -> None:
        rule = item(check="structural", rule="every claim unit carries at least one id")

        self.assertEqual(
            auto_verdict(
                rule,
                answer="A [c1].",
                gold_claims=[],
                cited_arxiv_ids=[],
                claim_units=[("a claim", ["c1"])],
            )[0],
            "pass",
        )
        self.assertEqual(
            auto_verdict(
                rule,
                answer="A.",
                gold_claims=[],
                cited_arxiv_ids=[],
                claim_units=[("a claim", []), ("another", ["c1"])],
            )[0],
            "fail",
        )

    def test_an_unknown_rule_is_human(self) -> None:
        verdict = auto_verdict(
            item(rule="value = 0"),
            answer="0",
            gold_claims=[],
            cited_arxiv_ids=[],
            claim_units=[],
        )

        self.assertEqual(verdict[0], "human")


class WorksheetTests(unittest.TestCase):
    def test_one_row_per_item_with_a_blank_label(self) -> None:
        instance = {
            "instance_id": "cmp-01",
            "gold": {"claims": [{"value": "7"}]},
            "bucket_checklist": [
                {"bucket": "objectives", "item": "count", "check": "gold", "claim_indexes": [0]},
                {
                    "bucket": "constraints",
                    "item": "year",
                    "check": "mechanical",
                    "rule": "year_from_arxiv_id(cited) in [2020, 2023]",
                },
            ],
        }

        rows = checklist_rows(
            instance,
            answer="There are 7 papers.",
            cited_arxiv_ids=["2101.00001"],
            claim_units=[],
            arm="host",
            run_id="r1",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["auto"] for row in rows}, {"pass"})
        self.assertTrue(all(row["label"] == "" for row in rows))
        self.assertTrue(all(row["arm"] == "host" for row in rows))


if __name__ == "__main__":
    unittest.main()
