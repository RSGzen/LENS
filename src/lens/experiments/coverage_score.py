"""Score a compound answer against a bucket checklist — mechanical part only.

`auto_verdict` handles the rules that can be checked exactly: a gold claim is
stated, a year range over the cited arXiv ids, a named paper set, and the
"every claim unit has an id" rule. Anything that needs a judgement returns
``human``; the worksheet keeps a blank ``label`` for it. The human label is the
result for those rows, not a fallback.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from lens.experiments.claim_score import claim_matches
from lens.tools.retrieval import year_from_arxiv_id

_YEAR_RANGE = re.compile(r"in\s*\[\s*(\d{4})\s*,\s*(\d{4})\s*\]")
_PAPER_SET = re.compile(r"in\s*\{([^}]+)\}")
_CLAIM_ID_RULE = "carries at least one id"


def auto_verdict(
    item: Mapping[str, Any],
    *,
    answer: str,
    gold_claims: Sequence[Mapping[str, Any]],
    cited_arxiv_ids: Sequence[str],
    claim_units: Sequence[tuple[str, Sequence[str]]],
) -> tuple[str, str]:
    """Return ``(pass|fail|human, detail)`` for one checklist item."""
    check = str(item.get("check") or "").strip().lower()
    rule = str(item.get("rule") or "")

    if check == "gold":
        indexes = [int(index) for index in item.get("claim_indexes") or []]

        if not indexes:
            return "human", "no claim_indexes"

        missing = [
            index
            for index in indexes
            if not (0 <= index < len(gold_claims)) or not claim_matches(gold_claims[index], answer)
        ]

        if missing:
            return "fail", f"gold claim(s) not stated: {missing}"

        return "pass", f"gold claim(s) stated: {indexes}"

    if check == "mechanical":
        years = _YEAR_RANGE.search(rule)

        if years:
            low, high = int(years.group(1)), int(years.group(2))
            cited = list(dict.fromkeys(str(item) for item in cited_arxiv_ids if item))

            if not cited:
                return "pass", "no cited arXiv id"

            outside = [
                (arxiv_id, year_from_arxiv_id(arxiv_id))
                for arxiv_id in cited
                if (year := year_from_arxiv_id(arxiv_id)) is None or not low <= year <= high
            ]

            if outside:
                return "fail", f"outside {low}-{high}: {outside}"

            return "pass", f"cited years within {low}-{high}"

        paper_set = _PAPER_SET.search(rule)

        if paper_set:
            allowed = {part.strip() for part in paper_set.group(1).split(",") if part.strip()}
            cited = list(dict.fromkeys(str(item) for item in cited_arxiv_ids if item))

            if not cited:
                return "pass", "no cited arXiv id"

            outside = sorted(set(cited) - allowed)

            if outside:
                return "fail", f"outside the named set: {outside}"

            return "pass", "all cited ids in the named set"

    if check in ("mechanical", "structural") and _CLAIM_ID_RULE in rule:
        if not claim_units:
            return "human", "no claim units"

        cited_units = sum(1 for _, ids in claim_units if ids)

        if not cited_units:
            return "human", "no cited claim units"

        missing = [index for index, (_, ids) in enumerate(claim_units) if not ids]

        if missing:
            return "fail", f"claim unit(s) without an id: {missing}"

        return "pass", "every claim unit has an id"

    return "human", "rule needs a human"


def checklist_rows(
    instance: Mapping[str, Any],
    *,
    answer: str,
    cited_arxiv_ids: Sequence[str],
    claim_units: Sequence[tuple[str, Sequence[str]]],
    arm: str = "",
    run_id: str = "",
) -> list[dict[str, str]]:
    """Return one worksheet row per checklist item."""
    gold_claims = list((instance.get("gold") or {}).get("claims") or [])
    rows: list[dict[str, str]] = []

    for item in instance.get("bucket_checklist") or []:
        auto, detail = auto_verdict(
            item,
            answer=answer,
            gold_claims=gold_claims,
            cited_arxiv_ids=cited_arxiv_ids,
            claim_units=claim_units,
        )
        rows.append(
            {
                "instance_id": str(instance.get("instance_id") or ""),
                "arm": arm,
                "run_id": run_id,
                "bucket": str(item.get("bucket") or ""),
                "item": str(item.get("item") or ""),
                "check": str(item.get("check") or ""),
                "auto": auto,
                "label": "",
                "detail": detail,
            }
        )

    return rows
