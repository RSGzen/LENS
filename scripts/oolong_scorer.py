"""The OOLONG release's `trec_coarse` scorer, vendored unchanged.

Source: `github.com/abertsch72/oolong`, `src/eval/eval_helpers.py`, MIT licence,
Copyright (c) 2025 Amanda Bertsch, fetched 14 Nov 2026. Only the two `synth_*`
functions are copied; the module's `litellm`/`datasets` imports belong to the
release's evaluation driver, not to the scoring rule.

The rule, unchanged: take the text after the **last** colon, strip the markdown
and bracket decorations models add, map the three comparison phrasings, then
exact-match against `ast.literal_eval(answer)[0]`, with `0.75 ** |gold - guess|`
partial credit for numeric answers.
"""

from __future__ import annotations

import ast
from datetime import datetime

SOURCE = "abertsch72/oolong src/eval/eval_helpers.py (MIT), fetched 2026-11-14"


def synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    """Parse one answer into a candidate; return it with a parse-confidence label."""
    parse_confidence = "low"
    if ":" not in answer:  # bad start
        if len(answer) < 20:  # it's short, return the whole thing
            return answer, parse_confidence
        else:
            return answer.split()[-1], parse_confidence
    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace("*", "")  # OpenAI models like bolding
    candidate_answer = candidate_answer.replace("[", "")
    candidate_answer = candidate_answer.replace("]", "")  # Anthropic uses brackets
    parse_confidence = "med"
    if "User:" in answer or "Answer:" in answer or "Date:" in answer or "Label" in answer:
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"

    return candidate_answer, parse_confidence


def synth_process_response(datapoint: dict, output: str, model: str = "lens") -> dict:
    """Score one answer against one datapoint; return the release's result dict."""
    score = 0
    gold = (
        ast.literal_eval(datapoint["answer"])[0]
        if "datetime" not in str(datapoint["answer"])
        else datetime.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
    )

    trimmed_output, parse_confidence = synth_attempt_answer_parse(output)
    if str(trimmed_output) == str(gold):
        score = 1
    elif str(trimmed_output) in ["more common", "less common", "same frequency"]:
        if str(trimmed_output) in str(gold):
            score = 1
    elif datapoint["answer_type"] == "ANSWER_TYPE.NUMERIC":
        try:
            trimmed_output = int(trimmed_output)
            gold = int(gold)
            score = 0.75 ** (abs(gold - trimmed_output))
        except Exception:  # noqa: BLE001 - the release records a low-confidence parse
            parse_confidence = "low"

    return {
        "id": datapoint["id"],
        "context_window_id": datapoint.get("context_window_id"),
        "dataset": datapoint.get("dataset"),
        "model": model,
        "attempted_parse": str(trimmed_output),
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "answer": str(gold),
    }


def score(answer: str, instance: dict) -> float:
    """Return the release's score for one answer against one instrument instance."""
    datapoint = {
        "id": instance["instance_id"],
        "answer": instance["gold"]["answer"],
        "answer_type": instance.get("answer_type"),
        "context_window_id": (instance.get("meta") or {}).get("context_window_id"),
        "dataset": (instance.get("meta") or {}).get("dataset"),
    }

    return float(synth_process_response(datapoint, answer)["score"])
