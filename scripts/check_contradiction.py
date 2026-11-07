"""Negative control: does the claim verifier catch a wrong number on real text?

Two checks, because they fail independently:

1. **Verifier control.** A sentence from a recorded chunk states a number; a
   frame is built with a different number and the same sentence. The verifier
   must return ``contradicted``. No model call.
2. **Extractor check.** A claim unit from the recorded answer is perturbed by
   changing its first number, then the real extractor runs on it with the
   original cited chunks. This reports what the extractor does with a claim the
   chunk contradicts; it does not fail the script, because a ``reports`` frame
   for a numeric claim is a finding about the extractor rather than about the
   verifier.

Run it against a recorded run with an answer and cited chunks.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone

from lens.clients import LensClient
from lens.observability.citations import claim_units, recorded_chunk_texts
from lens.observability.claim_extract import extract_frames
from lens.observability.claims import ClaimFrame, verify_frame, verify_units
from lens.observability.trajectory import TraceCtx, Trajectory

_DECIMAL = re.compile(r"\b\d+\.\d+\b")
#: One to three digits, not part of a longer number or an arXiv id. A four-digit
#: year is excluded because perturbing it would change the id, not the claim.
_INTEGER = re.compile(r"(?<![.\d])\b\d{1,3}\b(?![.\d])")

#: Chunk ids, arXiv ids and their bracketed forms. A number inside one is a
#: citation, not a claim, and must not be the one the control perturbs.
_CITATION = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|\b\d{4}\.\d{4,5}(?:v\d+)?\b"
    r"|\([^)]*\)|\[[^\]]*\]"
)


def claim_text(text: str) -> str:
    """Return ``text`` without chunk ids, arXiv ids or their brackets."""
    return " ".join(_CITATION.sub(" ", text).split())


def perturb_number(text: str) -> tuple[str, float, float] | None:
    """Return ``(text_with_a_new_number, old, new)``, or ``None`` if no number."""
    for pattern in (_DECIMAL, _INTEGER):
        match = pattern.search(text)

        if match is None:
            continue

        original = float(match.group(0))

        if re.fullmatch(r"\d{4}\.\d{4,5}", match.group(0)):
            # An arXiv id, not a measurement: perturbing it changes the citation.
            continue

        replacement = original + 17.0
        rendered = f"{replacement:.1f}" if "." in match.group(0) else str(int(replacement))

        return (
            text[: match.start()] + rendered + text[match.end() :],
            original,
            replacement,
        )

    return None


def _numeric_sentence(text: str, number: float) -> str | None:
    """Return the first sentence of ``text`` that states ``number``."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if f"{number:g}" in sentence:
            return " ".join(sentence.split())

    return None


def _stamp() -> str:
    """Return a filename-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="recorded run id")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    trajectory = Trajectory(args.run)
    answer_path = trajectory.run_dir / "answer.md"

    if not answer_path.is_file():
        print("run has no answer.md", file=sys.stderr)
        return 2

    answer = answer_path.read_text(encoding="utf-8", errors="replace")
    units = claim_units(answer, include_uncited=True)
    texts = recorded_chunk_texts(trajectory)

    selected = None

    for unit, ids in units:
        clean = claim_text(unit)
        perturbed = perturb_number(clean)

        if perturbed is None or not ids:
            continue

        chunk_id = next((str(value) for value in ids if str(value) in texts), None)

        if chunk_id is None:
            continue

        selected = (clean, ids, chunk_id, perturbed)
        break

    if selected is None:
        print("no numeric claim unit with a recorded chunk in this answer", file=sys.stderr)
        return 2

    unit, ids, chunk_id, (perturbed_text, original, replacement) = selected
    print(f"unit: {unit[:160]}")
    print(f"number: {original:g} -> {replacement:g}")

    sentence = _numeric_sentence(texts[chunk_id], original)

    if sentence is None:
        print("[control] no sentence in the cited chunk states the original number")
    else:
        frame = ClaimFrame(
            unit_index=0,
            text=unit,
            predicate="measures",
            subject="validation",
            value=str(original + 1),
            quote=sentence,
            chunk_id=chunk_id,
        )
        verdict = verify_frame(frame, texts)
        print(f"[control] verifier: {verdict.status} - {verdict.reason}")

        if verdict.status != "contradicted":
            print("the verifier did not catch a wrong number; stop", file=sys.stderr)
            return 1

    scratch = Trajectory(f"claims-contradiction-{_stamp()}")
    client = LensClient(trajectory=scratch)
    trace_ctx = TraceCtx(trace_id=scratch.run_id, step_idx=1, actor="sub")
    frames, errors = extract_frames(client, [(perturbed_text, ids)], trace_ctx, texts=texts)
    report = verify_units([(perturbed_text, ids)], frames, texts, run_id=args.run)

    print(f"[live] extractor: {report.counts} coverage {report.coverage:.0%}")

    for frame, verdict in zip(report.frames, report.verdicts):
        print(f"  {frame.predicate}: {verdict.status} - {verdict.reason}")

    for error in errors:
        print(f"  error: {error}")

    if not report.frames:
        print("the extractor returned no frame; the check is inconclusive", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
