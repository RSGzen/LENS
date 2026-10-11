r"""M4 — TEI section extraction + hybrid structural chunking (stdlib packer).

Spec: ``MVP Build Roadmap.md`` §4 (M4); ``Thesis Drafts/Tech Stack (Draft).md``
§2 (stages 5-6), §3 (``chunks`` columns), and the proposal §3.4 (deep matching).

What this module does
---------------------
Turns one GROBID ``.tei.xml`` into an ordered list of :class:`Chunk` rows ready
for embedding, in two steps:

1. :func:`extract_sections` — parse the TEI, walk the direct ``<body>`` ``<div>``
   children, and classify each one's ``<head>`` into one of the **6 canonical
   sections** (via the :data:`SECTION_KEYWORDS` registry) or ``Other``.
2. :func:`segment_section` — **hybrid** chunking: keep a whole section as one
   chunk when it is under :data:`config.CHUNK_TOKEN_THRESHOLD` (512 nomic
   tokens); otherwise pack its paragraphs into ``<= threshold`` chunks.

Design decision (C-01 amendment)
--------------------------------
The 512-token split uses this **stdlib paragraph-preserving packer** rather than
``langchain-text-splitters``. Rationale: paragraph integrity is guaranteed and
oversized paragraphs are sentence-aware (the default recursive splitter cuts at
spaces, i.e. mid-sentence); it adds no dependency; and the nomic tokenizer
(already present) is the only length source. The *outcome* — GROBID-section
structure + the frozen 512-token threshold — is unchanged. This is recorded in
``Tech Stack (Draft).md`` §2 stage 6 + §4i and the improvement log (M4 session).

In-text references (Option A, 9 Oct 2026)
-----------------------------------------
GROBID's ``<figure>``/``<formula>`` heads/labels are unreliable (~40% empty,
14% mismatched), but the ``ref @target`` -> asset ``xml:id`` link is exact
(99.6% resolve). So ``chunk_text`` keeps the **visible label** ("Table 2") and
the mention -> asset link is carried as **M5 metadata** (``asset_refs``), *not*
injected into the prose. Citation markers (``bibr``/``foot``) are blanked; every
``<formula>`` is blanked; figure/table/formula *refs* are left untouched.

Token counting
--------------
Chunking is length-agnostic about *which* tokenizer is used: callers pass a
``count_tokens`` callable (the runner builds it from the nomic tokenizer:
``lambda t: len(tok.encode(t, add_special_tokens=False, truncation=False))``).
Unit tests inject ``lambda s: len(s.split())``. This keeps the module pure and
testable without torch.

``tests/test_chunking.py`` is the contract (no key, zero cost).
"""

from __future__ import annotations

import re
import string
import uuid

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TypeVar, cast

from lxml import etree

# TEI namespace used by GROBID output (see a real ``.tei.xml`` under
# ``config.GROBID_XML_DIR_PATH``).
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Hardened parser for untrusted corpus XML. GROBID TEI needs no external entities,
# DTDs, or network access, so disabling them removes XXE (local-file read / SSRF)
# and billion-laughs expansion from files that may contain adversarial content.
_SAFE_XML_PARSER = etree.XMLParser(
    resolve_entities=False,
    no_network=True,
    load_dtd=False,
    dtd_validation=False,
    huge_tree=False,
)

# Fallback ``section_type`` for a top-level section whose head matches no
# canonical keyword (e.g. "Acknowledgements", "Reproducibility Statement").
# Such sections are still embedded and retrievable — no content is dropped.
OTHER_SECTION = "Other"

# Fixed namespace for the deterministic ``chunk_uuid`` (idempotent upsert on a
# multi-day load). Never change this once chunks exist — it would orphan rows.
CHUNK_UUID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# Matcher type: maps a piece of text to its nomic token count. Injected so the
# module never imports torch and tests stay fast.
TokenCounter = Callable[[str], int]

# Regex pattern for normalization of section header
# ^      : Start of the string
# \d+    : One or more digits
# (?:\.\d+)* : Zero or more groups of a dot followed by digits (e.g., .1, .4.2)
# \.?    : An optional trailing dot
# \s*    : Any optional trailing whitespace
header_pattern = r"^\d+(?:\.\d+)*\.?\s*"

# Splits on terminal punctuation (. ! ?) followed by whitespace or end of string
split_pattern = r"(?<=[.!?])\s+"


# -------------------------------------------------------- section registry
# The 6 canonical section types (Tech Stack §2 stage 5) and the head keywords
# that map into each. Matching is on the *normalized* head (see
# :func:`normalize_head`). Heads are matched case-insensitively as substrings;
# the **longest matching keyword wins** (so "experimental results" -> Result/
# Discussion rather than Experiment), and ties break by the order below.
#
# This is domain data, not logic — extend the tuples as you inspect real heads.
SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Background Study": (
        "introduction",
        "background",
        "motivation",
        "problem statement",
        "preliminaries",
        "preliminary",
        "notation",
    ),
    "Literature Review": (
        "related work",
        "related works",
        "prior work",
        "previous work",
        "state of the art",
        "state-of-the-art",
        "literature",
        "literature review",
        "review",
    ),
    "Methodology": (
        "methodology",
        "methods",
        "method",
        "approach",
        "proposed",
        "framework",
        "architecture",
        "model",
        "algorithm",
        "implementation",
        "system design",
    ),
    "Experiment": (
        "experimental setup",
        "experiments",
        "experiment",
        "experimental",
        "evaluation",
        "setup",
        "datasets",
        "dataset",
        "benchmarks",
        "benchmark",
        "training details",
    ),
    "Result/Discussion": (
        "results and discussion",
        "experimental results",
        "ablation study",
        "ablations",
        "ablation",
        "results",
        "result",
        "discussion",
        "analysis",
        "findings",
        "performance",
    ),
    "Future Work": (
        "conclusions and future work",
        "conclusion and future work",
        "future work",
        "concluding remarks",
        "conclusions",
        "conclusion",
        "limitations",
        "future",
        "summary",
    ),
}


# ------------------------------------------------------------- data model
@dataclass(frozen=True)
class Section:
    """One top-level classified section of a paper, paragraphs in reading order."""

    section_type: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class Chunk:
    """One embeddable chunk (maps to a ``chunks`` row minus the embedding).

    ``section_order`` is the 0-based, monotonic index of the chunk **within its
    paper** (section-major, then reading order). It doubles as the section
    sequence and is the stable component of the deterministic ``chunk_uuid``
    (idempotent re-run). If the chunking algorithm changes later, clear the
    paper's rows before re-loading (or the old ``section_order`` keys remain).
    """

    section_type: str
    section_order: int
    chunk_text: str
    token_count: int


# --------------------------------------------------------------- extraction
def normalize_head(head: str) -> str:
    """Normalize a ``<head>`` for keyword matching.

    1. Lowercase.
    2. Strip leading section numbering — ``"3.1 Experimental Setup"`` and
       ``"3.1. Experimental Setup"`` both -> ``"experimental setup"``; also drop
       a leading ``"chapter"``/``"section"`` word if present.
    3. Collapse internal whitespace and strip surrounding punctuation.

    Returns the empty string for a missing/empty head (callers map that to
    :data:`OTHER_SECTION`).
    """
    if not head:
        return ""

    normalized = head.lower()
    normalized = re.sub(header_pattern, "", normalized)          # leading numbering
    normalized = re.sub(r"^(chapter|section)\s+", "", normalized)  # leading word
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.strip(string.punctuation + " ").strip()


def classify_section(head: str) -> str:
    """Map a section head to one of the 6 canonical types, else :data:`OTHER_SECTION`.

    Match on :func:`normalize_head` output against :data:`SECTION_KEYWORDS`.
    Use **longest-matching-keyword wins**; ties break by the registry order.
    A blank head -> :data:`OTHER_SECTION`.

    Example
    -------
    ``"1. Introduction"`` -> ``"Background Study"`` ·
    ``"3.1 Method"`` -> ``"Methodology"`` ·
    ``"Experimental Results"`` -> ``"Result/Discussion"`` ·
    ``"Acknowledgements"`` -> ``"Other"``.
    """
    normalized = normalize_head(head)
    if not normalized:
        return OTHER_SECTION

    best_section = OTHER_SECTION
    best_length = 0
    for section_type, section_keywords in SECTION_KEYWORDS.items():
        for keyword in section_keywords:
            if keyword in normalized and len(keyword) > best_length:
                best_section = section_type
                best_length = len(keyword)

    return best_section


def replace_node_with_text(node: etree._Element, new_text: str = "") -> None:
    """Replace ``node`` in the tree with ``new_text`` (+ its own tail)."""
    # Require parent element to delete a child node
    parent = node.getparent()
    if parent is None:
        return

    # Grabs any text following the target node
    tail = node.tail or ""
    combined_text = new_text + tail

    # Find previous node to attach the text to
    prev = node.getprevious()

    # Have siblings --> Attach to sibling's tail
    if prev is not None:
        prev.tail = (prev.tail or "") + combined_text

    # No siblings --> Attach to parent's tail
    else:
        parent.text = (parent.text or "") + combined_text

    # Remove current node
    parent.remove(node)


def extract_sections(tei_path: str | Path) -> list[Section]:
    """Parse one ``.tei.xml`` into an ordered list of :class:`Section`.

    1. ``etree.parse(str(tei_path))`` and locate ``.//tei:body`` with
       :data:`TEI_NS`.
    2. Walk the **direct** ``<body>`` ``<div>`` children only (nested subsections
       belong to their parent section). Classify the child's ``<head>`` (if any)
       with :func:`classify_section` (which normalizes internally).
    3. Skip non-content divs: ``@type`` in ``{"references", "acknowledgement",
       "acknowledgments", "appendix"}``, and any div with no paragraphs.
    4. Collect paragraph text from every ``.//tei:p`` in the div (nested
       subsections included), element-aware (do **not** naively ``itertext()``):

       - every ``<formula>`` (incl. inline in ``<p>``) -> **blanked**.
       - ``<ref type="bibr">`` and ``<ref type="foot">`` -> **blanked**.
       - ``<ref type="figure" | "table" | "formula">`` -> kept as-is (their
         visible label, e.g. "Table 2"); the ref -> asset link is M5 metadata.
       - collapse runs of whitespace to a single space and drop empty paragraphs.
    5. Return the sections in document order. A paper with no body/divs -> ``[]``.
    """
    tree = etree.parse(str(tei_path), _SAFE_XML_PARSER)
    root = tree.getroot()

    # Blank every <formula> (GROBID math is garbled; covers inline formulas too).
    for formula in root.findall(".//tei:formula", TEI_NS):
        replace_node_with_text(formula, "")

    # Blank citation markers that are meaningless without the source bibliography.
    for ref in root.findall('.//tei:ref[@type="bibr"]', TEI_NS):
        replace_node_with_text(ref, "")
    for ref in root.findall('.//tei:ref[@type="foot"]', TEI_NS):
        replace_node_with_text(ref, "")

    # Find the body element (this automatically excludes the <abstract> which
    # lives in <profileDesc> / <front>).
    body = root.find(".//tei:body", TEI_NS)
    if body is None:
        return []

    skip_types = {"references", "acknowledgement", "acknowledgments", "appendix"}
    section_list: list[Section] = []

    for div in body.findall("tei:div", TEI_NS):
        if (div.get("type") or "") in skip_types:
            continue

        head = div.find("tei:head", TEI_NS)
        head_text = head.text if head is not None and head.text else ""
        section_type = classify_section(head_text) if head_text else OTHER_SECTION

        paragraphs: list[str] = []
        for paragraph in div.findall(".//tei:p", TEI_NS):

            p_iterText = paragraph.itertext()

            text = re.sub(r"\s+", " ", "".join(cast(Iterator[str], p_iterText))).strip()
            if text:
                paragraphs.append(text)

        if paragraphs:
            section_list.append(
                Section(section_type=section_type, paragraphs=tuple(paragraphs))
            )

    return section_list


# --------------------------------------------------------------- segmenting
# Counting is done on strings no longer than this many characters, so a single
# tokenizer call can never see a string long enough to trigger HF's
# "sequence length > max_seq_length" warning (#tokens <= #chars < 8192).
_SAFE_COUNT_CHARS = 4000


def _count_tokens_bounded(text: str, count_tokens: TokenCounter, cap: int = _SAFE_COUNT_CHARS) -> int:
    """``count_tokens(text)`` without ever tokenising one huge string.

    Splits long text by sentence -> whitespace -> fixed-width windows and sums
    the piece counts, so no single tokenizer call sees more than ``cap`` chars
    (hence at most ``cap`` tokens). The count of an oversized unit is therefore
    approximate — fine, since it is only used to route the unit to
    :func:`_split_oversize`.
    """
    if len(text) <= cap:
        return count_tokens(text)
    for splitter in (split_pattern, r"\s+"):
        parts = [p for p in re.split(splitter, text) if p]
        if len(parts) > 1:
            return sum(_count_tokens_bounded(p, count_tokens, cap) for p in parts)
    return sum(count_tokens(text[i:i + cap]) for i in range(0, len(text), cap))


def _split_oversize(text: str, count_tokens: TokenCounter, threshold: int) -> list[str]:
    """Split a single over-threshold string: sentences -> words -> characters."""
    sentences = [s for s in re.split(split_pattern, text) if s]
    if len(sentences) > 1:
        return _pack_flat(sentences, count_tokens, threshold, " ")

    words = text.split()
    if len(words) > 1:
        return _pack_flat(words, count_tokens, threshold, " ")

    # Last resort: an unbreakable token longer than the threshold -> characters.
    return _pack_flat(list(text), count_tokens, threshold, "")


def _pack_flat(
    units: list[str],
    count_tokens: TokenCounter,
    threshold: int,
    sep: str,
) -> list[str]:
    """Greedily pack pre-split ``units`` into chunks joined by ``sep`` (<= threshold).

    Each unit is tokenised **once** (its count is cached), so the packer is O(n)
    in the number of units and never tokenises a multi-unit concatenation. That
    avoids the HF tokenizer's "sequence length > max" warning (which the old code
    triggered by counting the joined section) and removes the O(n^2)
    re-tokenisation of the growing candidate. ``sep`` tokens are reserved per join
    so the boundary cost is accounted for.
    """
    sep_tokens = _count_tokens_bounded(sep, count_tokens) if sep else 0
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _count_tokens_bounded(unit, count_tokens)

        if unit_tokens <= threshold:
            extra = unit_tokens + (sep_tokens if current else 0)
            if current_tokens + extra <= threshold:
                current.append(unit)
                current_tokens += extra
                continue
            # The next unit would overflow -> seal the current chunk first.
            if current:
                chunks.append(sep.join(current))
                current = []
            current = [unit]
            current_tokens = unit_tokens
        else:
            # This unit alone exceeds the threshold -> seal, then split it.
            if current:
                chunks.append(sep.join(current))
                current = []
                current_tokens = 0
            chunks.extend(_split_oversize(unit, count_tokens, threshold))

    if current:
        chunks.append(sep.join(current))

    return chunks


def segment_section(
    section: Section,
    count_tokens: TokenCounter,
    threshold: int,
    overlap: int = 0,
) -> list[str]:
    """Split one :class:`Section` into chunk texts of ``<= threshold`` tokens.

    Hybrid rule (Roadmap §4 M4; proposal §3.4):

    - Join the section paragraphs with ``"\\n\\n"``. If the whole section is
      ``<= threshold`` tokens, return it as a **single** chunk (the common case:
      GROBID's own segmentation is kept intact).
    - Otherwise **pack whole paragraphs** greedily, in order, until adding the
      next paragraph would exceed ``threshold``; then start a new chunk. Paragraph
      boundaries are never crossed while packing.
    - A single paragraph that alone exceeds ``threshold`` cannot be packed, so it
      is split further: prefer sentence boundaries (``. `` / ``? `` / ``! ``),
      then whitespace, then (last resort) characters, so no chunk exceeds
      ``threshold``.
    - ``overlap`` (tokens) is carried forward from the tail of the previous chunk
      when ``> 0``; with the default ``0`` (structural chunking) chunks do not
      overlap. Implement overlap only if you enable it in config.

    Paragraph text is preserved verbatim within a chunk (joined with ``"\\n\\n"``)
    so the stored ``chunk_text`` reads as continuous prose.

    Tokenisation is per unit (paragraph / sentence / word), never over the whole
    joined section — so a large section does not trigger the HF tokenizer's
    ``> max_seq_length`` warning, and packing stays O(n).

    Returns the list of chunk texts in reading order (>= 1 for a non-empty
    section; ``[]`` for an empty section).
    """
    paragraphs = [p for p in section.paragraphs if p and p.strip()]
    if not paragraphs:
        return []

    return _pack_flat(paragraphs, count_tokens, threshold, "\n\n")


def chunk_tei(
    tei_path: str | Path,
    count_tokens: TokenCounter,
    threshold: int,
    overlap: int = 0,
) -> list[Chunk]:
    """Full M4 chunking for one paper: extract sections, segment, number them.

    Composes :func:`extract_sections` + :func:`segment_section` and assigns each
    chunk a monotonic ``section_order`` (0-based, across the whole paper), then
    its ``token_count`` via ``count_tokens``.
    """
    chunk_list: list[Chunk] = []
    section_order = 0

    for section in extract_sections(tei_path=tei_path):
        for chunk_text in segment_section(
            section=section, count_tokens=count_tokens, threshold=threshold
        ):
            chunk_list.append(
                Chunk(
                    section_type=section.section_type,
                    section_order=section_order,
                    chunk_text=chunk_text,
                    token_count=count_tokens(chunk_text),
                )
            )
            section_order += 1

    return chunk_list
