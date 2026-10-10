r"""Contract tests for M4 section classification + structural chunking.

No key, zero cost, no DB, no torch — ``count_tokens`` is injected as
``lambda s: len(s.split())``. These define the behavior ``chunking.py`` must
implement (they FAIL until the student fills the logic bodies — expected for a
scaffold-and-fill module).

Run:  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lens.chunking import (
    OTHER_SECTION,
    Section,
    chunk_tei,
    classify_section,
    extract_sections,
    normalize_head,
    segment_section,
)

WORDS = lambda s: len(s.split())  # noqa: E731 — test token counter


TEI_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc><titleStmt><title>t</title></titleStmt></fileDesc></teiHeader>
  <text>
    <front><div><p>This abstract is front matter and must be ignored.</p></div></front>
    <body>
      <div><head n="1">Introduction</head>
        <p>Intro paragraph one.</p><p>Intro paragraph two.</p></div>
      <div><head n="2">Related Work</head><p>Prior work paragraph.</p></div>
      <div><head n="3">Method</head><p>Our method paragraph.</p></div>
      <div type="references"><listBibl><bibl>ref</bibl></listBibl></div>
    </body>
  </text>
</TEI>
"""


class ClassifySectionTest(unittest.TestCase):
    def test_maps_canonical_heads(self) -> None:
        self.assertEqual(classify_section("1. Introduction"), "Background Study")
        self.assertEqual(classify_section("2. Related Work"), "Literature Review")
        self.assertEqual(classify_section("3. Methodology"), "Methodology")
        self.assertEqual(classify_section("4. Experimental Setup"), "Experiment")
        self.assertEqual(classify_section("5. Results and Discussion"), "Result/Discussion")
        self.assertEqual(classify_section("6. Conclusion and Future Work"), "Future Work")

    def test_unmatched_head_is_other(self) -> None:
        self.assertEqual(classify_section("Acknowledgements"), OTHER_SECTION)
        self.assertEqual(classify_section(""), OTHER_SECTION)

    def test_longest_keyword_wins(self) -> None:
        # "experimental results" must beat the shorter "experiment"/"results".
        self.assertEqual(classify_section("Experimental Results"), "Result/Discussion")

    def test_normalize_strips_numbering_and_case(self) -> None:
        self.assertEqual(normalize_head("3.1 Experimental Setup"), "experimental setup")
        self.assertEqual(normalize_head("1. Introduction"), "introduction")
        self.assertEqual(normalize_head(""), "")


class ExtractSectionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "x.tei.xml"
        self.path.write_text(TEI_TEMPLATE, encoding="utf-8")

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_extracts_typed_sections_in_order(self) -> None:
        sections = extract_sections(self.path)
        self.assertEqual(
            [s.section_type for s in sections],
            ["Background Study", "Literature Review", "Methodology"],
        )

    def test_groups_paragraphs_per_section(self) -> None:
        sections = extract_sections(self.path)
        self.assertEqual(
            sections[0].paragraphs,
            ("Intro paragraph one.", "Intro paragraph two."),
        )

    def test_skips_front_matter_and_references(self) -> None:
        sections = extract_sections(self.path)
        joined = " ".join(p for s in sections for p in s.paragraphs)
        self.assertNotIn("front matter", joined)
        self.assertNotIn("ref", joined)

    def test_strips_citation_markers_keeps_reference_labels(self) -> None:
        tei = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
  <div><head n="1">Introduction</head>
    <p>The foundation is here<ref type="bibr" target="#b13">[14]</ref>. Equation <ref type="formula" target="#formula_7">6</ref> holds.</p>
  </div>
</body></text></TEI>
"""
        path = Path(self._dir.name) / "refs.tei.xml"
        path.write_text(tei, encoding="utf-8")

        text = extract_sections(path)[0].paragraphs[0]

        self.assertNotIn("[14]", text)
        self.assertIn("6", text)
        self.assertEqual(text, "The foundation is here. Equation 6 holds.")


class SegmentSectionTest(unittest.TestCase):
    def test_under_threshold_is_single_chunk(self) -> None:
        section = Section("Background Study", ("one two three", "four five"))
        self.assertEqual(segment_section(section, WORDS, threshold=10), ["one two three\n\nfour five"])

    def test_packs_whole_paragraphs(self) -> None:
        section = Section("Methodology", ("aa bb", "cc dd", "ee ff"))
        # 2 + 2 = 4 fits (<=4); adding "ee ff" would exceed -> second chunk.
        self.assertEqual(segment_section(section, WORDS, threshold=4), ["aa bb\n\ncc dd", "ee ff"])

    def test_oversized_paragraph_is_split(self) -> None:
        words = " ".join(f"w{i}" for i in range(12))
        section = Section("Methodology", (words,))
        chunks = segment_section(section, WORDS, threshold=5)
        self.assertTrue(all(WORDS(c) <= 5 for c in chunks), chunks)
        self.assertEqual(" ".join(" ".join(chunks).split()), words)

    def test_empty_section_yields_no_chunks(self) -> None:
        self.assertEqual(segment_section(Section("Methodology", ()), WORDS, threshold=5), [])


class ChunkTeiTest(unittest.TestCase):
    def test_orders_chunks_across_sections(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.tei.xml"
            path.write_text(TEI_TEMPLATE, encoding="utf-8")
            chunks = chunk_tei(path, WORDS, threshold=3)
        self.assertEqual([c.section_order for c in chunks], list(range(len(chunks))))
        self.assertEqual(chunks[0].section_type, "Background Study")
        self.assertEqual(chunks[-1].section_type, "Methodology")
        self.assertTrue(all(c.token_count == WORDS(c.chunk_text) for c in chunks))


class PackerTokenizationTest(unittest.TestCase):
    def test_never_tokenises_a_joined_section(self) -> None:
        # 50 small paragraphs whose *join* is large. The packer must count each
        # paragraph (and the separator) individually, never the whole joined
        # string -- counting the join is what raised the >max_seq_length warning.
        paragraphs = tuple(f"p{i} word" for i in range(50))
        section = Section("Methodology", paragraphs)
        seen: list[int] = []

        def counter(text: str) -> int:
            seen.append(len(text))
            return len(text.split())

        segment_section(section, counter, threshold=5)

        self.assertTrue(seen)
        self.assertLess(max(seen), len("\n\n".join(paragraphs)))

    def test_bounded_counting_avoids_huge_tokenisation(self) -> None:
        # One ~25k-char paragraph (no sentence punctuation). The count must be
        # taken in pieces so no single tokenizer call sees the whole paragraph.
        big = " ".join(["word"] * 5000)
        section = Section("Methodology", (big,))
        seen: list[int] = []

        def counter(text: str) -> int:
            seen.append(len(text))
            return len(text.split())

        segment_section(section, counter, threshold=5)

        self.assertTrue(seen)
        self.assertLessEqual(max(seen), 4000)


if __name__ == "__main__":
    unittest.main()
