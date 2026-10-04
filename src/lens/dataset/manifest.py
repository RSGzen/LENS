"""M1 — build the arXiv cs.AI subset manifest.

Spec: ``MVP Build Roadmap.md`` §4 (M1), §2.1 (assets), §3 (runtime layout) and
``Thesis Drafts/Tech Stack (Draft).md`` §3 (``papers`` metadata fields).

Pipeline (single pass over the JSONL snapshot)
----------------------------------------------
``kaggle_arxiv-metadata-oai-snapshot.json`` (JSONL, ~3M records, ~4.9 GB)
  -> keep records with ``cs.AI`` in ``categories`` and ``id[:2]`` in
     :data:`config.PAPER_YEAR_RANGE` (the id prefix YYMM is the submission year,
     so no ``update_date`` is needed)
  -> reconstruct the TEI filename ``<id><latest_version>.tei.xml`` and keep the
     record only if that name exists in the flat XML directory
  -> append one manifest entry per kept paper
  -> ``manifests/manifest.json`` (provenance object + ``papers`` list)
     and ``manifests/manifest.checkpoint.json`` (append-only JSONL progress log).

The manifest is the join list for M3 (abstract embeddings) and M4 (chunking): each
entry carries the ``papers``-table fields those stages need plus the TEI filename.
``tests/test_manifest.py`` defines the contracts (no key, no network, zero cost).
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from lens import config

# ----------------------------------------------------------------- filtering
def is_candidate(
    record: dict[str, Any],
    target_category: str,
    paper_year_range: set
) -> bool:
    """True when a metadata record is in the target category and id-year window.

    - ``categories`` is a whitespace-joined **string** (e.g. ``"cs.AI cs.LG"``),
      so category membership is a substring test. A missing key -> ``""``.
    - The year is ``id[:2]`` (YYMM), checked as a set lookup against
      ``paper_year_range`` of two-digit strings. Old-style ids such as
      ``"acc-phys/9411001"`` are simply not members (no ``int()`` cast, no crash).
    """
    if target_category not in record.get("categories", ""):
        return False

    paper_year = record.get("id", "")[:2]
    
    if paper_year not in paper_year_range:
        return False

    return True

def tei_xml_filename_gen(record: dict[str, Any]) -> str:
    """Return the on-disk TEI basename ``<arxiv_id><latest_version>.tei.xml``.

    ``latest_version`` is ``record["versions"][-1]["version"]`` — the same value
    the downloader used, so the reconstructed name should match the extracted
    file. Callers must ensure ``record["versions"]`` is non-empty first.
    """
    paper_id = record["id"]
    
    paper_ver = record["versions"][-1]["version"]

    tei_xml_filename = f"{paper_id}{paper_ver}.tei.xml"

    return tei_xml_filename

# ----------------------------------------------------------------- logging
def checkpoint_logging(checkpoint_path: str | Path,
                       scanned_lines: int,
                       num_candidates: int,
                       done_flag: bool = False
) -> None:
    """Append one JSONL progress record to ``checkpoint_path``.

    Append-only by design: the file keeps the full scan history, and the last line
    (``done: true``) marks completion. A reader must parse the **last line** —
    ``json.loads`` on the whole file fails on multiple JSON values.
    """
    log_dict = {
        "lines_scanned": scanned_lines,
        "candidates": num_candidates,
        "done": done_flag
    }

    with open(checkpoint_path, "a", encoding="utf-8") as checkpoint_log:
        json.dump(log_dict, checkpoint_log, ensure_ascii=False)
        checkpoint_log.write("\n")

# ----------------------------------------------------------------- extracting
def extract_manifest_entry(record: dict[str, Any], tei_xml_filename: str) -> dict[str, Any]:
    """Build one manifest entry dict from a metadata record.

    Keys (the ``papers``-table fields M3 needs + the file M4 needs):

    - ``arxiv_id``    ``record["id"]``
    - ``version``     ``record["versions"][-1]["version"]`` (latest; == DB ``current_ver``)
    - ``tei_xml``     ``tei_xml_filename`` (basename passed by the caller)
    - ``title``       ``record["title"]``
    - ``authors``     ``record["authors"]`` (raw string)
    - ``categories``  ``record["categories"].split()`` -> list[str]
    - ``arxiv_doi``   ``record["doi"]`` (nullable)
    - ``journal_ref`` ``record["journal-ref"]`` (nullable)
    - ``update_date`` ``record["update_date"]``
    - ``abstract``    ``record["abstract"]``
    """
    entry_dict = {
        "arxiv_id": record["id"],
        "version": record["versions"][-1]["version"],
        "tei_xml": tei_xml_filename,
        "title": record["title"],
        "authors": record["authors"],
        "categories": record["categories"].split(),
        "arxiv_doi": record.get("doi"),
        "journal_ref": record.get("journal-ref"),
        "update_date": record.get("update_date"),
        "abstract": record["abstract"]
    }
    return entry_dict

# ---------------------------------------------------------------- build (pass)
def build_manifest(
    metadata_path: str | Path,
    xml_dir: str | Path,
    manifest_path: str | Path,
    *,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Stream the metadata once and write ``manifest.json`` + a checkpoint.

    Steps:
      1. ``xml_names = {p.name for p in Path(xml_dir).iterdir()}`` — one listing,
         then an O(1) membership test per candidate (no per-record ``stat``).
      2. For each JSONL line: ``json.loads`` the record; if :func:`is_candidate`
         and ``record.get("versions")`` is non-empty, build
         ``f"{id}{versions[-1]['version']}.tei.xml"``; if that name is in
         ``xml_names``, append :func:`extract_manifest_entry` to ``entries``.
      3. Every ``config.MANIFEST_CHECKPOINT_INTERVAL`` lines, append one JSONL
         checkpoint record ``{"lines_scanned", "candidates", "done": false}``
         (``candidates`` counts kept papers, i.e. matched **and** XML present).
      4. Write the manifest object:
         ``{"generated_at", "target_category", "year_range", "metadata_source",
         "xml_dir", "count", "papers"}`` (``generated_at`` = UTC ISO-8601,
         ``count`` = ``len(papers)``), then the final checkpoint with
         ``"done": true``.
      5. Return the manifest dict.
    """
    target_categories = config.TARGET_CATEGORY
    paper_yr_range = config.PAPER_YEAR_RANGE
    
    scanned_lines = 0
    num_candidates = 0
    entries_list = []

    xml_names = {p.name for p in Path(xml_dir).iterdir()}

    with open(metadata_path, "r", encoding="utf-8") as metadata_f:
        for line in metadata_f:
            one_paper_metadata = json.loads(line)
            scanned_lines += 1

            # Check if its a candidate paper (Category + Time range)
            if is_candidate(one_paper_metadata, target_categories, paper_yr_range):

                # Skip if version not available
                versions = one_paper_metadata.get("versions")
                if not versions:
                    continue

                tei_xml_filename = tei_xml_filename_gen(one_paper_metadata)

                # Check if candidate paper is downloaded and processed as XML papers
                if tei_xml_filename in xml_names:
                    num_candidates += 1

                    # Generate manifest entry dictionary and append to list
                    entries_list.append(extract_manifest_entry(one_paper_metadata, tei_xml_filename))

            # Log progress progressively
            if scanned_lines % config.MANIFEST_CHECKPOINT_INTERVAL == 0:
                checkpoint_logging(checkpoint_path, scanned_lines, num_candidates)

        checkpoint_logging(checkpoint_path, scanned_lines, num_candidates, done_flag=True)

    manifest = {                                  
        "generated_at":     datetime.now(timezone.utc).isoformat(),                  
        "target_category":  config.TARGET_CATEGORY,               
        "year_range":       [config.PAPER_START_YEAR, config.PAPER_END_YEAR],
        "metadata_source":  str(metadata_path),    
        "xml_dir":          str(xml_dir),           
        "count":            len(entries_list),           
        "papers":           entries_list,                
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    return manifest

def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read ``manifest.json`` back into a dict (M3/M4 entry point)."""

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

        manifest = {                                  
            "generated_at":     data["generated_at"],                  
            "target_category":  data["target_category"],               
            "year_range":       data["year_range"],
            "metadata_source":  data["metadata_source"],    
            "xml_dir":          data["xml_dir"],           
            "count":            data["count"],           
            "papers":           data["papers"]            
        }
        del data

        return manifest
