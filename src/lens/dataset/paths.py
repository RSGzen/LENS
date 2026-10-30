"""Resolve where the corpus lives, with one explicit precedence chain.

Three locations matter to the pipeline: the GROBID TEI directory (read by the chunk
loader), the PDF directory (written by the downloader, then deleted), and the arXiv
metadata dump (read when the manifest is built). Each can be given four ways, and
they are resolved in the same order every time:

    CLI flag  >  specific environment variable  >  LENS_CORPUS_DIR / <role>  >  default

The default is ``<data_root>/corpus/{xml,pdf,metadata.jsonl}``, so a fresh machine
needs no configuration at all: ``fetch`` writes there, the loaders read there. The
environment variables remain overrides, which is what an existing install (with a
corpus on another drive) uses unchanged.

Every resolution records **where the value came from**, because "which corpus am I
actually indexing" is the question that costs an afternoon when it goes wrong.
:func:`check_environment` prints that provenance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from lens import config

#: Suffix the pipeline uses for a GROBID document. The basename is the join key
#: between the manifest, this directory and every ``chunk`` row.
TEI_SUFFIX = ".tei.xml"

#: Role names, used in messages and in the ``source`` map.
XML = "xml_dir"
PDF = "pdf_dir"
METADATA = "metadata"

_ENV = {
    XML: "LENS_GROBID_XML_DIR",
    PDF: "LENS_PDF_DIR",
    METADATA: "LENS_ARXIV_METADATA",
}

_DEFAULT_NAME = {
    XML: "xml",
    PDF: "pdf",
    METADATA: "metadata.jsonl",
}


class CorpusError(RuntimeError):
    """A corpus location was asked for and could not be used.

    Carries the lines to print (what was tried, what to do next) so callers can
    show one actionable message instead of a stack trace. ``scripts/`` catch it and
    exit 2; the library never swallows it.
    """

    def __init__(self, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.details = details or []


@dataclass(frozen=True)
class CorpusPaths:
    """The three corpus locations and where each was decided."""

    xml_dir: Path
    pdf_dir: Path
    metadata: Path
    source: Mapping[str, str] = field(default_factory=dict)

    def describe(self) -> list[str]:
        """Return ``role = value  (source)`` lines, safe to print."""
        return [
            f"{role} = {getattr(self, role)}  ({self.source.get(role, 'unknown')})"
            for role in (XML, PDF, METADATA)
        ]


def corpus_root() -> Path:
    """Return the corpus root: ``LENS_CORPUS_DIR`` or ``<data_root>/corpus``."""
    override = (os.environ.get("LENS_CORPUS_DIR") or "").strip()

    return Path(override).expanduser() if override else Path(config.DATA_ROOT) / "corpus"


def _resolve(role: str, cli_value: str | Path | None) -> tuple[Path, str]:
    """Return ``(path, source_label)`` for one role, honouring the chain."""
    if cli_value:
        return Path(cli_value).expanduser(), "command line"

    env_value = (os.environ.get(_ENV[role]) or "").strip()

    if env_value:
        return Path(env_value).expanduser(), f"env:{_ENV[role]}"

    return corpus_root() / _DEFAULT_NAME[role], "default"


def resolve_corpus(
    *,
    xml_dir: str | Path | None = None,
    pdf_dir: str | Path | None = None,
    metadata: str | Path | None = None,
) -> CorpusPaths:
    """Resolve all three locations; do not require that any of them exists.

    Resolution is separate from validation on purpose: ``fetch`` must be able to
    report "I will write to <dir>" before the directory exists, and the read-only
    commands must not fail on a machine with no corpus at all.
    """
    resolved = {}

    for role, cli_value in ((XML, xml_dir), (PDF, pdf_dir), (METADATA, metadata)):
        resolved[role] = _resolve(role, cli_value)

    return CorpusPaths(
        xml_dir=resolved[XML][0],
        pdf_dir=resolved[PDF][0],
        metadata=resolved[METADATA][0],
        source={role: resolved[role][1] for role in resolved},
    )


def count_tei(directory: str | Path) -> int:
    """Count ``*.tei.xml`` files directly under ``directory`` (0 when absent)."""
    path = Path(directory)

    if not path.is_dir():
        return 0

    return sum(1 for entry in os.scandir(path) if entry.name.endswith(TEI_SUFFIX))


def tei_files(directory: str | Path) -> list[Path]:
    """Return the TEI files under ``directory``, flat first then recursively.

    The pipeline writes a flat directory, which is the fast path. A one-off nested
    layout (``xml/1501/...``) is still accepted, but only walked when the flat
    listing is empty, so the common case stays a single scandir.
    """
    path = Path(directory)

    if not path.is_dir():
        return []

    flat = sorted(p for p in path.glob(f"*{TEI_SUFFIX}") if p.is_file())

    return flat or sorted(p for p in path.rglob(f"*{TEI_SUFFIX}") if p.is_file())


def require_xml_corpus(
    paths: CorpusPaths | None = None,
    *,
    minimum: int = 1,
) -> tuple[Path, int]:
    """Return ``(xml_dir, count)`` or raise :class:`CorpusError`.

    This is the check run before a database is set up or a load starts: a corpus
    directory with no ``*.tei.xml`` is not a corpus, and finding that out at the
    start beats finding it out after a load has been launched.
    """
    resolved = paths or resolve_corpus()
    directory = Path(resolved.xml_dir)

    if not directory.is_dir():
        raise CorpusError(
            f"corpus XML directory does not exist: {directory}",
            [
                f"source: {resolved.source.get(XML, 'unknown')}",
                "create it by processing PDFs:  python scripts/fetch_corpus.py fetch",
                "or point at an existing directory:  set LENS_GROBID_XML_DIR=<dir>",
            ],
        )

    count = len(tei_files(directory))

    if count < minimum:
        raise CorpusError(
            f"no {TEI_SUFFIX} files under {directory} (found {count})",
            [
                f"source: {resolved.source.get(XML, 'unknown')}",
                "process PDFs into TEI:  python scripts/fetch_corpus.py grobid --pdf-dir <dir>",
                "or download and process:  python scripts/fetch_corpus.py fetch",
            ],
        )

    return directory, count


__all__ = [
    "METADATA",
    "PDF",
    "TEI_SUFFIX",
    "XML",
    "CorpusError",
    "CorpusPaths",
    "corpus_root",
    "count_tei",
    "require_xml_corpus",
    "resolve_corpus",
    "tei_files",
]
