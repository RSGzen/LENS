r"""Build a corpus: download arXiv PDFs, convert them with GROBID, keep the TEI.

Wiring only: the work lives in :mod:`lens.dataset.fetch` and the paths in
:mod:`lens.dataset.paths`. This file parses arguments, wires logging, and prints
the outcome tally.

One command, five jobs::

    # 1. the whole pipeline: metadata -> download -> GROBID -> burn the PDF
    python scripts/fetch_corpus.py fetch --limit 100 --dry-run
    python scripts/fetch_corpus.py fetch

    # 2. convert PDFs you already have (no network, no metadata needed)
    python scripts/fetch_corpus.py grobid --pdf-dir D:\papers

    # 3. retry what failed; the file is rewritten with what is still failing
    python scripts/fetch_corpus.py retry

    # 4. build manifest.json from the metadata and the TEI directory
    python scripts/fetch_corpus.py manifest

    # 5. what is on disk, without touching the network
    python scripts/fetch_corpus.py status

GROBID runs in Docker. Start it once, before any of the above::

    docker run -d --restart unless-stopped --init -m 12g ^
        -e JAVA_OPTS=-Xmx7G -p 8070:8070 lfoppiano/grobid:0.8.0

Every location defaults to ``<data_root>/corpus/{xml,pdf,metadata.jsonl}`` and can be
overridden per command (``--xml-dir``, ``--pdf-dir``, ``--metadata``) or through
``LENS_GROBID_XML_DIR`` / ``LENS_PDF_DIR`` / ``LENS_ARXIV_METADATA``. The precedence
chain and the provenance of each value are printed before anything runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

from lens import config
from lens.dataset import fetch, paths
from lens.dataset.manifest import build_manifest, load_manifest
from lens.preflight import DB, METADATA, require_environment

LOGGER_NAME = "lens.fetch_corpus"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for all five subcommands."""
    parser = argparse.ArgumentParser(
        prog="python scripts/fetch_corpus.py",
        description="Download arXiv PDFs, convert them with GROBID, and keep the TEI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_locations(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--xml-dir", default=None, help="where TEI files are written")
        sub.add_argument("--pdf-dir", default=None, help="where PDFs are downloaded")
        sub.add_argument("--metadata", default=None, help="the arXiv metadata dump (JSONL)")
        sub.add_argument("--log-file", default=None, help="append a per-paper log here")

    fetch_parser = subparsers.add_parser("fetch", help="metadata -> download -> GROBID -> burn")
    add_locations(fetch_parser)
    fetch_parser.add_argument("--limit", type=int, default=None, help="stop after N papers")
    fetch_parser.add_argument(
        "--dry-run", action="store_true", help="count the targets, download nothing"
    )
    fetch_parser.add_argument("--category", default=config.TARGET_CATEGORY)
    fetch_parser.add_argument("--years", default="15-26", help="two-digit year range, e.g. 15-26")
    fetch_parser.add_argument("--workers", type=int, default=10)
    fetch_parser.add_argument("--retries", type=int, default=3)
    fetch_parser.add_argument("--grobid-url", default=fetch.DEFAULT_GROBID_URL)
    fetch_parser.add_argument(
        "--keep-pdf", action="store_true", help="do not delete the PDF afterwards"
    )
    fetch_parser.add_argument(
        "--failed-file",
        default=None,
        help="record permanent failures here (default: <pdf-dir>/failed_papers.txt)",
    )

    grobid_parser = subparsers.add_parser("grobid", help="convert PDFs you already have")
    add_locations(grobid_parser)
    grobid_parser.add_argument("--workers", type=int, default=10)
    grobid_parser.add_argument("--retries", type=int, default=3)
    grobid_parser.add_argument("--grobid-url", default=fetch.DEFAULT_GROBID_URL)
    grobid_parser.add_argument("--keep-pdf", action="store_true")
    grobid_parser.add_argument("--failed-file", default=None)

    retry_parser = subparsers.add_parser("retry", help="retry the papers in a failures file")
    add_locations(retry_parser)
    retry_parser.add_argument("--failed-file", default=None, help="read and rewrite this file")
    retry_parser.add_argument(
        "--exhausted-file",
        default=None,
        help="papers still failing go here (default: <failed-file>.exhausted)",
    )
    retry_parser.add_argument("--workers", type=int, default=10)
    retry_parser.add_argument("--retries", type=int, default=3)
    retry_parser.add_argument("--grobid-url", default=fetch.DEFAULT_GROBID_URL)
    retry_parser.add_argument("--keep-pdf", action="store_true")

    manifest_parser = subparsers.add_parser(
        "manifest", help="build manifest.json from metadata + TEI"
    )
    add_locations(manifest_parser)
    manifest_parser.add_argument("--category", default=config.TARGET_CATEGORY)
    manifest_parser.add_argument("--years", default="15-26")
    manifest_parser.add_argument(
        "--out",
        default=None,
        help="where to write manifest.json (default: <data_root>/manifests/manifest.json)",
    )

    status_parser = subparsers.add_parser("status", help="report what is on disk (no network)")
    add_locations(status_parser)
    status_parser.add_argument("--failed-file", default=None)
    status_parser.add_argument(
        "--probe", action="store_true", help="also check the bucket and GROBID reachability"
    )
    status_parser.add_argument("--grobid-url", default=fetch.DEFAULT_GROBID_URL)

    return parser


def parse_years(spec: str) -> list[str]:
    """Expand ``15-26`` (or ``15,17,20``) into two-digit year prefixes."""
    spec = spec.strip()

    if "-" in spec:
        start, _, end = spec.partition("-")
        return [f"{year:02d}" for year in range(int(start), int(end) + 1)]

    return [piece.strip() for piece in spec.split(",") if piece.strip()]


def configure_logging(log_file: str | None) -> logging.Logger:
    """Return the module logger: per-paper lines to a file, warnings to the console."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(threadName)-17s | %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def resolve(args: argparse.Namespace) -> paths.CorpusPaths:
    """Resolve the three corpus locations from this command's arguments and the env."""
    return paths.resolve_corpus(
        xml_dir=getattr(args, "xml_dir", None),
        pdf_dir=getattr(args, "pdf_dir", None),
        metadata=getattr(args, "metadata", None),
    )


def print_paths(resolved: paths.CorpusPaths) -> None:
    """Print the effective locations and where each came from."""
    print("corpus locations")

    for line in resolved.describe():
        print(f"  {line}")


def failed_path(args: argparse.Namespace, resolved: paths.CorpusPaths) -> Path:
    """Return the failures file for this command (explicit, else beside the PDFs)."""
    return (
        Path(args.failed_file) if args.failed_file else Path(resolved.pdf_dir) / "failed_papers.txt"
    )


def command_fetch(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Download, convert, burn. The whole pipeline."""
    require_environment(need=(METADATA,))
    resolved = resolve(args)
    print_paths(resolved)

    try:
        fetch.probe_metadata(resolved.metadata)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    years = parse_years(args.years)
    targets = fetch.iter_targets(
        resolved.metadata,
        category=args.category,
        years=years,
        limit=args.limit,
        log=logger,
    )

    if args.dry_run:
        count = 0
        first = last = None

        for target in targets:
            first = first or target
            last = target
            count += 1

        print(f"dry run: {count} target(s), category {args.category}, years {years[0]}-{years[-1]}")

        if first is not None:
            print(f"  first: {first.url}")

        if last is not None:
            print(f"  last:  {last.url}")

        return 0

    started = time.time()

    report = fetch.run_batch(
        list(targets),
        xml_dir=Path(resolved.xml_dir),
        pdf_dir=Path(resolved.pdf_dir),
        grobid_url=args.grobid_url,
        workers=args.workers,
        retries=args.retries,
        keep_pdf=args.keep_pdf,
        log=logger,
    )

    failures = failed_path(args, resolved)
    written = fetch.write_targets(failures, report.failed)

    print(report.summary())
    print(f"\nfinished in {time.time() - started:.0f}s")

    if written:
        print(f"{written} paper(s) still failing, recorded in {failures}")
        print(f"retry them with:  python scripts/fetch_corpus.py retry --failed-file {failures}")

    return 0


def command_grobid(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Convert PDFs that are already on disk (no network, no metadata)."""
    resolved = resolve(args)
    print_paths(resolved)

    pdf_dir = Path(resolved.pdf_dir)

    if not pdf_dir.is_dir():
        print(f"no PDF directory at {pdf_dir}", file=sys.stderr)
        return 2

    targets = []

    for pdf in sorted(pdf_dir.rglob("*.pdf")):
        target = fetch.parse_target(pdf.name)

        if target is None:
            logger.warning("%s | skipped | name is not <arxiv_id><version>.pdf", pdf.name)
            continue

        targets.append(target)

    if not targets:
        print(f"no arXiv-shaped PDFs under {pdf_dir}", file=sys.stderr)
        return 2

    report = fetch.run_batch(
        targets,
        xml_dir=Path(resolved.xml_dir),
        pdf_dir=pdf_dir,
        grobid_url=args.grobid_url,
        workers=args.workers,
        retries=args.retries,
        keep_pdf=args.keep_pdf,
        log=logger,
    )

    failures = failed_path(args, resolved)
    written = fetch.write_targets(failures, report.failed)

    print(report.summary())

    if written:
        print(f"{written} paper(s) still failing, recorded in {failures}")

    return 0


def command_retry(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Retry a failures file, rewriting it with what is still failing."""
    resolved = resolve(args)
    print_paths(resolved)

    source = failed_path(args, resolved)

    if not source.is_file():
        print(f"no failures file at {source} — nothing to retry", file=sys.stderr)
        return 2

    exhausted = (
        Path(args.exhausted_file) if args.exhausted_file else source.with_suffix(".exhausted")
    )
    targets = fetch.read_targets(source)
    print(f"retrying {len(targets)} paper(s) from {source}")

    report = fetch.run_batch(
        targets,
        xml_dir=Path(resolved.xml_dir),
        pdf_dir=Path(resolved.pdf_dir),
        grobid_url=args.grobid_url,
        workers=args.workers,
        retries=args.retries,
        keep_pdf=args.keep_pdf,
        log=logger,
    )

    remaining = fetch.write_targets(source, report.failed)
    previous = fetch.read_targets(exhausted) if exhausted.is_file() else []
    fetch.write_targets(exhausted, [*previous, *report.failed])

    print(report.summary())
    print(
        f"\n{remaining} paper(s) still failing; {len(fetch.read_targets(exhausted))} in {exhausted}"
    )

    return 0


def command_manifest(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Build manifest.json from the metadata dump and the TEI directory."""
    require_environment(need=(METADATA,))
    resolved = resolve(args)
    print_paths(resolved)

    directory, count = paths.require_xml_corpus(resolved)
    print(f"TEI files: {count} in {directory}")

    try:
        fetch.probe_metadata(resolved.metadata)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    out = (
        Path(args.out)
        if args.out
        else Path(config.LENS_MANIFESTS_ROOT_PATH) / config.MANIFEST_FILENAME
    )
    checkpoint = Path(config.LENS_MANIFESTS_ROOT_PATH) / config.CHECKPOINT_FILENAME
    # The default lives under the data root, which may not exist on a fresh machine;
    # the checkpoint writer appends and does not create its directory.
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"scanning {resolved.metadata} (this is one pass, several minutes on the full dump)")
    manifest = build_manifest(
        metadata_path=resolved.metadata,
        xml_dir=directory,
        manifest_path=out,
        checkpoint_path=checkpoint,
    )

    print(f"manifest: {manifest['count']} papers -> {out}")
    print(f"checkpoint: {checkpoint}")

    if manifest["count"] < count:
        print(
            f"note: {count - manifest['count']} TEI file(s) are not in the manifest "
            "(not in the category/year window, or no longer in the dump)"
        )

    return 0


def command_status(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Report what is on disk without touching the network, unless ``--probe`` is given."""
    resolved = resolve(args)
    print_paths(resolved)

    tei = paths.tei_files(resolved.xml_dir)
    print(f"TEI files      : {len(tei)}")

    pdf_dir = Path(resolved.pdf_dir)
    pending = sorted(pdf_dir.rglob("*.pdf")) if pdf_dir.is_dir() else []
    print(f"PDFs on disk   : {len(pending)} (not yet converted)")

    manifest_path = Path(config.LENS_MANIFESTS_ROOT_PATH) / config.MANIFEST_FILENAME

    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        print(f"manifest       : {manifest['count']} papers ({manifest_path})")
    else:
        print(f"manifest       : not built ({manifest_path})")

    failures = failed_path(args, resolved)

    if failures.is_file():
        print(f"failures file  : {len(fetch.read_targets(failures))} paper(s) ({failures})")
    else:
        print(f"failures file  : none ({failures})")

    if args.probe:
        probe = fetch.Target(arxiv_id="1501.00601", version="v3")
        print(f"\nprobe bucket   : {probe.url}")
        print("probe GROBID   : " + args.grobid_url)

    return 0


COMMANDS = {
    "fetch": command_fetch,
    "grobid": command_grobid,
    "retry": command_retry,
    "manifest": command_manifest,
    "status": command_status,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the five subcommands."""
    args = build_parser().parse_args(argv)

    require_environment(need=(DB,))

    logger = configure_logging(getattr(args, "log_file", None))

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    try:
        return COMMANDS[args.command](args, logger)
    except paths.CorpusError as exc:
        print(str(exc), file=sys.stderr)

        for detail in exc.details:
            print(f"  {detail}", file=sys.stderr)

        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
