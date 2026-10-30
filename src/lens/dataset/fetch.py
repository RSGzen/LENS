"""Build a corpus from arXiv: download PDFs, convert them with GROBID, delete them.

The unit of work is one paper, named ``<arxiv_id><version>`` throughout:

    1501.00601v3  ->  arxiv/arxiv/pdf/1501/1501.00601v3.pdf   (source object key)
                  ->  <pdf_dir>/1501.00601v3.pdf              (temporary, burned)
                  ->  <xml_dir>/1501.00601v3.tei.xml          (kept, the join key)

Everything here is a function of explicit parameters plus two injected callables,
so the whole pipeline can be exercised with no network and no GROBID container:

- ``fetch_bytes``  : ``(url) -> bytes``, replaced by a fake in tests;
- ``post_pdf``     : ``(pdf_path) -> (status_code, body)``, the GROBID call.

Resume is decided by what is on disk, which is what makes the pipeline safe to
re-run: TEI present means done (whatever happened to the PDF), a leftover PDF means
"downloaded but not converted", nothing means "download it". Downloads and XML are
written to ``.part`` files and moved into place, so an interrupted run cannot leave
a truncated file that a later run mistakes for finished work.

The arXiv bucket is public: objects are read over plain HTTPS with an anonymous
client, which keeps the dependency list to ``requests``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import requests

from lens.dataset.manifest import is_candidate
from lens.dataset.paths import TEI_SUFFIX

#: Public bucket and prefix. Both are pinned by ``CORPUS.md`` and the manifest, so a
#: change here is a corpus-version change, not a refactor.
GCS_BUCKET_URL = "https://storage.googleapis.com/arxiv-dataset"
PDF_PREFIX = "arxiv/arxiv/pdf"

#: The GROBID container's full-text endpoint (multipart field name: ``input``).
GROBID_PATH = "/api/processFulltextDocument"
DEFAULT_GROBID_URL = "http://localhost:8070"

#: Seconds. A full-text conversion of a long paper can take a couple of minutes, so
#: the read timeout is generous; the container is the slow party, not the network.
DOWNLOAD_TIMEOUT_S = 120
GROBID_TIMEOUT_S = 180

#: Backoff after a retryable failure, keyed by cause. A 503 is the container being
#: busy (wait for it), a connection error usually means it is restarting after an
#: out-of-memory kill (wait longer).
SLEEP_BUSY_S = 15
SLEEP_TIMEOUT_S = 5
SLEEP_CONNECT_S = 25

#: Outcomes. The vocabulary is printed as a tally, so it is part of the interface.
SUCCESS_FIRST_TRY = "SUCCESS_FIRST_TRY"
SUCCESS_AFTER_RETRY = "SUCCESS_AFTER_RETRY"
ALREADY_PROCESSED = "ALREADY_PROCESSED"
MISSING_UPSTREAM = "MISSING_UPSTREAM"
PERMANENT_FAILURE = "PERMANENT_FAILURE"
FATAL_EXCEPTION = "FATAL_EXCEPTION"
THREAD_CRASH = "THREAD_CRASH"


@dataclass(frozen=True)
class Target:
    """One paper to build: its identity and the two file names derived from it."""

    arxiv_id: str
    version: str

    @property
    def stem(self) -> str:
        return f"{self.arxiv_id}{self.version}"

    @property
    def pdf_name(self) -> str:
        return f"{self.stem}.pdf"

    @property
    def tei_name(self) -> str:
        return f"{self.stem}{TEI_SUFFIX}"

    @property
    def object_key(self) -> str:
        """Return the bucket key for this paper's PDF."""
        return f"{PDF_PREFIX}/{self.arxiv_id.split('.')[0]}/{self.pdf_name}"

    @property
    def url(self) -> str:
        """Return the public HTTPS URL for this paper's PDF."""
        return f"{GCS_BUCKET_URL}/{self.object_key}"


def parse_target(text: str) -> Target | None:
    """Parse ``1501/1501.00601v3.pdf`` or ``1501.00601v3`` into a :class:`Target`.

    Accepts the shapes that appear in a failures file (an object key) and that a
    user types by hand (a bare stem). Anything else returns ``None`` so a damaged
    line is skipped rather than crashing a 200-line retry batch.
    """
    name = Path(text.strip()).name

    if name.endswith(".pdf"):
        name = name[: -len(".pdf")]

    if not name or ".." in name or "/" in name:
        return None

    identifier, _, version = name.rpartition("v")

    if not version.isdigit() or not identifier or "." not in identifier:
        return None

    return Target(arxiv_id=identifier, version=f"v{version}")


def target_from_record(record: dict[str, Any]) -> Target | None:
    """Return the target for a metadata record, or ``None`` when it has no versions."""
    versions = record.get("versions") or []

    if not versions:
        return None

    return Target(arxiv_id=record["id"], version=versions[-1]["version"])


def probe_metadata(path: str | Path, *, sample: int = 5) -> None:
    """Raise ``ValueError`` when ``path`` is not the JSONL metadata dump.

    A common mistake is to point at the Kaggle CSV, an extracted JSON array, or a
    gzipped file. Catching that here turns a ``KeyError: 'title'`` three functions
    deep into one sentence the user can act on.
    """
    seen = 0

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"{path} is not JSONL: line {seen + 1} does not parse as JSON ({exc}). "
                    "The loader expects one JSON object per line "
                    "(arXiv-metadata-oai-snapshot.json from Kaggle)."
                ) from exc

            if not isinstance(record, dict) or "id" not in record or "versions" not in record:
                keys = sorted(record)[:8] if isinstance(record, dict) else type(record).__name__

                raise ValueError(
                    f"{path} does not look like the arXiv metadata dump: the first record has "
                    f"keys {keys}, expected at least 'id' and 'versions'."
                )

            seen += 1

            if seen >= sample:
                return

    if seen == 0:
        raise ValueError(f"{path} has no records (it is empty)")


def iter_targets(
    metadata_path: str | Path,
    *,
    category: str,
    years: Iterable[str],
    limit: int | None = None,
    log: logging.Logger | None = None,
) -> Iterator[Target]:
    """Stream the metadata dump and yield one target per candidate paper.

    The filter is :func:`lens.dataset.manifest.is_candidate`, the same one the
    manifest builder uses, so "which papers are in scope" has exactly one
    definition. ``limit`` exists so a first run can be measured on a hundred papers
    before committing hours to the full set.
    """
    wanted = set(years)
    yielded = 0
    scanned = 0

    with open(metadata_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            scanned += 1
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except ValueError:
                continue

            if not is_candidate(record, category, wanted):
                continue

            target = target_from_record(record)

            if target is None:
                continue

            yield target
            yielded += 1

            if limit is not None and yielded >= limit:
                break

    if log is not None:
        log.info("scanned %d metadata lines, kept %d targets", scanned, yielded)


def plan_action(pdf_path: Path, xml_path: Path) -> str:
    """Return ``"skip"``, ``"download"`` or ``"process"`` for one paper.

    The three states the pipeline can find on disk. Kept as a separate function
    because it is the part that must be right after a crash: a PDF with no TEI is
    work to finish, never work to redo from the network.
    """
    if xml_path.exists():
        return "skip"

    return "process" if pdf_path.exists() else "download"


def write_atomic(path: Path, data: bytes | str) -> None:
    """Write ``data`` to ``path`` through a ``.part`` file, then move it into place.

    A truncated PDF would be indistinguishable from a downloaded one, and GROBID
    would report the failure as a conversion error rather than a download error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix + ".part"
    temporary = path.with_suffix(suffix)
    payload = data.encode("utf-8") if isinstance(data, str) else data

    temporary.write_bytes(payload)
    os.replace(temporary, path)


def fetch_bytes(url: str, *, timeout: int = DOWNLOAD_TIMEOUT_S) -> bytes:
    """Download one object. Raises :class:`UpstreamMissing` when the bucket says 404."""
    response = requests.get(url, timeout=timeout)

    if response.status_code == 404:
        raise UpstreamMissing(url)

    response.raise_for_status()

    return response.content


class UpstreamMissing(FileNotFoundError):
    """The paper's PDF is not in the archive (withdrawn, or never deposited)."""


def post_pdf(
    pdf_path: Path,
    *,
    url: str,
    timeout: int = GROBID_TIMEOUT_S,
) -> tuple[int, str]:
    """Send one PDF to GROBID and return ``(status_code, body)``."""
    with open(pdf_path, "rb") as handle:
        files = {"input": (pdf_path.name, handle, "application/pdf")}
        response = requests.post(url, files=files, timeout=timeout)

    return response.status_code, response.text


def grobid_convert(
    pdf_path: Path,
    *,
    url: str,
    retries: int = 3,
    post: Callable[[Path], tuple[int, str]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: logging.Logger | None = None,
) -> tuple[str | None, str | None]:
    """Convert one PDF, returning ``(tei_text, failure_reason)``.

    Status handling mirrors what the container actually does: 200 is success, 503
    means busy (retry), a connection error usually means it is restarting after an
    out-of-memory kill (retry with a longer wait), and any other status is a hard
    failure that another attempt would only repeat. Timeouts are retried because
    they are the common case for very long papers.
    """
    call = post or (lambda path: post_pdf(path, url=url))

    for attempt in range(1, retries + 1):
        try:
            status, body = call(pdf_path)
        except requests.exceptions.Timeout:
            if log is not None:
                log.warning("%s | timeout | attempt %d/%d", pdf_path.stem, attempt, retries)
            sleep(SLEEP_TIMEOUT_S)
            continue
        except requests.exceptions.ConnectionError:
            if log is not None:
                log.warning("%s | container down | attempt %d/%d", pdf_path.stem, attempt, retries)
            sleep(SLEEP_CONNECT_S)
            continue
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            if log is not None:
                log.error("%s | fatal: %s", pdf_path.stem, exc)
            return None, f"{FATAL_EXCEPTION}: {exc}"

        if status == 200:
            return body, None

        if status == 503:
            if log is not None:
                log.warning("%s | busy (503) | attempt %d/%d", pdf_path.stem, attempt, retries)
            sleep(SLEEP_BUSY_S)
            continue

        if log is not None:
            log.error(
                "%s | failed | status %d | attempt %d/%d", pdf_path.stem, status, attempt, retries
            )

        return None, f"status {status}"

    return None, f"exhausted {retries} attempts"


@dataclass
class BatchReport:
    """Outcome tally plus the papers that need another attempt."""

    tally: Counter[str] = field(default_factory=Counter)
    failed: list[Target] = field(default_factory=list)

    def add(self, status: str) -> None:
        self.tally[status] += 1

    def summary(self) -> str:
        """Return one line per status, widest name first, for a run log."""
        if not self.tally:
            return "nothing to do"

        width = max(len(name) for name in self.tally)

        return "\n".join(
            f"{name.ljust(width)} : {count}" for name, count in sorted(self.tally.items())
        )


def build_one(
    target: Target,
    *,
    xml_dir: Path,
    pdf_dir: Path,
    grobid_url: str,
    retries: int,
    keep_pdf: bool,
    fetch: Callable[[str], bytes] | None = None,
    post: Callable[[Path], tuple[int, str]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: logging.Logger | None = None,
) -> tuple[str, Target | None]:
    """Build one paper: download if needed, convert, burn. Return ``(status, retry?)``."""
    pdf_path = Path(pdf_dir) / target.pdf_name
    xml_path = Path(xml_dir) / target.tei_name
    action = plan_action(pdf_path, xml_path)

    if action == "skip":
        return ALREADY_PROCESSED, None

    if action == "download":
        getter = fetch or fetch_bytes

        try:
            write_atomic(pdf_path, getter(target.url))
        except UpstreamMissing:
            if log is not None:
                log.warning("%s | missing upstream", target.stem)
            return MISSING_UPSTREAM, None
        except Exception as exc:  # noqa: BLE001 - one paper's failure is not the run's
            if log is not None:
                log.error("%s | download failed: %s", target.stem, exc)
            return PERMANENT_FAILURE, target

    tei, failure = grobid_convert(
        pdf_path,
        url=f"{grobid_url.rstrip('/')}{GROBID_PATH}",
        retries=retries,
        post=post,
        sleep=sleep,
        log=log,
    )

    if tei is None:
        if log is not None:
            log.error("%s | permanent failure (%s)", target.stem, failure)
        return PERMANENT_FAILURE, target

    write_atomic(xml_path, tei)

    if not keep_pdf:
        try:
            pdf_path.unlink()
        except OSError as exc:
            if log is not None:
                log.warning("%s | could not delete the PDF: %s", target.stem, exc)

    return (SUCCESS_FIRST_TRY if action == "download" else SUCCESS_AFTER_RETRY), None


def run_batch(
    targets: Sequence[Target],
    *,
    xml_dir: Path,
    pdf_dir: Path,
    grobid_url: str = DEFAULT_GROBID_URL,
    workers: int = 10,
    retries: int = 3,
    keep_pdf: bool = False,
    fetch: Callable[[str], bytes] | None = None,
    post: Callable[[Path], tuple[int, str]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: logging.Logger | None = None,
) -> BatchReport:
    """Build every target, in a thread pool, and tally the outcomes.

    Threads are used because the work is dominated by two waits: the download and
    the (single) GROBID container. Ten in flight is deliberate; the public bucket
    treats it as polite, and more does not help when one container is the bottleneck.
    """
    report = BatchReport()

    if not targets:
        return report

    Path(xml_dir).mkdir(parents=True, exist_ok=True)
    Path(pdf_dir).mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()

    def submit(target: Target) -> tuple[str, Target | None]:
        return build_one(
            target,
            xml_dir=Path(xml_dir),
            pdf_dir=Path(pdf_dir),
            grobid_url=grobid_url,
            retries=retries,
            keep_pdf=keep_pdf,
            fetch=fetch,
            post=post,
            sleep=sleep,
            log=log,
        )

    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(submit, target): target for target in targets if target.stem not in seen
        }

        for future in as_completed(futures):
            target = futures[future]

            try:
                status, retry_target = future.result()
            except Exception as exc:  # noqa: BLE001 - a crashed thread must not lose the run
                if log is not None:
                    log.error("%s | thread crash: %s", target.stem, exc)
                report.add(THREAD_CRASH)
                continue

            report.add(status)

            if retry_target is not None:
                with lock:
                    report.failed.append(retry_target)

    return report


def read_targets(path: str | Path) -> list[Target]:
    """Read a failures file into targets, dropping duplicates and blank lines."""
    target_list: list[Target] = []
    seen: set[str] = set()

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            target = parse_target(line)

            if target is None or target.stem in seen:
                continue

            seen.add(target.stem)
            target_list.append(target)

    return target_list


def write_targets(path: str | Path, targets: Iterable[Target]) -> int:
    """Replace ``path`` with the given targets; return how many were written.

    A rewrite rather than an append: a retry that fixed 190 of 204 papers should
    leave 14 lines, not 204 plus 14.
    """
    ordered = sorted({target.object_key for target in targets})
    destination = Path(path)

    if not ordered:
        if destination.exists():
            destination.unlink()

        return 0

    write_atomic(destination, "\n".join(ordered) + "\n")

    return len(ordered)


__all__ = [
    "ALREADY_PROCESSED",
    "DEFAULT_GROBID_URL",
    "FATAL_EXCEPTION",
    "GCS_BUCKET_URL",
    "GROBID_PATH",
    "MISSING_UPSTREAM",
    "PDF_PREFIX",
    "PERMANENT_FAILURE",
    "SUCCESS_AFTER_RETRY",
    "SUCCESS_FIRST_TRY",
    "THREAD_CRASH",
    "BatchReport",
    "Target",
    "UpstreamMissing",
    "build_one",
    "fetch_bytes",
    "grobid_convert",
    "iter_targets",
    "parse_target",
    "plan_action",
    "post_pdf",
    "read_targets",
    "run_batch",
    "write_atomic",
    "write_targets",
]
