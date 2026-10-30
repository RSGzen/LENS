"""Check that the environment can support the command, before the command runs.

Two failure modes cost real time and both are cheap to prevent: a missing
credential that surfaces as a confusing error halfway through a run, and a corpus
path that points somewhere empty so a load reports "0 papers" instead of "your
directory is wrong". Each script declares what it needs:

    from lens.preflight import require_environment
    require_environment(need=("db", "corpus"))

Requirements are checked together, not one at a time, so a fresh clone reports every
missing setting in one pass. Secrets are never printed: a key is reduced to its
first eight and last four characters.

Nothing here reads the corpus's contents (that is ``dataset.paths.require_xml_corpus``,
which counts TEI files), and nothing connects to the database, so the check itself
cannot fail for the reason it exists to diagnose.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import find_dotenv, load_dotenv

from lens import config
from lens.dataset.paths import CorpusError, require_xml_corpus, resolve_corpus

#: The requirement names a caller may ask for.
API_KEY = "api_key"
DB = "db"
CORPUS = "corpus"
METADATA = "metadata"

ALL: tuple[str, ...] = (API_KEY, DB, CORPUS, METADATA)

#: Shortest value accepted as a real key. A placeholder such as ``sk-or-v1-...``
#: from ``.env.example`` is 12 characters, so this rejects copy-paste mistakes
#: without pretending to validate the vendor's format.
MIN_KEY_LENGTH = 30


@dataclass(frozen=True)
class Check:
    """One requirement: what was looked for, what was found, what to do."""

    name: str
    ok: bool
    detail: str
    hint: str = ""

    def line(self) -> str:
        """Return the one-line form used in the report."""
        return f"  {self.name.ljust(16)} {'ok  ' if self.ok else 'FAIL'}  {self.detail}"


def mask(secret: str | None) -> str:
    """Return a printable fingerprint of a secret: first 8, last 4, never the middle."""
    if not secret:
        return "not set"

    if len(secret) <= MIN_KEY_LENGTH // 2:
        return f"set ({len(secret)} chars, too short)"

    return f"{secret[:8]}...{secret[-4:]} ({len(secret)} chars)"


def env_file() -> str:
    """Return where ``.env`` was found, or say that the OS environment is used."""
    path = find_dotenv(usecwd=True)

    if not path:
        return f"no .env found from {Path.cwd()} (using the OS environment)"

    return path


def _check_api_key() -> Check:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    if not key:
        return Check(
            API_KEY,
            False,
            "OPENROUTER_API_KEY is not set",
            "put it in .env (see .env.example) or export it in the shell",
        )

    if len(key) < MIN_KEY_LENGTH or key.endswith("..."):
        return Check(
            API_KEY,
            False,
            f"OPENROUTER_API_KEY looks like a placeholder: {mask(key)}",
            "replace the example value with a real key",
        )

    return Check(API_KEY, True, mask(key))


def _check_db() -> Check:
    host = config.LENS_DB_HOST
    port = config.LENS_DB_PORT
    name = config.LENS_DB_NAME
    admin = config.LENS_DB_PASSWORD
    readonly = config.LENS_DB_RO_PASSWORD
    missing = [
        var
        for var, value in (("LENS_DB_PASSWORD", admin), ("LENS_DB_RO_PASSWORD", readonly))
        if not value
    ]

    if missing:
        return Check(
            DB,
            False,
            f"{', '.join(missing)} not set",
            "copy .env.example to .env and fill both passwords "
            "(the read-only role lens_ro is created by the schema)",
        )

    return Check(
        DB, True, f"{config.LENS_DB_USER}@{host}:{port}/{name} (lens_ro credential present)"
    )


def _check_corpus() -> Check:
    paths = resolve_corpus()

    try:
        directory, count = require_xml_corpus(paths)
    except CorpusError as exc:
        return Check(CORPUS, False, str(exc), "; ".join(exc.details))

    return Check(
        CORPUS, True, f"{count} TEI file(s) in {directory} ({paths.source.get('xml_dir')})"
    )


def _check_metadata() -> Check:
    paths = resolve_corpus()
    path = Path(paths.metadata)

    if not path.is_file():
        return Check(
            METADATA,
            False,
            f"metadata dump not found: {path} ({paths.source.get('metadata')})",
            "download arXiv-metadata-oai-snapshot.json from Kaggle "
            "(https://www.kaggle.com/datasets/Cornell-University/arxiv) and "
            "either put it at that path or set LENS_ARXIV_METADATA",
        )

    size_mb = path.stat().st_size / 1e6

    return Check(METADATA, True, f"{path} ({size_mb:.0f} MB)")


_CHECKS = {
    API_KEY: _check_api_key,
    DB: _check_db,
    CORPUS: _check_corpus,
    METADATA: _check_metadata,
}


def check_environment(*, need: Iterable[str] = ALL) -> list[Check]:
    """Return one :class:`Check` per requested requirement, in a stable order."""
    requested = [name for name in ALL if name in set(need)]

    return [_CHECKS[name]() for name in requested]


def report(checks: Sequence[Check]) -> str:
    """Return the printable report: where the settings came from, then each check."""
    lines = [f"environment check ({', '.join(check.name for check in checks)})"]
    lines.append(f"  {'.env'.ljust(16)} {'':6} {env_file()}")

    for check in checks:
        lines.append(check.line())

        if not check.ok and check.hint:
            lines.append(f"  {'':16} {'':6} hint: {check.hint}")

    failed = [check for check in checks if not check.ok]
    lines.append("")
    lines.append(
        "all required settings are present"
        if not failed
        else f"{len(failed)} requirement(s) missing"
    )

    return "\n".join(lines)


def require_environment(*, need: Iterable[str] = ALL) -> None:
    """Print the report and exit with status 2 when a requirement is missing.

    Exiting here, rather than raising, keeps the CLI behaviour predictable: every
    script fails the same way, with the same report, before doing any work.
    """
    checks = check_environment(need=need)

    if all(check.ok for check in checks):
        return

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    print(report(checks), file=sys.stderr)

    raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``scripts/check_env.py``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--need",
        default=",".join(ALL),
        help=f"comma-separated requirements, any of: {', '.join(ALL)} (default: all)",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    requested = [name.strip() for name in args.need.split(",") if name.strip()]
    unknown = [name for name in requested if name not in _CHECKS]

    if unknown:
        print(f"unknown requirement(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    checks = check_environment(need=requested)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    print(report(checks))

    return 0 if all(check.ok for check in checks) else 2


__all__ = [
    "ALL",
    "API_KEY",
    "CORPUS",
    "DB",
    "METADATA",
    "Check",
    "check_environment",
    "env_file",
    "main",
    "mask",
    "report",
    "require_environment",
]


if __name__ == "__main__":
    raise SystemExit(main())
