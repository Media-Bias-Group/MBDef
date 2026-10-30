"""What every Step 5 run records about itself: the code that ran and the exact input and output files."""

import platform
import sys
from pathlib import Path

from mbdef import repo
from mbdef.files import file_sha256

from . import REPO

PACKAGE = Path(__file__).resolve().parent


def file_record(path: Path) -> dict:
    path = path.resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path))


def _git(*args: str) -> str | None:
    """git's output in this checkout, or None if git failed or is not installed."""
    try:
        out = repo.git_or_none(REPO, *args)
    except OSError:
        return None
    return None if out is None else out.decode().strip()


def code_version() -> dict:
    """Commit and branch of the checkout, whether the Step 5 package differs from that commit, and its file hashes."""
    package = PACKAGE.relative_to(REPO)
    status = _git("status", "--porcelain", "--", str(package))
    return dict(
        git_commit=_git("rev-parse", "HEAD"),
        git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_dirty=None if status is None else bool(status),
        package_sha256={
            str(p.relative_to(REPO)): file_sha256(p)
            for p in sorted(PACKAGE.rglob("*.py"))
        },
        python=sys.version.split()[0],
        platform=platform.platform(),
    )
