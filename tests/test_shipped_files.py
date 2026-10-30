"""What the repository ships names no private repository and no person behind the relevance step.

The shipped files are those git tracks. The words are spelled in two halves here, so that this file does not name them either.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
UNSHIPPED_WORDS = ("Isa" + "Hab", "Haber" + "mann", "Relevance" + "-Classification")


def shipped_files() -> list[Path]:
    done = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=False
    )
    if done.returncode != 0:
        pytest.skip("not a git checkout")
    return [REPO / name for name in done.stdout.decode().split("\0") if name]


def test_no_shipped_file_names_the_private_repository_or_its_author():
    found = []
    for path in shipped_files():
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore").casefold()
        found += [
            f"{path.relative_to(REPO)}: {word}"
            for word in UNSHIPPED_WORDS
            if word.casefold() in text
        ]
    assert not found, "\n".join(found)
