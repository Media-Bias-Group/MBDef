"""The git helpers of the package: run git, list a tree's files and read their contents."""

import subprocess
from pathlib import Path, PurePosixPath


def git(
    root: Path, *args: str, env: dict | None = None, data: bytes | None = None
) -> str:
    """Run git in ROOT (with an optional environment, for example a temporary index, and standard input) and return its output; stop with git's message if it fails. The package's git calls all go through here or `git_or_none`."""
    done = subprocess.run(
        ["git", *args], cwd=root, env=env, input=data, capture_output=True
    )
    if done.returncode != 0:
        raise SystemExit(f"git {args[0]} failed: {done.stderr.decode().strip()}")
    return done.stdout.decode()


def git_or_none(root: Path, *args: str, data: bytes | None = None) -> bytes | None:
    """The output of a git command that may fail for an expected reason (a file that is not at a commit, a commit without a note), or None if it failed."""
    done = subprocess.run(["git", *args], cwd=root, input=data, capture_output=True)
    return done.stdout if done.returncode == 0 else None


def tree_files(root: Path, rev: str | None) -> list[tuple[str, int]]:
    """Return (path, size in bytes) for every file in `rev`'s tree, or in the index if `rev` is None."""
    fields = "--format=%(objectmode) %(objectsize) %(path)"
    if rev is None:
        out = git(root, "ls-files", "-z", fields)
    else:
        out = git(root, "ls-tree", "-r", "-z", "--full-tree", fields, rev)
    files = []
    for record in out.split("\0"):
        if not record:
            continue
        mode, size, path = record.split(" ", 2)
        if mode != "160000":  # a submodule has no file of its own
            files.append((path, int(size)))
    return files


def read_blobs(root: Path, rev: str | None, paths: list[str]) -> dict[str, bytes]:
    """Return the content of each path in `rev`'s tree (in the index if `rev` is None), read in one `git cat-file --batch` call."""
    request = "".join(f"{rev or ''}:{path}\n" for path in paths).encode()
    out = git_or_none(root, "cat-file", "--batch", data=request)
    if out is None:
        raise SystemExit(
            f"git cat-file could not read the files of {rev or 'the index'}."
        )
    blobs, at = {}, 0
    for path in paths:
        header_end = out.index(b"\n", at)
        size = int(out[at:header_end].split()[2])
        blobs[path] = out[header_end + 1 : header_end + 1 + size]
        at = header_end + 1 + size + 1
    return blobs


def matches_any(path: str, patterns: list[str]) -> bool:
    """A pattern must match the whole path: `*` stays inside one folder, `**` spans folders."""
    return any(PurePosixPath(path).full_match(pattern) for pattern in patterns)
