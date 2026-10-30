import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """Build a throwaway git repository in tmp_path, cut off from the user's git config.

    Returns `commit(files, message="...")`, which writes `files` ({path: text or bytes}) into the repository, commits everything and returns the repository folder.
    Each call adds one commit, with fixed author, committer and date.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.org")
        monkeypatch.setenv(f"GIT_{role}_DATE", "2026-10-16T18:10:00+02:00")
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)

    def commit(files: dict[str, str | bytes], message: str = "Add files") -> Path:
        for name, content in files.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
        return repo

    return commit
