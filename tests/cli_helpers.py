"""Run the `mbdef` command line the way a user does: in a folder, with its own `.env`."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# The machine-specific keys of .env.example. Tests remove them from the environment, so a developer's own values can never leak into a test.
ENV_KEYS = (
    "DATA_ROOT",
    "HF_TOKEN",
    "HF_ENDPOINT",
    "DSPY_API_BASE",
    "DSPY_API_KEY",
    "SCIDEF_NLI_DEVICE",
    "UNPAYWALL_EMAIL",
    "OPENALEX_API_KEY",
    "CROSSREF_MAILTO",
    "SEMANTIC_SCHOLAR_API_KEY",
)
# And every cluster site's keys, CLUSTER_<SITE>_ROOT and CLUSTER_<SITE>_ACCOUNT (docs/setup/cluster.md).
CLUSTER_KEYS = "CLUSTER_"


def checkout_commit(folder: Path) -> str | None:
    """The commit checked out in the repository that holds FOLDER, or None outside a checkout (a copy of the repository without git)."""
    done = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True
    )
    return done.stdout.strip() if done.returncode == 0 else None


def package_commit() -> str | None:
    """The commit of the checkout that holds the package the commands run, as their receipts record it, or None outside a checkout."""
    import mbdef

    return checkout_commit(Path(mbdef.__file__).resolve().parent)


def run_cli(
    cwd: Path,
    *args: str,
    path: str | None = None,
    env: dict[str, str] | None = None,
    input: str = "",
):
    """Run `mbdef ARGS` in `cwd`. `path` replaces PATH (to hide or fake tools); `env` adds or replaces environment variables; `input` is standard input (a pipe, never a terminal)."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ENV_KEYS and not k.startswith(CLUSTER_KEYS)
    } | (env or {})
    if path is not None:
        env["PATH"] = path
    return subprocess.run(
        [sys.executable, "-m", "mbdef", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        input=input,
    )


def real_data_root() -> Path | None:
    """The developer's data root: DATA_ROOT in the repository's .env, or else in the environment.
    None if neither sets it."""
    env = Path(__file__).resolve().parents[1] / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("DATA_ROOT="):
            return Path(line.split("=", 1)[1].strip()).expanduser()
    value = os.environ.get("DATA_ROOT")
    return Path(value).expanduser() if value else None


def real_mbdef_gold():
    """MBDef-Gold's folder in the developer's data root (`mbdef data fetch MBDef-Gold/`), and a decorator for a test that reads it.
    The decorator marks the test slow (it needs the data root) and skips it where the data root does not hold MBDef-Gold."""
    import pytest

    root = real_data_root()
    folder = root / "MBDef-Gold" if root else Path("MBDef-Gold")
    missing = pytest.mark.skipif(
        not (folder / "validated_gt.json").is_file(),
        reason="needs MBDef-Gold in the data root (mbdef data fetch MBDef-Gold/)",
    )
    return folder, lambda test: pytest.mark.slow(missing(test))


def mini_gold(root: Path) -> Path:
    """A made-up MBDef-Gold in the data root ROOT: its two record files, with one train paper each (ROOT/MBDef-Gold/)."""
    import json

    folder = root / "MBDef-Gold"
    folder.mkdir(parents=True, exist_ok=True)
    for name, term in (
        ("validated_gt.json", "framing bias"),
        ("validated_gt_non_relevant.json", "LSTM"),
    ):
        paper = {
            "paper_id": "p1",
            "split": "train",
            "metadata": {"title": "Title of p1"},
            "definition_records": [
                {"term": term, "definitions": [{"text": f"{term} is a slant."}]}
            ],
        }
        (folder / name).write_text(json.dumps({"papers": [paper]}), encoding="utf-8")
    return root


def write_env(folder: Path, **values: str) -> None:
    """Write `folder/.env` with one KEY=value line per keyword."""
    lines = [f"{key}={value}" for key, value in values.items()]
    (folder / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tool_dir(folder: Path, *tools: str, fakes: dict[str, str] | None = None) -> str:
    """A folder for PATH holding links to the real `tools` and the given fake scripts.

    Each fake is Python source, run by this interpreter. Returns the folder as a PATH value.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for tool in tools:
        real = shutil.which(tool)
        assert real, f"{tool} is needed by the tests"
        (folder / tool).symlink_to(real)
    for name, source in (fakes or {}).items():
        script = folder / name
        script.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
        script.chmod(0o755)
    return str(folder)
