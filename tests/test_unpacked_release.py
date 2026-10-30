"""A copy of the repository without git works: outside a git checkout, and added to a new repository.

The repository can be downloaded as a zip, which has no `.git`. Its commands then run on code that no checkout holds: a receipt records that there is no commit (`"commit": null`), and `cluster render` writes a job that runs the code in its folder as it is, instead of refusing. Inside a checkout both behave as usual (tests/test_cluster.py and each step's receipt test). And when the unpacked files are added to a new repository, `git add -A` keeps every file: the repository's `.gitignore` hides no tracked file.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from cli_helpers import run_cli
from mini_scoring import paired_inputs

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_KEYS = {
    "CLUSTER_EXAMPLE_ROOT": "/projects/example/taxonomy",
    "CLUSTER_EXAMPLE_ACCOUNT": "example-project",
}


@pytest.fixture
def plain(tmp_path) -> Path:
    """A folder with the package and what the commands below read, and no git checkout above it."""
    tree = tmp_path / "copy"
    shutil.copytree(REPO / "src/mbdef", tree / "src/mbdef")
    for name in (
        "cluster/launcher.sbatch",
        "cluster/example.toml",
    ):
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, tree / name)
    outside = subprocess.run(
        ["git", "rev-parse", "--git-dir"], cwd=tree, capture_output=True, text=True
    )
    assert outside.returncode != 0, "the test folder is inside a git checkout"
    return tree


def run_plain(tree: Path, *args: str, env: dict[str, str] | None = None):
    """The command line of the copied package (not the installed one), run in TREE."""
    return run_cli(tree, *args, env={"PYTHONPATH": str(tree / "src"), **(env or {})})


def test_the_copied_package_is_the_one_that_runs(plain):
    done = run_plain(plain, "--help")

    assert done.returncode == 0, done.stderr
    where = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mbdef, pathlib; print(pathlib.Path(mbdef.__file__).parent)",
        ],
        cwd=plain,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(plain / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert where.stdout.strip() == str(plain / "src/mbdef")


def test_a_receipt_records_that_there_is_no_checkout(plain, tmp_path):
    selection, test_results, opus_dev = paired_inputs(tmp_path / "inputs")

    done = run_plain(
        plain,
        *("evaluate", "paired", "--selection", str(selection)),
        *("--test-results", str(test_results), "--opus-dev", str(opus_dev)),
        *("--out", "out"),
    )

    assert done.returncode == 0, done.stderr
    receipt = json.loads((plain / "out/receipt.json").read_text())
    assert receipt["code"] == {"commit": None, "dirty": None}


def test_render_writes_a_job_that_runs_the_code_as_it_is(plain):
    out = plain / "job.sbatch"

    done = run_plain(
        plain,
        *("cluster", "render", "--site", "example", "--gpus", "1"),
        *("--time", "01:00:00", "--step", "true", "--out", str(out)),
        env=EXAMPLE_KEYS,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == (
        f"Wrote {out} for example outside a git checkout: the job runs the code in its "
        "repo folder as it is, without checking a commit. Nothing was submitted."
    )
    script = out.read_text()
    assert "\ncommit=\n" in script
    checked = subprocess.run(["bash", "-n", str(out)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr


def test_the_postprocess_provenance_names_no_commit_outside_a_checkout(plain):
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from mbdef.postprocess import compare; "
            "print(json.dumps(compare.provenance()))",
        ],
        cwd=plain,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(plain / "src"), "PATH": "/usr/bin:/bin"},
    )

    assert done.returncode == 0, done.stderr
    provenance = json.loads(done.stdout)
    assert provenance["commit"] is None and provenance["postprocess_dirty"] is None


def test_no_tracked_file_is_hidden_by_the_gitignore():
    # `git add -A` on the unpacked files must keep every file the repository tracks.
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--ignored", "--exclude-standard"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        pytest.skip("not a git checkout: nothing is tracked here")
    assert listed.stdout.splitlines() == []
