"""The cluster job of the extractor selection, without a cluster.

The job is `selection.JOB_STEP`, rendered through the cluster template (docs/setup/cluster.md): it unpacks the scorings that `evaluate grid --write-scorings` wrote (one list per split and model) onto the node's own disk, scores the four lists with the general scorer, four processes at once on the node's one GPU, and records every pair score. Here render runs with a stand-in `sbatch` (it must never be called) and the step runs under bash with a stand-in `mbdef`, which notes how it was called.
"""

import io
import subprocess
import tarfile
from pathlib import Path

from cli_helpers import run_cli, tool_dir

from mbdef.evaluate import selection

REPO = Path(__file__).resolve().parents[1]
JOB = "selection/rescore/2026-10-20"
FAMILIES = ("test-qwen", "test-gemma12b", "dev-qwen", "dev-gemma12b")

# Notes the call (one file per call), writes the pair-score file it was asked for, and fails for the scorings list named in FAIL_ON, as a scorer that cannot score a setup does.
FAKE_SCORER = """\
import os, pathlib, sys
args = sys.argv[1:]
assert pathlib.Path(args[args.index("--scorings") + 1]).is_file()
calls = pathlib.Path(os.environ["CALLS"])
calls.mkdir(exist_ok=True)
(calls / str(os.getpid())).write_text("\\n".join(args))
pathlib.Path(args[args.index("--record-scores") + 1]).write_text("{}")
if os.environ["FAIL_ON"] in args[args.index("--scorings") + 1]:
    sys.exit(1)
"""

# Records the call, so a test can see that render never submits anything.
FAKE_SBATCH = """\
import pathlib, sys
pathlib.Path(sys.argv[0]).with_name("sbatch-was-called").write_text(" ".join(sys.argv))
sys.exit(1)
"""


def stage_scorings(tmp_path) -> None:
    """What the job finds in DATA_ROOT: the scorings of every family, as a zstd tar."""
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w") as archive:
        for family in FAMILIES:
            data = b"[]"
            member = tarfile.TarInfo(f"scorings/{family}.json")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    packed = subprocess.run(
        ["zstd", "-q", "-c"], input=tar.getvalue(), capture_output=True, check=True
    )
    target = tmp_path / "data-root" / JOB / "scorings.tar.zst"
    target.parent.mkdir(parents=True)
    target.write_bytes(packed.stdout)


def run_step(tmp_path, fail_on: str = "@"):
    """Run the job's step under bash as the job does: DATA_ROOT and the node's TMPDIR in tmp_path."""
    stage_scorings(tmp_path)
    (tmp_path / "node").mkdir()
    bin_dir = tool_dir(
        tmp_path / "bin",
        "bash",
        "printf",
        "xargs",
        "mkdir",
        "sha256sum",
        "tar",
        "zstd",
        fakes={"mbdef": FAKE_SCORER},
    )
    env = {
        "PATH": bin_dir,
        "DATA_ROOT": str(tmp_path / "data-root"),
        "TMPDIR": str(tmp_path / "node"),
        "CALLS": str(tmp_path / "calls"),
        "FAIL_ON": fail_on,
    }
    return subprocess.run(
        ["bash", "-c", selection.JOB_STEP],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )


def test_the_job_step_scores_each_scorings_list_and_records_its_pair_scores(tmp_path):
    done = run_step(tmp_path)

    assert done.returncode == 0, done.stderr
    root = tmp_path / "data-root" / JOB
    calls = sorted(
        path.read_text().split("\n") for path in (tmp_path / "calls").iterdir()
    )
    assert calls == sorted(
        [
            "evaluate",
            "score",
            "--scorings",
            f"{tmp_path}/node/scorings/{family}.json",
            "--record-scores",
            f"{root}/pair-scores/{family}.json",
            "--out",
            f"{root}/scores/{family}",
        ]
        for family in FAMILIES
    )
    # The pair scores' hashes, to compare after the files are copied back from the cluster.
    listed = (root / "pair-scores.sha256").read_text().splitlines()
    assert [line.split()[-1] for line in listed] == sorted(
        f"{family}.json" for family in FAMILIES
    )


def test_one_scorings_list_that_fails_fails_the_job(tmp_path):
    done = run_step(tmp_path, fail_on="dev-qwen")

    assert done.returncode != 0
    assert not (tmp_path / "data-root" / JOB / "pair-scores.sha256").exists()


def test_the_job_renders_at_the_normal_qos_and_submits_nothing(tmp_path):
    bin_dir = tool_dir(tmp_path / "bin", "git", fakes={"sbatch": FAKE_SBATCH})
    out = tmp_path / "selection.sbatch"

    done = run_cli(
        REPO,
        *("cluster", "render", "--site", "example", "--gpus", "1"),
        *("--time", selection.JOB_TIME, "--job-name", "selection-rescore"),
        *("--model", "nli", "--model", "bge"),
        *("--step", selection.JOB_STEP, "--out", str(out)),
        path=bin_dir,
        # The folder and the account are a person's settings, not the repository's: made up here.
        env={
            "CLUSTER_EXAMPLE_ROOT": "/projects/example/taxonomy",
            "CLUSTER_EXAMPLE_ACCOUNT": "example-project",
        },
    )

    assert done.returncode == 0, done.stderr
    assert not (tmp_path / "bin" / "sbatch-was-called").exists()
    script = out.read_text()
    sbatch = [line for line in script.splitlines() if line.startswith("#SBATCH ")]
    assert "#SBATCH --qos=normal" in sbatch
    assert "#SBATCH --gres=gpu:1" in sbatch
    assert f"#SBATCH --time={selection.JOB_TIME}" in sbatch
    assert "\ngpus=1\n" in script
    assert "tasksource/ModernBERT-large-nli@ca476cb9" in script
    assert "BAAI/bge-base-en-v1.5@a5beb1e3" in script
    checked = subprocess.run(["bash", "-n", str(out)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr


def test_the_docs_give_the_command_the_job_runs():
    docs = (REPO / "docs/step-3-extraction/selection.md").read_text(encoding="utf-8")

    assert selection.JOB_STEP in docs
