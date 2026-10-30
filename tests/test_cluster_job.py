"""A rendered job script, run with bash on a stand-in compute node (no Slurm, GPU or model).

The stand-ins: `nvidia-smi` and `curl` answer as on a healthy node, the venv's `python` runs the job's inline Python with a fake `torch` but only logs the `mbdef` checks (their own tests are in test_cluster.py), and `vllm` only records its arguments and sleeps. What runs for real is the job's own logic: the commit check, the environment, copying the models to node-local disk, starting and stopping the server, the step and its terminal record.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from cli_helpers import run_cli

REPO = Path(__file__).resolve().parents[1]
JOB = "4242"
PORT = 20000 + int(JOB) % 20000

FAKE_TORCH = """\
import os


class _Matrix:
    def __matmul__(self, other):
        return self

    def __getitem__(self, index):
        return 64.0


def ones(shape, device=None):
    return _Matrix()


class _Properties:
    name = "NVIDIA H100 80GB HBM3"
    total_memory = 80 * 1024**3


class cuda:
    class device:
        def __init__(self, index):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    @staticmethod
    def device_count():
        return int(os.environ["FAKE_GPU_COUNT"])

    @staticmethod
    def synchronize(index):
        pass

    @staticmethod
    def get_device_properties(index):
        return _Properties()

    @staticmethod
    def mem_get_info(index):
        return 80 * 1024**3, 80 * 1024**3
"""

FAKE_PYTHON = """\
#!/usr/bin/env bash
if [ "$1" = -m ]; then shift; echo "$*" >> "$FAKE_CALLS"; exit 0; fi
PYTHONPATH=$FAKE_TORCH exec {python} "$@"
"""

FAKE_NVIDIA_SMI = """\
#!/usr/bin/env bash
case "$*" in
  *-x*) echo '<nvidia_smi_log><gpu><gpu_recovery_action>None</gpu_recovery_action></gpu></nvidia_smi_log>' ;;
  *) echo 'name, driver_version, memory.total'; echo 'NVIDIA H100 80GB HBM3, 590.48, 81559 MiB' ;;
esac
"""

FAKE_VLLM = """\
#!/usr/bin/env bash
echo "$*" > "$FAKE_VLLM_ARGS"
exec sleep 300
"""

BGE = ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")
GEMMA = (
    "google/gemma-4-12B-it-qat-w4a16-ct",
    "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee",
)


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def cached_model(hub: Path, repo: str, revision: str) -> None:
    folder = hub / ("models--" + repo.replace("/", "--"))
    (folder / "blobs").mkdir(parents=True)
    (folder / "blobs/0123abcd").write_text(f"config of {repo}")
    (folder / "snapshots" / revision).mkdir(parents=True)
    (folder / "snapshots" / revision / "config.json").symlink_to("../../blobs/0123abcd")


def node(tmp_path, git_repo):
    """A cluster root in tmp_path: the checkout (repo/), a stand-in venv, two cached models."""
    config = """\
qos = "normal"
stage_models = true

[gpu]
partition = "h100"
gres = "gpu:H100:{gpus}"
names = ["H100"]
min_memory_gb = 79
cpus_per_gpu = 8
memory_gb_per_gpu = 100

[env]
SITE_SETTING = "{root}/site"
"""
    repo = git_repo(
        {
            "cluster/launcher.sbatch": (REPO / "cluster/launcher.sbatch").read_text(),
            "cluster/node.toml": config,
        }
    )
    executable(tmp_path / "venv/bin/python", FAKE_PYTHON.format(python=sys.executable))
    executable(tmp_path / "venv/bin/vllm", FAKE_VLLM)
    executable(tmp_path / "venv/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc", "")
    executable(tmp_path / "tools/nvidia-smi", FAKE_NVIDIA_SMI)
    executable(tmp_path / "tools/curl", "#!/usr/bin/env bash\nexit 0\n")
    (tmp_path / "fake-torch/torch").mkdir(parents=True)
    (tmp_path / "fake-torch/torch/__init__.py").write_text(FAKE_TORCH)
    for model in (BGE, GEMMA):
        cached_model(tmp_path / "hf-cache/hub", *model)
    return repo


def run_job(tmp_path, script: Path) -> subprocess.CompletedProcess:
    env = {
        "PATH": f"{tmp_path / 'tools'}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "SLURM_JOB_ID": JOB,
        "FAKE_CALLS": str(tmp_path / "calls.txt"),
        "FAKE_TORCH": str(tmp_path / "fake-torch"),
        "FAKE_VLLM_ARGS": str(tmp_path / "vllm-args.txt"),
        "FAKE_GPU_COUNT": "1",
        "OPENROUTER_API_KEY": "a key that must not reach the step",
    }
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env, timeout=120
    )


def test_the_rendered_job_checks_the_node_serves_the_model_and_runs_the_step(
    tmp_path, git_repo
):
    repo = node(tmp_path, git_repo)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    step = (
        f"env > {tmp_path}/step-env.txt; "
        f'find "$HF_HUB_CACHE" | sort > {tmp_path}/staged.txt; echo step ran'
    )
    rendered = run_cli(
        repo,
        "cluster",
        "render",
        "--site",
        "node",
        "--time",
        "01:00:00",
        *("--model", "bge", "--serve", "gemma", "--step", step),
        *("--out", str(tmp_path / "job.sbatch")),
        env={"CLUSTER_NODE_ROOT": str(tmp_path)},
    )
    assert rendered.returncode == 0, rendered.stderr

    done = run_job(tmp_path, tmp_path / "job.sbatch")

    assert done.returncode == 0, done.stdout + done.stderr
    logs = tmp_path / f"logs/job-{JOB}"
    terminal = json.loads((logs / "terminal.json").read_text())
    assert {k: terminal[k] for k in ("job", "site", "commit", "step", "exit_code")} == {
        "job": JOB,
        "site": "node",
        "commit": commit,
        "step": step,
        "exit_code": 0,
    }
    assert terminal["seconds"] >= 0
    assert (logs / "step.log").read_text() == "step ran\n"
    assert (
        json.loads((logs / "gpu-health-canary.json").read_text())["status"] == "passed"
    )

    calls = (tmp_path / "calls.txt").read_text().splitlines()
    staged_hub = next(
        line.split("=", 1)[1]
        for line in (tmp_path / "step-env.txt").read_text().splitlines()
        if line.startswith("HF_HUB_CACHE=")
    )
    assert calls == [
        f"mbdef cluster check-runtime --out {logs}/runtime.json",
        f"mbdef cluster download --offline --cache {staged_hub} "
        f"--model {BGE[0]}@{BGE[1]} --model {GEMMA[0]}@{GEMMA[1]} --out {logs}/models.json",
        f"mbdef cluster vllm-check --base http://127.0.0.1:{PORT} "
        f"--model {GEMMA[0]} --max-model-len 65536 --log {logs}/vllm.log "
        f"--out {logs}/vllm-health.json",
    ]
    vllm_args = (tmp_path / "vllm-args.txt").read_text().split()
    assert vllm_args[:3] == ["serve", GEMMA[0], "--revision"]
    assert vllm_args[-6:] == [
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--data-parallel-rpc-port",
        str(PORT + 1),
    ]

    step_env = dict(
        line.split("=", 1)
        for line in (tmp_path / "step-env.txt").read_text().splitlines()
        if "=" in line
    )
    assert step_env["DSPY_API_BASE"] == f"http://127.0.0.1:{PORT}/v1"
    assert step_env["DATA_ROOT"] == f"{tmp_path}/data-root"
    assert step_env["OPENROUTER_API_KEY"] == ""
    assert step_env["HF_HUB_OFFLINE"] == "1"
    assert step_env["SCIDEF_NLI_DEVICE"] == "cuda"
    assert step_env["PYTHONHASHSEED"] == "0"
    assert step_env["SITE_SETTING"] == f"{tmp_path}/site"
    assert staged_hub.startswith(f"/tmp/j{JOB}.") and staged_hub.endswith("/hub")
    staged = (tmp_path / "staged.txt").read_text().splitlines()
    for repo_id, revision in (BGE, GEMMA):
        folder = "models--" + repo_id.replace("/", "--")
        assert f"{staged_hub}/{folder}/blobs/0123abcd" in staged
        assert f"{staged_hub}/{folder}/snapshots/{revision}/config.json" in staged
    assert not Path(staged_hub).exists()  # TMPDIR is removed at the end


def test_the_rendered_job_lets_flashinfer_link_the_cuda_runtime_of_the_wheels(
    tmp_path, git_repo
):
    """vLLM's FlashInfer builds some kernels on first use and links them with `-lcudart`. The CUDA wheels hold the runtime only as `lib/libcudart.so.13`, so on a node without a system CUDA toolkit the link fails (`ld: cannot find -lcudart`) unless the job names it."""
    repo = node(tmp_path, git_repo)
    runtime = (
        tmp_path / "venv/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so.13"
    )
    runtime.parent.mkdir(parents=True)
    runtime.write_text("the CUDA runtime of the wheels")
    found = tmp_path / "linker-finds.txt"
    step = (
        'for d in ${LIBRARY_PATH//:/ }; do [ -e "$d/libcudart.so" ] && '
        f'readlink -f "$d/libcudart.so"; done > {found}'
    )
    rendered = run_cli(
        repo,
        "cluster",
        "render",
        "--site",
        "node",
        "--time",
        "01:00:00",
        *("--serve", "gemma", "--step", step, "--out", str(tmp_path / "job.sbatch")),
        env={"CLUSTER_NODE_ROOT": str(tmp_path)},
    )
    assert rendered.returncode == 0, rendered.stderr

    done = run_job(tmp_path, tmp_path / "job.sbatch")

    assert done.returncode == 0, done.stdout + done.stderr
    assert found.read_text().splitlines() == [str(runtime)]


def test_the_rendered_job_lets_nvrtc_open_the_builtins_of_the_wheels(
    tmp_path, git_repo
):
    """CUDA's runtime compiler (NVRTC), which vLLM's 4-bit Gemma kernel on H200 calls while it loads the weights, opens `libnvrtc-builtins.so.13.0` by name. The job unsets LD_LIBRARY_PATH and the node has no system CUDA, so without the wheels' copy on the loader's path it is not found."""
    repo = node(tmp_path, git_repo)
    lib = tmp_path / "venv/lib/python3.13/site-packages/nvidia/cu13/lib"
    lib.mkdir(parents=True)
    builtins = lib / "libnvrtc-builtins.so.13.0"
    builtins.write_text("the NVRTC builtins of the wheels")
    found = tmp_path / "loader-finds.txt"
    step = (
        'for d in ${LD_LIBRARY_PATH//:/ }; do [ -e "$d/libnvrtc-builtins.so.13.0" ] && '
        f'readlink -f "$d/libnvrtc-builtins.so.13.0"; done > {found}'
    )
    rendered = run_cli(
        repo,
        "cluster",
        "render",
        "--site",
        "node",
        "--time",
        "01:00:00",
        *("--serve", "gemma", "--step", step, "--out", str(tmp_path / "job.sbatch")),
        env={"CLUSTER_NODE_ROOT": str(tmp_path)},
    )
    assert rendered.returncode == 0, rendered.stderr

    done = run_job(tmp_path, tmp_path / "job.sbatch")

    assert done.returncode == 0, done.stdout + done.stderr
    assert found.read_text().splitlines() == [str(builtins)]


def test_the_rendered_job_refuses_another_commit(tmp_path, git_repo):
    repo = node(tmp_path, git_repo)
    rendered = run_cli(
        repo,
        "cluster",
        "render",
        "--site",
        "node",
        "--time",
        "01:00:00",
        *("--step", "echo step ran", "--out", str(tmp_path / "job.sbatch")),
        env={"CLUSTER_NODE_ROOT": str(tmp_path)},
    )
    assert rendered.returncode == 0, rendered.stderr
    git_repo({"later.txt": "a later change"}, "A later commit")
    later = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()

    done = run_job(tmp_path, tmp_path / "job.sbatch")

    assert done.returncode == 1
    assert f"FATAL {repo} is at {later}, not " in done.stderr
    assert not (tmp_path / f"logs/job-{JOB}/step.log").exists()


def test_a_job_rendered_outside_a_checkout_runs_its_folder_as_it_is(tmp_path, git_repo):
    # A copy of the repository without git has no checkout: render leaves `commit=` empty, and the job runs the code in its repo folder without the commit check (tests/test_unpacked_release.py).
    repo = node(tmp_path, git_repo)
    shutil.rmtree(repo / ".git")
    rendered = run_cli(
        repo,
        "cluster",
        "render",
        "--site",
        "node",
        "--time",
        "01:00:00",
        *("--step", "echo step ran", "--out", str(tmp_path / "job.sbatch")),
        env={"CLUSTER_NODE_ROOT": str(tmp_path)},
    )
    assert rendered.returncode == 0, rendered.stderr
    assert "outside a git checkout" in rendered.stdout

    done = run_job(tmp_path, tmp_path / "job.sbatch")

    assert done.returncode == 0, done.stdout + done.stderr
    terminal = json.loads((tmp_path / f"logs/job-{JOB}/terminal.json").read_text())
    assert (terminal["commit"], terminal["exit_code"]) == ("", 0)
    assert "step ran" in (tmp_path / f"logs/job-{JOB}/step.log").read_text()
