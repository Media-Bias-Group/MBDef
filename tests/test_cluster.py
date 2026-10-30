"""`mbdef cluster`: the launcher render, the model download, the runtime and vLLM checks.

Render is tested on the site-neutral example site file, `cluster/example.toml`, and on made-up site files in throwaway repositories. Nothing here submits a job or reaches a cluster, the Hub or a GPU: a stand-in `sbatch` fails the test if anything calls it, a stand-in `huggingface_hub` writes a tiny cache, and a stand-in vLLM server answers on localhost.
"""

import hashlib
import http.server
import importlib.metadata
import json
import platform
import re
import subprocess
import threading
import tomllib
from pathlib import Path
from typing import Any

import pytest
from cli_helpers import checkout_commit, run_cli, tool_dir

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "cluster/example.toml"

# Records the call, so a test can see that render never submits anything.
FAKE_SBATCH = """\
import pathlib, sys
pathlib.Path(sys.argv[0]).with_name("sbatch-was-called").write_text(" ".join(sys.argv))
sys.exit(1)
"""


# A made-up account and folder for the example site. The real ones are in each person's `.env` (docs/setup/cluster.md), never in the repository: `cluster render` reads them from these keys.
SITE_ENV = {
    "CLUSTER_EXAMPLE_ROOT": "/projects/example/taxonomy",
    "CLUSTER_EXAMPLE_ACCOUNT": "example-project",
}
ROOT = SITE_ENV["CLUSTER_EXAMPLE_ROOT"]


def render(tmp_path, *args: str, cwd: Path = REPO, env: dict[str, str] | None = None):
    """Run `cluster render ARGS` with only git and a stand-in sbatch on PATH. ENV replaces the site keys (default: SITE_ENV)."""
    bin_dir = tmp_path / "bin"
    if not bin_dir.exists():
        tool_dir(bin_dir, "git", fakes={"sbatch": FAKE_SBATCH})
    done = run_cli(
        cwd, "cluster", "render", *args, path=str(bin_dir), env=env or SITE_ENV
    )
    assert not (tmp_path / "bin/sbatch-was-called").exists()
    return done


def sbatch_lines(script: str) -> list[str]:
    return [line for line in script.splitlines() if line.startswith("#SBATCH ")]


def assert_valid_bash(tmp_path, script: str) -> None:
    job = tmp_path / "job.sbatch"
    job.write_text(script, encoding="utf-8")
    checked = subprocess.run(["bash", "-n", str(job)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr


def site_repo(git_repo, name: str, change: dict[str, str]) -> Path:
    """A throwaway checkout with the launcher and cluster/NAME.toml: the example site file with each text in CHANGE replaced."""
    text = EXAMPLE.read_text(encoding="utf-8")
    for old, new in change.items():
        assert old in text, old
        text = text.replace(old, new)
    return git_repo(
        {
            "cluster/launcher.sbatch": (REPO / "cluster/launcher.sbatch").read_text(),
            f"cluster/{name}.toml": text,
        }
    )


def test_render_for_the_example_site_uses_its_account_partition_and_the_normal_qos(
    tmp_path,
):
    done = render(
        tmp_path,
        *("--site", "example", "--gpus", "4", "--time", "02:00:00"),
        *("--step", "mbdef evaluate score --help"),
    )

    assert done.returncode == 0, done.stderr
    assert sbatch_lines(done.stdout) == [
        "#SBATCH --job-name=taxonomy",
        "#SBATCH --account=example-project",
        "#SBATCH --partition=gpu",
        "#SBATCH --qos=normal",
        "#SBATCH --time=02:00:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gres=gpu:4",
        "#SBATCH --cpus-per-task=64",
        "#SBATCH --mem=480G",
        f"#SBATCH --output={ROOT}/logs/slurm-%j.out",
    ]
    assert f"\nroot={ROOT}\n" in done.stdout
    assert "\ngpus=4\n" in done.stdout
    assert "step='mbdef evaluate score --help'" in done.stdout
    assert "\n# (nothing for this site)\n" in done.stdout
    assert_valid_bash(tmp_path, done.stdout)


@pytest.mark.parametrize("key", sorted(SITE_ENV))
def test_render_stops_with_a_pointer_to_env_example_when_a_site_key_is_unset(
    tmp_path, key
):
    done = render(
        tmp_path,
        *("--site", "example", "--time", "01:00:00", "--step", "true"),
        env={**SITE_ENV, key: ""},
    )

    assert done.returncode == 1
    assert done.stdout == ""
    assert done.stderr.strip() == (
        f"{key} is not set. Copy .env.example to .env and set {key} there; "
        ".env.example says what it is."
    )


def test_a_site_that_bills_no_account_asks_only_for_its_root(tmp_path, git_repo):
    repo = site_repo(git_repo, "own", {"needs_account = true\n": ""})

    done = render(
        tmp_path,
        *("--site", "own", "--time", "01:00:00", "--step", "true"),
        cwd=repo,
        env={"CLUSTER_OWN_ROOT": "/projects/own"},
    )

    assert done.returncode == 0, done.stderr
    assert not [
        x for x in sbatch_lines(done.stdout) if x.startswith("#SBATCH --account")
    ]
    assert "\nroot=/projects/own\n" in done.stdout


def test_every_committed_site_file_has_the_normal_qos_and_no_account_or_home_folder(
    tmp_path,
):
    """A site file says what a cluster is; an account and a root folder name a person and a cluster user, so they live in `.env`."""
    configs = sorted((REPO / "cluster").glob("*.toml"))

    assert EXAMPLE in configs
    for config in configs:
        text = config.read_text(encoding="utf-8")
        table = tomllib.loads(text)
        assert table.get("qos", "normal") == "normal", config
        assert "root" not in table and "account" not in table, config
        assert not re.search(r"/(home|mnt|scratch)/", text), config
        done = render(
            tmp_path,
            *("--site", config.stem, "--time", "01:00:00", "--step", "true"),
            env={
                f"CLUSTER_{config.stem.upper()}_{k}": "x" for k in ("ROOT", "ACCOUNT")
            },
        )
        assert "#SBATCH --qos=normal" in sbatch_lines(done.stdout), config


def test_env_example_lists_every_key_that_render_reads_for_the_example():
    lines = (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
    keys = {line.split("=")[0] for line in lines if "=" in line and line[0] != "#"}

    assert set(SITE_ENV) <= keys


def test_a_site_config_with_another_qos_is_refused(tmp_path, git_repo):
    repo = site_repo(git_repo, "fast", {'qos = "normal"': 'qos = "high"'})

    done = render(
        tmp_path,
        *("--site", "fast", "--time", "01:00:00", "--step", "true"),
        cwd=repo,
        env={"CLUSTER_FAST_ROOT": "/projects/fast", "CLUSTER_FAST_ACCOUNT": "fast"},
    )

    assert done.returncode == 1
    assert done.stderr.strip() == (
        "cluster/fast.toml sets qos = 'high'. A site's QOS must be the normal one "
        "(normal), the only one render writes."
    )


@pytest.mark.parametrize("qos", ["high", "preempt"])
def test_render_writes_only_the_normal_qos(tmp_path, qos):
    common = ("--site", "example", "--time", "01:00:00", "--step", "true")

    normal = render(tmp_path, *common, "--qos", "normal")
    other = render(tmp_path, *common, "--qos", qos)

    assert normal.returncode == 0, normal.stderr
    assert "#SBATCH --qos=normal" in sbatch_lines(normal.stdout)
    assert other.returncode == 1
    assert other.stdout == ""
    assert other.stderr.strip() == (
        f"--qos {qos}: render writes only the normal QOS (--qos=normal), never one that "
        "pre-empts other users' jobs or jumps ahead of them. Another QOS needs the "
        "agreement of whoever answers for the project's use of the cluster; with it, edit "
        "the rendered file by hand."
    )


NLI = "tasksource/ModernBERT-large-nli@ca476cb923a8637073d4ceb0f19f7fc236e260d4"
BGE = "BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
QWEN = "Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0"


def test_render_writes_a_one_gpu_job_that_reads_two_pinned_models(tmp_path):
    # The shape of the selection rescoring: one GPU, the entailment and embedding models.
    out = tmp_path / "selection.sbatch"
    done = render(
        tmp_path,
        *("--site", "example", "--gpus", "1", "--time", "02:00:00"),
        *("--model", "nli", "--model", "bge", "--job-name", "selection-rescore"),
        *("--step", "true", "--out", str(out)),
    )

    assert done.returncode == 0, done.stderr
    script = out.read_text()
    commit = checkout_commit(REPO)
    assert done.stdout.startswith(
        f"Wrote {out} for example at commit {commit[:12]}"
        if commit
        else f"Wrote {out} for example outside a git checkout"
    )
    assert done.stdout.rstrip().endswith("Nothing was submitted.")
    assert sbatch_lines(script)[:2] == [
        "#SBATCH --job-name=selection-rescore",
        "#SBATCH --account=example-project",
    ]
    assert "#SBATCH --gres=gpu:1" in sbatch_lines(script)
    assert "#SBATCH --cpus-per-task=16" in sbatch_lines(script)
    assert "#SBATCH --mem=120G" in sbatch_lines(script)
    assert f"\nmodels=({NLI} {BGE})\n" in script
    assert (
        "\nmodel_dirs=(models--tasksource--ModernBERT-large-nli models--BAAI--bge-base-en-v1.5)\n"
        in script
    )
    assert "\nstage_models=0\n" in script
    assert "\nserve=()\n" in script
    assert_valid_bash(tmp_path, script)

    again = render(
        tmp_path,
        *(
            "--site",
            "example",
            "--time",
            "1:00:00",
            "--step",
            "true",
            "--out",
            str(out),
        ),
    )
    assert again.returncode == 1
    assert again.stderr.strip() == f"{out} exists; render never overwrites a file."
    assert out.read_text() == script


def test_a_site_that_stages_models_and_sets_its_environment(tmp_path, git_repo):
    repo = site_repo(
        git_repo,
        "slow-disk",
        {
            "stage_models = false": "stage_models = true",
            "[env]\n": '[env]\nLD_LIBRARY_PATH = "{root}/cuda-compat/lib"\n',
        },
    )

    done = render(
        tmp_path,
        *(
            "--site",
            "slow-disk",
            "--time",
            "01:00:00",
            "--model",
            "bge",
            "--step",
            "true",
        ),
        cwd=repo,
        env={"CLUSTER_SLOW_DISK_ROOT": "/projects/s", "CLUSTER_SLOW_DISK_ACCOUNT": "s"},
    )

    assert done.returncode == 0, done.stderr
    assert "\nstage_models=1\n" in done.stdout
    assert "\nexport LD_LIBRARY_PATH=/projects/s/cuda-compat/lib\n" in done.stdout
    assert_valid_bash(tmp_path, done.stdout)


def test_render_serves_the_pinned_qwen_with_vllm_and_checks_it(tmp_path):
    done = render(
        tmp_path,
        *("--site", "example", "--gpus", "4", "--time", "06:00:00"),
        *("--serve", "qwen", "--think", "--step", "true"),
    )

    assert done.returncode == 0, done.stderr
    assert (
        "\nserve=(Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 "
        "--max-model-len 65536 --max-num-seqs 1024 --max-num-batched-tokens 16384 "
        "--gpu-memory-utilization 0.82 --enable-prefix-caching --generation-config auto "
        "--enable-prompt-tokens-details --language-model-only "
        "--tensor-parallel-size 2 --data-parallel-size 2 --reasoning-parser qwen3)\n"
    ) in done.stdout
    assert (
        "\nserve_check=(--model Qwen/Qwen3.6-35B-A3B --max-model-len 65536 --think)\n"
        in done.stdout
    )
    assert f"\nmodels=({QWEN})\n" in done.stdout
    assert_valid_bash(tmp_path, done.stdout)

    odd = render(
        tmp_path,
        *("--site", "example", "--gpus", "3", "--time", "1:00:00"),
        *("--serve", "qwen", "--step", "true"),
    )
    assert odd.returncode == 1
    assert odd.stderr.strip() == (
        "qwen is served with tensor parallelism 2, so --gpus must be a multiple of 2."
    )


def test_a_cpu_job_uses_the_site_cpu_partition_and_no_gpu(tmp_path, git_repo):
    cpu = render(
        tmp_path,
        *("--site", "example", "--gpus", "0", "--time", "06:00:00", "--step", "true"),
    )
    repo = site_repo(git_repo, "gpus-only", {"[cpu]\n": "[cpu-unused]\n"})
    gpus_only = render(
        tmp_path,
        *("--site", "gpus-only", "--gpus", "0", "--time", "06:00:00", "--step", "true"),
        cwd=repo,
        env={"CLUSTER_GPUS_ONLY_ROOT": "/projects/g", "CLUSTER_GPUS_ONLY_ACCOUNT": "g"},
    )

    assert cpu.returncode == 0, cpu.stderr
    assert sbatch_lines(cpu.stdout) == [
        "#SBATCH --job-name=taxonomy",
        "#SBATCH --account=example-project",
        "#SBATCH --partition=cpu",
        "#SBATCH --qos=normal",
        "#SBATCH --time=06:00:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=48",
        "#SBATCH --mem=120G",
        f"#SBATCH --output={ROOT}/logs/slurm-%j.out",
    ]
    assert "\ngpus=0\n" in cpu.stdout
    assert_valid_bash(tmp_path, cpu.stdout)
    assert gpus_only.returncode == 1
    assert gpus_only.stderr.strip() == (
        "cluster/gpus-only.toml has no [cpu] table, so this site takes no CPU-only job (--gpus 0)."
    )


def test_the_example_site_accepts_the_80_gb_gpus_the_released_runs_used(tmp_path):
    """The job's GPU check passes a device when one accepted name is part of its name (torch's `get_device_properties(i).name`) and it has at least `min_memory_gb`."""
    done = render(
        tmp_path, *("--site", "example", "--time", "01:00:00", "--step", "true")
    )

    found = re.search(r"\ngpu_names=(\S+)\n", done.stdout)
    assert found, done.stdout
    names = found.group(1).split(",")
    for device in ("NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3", "NVIDIA H200"):
        assert any(name in device for name in names), device
    assert "\nmin_gpu_gb=79\n" in done.stdout


# A stand-in for huggingface_hub.snapshot_download: it logs each call and writes a two-file snapshot laid out like the real cache, with each blob named by its Hub object id (the sha256 of a large file, the git blob sha1 of a small one).
FAKE_HUB = """\
import hashlib, json, os
from pathlib import Path
from typing import Any

__version__ = "0.0-fake"


def snapshot_download(repo_id, revision=None, cache_dir=None, local_files_only=False, **kwargs):
    with open(os.environ["FAKE_HUB_LOG"], "a") as log:
        log.write(json.dumps([repo_id, revision, local_files_only]) + "\\n")
    folder = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
    snapshot = folder / "snapshots" / revision
    if local_files_only:
        if not snapshot.is_dir():
            raise FileNotFoundError(f"{repo_id}@{revision} is not in the cache")
        return str(snapshot)
    files = {"config.json": b'{"model": "%s"}' % repo_id.encode(),
             "model.safetensors": b"weights of " + repo_id.encode()}
    for name, data in files.items():
        if name.endswith(".json"):
            blob = hashlib.sha1(b"blob %d\\0" % len(data) + data).hexdigest()
        else:
            blob = hashlib.sha256(data).hexdigest()
        (folder / "blobs").mkdir(parents=True, exist_ok=True)
        (folder / "blobs" / blob).write_bytes(data)
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / name).symlink_to(Path("../../blobs") / blob)
    return str(snapshot)
"""

PINNED = [
    ("google/gemma-4-12B-it-qat-w4a16-ct", "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee"),
    ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
    ("tasksource/ModernBERT-large-nli", "ca476cb923a8637073d4ceb0f19f7fc236e260d4"),
]


def fake_hub(tmp_path, monkeypatch) -> Path:
    package = tmp_path / "fake-hub/huggingface_hub"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(FAKE_HUB)
    monkeypatch.setenv("PYTHONPATH", str(package.parent))
    log = tmp_path / "hub-calls.jsonl"
    monkeypatch.setenv("FAKE_HUB_LOG", str(log))
    return log


def hub_calls(log: Path) -> list:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_download_fetches_the_pinned_models_and_records_every_file(
    tmp_path, monkeypatch
):
    log = fake_hub(tmp_path, monkeypatch)
    cache, receipt = tmp_path / "hub", tmp_path / "models.json"

    done = run_cli(
        tmp_path, "cluster", "download", "--cache", str(cache), "--out", str(receipt)
    )

    assert done.returncode == 0, done.stderr
    assert hub_calls(log) == [[repo, rev, False] for repo, rev in PINNED]
    written = json.loads(receipt.read_text())
    assert written["command"] == "cluster download"
    assert written["settings"] == {
        "models": [f"{repo}@{rev}" for repo, rev in PINNED],
        "cache": str(cache),
        "offline": False,
        "huggingface_hub": "0.0-fake",
    }
    gemma = written["models"][0]
    weights = b"weights of google/gemma-4-12B-it-qat-w4a16-ct"
    assert (gemma["repo"], gemma["revision"]) == PINNED[0]
    assert gemma["verified"] is True and gemma["problems"] == []
    assert gemma["files"][1] == {
        "path": "model.safetensors",
        "blob": hashlib.sha256(weights).hexdigest(),
        "bytes": len(weights),
        "sha256": hashlib.sha256(weights).hexdigest(),
    }
    assert all(m["verified"] for m in written["models"])
    assert (
        "4 models are in the cache and every file matches its Hub object id"
        in done.stdout
    )


def test_download_offline_finds_a_changed_or_missing_model(tmp_path, monkeypatch):
    log = fake_hub(tmp_path, monkeypatch)
    cache = tmp_path / "hub"
    first = run_cli(
        tmp_path,
        "cluster",
        "download",
        "--model",
        "bge",
        "--cache",
        str(cache),
        "--out",
        str(tmp_path / "one.json"),
    )
    assert first.returncode == 0, first.stderr
    blob = cache / "models--BAAI--bge-base-en-v1.5/blobs"
    weights = next(p for p in blob.iterdir() if len(p.name) == 64)
    weights.write_bytes(b"changed on the way")

    done = run_cli(
        tmp_path,
        "cluster",
        "download",
        "--offline",
        "--cache",
        str(cache),
        *("--model", "bge", "--model", "nli", "--out", str(tmp_path / "two.json")),
    )

    assert done.returncode == 1
    assert hub_calls(log)[1:] == [[*PINNED[2], True], [*PINNED[3], True]]
    written = json.loads((tmp_path / "two.json").read_text())
    bge, nli = written["models"]
    assert bge["verified"] is False
    assert bge["problems"] == [
        "model.safetensors: its content does not match its Hub object id"
    ]
    assert nli["verified"] is False
    assert nli["problems"] == [
        "not found in the cache: tasksource/ModernBERT-large-nli@"
        "ca476cb923a8637073d4ceb0f19f7fc236e260d4 is not in the cache"
    ]
    assert "2 of 2 models are missing or differ" in done.stdout


GEMMA = "google/gemma-4-12B-it-qat-w4a16-ct"
METRICS = f"""\
# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{{engine="0",model_name="{GEMMA}"}} 0.0
vllm:num_requests_running{{engine="1",model_name="{GEMMA}"}} 0.0
"""
VLLM_LOG = """\
INFO 10-20 12:00:01 [gpu_worker.py:298] Available KV cache memory: 52.10 GiB
INFO 10-20 12:00:01 [kv_cache_utils.py:1087] GPU KV cache size: 1,138,688 tokens
INFO 10-20 12:00:01 [kv_cache_utils.py:1091] Maximum concurrency for 65,536 tokens per request: 17.38x
INFO 10-20 12:00:09 [api_server.py:1880] Starting vLLM API server 0 on http://127.0.0.1:8123
"""


@pytest.fixture
def fake_vllm():
    """A stand-in vLLM server on localhost; `served` sets what /v1/models reports."""
    served: dict[str, Any] = {"id": GEMMA, "max_model_len": 65536, "tokenize": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def answer(self, body: bytes, kind: str = "application/json"):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                model = {"id": served["id"], "max_model_len": served["max_model_len"]}
                self.answer(json.dumps({"data": [model]}).encode())
            elif self.path == "/metrics":
                self.answer(METRICS.encode(), "text/plain")
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            served["tokenize"].append(json.loads(self.rfile.read(length)))
            self.answer(json.dumps({"count": 9, "tokens": list(range(9))}).encode())

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    served["base"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield served
    server.shutdown()


def vllm_check(tmp_path, served, *args):
    log = tmp_path / "vllm.log"
    log.write_text(VLLM_LOG)
    return run_cli(
        tmp_path,
        "cluster",
        "vllm-check",
        "--base",
        served["base"],
        *("--log", str(log), "--out", str(tmp_path / "health.json"), *args),
    )


def test_vllm_check_passes_a_server_that_serves_the_expected_model(tmp_path, fake_vllm):
    done = vllm_check(tmp_path, fake_vllm, "--model", GEMMA, "--think")

    assert done.returncode == 0, done.stdout + done.stderr
    assert fake_vllm["tokenize"] == [
        {
            "model": GEMMA,
            "messages": [{"role": "user", "content": "health"}],
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ]
    health = json.loads((tmp_path / "health.json").read_text())
    assert health["base"] == fake_vllm["base"]
    assert health["model"] == GEMMA
    assert health["max_model_len"] == 65536
    assert health["tokenize_count"] == 9
    assert health["engines_in_metrics"] == ["0", "1"]
    assert health["kv_cache_lines"] == [
        "GPU KV cache size: 1,138,688 tokens",
        "Maximum concurrency for 65,536 tokens per request: 17.38x",
    ]
    assert health["passed"] is True and health["problems"] == []
    assert done.stdout.strip() == (
        f"{fake_vllm['base']} serves {GEMMA} with max_model_len 65536, tokenizes a "
        "request (9 tokens) and reports 2 engines."
    )


@pytest.mark.parametrize(
    "served, problem",
    [
        (
            {"id": "Qwen/Qwen3-8B"},
            f"/v1/models lists ['Qwen/Qwen3-8B'], not {GEMMA} alone",
        ),
        (
            {"max_model_len": 16384},
            "/v1/models says max_model_len 16384, not 65536",
        ),
    ],
)
def test_vllm_check_fails_a_wrong_model_or_context(
    tmp_path, fake_vllm, served, problem
):
    fake_vllm.update(served)

    done = vllm_check(tmp_path, fake_vllm, "--model", GEMMA)

    assert done.returncode == 1
    health = json.loads((tmp_path / "health.json").read_text())
    assert health["passed"] is False
    assert health["problems"] == [problem]
    assert done.stdout.strip() == f"FAILED: {fake_vllm['base']}: {problem}"


def test_vllm_check_fails_when_nothing_answers(tmp_path):
    done = run_cli(
        tmp_path,
        "cluster",
        "vllm-check",
        "--base",
        "http://127.0.0.1:9",
        *("--model", GEMMA, "--out", str(tmp_path / "health.json")),
    )

    assert done.returncode == 1
    assert done.stdout.startswith(
        "FAILED: http://127.0.0.1:9: /v1/models did not answer"
    )


def lock(tmp_path, requires_python: str, **versions: str) -> Path:
    path = tmp_path / "uv.lock"
    packages = "".join(
        f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n'
        for name, version in versions.items()
    )
    path.write_text(f'version = 1\nrequires-python = "{requires_python}"\n{packages}')
    return path


def test_check_runtime_passes_when_the_installed_packages_match_uv_lock(tmp_path):
    installed = importlib.metadata.version("pytest")
    uv_lock = lock(tmp_path, ">=3.14", pytest=installed, dspy="3.3.1")

    done = run_cli(
        tmp_path,
        "cluster",
        "check-runtime",
        "--lock",
        str(uv_lock),
        *("--package", "pytest", "--out", str(tmp_path / "runtime.json")),
    )

    assert done.returncode == 0, done.stdout + done.stderr
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["packages"] == {"pytest": installed}
    assert runtime["python"] == platform.python_version()
    assert runtime["tested_python"] == "3.14"
    assert runtime["passed"] is True
    assert done.stdout.strip() == (
        f"Python {platform.python_version()} and 1 package match {uv_lock}."
    )


def test_check_runtime_fails_on_a_package_that_differs(tmp_path):
    # A package no environment has: every environment installs DSPy, so dspy cannot be the missing one.
    uv_lock = lock(tmp_path, ">=3.14", pytest="1.0", absent_package="3.3.1")

    done = run_cli(
        tmp_path,
        "cluster",
        "check-runtime",
        "--lock",
        str(uv_lock),
        *("--package", "pytest", "--package", "absent_package"),
        *("--out", str(tmp_path / "runtime.json")),
    )

    assert done.returncode == 1
    installed = importlib.metadata.version("pytest")
    problems = [
        f"pytest {installed} is installed; {uv_lock} has 1.0",
        f"absent_package is not installed; {uv_lock} has 3.3.1",
    ]
    assert json.loads((tmp_path / "runtime.json").read_text())["problems"] == problems
    assert done.stdout.splitlines() == [f"FAILED: {p}" for p in problems]


def test_check_runtime_reads_the_released_runs_packages_from_the_lock(tmp_path):
    # The versions come from ./uv.lock, never from the code, so a toolchain refresh moves them.
    names = ["dspy", "gepa", "litellm", "optuna", "torch", "transformers", "vllm"]
    lock(tmp_path, "==3.13.*", **{name: "99.0" for name in names}, pytest="1.0")

    done = run_cli(tmp_path, "cluster", "check-runtime", "--out", "runtime.json")

    assert done.returncode == 1
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["wanted"] == {name: "99.0" for name in names}
