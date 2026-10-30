"""Running a step on a cluster: the launcher render, the model download, the runtime and vLLM checks.

Every step runs on one machine; a cluster is optional. For a step that wants larger GPUs, one template (`cluster/launcher.sbatch`) and one plain file per site (`cluster/<site>.toml`; `cluster/example.toml` is a site-neutral example to copy) make a Slurm job that runs the same command (docs/setup/cluster.md). `render` fills the template in and prints it; it never submits anything. The other commands run inside the rendered job, and `download` also on a login node.
A site file holds what the cluster is (partitions, GPUs, QOS); who uses it and where (the account and the one folder under which the job lives) come from `.env`: `CLUSTER_<SITE>_ROOT` and, for a site that bills an account, `CLUSTER_<SITE>_ACCOUNT` (`.env.example`).
"""

import hashlib
import importlib.metadata
import json
import platform
import re
import shlex
import sys
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import mbdef
from mbdef import files
from mbdef.models import PINNED

# Read from the folder the command runs from, normally the repository root.
CLUSTER = Path("cluster")
TEMPLATE = CLUSTER / "launcher.sbatch"
# The only QOS render writes (the rule for every site: never a QOS that pre-empts other users' jobs, as some clusters' default QOS does). Another QOS needs the agreement of whoever answers for the project's use of the cluster, and then a person edits the rendered file by hand.
NORMAL_QOS = "normal"


# How the released corpus runs served the extractor models with vLLM: these flags for both models, then per model its tensor parallelism and, for thinking runs, its reasoning parser. Every GPU of the job serves: data parallelism is GPUs / tensor parallelism.
MAX_MODEL_LEN = 65536
SERVE_FLAGS = (
    *("--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", "1024"),
    *("--max-num-batched-tokens", "16384", "--gpu-memory-utilization", "0.82"),
    *("--enable-prefix-caching", "--generation-config", "auto"),
    *("--enable-prompt-tokens-details", "--language-model-only"),
)
# model: (tensor parallelism, reasoning parser)
SERVE = {"gemma": (1, "gemma4"), "qwen": (2, "qwen3")}


def model_ref(name: str) -> tuple[str, str]:
    """(repo, revision) of a pinned model's short name, or of REPO@REVISION."""
    if name in PINNED:
        return PINNED[name]
    repo, _, revision = name.partition("@")
    if not repo or not revision:
        raise SystemExit(
            f"Unknown model {name!r}: give one of {', '.join(PINNED)} or REPO@REVISION."
        )
    return repo, revision


def cache_folder(repo: str) -> str:
    """The folder that holds REPO in a Hugging Face hub cache."""
    return "models--" + repo.replace("/", "--")


def cached_snapshot(repo: str, revision: str, needed: tuple[str, ...]) -> Path:
    """The folder of REPO at commit REVISION in the local Hugging Face cache, if it holds `needed`.

    Only these files are looked for, so a cache with just a model's tokenizer files will do (the Hub client's own offline lookup refuses a snapshot that lacks any file of the repository).
    """
    from huggingface_hub import constants  # ty: ignore[unresolved-import]

    folder = Path(constants.HF_HUB_CACHE) / cache_folder(repo) / "snapshots" / revision
    missing = [name for name in needed if not (folder / name).is_file()]
    if missing:
        raise SystemExit(
            f"{repo} at revision {revision} is not in the Hugging Face cache "
            f"({', '.join(missing)} missing under {folder}). Download it "
            f"(mbdef cluster download {repo}@{revision}, or only these files with "
            f"hf download {repo} {' '.join(missing)} --revision {revision}), or give its folder."
        )
    return folder


def render(
    site: str,
    step: str,
    gpus: int,
    time: str,
    job_name: str,
    partition: str | None,
    qos: str | None,
    models: list[str],
    serve: str | None,
    think: bool,
    out: str | None,
) -> int:
    """Fill the launcher template in for SITE and print it (or write it to OUT). Submits nothing."""
    config = read_site(site)
    if (config.get("qos") or NORMAL_QOS) != NORMAL_QOS:
        raise SystemExit(
            f"{CLUSTER / site}.toml sets qos = {config['qos']!r}. A site's QOS must be the "
            f"normal one ({NORMAL_QOS}), the only one render writes."
        )
    root, account = site_keys(site, config)
    qos = qos or NORMAL_QOS
    if qos != NORMAL_QOS:
        raise SystemExit(
            f"--qos {qos}: render writes only the normal QOS (--qos={NORMAL_QOS}), never one "
            "that pre-empts other users' jobs or jumps ahead of them. Another QOS needs the "
            "agreement of whoever answers for the project's use of the cluster; with it, edit the "
            "rendered file by hand."
        )
    if "\n" in step:
        raise SystemExit("--step must be one line.")
    if gpus < 0:
        raise SystemExit("--gpus cannot be negative.")
    # The job pins the commit it is rendered at. A copy of the repository without git has no commit: its job runs the code in its repo folder as it is (the launcher skips the check).
    state = files.git_state(Path.cwd())
    refs = [model_ref(name) for name in models]
    serve_args, serve_check = vllm_arguments(serve, gpus, think) if serve else ([], [])
    if serve and PINNED[serve] not in refs:
        refs.append(PINNED[serve])
    if gpus:
        gpu = config["gpu"]
        names, min_gb = gpu["names"], gpu["min_memory_gb"]
        resources = {
            "partition": partition or gpu["partition"],
            "gres": gpu["gres"].format(gpus=gpus),
            "constraint": gpu.get("constraint"),
            "cpus-per-task": gpu["cpus_per_gpu"] * gpus,
            "mem": f"{gpu['memory_gb_per_gpu'] * gpus}G",
        }
    elif "cpu" in config:
        cpu, names, min_gb = config["cpu"], [], 0
        resources = {
            "partition": partition or cpu["partition"],
            "cpus-per-task": cpu["cpus"],
            "mem": f"{cpu['memory_gb']}G",
        }
    else:
        raise SystemExit(
            f"{CLUSTER / site}.toml has no [cpu] table, so this site takes no CPU-only job (--gpus 0)."
        )
    sbatch = {
        "job-name": job_name,
        "account": account,
        "partition": resources.pop("partition"),
        "qos": qos,
        "time": time,
        "nodes": 1,
        "ntasks": 1,
        **resources,
        "output": f"{root}/logs/slurm-%j.out",
    }
    values = {
        "sbatch": "\n".join(
            f"#SBATCH --{key}={value}"
            for key, value in sbatch.items()
            if value is not None
        ),
        "site": shlex.quote(site),
        "root": shlex.quote(root),
        "commit": state["commit"] or "",
        "gpus": str(gpus),
        "gpu_names": shlex.quote(",".join(names)),
        "min_gpu_gb": str(min_gb),
        "models": shlex.join(f"{repo}@{revision}" for repo, revision in refs),
        "model_dirs": shlex.join(cache_folder(repo) for repo, _ in refs),
        "stage_models": str(int(config.get("stage_models", False))),
        "serve": shlex.join(serve_args),
        "serve_check": shlex.join(serve_check),
        "step": shlex.quote(step),
        "env": "\n".join(
            f"export {key}={shlex.quote(value.format(root=root))}"
            for key, value in config.get("env", {}).items()
        )
        or "# (nothing for this site)",
    }
    script = fill(TEMPLATE.read_text(encoding="utf-8"), values)
    if out is None:
        sys.stdout.write(script)
        return 0
    target = Path(out)
    if target.exists():
        raise SystemExit(f"{out} exists; render never overwrites a file.")
    target.write_text(script, encoding="utf-8")
    where = (
        f"at commit {state['commit'][:12]}"
        if state["commit"]
        else "outside a git checkout: the job runs the code in its repo folder as it is, "
        "without checking a commit"
    )
    print(f"Wrote {out} for {site} {where}. Nothing was submitted.")
    return 0


def vllm_arguments(serve: str, gpus: int, think: bool) -> tuple[list[str], list[str]]:
    """The `vllm serve` arguments for a pinned model on GPUS, and what `vllm-check` expects."""
    tensor_parallel, parser = SERVE[serve]
    if gpus < 1 or gpus % tensor_parallel:
        raise SystemExit(
            f"{serve} is served with tensor parallelism {tensor_parallel}, "
            f"so --gpus must be a multiple of {tensor_parallel}."
        )
    repo, revision = PINNED[serve]
    serve_args = [repo, "--revision", revision, *SERVE_FLAGS]
    serve_args += ["--tensor-parallel-size", str(tensor_parallel)]
    serve_args += ["--data-parallel-size", str(gpus // tensor_parallel)]
    check = ["--model", repo, "--max-model-len", str(MAX_MODEL_LEN)]
    if think:
        serve_args += ["--reasoning-parser", parser]
        check.append("--think")
    return serve_args, check


def site_keys(site: str, config: dict) -> tuple[str, str | None]:
    """The site's root folder and account, from `.env` (`CLUSTER_<SITE>_ROOT`, and `CLUSTER_<SITE>_ACCOUNT` where the site file says `needs_account = true`). An unset key stops with a pointer to `.env.example`; only the chosen site's keys are asked for."""
    prefix = "CLUSTER_" + site.upper().replace("-", "_")
    root = files.env(f"{prefix}_ROOT")
    account = files.env(f"{prefix}_ACCOUNT") if config.get("needs_account") else None
    return root, account


def read_site(site: str) -> dict:
    path = CLUSTER / f"{site}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in CLUSTER.glob("*.toml"))
        raise SystemExit(f"No site config {path}. The sites are: {', '.join(known)}.")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def fill(template: str, values: dict[str, str]) -> str:
    """Replace every {{name}} in TEMPLATE; stop if one has no value or a value is not used."""
    names = set(re.findall(r"\{\{(\w+)\}\}", template))
    if names != set(values):
        raise SystemExit(
            f"{TEMPLATE} and render disagree: placeholders {sorted(names ^ set(values))}."
        )
    return re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], template)


def download(models: list[str], cache: str | None, offline: bool, out: str) -> int:
    """Put each model (default: all of PINNED) at its pinned revision into the Hugging Face cache.

    With `offline`, only find it there (inside a job, where nodes have no internet). Then check every file of each snapshot against the Hub object id that names its blob: the sha256 of a large file, the git blob sha1 of a small one. Writes a receipt with every file's size and sha256 to OUT, so copies on other machines can be compared. Returns 1 if a model is missing or differs.
    """
    # Only this command needs the Hub client, so it is imported here.
    import huggingface_hub  # ty: ignore[unresolved-import]

    refs = [model_ref(name) for name in models] or list(PINNED.values())
    checked, bad = [], 0
    for repo, revision in refs:
        try:
            snapshot = Path(
                huggingface_hub.snapshot_download(
                    repo, revision=revision, cache_dir=cache, local_files_only=offline
                )
            )
        except (
            Exception
        ) as error:  # the Hub's errors differ by version; report any of them
            problems = [f"not found in the cache: {error}"]
            checked.append(
                {
                    "repo": repo,
                    "revision": revision,
                    "verified": False,
                    "problems": problems,
                }
            )
            bad += 1
            print(f"{repo}@{revision[:8]}: FAILED, {problems[0]}")
            continue
        file_set, problems = check_snapshot(snapshot)
        size = sum(f["bytes"] for f in file_set)
        checked.append(
            {
                "repo": repo,
                "revision": revision,
                "snapshot": str(snapshot),
                "files": file_set,
                "bytes": size,
                "verified": not problems,
                "problems": problems,
            }
        )
        if problems:
            bad += 1
            print(f"{repo}@{revision[:8]}: FAILED, " + "; ".join(problems))
        else:
            print(
                f"{repo}@{revision[:8]}: {len(file_set)} files, {size / 1e9:.2f} GB, "
                "each matches its Hub object id"
            )
    settings = {
        "models": [f"{repo}@{revision}" for repo, revision in refs],
        "cache": cache,
        "offline": offline,
        "huggingface_hub": huggingface_hub.__version__,
    }
    receipt = files.receipt("cluster download", settings, {}, {})
    files.write_json(Path(out), {**receipt, "models": checked})
    if bad:
        print(f"{bad} of {len(refs)} models are missing or differ; see {out}.")
        return 1
    print(
        f"All {len(refs)} models are in the cache and every file matches its Hub object id."
    )
    return 0


def check_snapshot(snapshot: Path) -> tuple[list[dict], list[str]]:
    """Each file of a cached snapshot with its size and sha256, and what is wrong with any of them."""
    folder = snapshot.parents[
        1
    ].resolve()  # models--ORG--NAME, which holds blobs/ and snapshots/
    links = sorted(p for p in snapshot.rglob("*") if not p.is_dir())
    problems = [] if (snapshot / "config.json").is_file() else ["config.json: missing"]
    with ThreadPoolExecutor(max_workers=8) as pool:  # large files hash in parallel
        hashed = list(pool.map(lambda link: object_ids(link, folder), links))
    file_set = []
    for link, ids in zip(links, hashed, strict=True):
        name = link.relative_to(snapshot).as_posix()
        if ids is None:
            problems.append(f"{name}: missing, or outside {folder.name}")
            continue
        sha256, git_sha1, size = ids
        blob = link.resolve().name
        if blob not in (sha256, git_sha1):
            problems.append(f"{name}: its content does not match its Hub object id")
        file_set.append({"path": name, "blob": blob, "bytes": size, "sha256": sha256})
    return file_set, problems


def object_ids(link: Path, folder: Path) -> tuple[str, str, int] | None:
    """sha256, git blob sha1 and size of the blob a snapshot entry points to.

    None if the blob is absent or lies outside the model's cache FOLDER.
    """
    target = link.resolve()
    if not target.is_file() or folder not in target.parents:
        return None
    size = target.stat().st_size
    sha256, git_sha1 = hashlib.sha256(), hashlib.sha1(b"blob %d\0" % size)
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            sha256.update(block)
            git_sha1.update(block)
    return sha256.hexdigest(), git_sha1.hexdigest(), size


LOCK = Path("uv.lock")
# The packages whose versions the released runs depend on most. Their versions are read from uv.lock, so a toolchain refresh changes them in one place.
RUNTIME_PACKAGES = (
    "dspy",
    "gepa",
    "optuna",
    "litellm",
    "transformers",
    "vllm",
    "torch",
)


def check_runtime(lock: str | None, packages: list[str], out: str) -> int:
    """Compare the installed PACKAGES with uv.lock, write what was found to OUT.

    Returns 1 if a package is missing or differs. A Python other than the tested one is named, with a warning, but passes.
    """
    lock_path = Path(lock) if lock else LOCK
    locked = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    versions = {p["name"]: p.get("version") for p in locked.get("package", [])}
    names = packages or list(RUNTIME_PACKAGES)
    wanted = {name: versions.get(name) for name in names}
    python = platform.python_version()
    warning = mbdef.untested_python(sys.version_info[:3])
    problems = []
    installed = {}
    for name in names:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name} is not installed; {lock_path} has {wanted[name]}")
            continue
        if installed[name] != wanted[name]:
            problems.append(
                f"{name} {installed[name]} is installed; {lock_path} has {wanted[name]}"
            )
    report = {
        "lock": str(lock_path),
        "python": python,
        "tested_python": ".".join(map(str, mbdef.TESTED_PYTHON)),
        "packages": installed,
        "wanted": wanted,
        "passed": not problems,
        "problems": problems,
    }
    files.write_json(Path(out), report)
    if warning:
        print(warning)
    for problem in problems:
        print(f"FAILED: {problem}")
    if problems:
        return 1
    count = f"{len(names)} package" + ("s" if len(names) != 1 else "")
    print(f"Python {python} and {count} match {lock_path}.")
    return 0


def vllm_check(
    base: str,
    model: str,
    max_model_len: int,
    think: bool,
    log: str | None,
    out: str,
) -> int:
    """Check a running vLLM server the way the released corpus runs did, and write what it found.

    /v1/models must list MODEL alone, with MAX_MODEL_LEN; /tokenize must count the tokens of a one-message chat; /metrics must list at least one engine. The KV-cache lines of the server's LOG are recorded. Writes OUT and returns 1 at the first check that fails.
    """
    base = base.rstrip("/")
    health: dict = {"base": base, "model": model, "max_model_len": max_model_len}
    problems, count, engines = [], 0, []
    try:
        listed = json.loads(http_get(f"{base}/v1/models"))["data"]
        ids = [entry["id"] for entry in listed]
        if ids != [model]:
            problems.append(f"/v1/models lists {ids}, not {model} alone")
        elif listed[0].get("max_model_len") != max_model_len:
            problems.append(
                f"/v1/models says max_model_len {listed[0].get('max_model_len')}, "
                f"not {max_model_len}"
            )
    except (OSError, ValueError, KeyError) as error:
        problems.append(f"/v1/models did not answer as vLLM does: {error}")
    if not problems:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "health"}],
            "chat_template_kwargs": {"enable_thinking": think},
        }
        try:
            count = json.loads(http_post(f"{base}/tokenize", body)).get("count", 0)
            health["tokenize_count"] = count
            if not count > 0:
                problems.append("/tokenize counted no token")
        except (OSError, ValueError) as error:
            problems.append(f"/tokenize did not answer: {error}")
    if not problems:
        try:
            metrics = http_get(f"{base}/metrics").decode()
            engines = sorted(set(re.findall(r'engine="(\d+)"', metrics)), key=int)
            health["engines_in_metrics"] = engines
            if not engines:
                problems.append("/metrics lists no engine")
        except (OSError, ValueError) as error:
            problems.append(f"/metrics did not answer: {error}")
    if log:
        lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
        health["kv_cache_lines"] = [
            line.rsplit("] ", 1)[-1].strip()
            for line in lines
            if "KV cache size" in line or "Maximum concurrency" in line
        ][:8]
    health.update(passed=not problems, problems=problems)
    health["utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    files.write_json(Path(out), health)
    if problems:
        print(f"FAILED: {base}: {problems[0]}")
        return 1
    print(
        f"{base} serves {model} with max_model_len {max_model_len}, tokenizes a request "
        f"({count} tokens) and reports {len(engines)} engine{'s' * (len(engines) != 1)}."
    )
    return 0


def http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def http_post(url: str, body: dict) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()
