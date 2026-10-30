# Running a step on a cluster: `mbdef cluster`

| | |
|---|---|
| **Paper step** | None of its own: this runs any step's command as a job on a Slurm GPU cluster, if you have one |
| **Code** | the template [`cluster/launcher.sbatch`](../../cluster/launcher.sbatch), one file per cluster (a "site file"; [`cluster/example.toml`](../../cluster/example.toml) is the example to copy) and [`src/mbdef/cluster.py`](../../src/mbdef/cluster.py) |
| **Commands** | `cluster render` (write a job script), `cluster download` (models), `cluster check-runtime`, `cluster vllm-check` |
| **Checked by** | [`tests/test_cluster.py`](../../tests/test_cluster.py) and [`tests/test_cluster_job.py`](../../tests/test_cluster_job.py), in the fast test set, without a cluster; the selection rescoring ran as such a job |

Nothing requires a cluster: every step runs on one machine with `uv run mbdef …`. A few steps want more or larger GPUs than a workstation has, for example the selection rescoring or an extraction run that serves a model with vLLM. If you have a Slurm cluster, `cluster render` writes a job script that runs the same command there.

A job does the same few things around its one command: set up the environment, check the GPUs, check the installed packages, check the model files, maybe start a vLLM server, run the command, write a record of how it ended. There is **one template** for those things and **one plain file per cluster** for what differs between clusters (partition, GPU type, QOS). Who you are on a cluster and where your files live (the account and the root folder) are in your `.env`, not in the repository. `cluster render` fills the template in for one site and one command and prints the job script. **It never submits anything**: you read the script and submit it yourself on the login node.

## The rule for submitting

- **Every job runs at the normal QOS.** The QOS (quality of service) is the Slurm setting that decides a job's priority and limits. On some clusters the default QOS, if a job names none, *pre-empts* other users: it stops their running jobs and puts them back in the queue. So every rendered script names `--qos=normal` (a site file says `qos = "normal"`, or nothing, and then render writes `normal`), and render refuses any other QOS, both as `--qos` (for example `--qos high`) and in a site file. Another QOS needs the agreement of whoever answers for the project's use of the cluster; with it, a person edits the rendered file by hand.
- Ask for the walltime the step needs (`--time` is required) and the GPUs it can keep busy.

## Inputs: the site file and your `.env`

A **site file**, `cluster/<name>.toml`, says what a cluster is. [`cluster/example.toml`](../../cluster/example.toml) is one to copy; each line says what it is for:

| Key | What it says | In `example.toml` |
|---|---|---|
| `needs_account` | jobs are billed to a project account: render asks for `CLUSTER_<NAME>_ACCOUNT` and writes `--account` | `true` |
| `qos` | `"normal"`, or no line (render then writes `normal`); any other value is refused | `"normal"` |
| `stage_models` | copy the models to node-local disk before the step, for a shared disk that is slow to read | `false` |
| `[gpu]` `partition`, `gres` | the GPU partition and what `--gres` asks for (`{gpus}` becomes `--gpus`); `constraint` is optional | `gpu`, `gpu:{gpus}` |
| `[gpu]` `names`, `min_memory_gb` | the GPUs the job's GPU check accepts: a name that is part of the device's name, and at least this much memory | `A100`, `H100`, `H200`; 79 GB |
| `[gpu]` `cpus_per_gpu`, `memory_gb_per_gpu` | each GPU's share of the node | 16 CPUs, 120 GB |
| `[cpu]` | the partition, CPUs and memory of a CPU-only job (`--gpus 0`); without the table the site takes none | `cpu`, 48 CPUs, 120 GB |
| `[env]` | extra environment for every job, `{root}` being the site's root folder | nothing |

A site file names no account and no home folder, because those belong to a person. Each person sets them in `.env` ([`.env.example`](../../.env.example) lists the keys):

| Key | What it is |
|---|---|
| `CLUSTER_<NAME>_ROOT` | The one folder that holds everything of yours on that cluster (below), as an absolute path on the cluster |
| `CLUSTER_<NAME>_ACCOUNT` | The project account Slurm bills the job to, for a site file that says `needs_account = true` |

`<NAME>` is the site file's name in capitals, with `-` written `_`: `CLUSTER_EXAMPLE_ROOT` for `example.toml`.
`cluster render --site example` asks for the two example keys only. If one is unset it stops before it writes anything:

```text
CLUSTER_EXAMPLE_ACCOUNT is not set. Copy .env.example to .env and set CLUSTER_EXAMPLE_ACCOUNT there; .env.example says what it is.
```

## One folder per cluster

Everything on a cluster lives under one folder, the site's root (`CLUSTER_<NAME>_ROOT`). Bigger things can live elsewhere and be linked from there, so nothing is spread around.

| Path under `root` | What it is |
|---|---|
| `repo/` | a checkout of this repository; the job runs the exact commit it was rendered at (rendered outside a git checkout, for example from a copy of the repository downloaded without git, it runs the folder as it is) |
| `venv/` | the Python environment, made by `uv sync` from `repo/` |
| `hf-cache/hub/` | the Hugging Face model cache (`models--ORG--NAME/` folders) |
| `data-root/` | the data root ([docs/setup/data-root.md](data-root.md)); the job sets `DATA_ROOT` to it |
| `logs/` | `slurm-<job>.out`, `jobs.tsv` (one line per job start) and `job-<job>/` with every check's output |

### Setting a cluster up once

On the login node (compute nodes often have no internet):

```bash
cd $ROOT
git clone <repository-url> repo
cd repo && git checkout <commit>
UV_PROJECT_ENVIRONMENT=$ROOT/venv uv sync           # the locked packages, into $ROOT/venv
hf auth login                                       # once: Gemma is gated on the Hub
$ROOT/venv/bin/mbdef cluster download --cache $ROOT/hf-cache/hub \
    --out $ROOT/logs/models-$(date +%F).json         # the four pinned models (below)
ln -s <the data root on this cluster> $ROOT/data-root
```

A cluster whose nodes run an older NVIDIA driver than the locked CUDA 13 packages need (for example driver 570, which is CUDA 12.8) also needs NVIDIA's CUDA 13.1 forward-compatibility libraries, unpacked under the root and put on `LD_LIBRARY_PATH` by the site file's `[env]` table (`example.toml` shows the line).

## Render a job

The selection rescoring ([the cluster job](../step-3-extraction/selection.md#the-cluster-job)) needs one GPU for about half an hour (28 min 30 s on one H100) and reads the entailment and embedding models. On your own machine, with `CLUSTER_EXAMPLE_ROOT` and `CLUSTER_EXAMPLE_ACCOUNT` set in `.env` (or your own site and its keys), at the commit you want to run (pushed, so the cluster can check it out), with the job's command in `$STEP` (`selection.JOB_STEP`):

```bash
uv run mbdef cluster render --site example --gpus 1 --time 02:00:00 \
    --model nli --model bge --job-name selection-rescore --step "$STEP" --out selection.sbatch
```

It prints `Wrote selection.sbatch for example at commit <the first 12 characters of the commit>. Nothing was submitted.`
Rendered outside a git checkout (a copy of the repository without git), it prints `Wrote selection.sbatch for example outside a git checkout: the job runs the code in its repo folder as it is, without checking a commit. Nothing was submitted.`, the script's `commit=` is empty, and the job skips its commit check. The script starts with what render filled in (here with `CLUSTER_EXAMPLE_ROOT=/projects/example/taxonomy` and `CLUSTER_EXAMPLE_ACCOUNT=example-project`):

```bash
#SBATCH --job-name=selection-rescore
#SBATCH --account=example-project
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/projects/example/taxonomy/logs/slurm-%j.out
...
site=example
root=/projects/example/taxonomy
commit=b8efdd84bd721bfac21ebfd235aef1c05809409e
gpus=1
gpu_names=A100,H100,H200
min_gpu_gb=79
models=(tasksource/ModernBERT-large-nli@ca476cb923a8637073d4ceb0f19f7fc236e260d4 BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a)
model_dirs=(models--tasksource--ModernBERT-large-nli models--BAAI--bge-base-en-v1.5)
stage_models=0
serve=()
serve_check=()
step='O=$DATA_ROOT/selection/rescore/2026-10-20; S=$TMPDIR/scorings; mkdir -p …'
```

The 16 CPUs and 120 GB are the example's share per GPU (`cpus_per_gpu`, `memory_gb_per_gpu`), so `--gpus 4` asks for 64 CPUs and 480 GB.

Copy it to the cluster (or render it on the login node, in `$ROOT/repo`), check out that commit in `$ROOT/repo`, and submit it there: `sbatch selection.sbatch`. `squeue --start -j <job>` shows when Slurm expects it to start.

### What the job does, in order

Each check stops the job with a line `FATAL <reason>` (also in `job-<id>/FATAL.txt`) before the step runs.

| # | What | Its output in `logs/job-<id>/` |
|---|---|---|
| 1 | Sets the environment: models from the cache only (`HF_HUB_OFFLINE=1`), no telemetry, `PYTHONHASHSEED=0`, caches in a short node-local `TMPDIR` (at most 40 characters, for vLLM's sockets), `DATA_ROOT`, the NLI device, the CUDA toolkit of the locked wheels (`CUDA_HOME`) with a `libcudart.so` on `LIBRARY_PATH` (vLLM's FlashInfer links the kernels it builds on first use with it, and the wheels name it only `libcudart.so.13`: on a node without a system CUDA toolkit a job stopped with `ld: cannot find -lcudart`) and the wheels' NVRTC builtins on `LD_LIBRARY_PATH` (CUDA's runtime compiler opens `libnvrtc-builtins.so.13.0` by name while the 4-bit Gemma loads on H200), the site's `[env]`, and **empty** paid API keys, so a `.env` in the checkout cannot set them | |
| 2 | Refuses to run if `repo/` is not at the rendered commit, or has uncommitted changes in tracked files | |
| 3 | With GPUs (`--gpus` above 0), the GPU check: the expected number of GPUs, each with an accepted name, enough memory, 95% of it free, no pending recovery action, and one small matrix product | `gpu-health.xml`, `gpu.csv`, `gpu-health-canary.json` |
| 4 | `cluster check-runtime`: Python and the released runs' packages are the versions in `uv.lock` | `runtime.json` |
| 5 | On a site that stages models (`stage_models = true`), with models listed, copies each model to node-local disk (a shared disk that reads one file stream at 45 MB/s takes about 26 minutes for a 70 GB model), and checks that every file arrived with its size | `models--…-shared.txt`, `-local.txt` |
| 6 | With models listed, `cluster download --offline`: every file of every listed model matches its Hub object id | `models.json` |
| 7 | With `--serve`: starts `vllm serve` with the released settings, waits up to 40 minutes, runs `cluster vllm-check`, and sets `DSPY_API_BASE` for the step | `vllm.log`, `vllm-health.json` |
| 8 | Runs the step in `repo/` | `step.log` |
| 9 | Writes how it ended, and stops the server and removes `TMPDIR` | `terminal.json` |

`terminal.json` is the job's record: the job id, the site, the commit, the step, its exit code, the start and end times and the seconds in between. The selection rescoring's says `"exit_code": 0`, `"start_utc": "2026-10-20T08:26:33Z"`, `"end_utc": "2026-10-20T08:55:01Z"` and `"seconds": 1708`: 28 min 28 s from the script's first line to its last, of the 28 min 30 s that Slurm counted for the job. The step writes its own receipt into its output folder as on any other machine.

### Other kinds of job

- **A step that needs a language model**: `--serve gemma` or `--serve qwen`, with `--think` for the thinking runs. vLLM serves on every GPU of the job: Gemma on one GPU per copy, Qwen on two (tensor parallelism 2), so `--gpus 4 --serve qwen` serves two copies. The settings are the released corpus runs' (65,536 tokens of context, up to 1,024 requests at once, 82% of GPU memory, prefix caching; `SERVE_FLAGS` in `cluster.py`). The served model is checked and, on a site that stages models, copied like a `--model`.
- **A short test**: `--partition <name>` picks another partition of the site, such as a test partition that takes jobs up to an hour and often starts sooner.
- **A CPU-only job** such as GROBID: `--gpus 0` uses the site's `[cpu]` partition (`cpu`, 48 CPUs, 120 GB in `example.toml`). A site file without a `[cpu]` table refuses it.

## Outputs

`cluster render` prints the job script, or with `--out` writes it and prints one line, `Wrote <file> for <site> at commit <sha>. Nothing was submitted.` A job leaves, under `logs/` of its site's root, `slurm-<job>.out`, one line in `jobs.tsv` and the folder `job-<job>/` with the output of every check (the table above) and `terminal.json`. `cluster download`, `cluster check-runtime` and `cluster vllm-check` each write what they found as JSON to `--out` (required) and exit 0 or 1; the first prints a line per model, `check-runtime` one line, or one `FAILED:` line per difference, and `vllm-check` one line.

## The models: `cluster download`

The pinned models are the revisions the released runs used (`PINNED` in `models.py`):

| Name | Model | Revision | Used for |
|---|---|---|---|
| `gemma` | `google/gemma-4-12B-it-qat-w4a16-ct` | `1d2c2d7f…` | the extractor of the picks |
| `qwen` | `Qwen/Qwen3.6-35B-A3B` | `995ad96e…` | the other grid extractor and the verifier |
| `bge` | `BAAI/bge-base-en-v1.5` | `a5beb1e3…` | the embeddings of DefScore |
| `nli` | `tasksource/ModernBERT-large-nli` | `ca476cb9…` | the entailment model of the scoring |

`cluster download` downloads them (on a login node), or with `--offline` only finds them in the cache (in a job).
Then it checks every file: the Hub names each stored file by its object id, the sha256 of a large file or the git blob sha1 of a small one, so a file whose content no longer matches its name is caught. It writes a receipt with each file's size and sha256 and exits 1 if a model is missing or differs.

**A worked example.** Run offline on the cache of a workstation with one 24 GB GPU, it printed:

```text
google/gemma-4-12B-it-qat-w4a16-ct@1d2c2d7f: 10 files, 10.30 GB, each matches its Hub object id
BAAI/bge-base-en-v1.5@a5beb1e3: FAILED, not found in the cache: The cached snapshot for 'BAAI/bge-base-en-v1.5' (revision 'a5beb1e3…') is incomplete: 4 file(s) are missing (.gitattributes, README.md, onnx/model.onnx, ... (1 more)). …
tasksource/ModernBERT-large-nli@ca476cb9: 7 files, 1.59 GB, each matches its Hub object id
Qwen/Qwen3.6-35B-A3B@995ad96e: FAILED, not found in the cache: Cannot find an appropriate cached snapshot folder …
```

and a last line, `2 of 4 models are missing or differ; see <receipt>.` It took 9 s for the 12 GB. The 10 Gemma files have exactly the sizes and object ids of the Hub listing that the released test evaluation saved: 10 of 10 match. The workstation held only the files of BGE that the embedding library loads, so the snapshot counts as incomplete, and Qwen had never been downloaded there. On a cluster, running `cluster download` online on the login node completes both, and the job's check then passes.

## The runtime: `cluster check-runtime`

It compares the installed versions of dspy, gepa, optuna, litellm, transformers, vLLM and torch with `uv.lock` (`RUNTIME_PACKAGES` in `cluster.py`), so a toolchain refresh ([docs/setup/toolchain.md](toolchain.md)) changes what a job expects in one place. It records the running Python next to the tested one (3.14) in `runtime.json`; a newer Python passes with a warning line, and a package that is missing or has another version fails. In an environment made by `uv sync --locked` on Python 3.14 it prints `Python 3.14.8 and 7 packages match uv.lock.`

## The vLLM server: `cluster vllm-check`

It checks a running server the way the released corpus runs did before sending it any work: `/v1/models` lists the expected model alone with the expected context length, `/tokenize` counts the tokens of a one-message chat, and `/metrics` lists the server's engines. It records the KV-cache lines of the server's log (how many tokens fit in GPU memory) in `vllm-health.json`.

**A worked example.** The pinned Gemma, served on one 24 GB GPU by vLLM 0.28.0 with the template's settings, passed:

```text
http://127.0.0.1:28123 serves google/gemma-4-12B-it-qat-w4a16-ct with max_model_len 16384, tokenizes a request (17 tokens) and reports 1 engine.
```

with `"kv_cache_lines": ["GPU KV cache size: 22,272 tokens, Maximum concurrency for 16,384 tokens per request: 1.36x"]`.
Asked to expect twice the context, the same server failed:
`FAILED: http://127.0.0.1:28123: /v1/models says max_model_len 16384, not 32768`. It ran at 16,384 tokens because the released 65,536 do not fit next to the weights in 24 GB (7.1 GB were left for the KV cache); 80 GB GPUs hold them.

## How it was checked

The tests run in the fast test set without network, GPU or Slurm, and submit nothing:

- **Render** (`tests/test_cluster.py`): with the example site file and a made-up account and folder in the environment, the rendered `#SBATCH` lines are exactly the expected ones (account, partition, `--qos=normal`, GPUs, CPUs, memory, log path), an unset key stops the command with the pointer to `.env.example`, a site that bills no account asks for its root only, no committed site file holds an account or a home folder, the script passes `bash -n`, and a stand-in `sbatch` on `PATH` fails the test if anything calls it. Every committed site file has `qos = "normal"` or none; a site file with another QOS and every `--qos` other than `normal` are refused; `--serve`, `--model`, `--gpus 0`, `stage_models`, `[env]` and `--out` render what they should, on the example or on made-up site files in throwaway repositories.
- **Outside a git checkout** (`tests/test_unpacked_release.py`, on a copy of the package and the example site in a folder that no checkout holds): render writes a script with an empty `commit=` that passes `bash -n`, and a job rendered that way runs its repo folder without the commit check (`tests/test_cluster_job.py`).
- **The job script itself** (`tests/test_cluster_job.py`) runs under bash on a stand-in node (fake `nvidia-smi`, `curl`, `vllm` and `torch`): it checks the commit, copies the models to node-local disk, starts and checks the server, gives the step the environment above, writes `terminal.json`, removes `TMPDIR`, and refuses a checkout at another commit.
- **Download**, with a stand-in `huggingface_hub`: the four pinned models in order, each file's size and sha256 in the receipt, and a changed or missing file found offline.
- **vLLM check**, against a stand-in server: it passes the right model and fails a wrong model id or context length. **Runtime check**, with small lock files: versions are read from the lock, a different version or a missing package fails, and a newer Python passes with a warning.

## How to adapt it

- **Your cluster**: copy `cluster/example.toml` to `cluster/<name>.toml` and change its values to your cluster's (the table above; the cluster's documentation, `sinfo` and `scontrol show partition` give them); keep `qos = "normal"`, and keep `needs_account = true` only if jobs are billed to an account. Put `CLUSTER_<NAME>_ROOT` (and `CLUSTER_<NAME>_ACCOUNT`) into `.env`. `render --site <name>` then works.
- **Another model**: `--model ORG/NAME@REVISION` for a model the step reads. To serve another model, add it to `PINNED` in `models.py` and to `SERVE` (its tensor parallelism and reasoning parser) in `cluster.py`.
- **Other vLLM settings**: `SERVE_FLAGS` in `cluster.py`. Changing them changes the served runs, so record why.
- **Something every job should do**: edit `cluster/launcher.sbatch`; `tests/test_cluster_job.py` runs it.
- **Not covered**: two different servers in one job, and multi-node jobs. The template serves one model on one node.
