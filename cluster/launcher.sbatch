#!/usr/bin/env bash
# One step of the MBDef pipeline as one Slurm job (docs/setup/cluster.md).
# Rendered by `mbdef cluster render` from cluster/launcher.sbatch and cluster/{{site}}.toml.
# Rendering submits nothing: submit this file on the login node with `sbatch FILE`.
{{sbatch}}
set -euo pipefail

# ---- Filled in by render ---------------------------------------------------------------------
# The one folder on this cluster that holds everything (docs/setup/cluster.md), and the commit of $root/repo that the job runs: it refuses any other. Empty when render ran outside a git checkout (a copy of the repository without git): then the job runs $root/repo as it is.
site={{site}}
root={{root}}
commit={{commit}}
# GPUs on the node, and the GPU names and memory per GPU (GB) that the GPU check accepts.
gpus={{gpus}}
gpu_names={{gpu_names}}
min_gpu_gb={{min_gpu_gb}}
# repo@revision of every model the step reads, their folders in the Hugging Face cache, and whether to copy them to node-local disk before the step (1) or read them where they are (0).
models=({{models}})
model_dirs=({{model_dirs}})
stage_models={{stage_models}}
# The `vllm serve` arguments, and what `cluster vllm-check` expects of that server (empty: no server).
serve=({{serve}})
serve_check=({{serve_check}})
# The step: one command line, run in $root/repo.
step={{step}}

# ---- The rest is the same for every site and step --------------------------------------------
repo=$root/repo
venv=$root/venv
hub=$root/hf-cache/hub
logs=$root/logs/job-${SLURM_JOB_ID:?}
python=$venv/bin/python
mkdir -p "$logs"
fatal() { echo "FATAL $*" | tee -a "$logs/FATAL.txt" >&2; exit 1; }
start_utc=$(date -u +%FT%TZ)
start_s=$(date +%s)
server=""

# Environment: models from the cache only, no telemetry, caches in a short node-local TMPDIR, and no paid API key (empty values, so a .env in the checkout cannot set them either).
export HF_HOME=$root/hf-cache HF_HUB_CACHE=$hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 WANDB_MODE=disabled PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_ALLREDUCE_USE_FLASHINFER=0
TMPDIR="$(mktemp -d "/tmp/j${SLURM_JOB_ID}.XXXXXX")"
(( ${#TMPDIR} <= 40 )) || fatal "TMPDIR is too long for vLLM's IPC sockets: $TMPDIR"
export TMPDIR VLLM_RPC_BASE_PATH="$TMPDIR" XDG_CACHE_HOME="$TMPDIR/x" TRITON_CACHE_DIR="$TMPDIR/tc"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/ti" CUDA_CACHE_PATH="$TMPDIR/cu" VLLM_CACHE_ROOT="$TMPDIR/v"
export UV_CACHE_DIR="$TMPDIR/uv" FLASHINFER_WORKSPACE_BASE="$TMPDIR/fi" DSPY_CACHEDIR="$TMPDIR/dspy"
unset PYTHONPATH LIBRARY_PATH LD_LIBRARY_PATH VLLM_PORT
export OPENROUTER_API_KEY= ANTHROPIC_API_KEY= OPENAI_API_KEY= DSPY_API_KEY=EMPTY
export DATA_ROOT=$root/data-root SCIDEF_NLI_DEVICE=cpu
if (( gpus > 0 )); then export SCIDEF_NLI_DEVICE=cuda; fi
# This site's own settings ([env] in cluster/{{site}}.toml):
{{env}}
cuda=("$venv"/lib/python3.*/site-packages/nvidia/cu13)   # the CUDA toolkit of the locked wheels
export CUDA_HOME=${cuda[0]}
export PATH="$CUDA_HOME/bin:$venv/bin:$PATH"
# vLLM's FlashInfer builds some kernels on first use and links them with -lcudart, which it looks for in $CUDA_HOME/lib64; the wheels hold the runtime only as lib/libcudart.so.<major>. Give the linker that name (a node without a system CUDA toolkit has no other, and the link fails without it).
cudart=("$CUDA_HOME"/lib/libcudart.so.*)
mkdir -p "$TMPDIR/cuda-link"
ln -sf "${cudart[0]}" "$TMPDIR/cuda-link/libcudart.so"
export LIBRARY_PATH="$TMPDIR/cuda-link"
# CUDA's runtime compiler (NVRTC), which some of vLLM's kernels call while loading the weights (the 4-bit Gemma's on H200), opens libnvrtc-builtins.so.<version> by name, and LD_LIBRARY_PATH is unset above: give the loader the wheels' copy too (without it, the load stops on a node without a system CUDA).
for lib in "$CUDA_HOME"/lib/libnvrtc-builtins.so.*; do
  if [ -e "$lib" ]; then ln -sf "$lib" "$TMPDIR/cuda-link/"; fi
done
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$TMPDIR/cuda-link"
cleanup() {
  if [ -n "$server" ]; then kill "$server" 2>/dev/null || true; fi
  rm -rf "$TMPDIR" 2>/dev/null || true
}
trap cleanup EXIT
cd "$repo"
printf '%s\t%s\t%s\n' "$start_utc" "$SLURM_JOB_ID" "$(hostname)" >> "$root/logs/jobs.tsv"

# The code: exactly the rendered commit, with no uncommitted change.
if [ -n "$commit" ]; then
  [ "$(git rev-parse HEAD)" = "$commit" ] || fatal "$repo is at $(git rev-parse HEAD), not $commit: check out $commit"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || fatal "$repo has uncommitted changes"
fi

# The GPUs: the expected count, name and free memory, and one small product on each.
if (( gpus > 0 )); then
  timeout --kill-after=5s 45s nvidia-smi -q -x > "$logs/gpu-health.xml" || fatal "nvidia-smi failed"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$logs/gpu.csv" || true
  timeout --kill-after=5s 180s "$python" - "$logs" "$gpus" "$min_gpu_gb" "$gpu_names" <<'PY' || fatal "GPU check failed"
import json, sys, xml.etree.ElementTree as ET
from pathlib import Path
import torch
logs, n, gb, names = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4].split(",")
actions = [str(e.text).strip() for e in ET.parse(logs / "gpu-health.xml").iter() if e.tag == "gpu_recovery_action"]
assert all(a in ("None", "N/A", "0", "Not Supported") for a in actions), actions
assert torch.cuda.device_count() == n, torch.cuda.device_count()
devices = []
for index in range(n):
    with torch.cuda.device(index):
        x = torch.ones((64, 64), device=f"cuda:{index}")
        assert float((x @ x)[0, 0]) == 64
        torch.cuda.synchronize(index)
        props = torch.cuda.get_device_properties(index)
        assert any(name in props.name for name in names), props.name
        assert props.total_memory >= gb * 1024**3, (props.name, props.total_memory)
        free, total = torch.cuda.mem_get_info(index)
        assert free >= 0.95 * total, (index, free, total)
        devices.append({"index": index, "name": props.name, "free_bytes": free})
report = {"status": "passed", "devices": devices, "recovery_actions": actions}
(logs / "gpu-health-canary.json").write_text(json.dumps(report, indent=2) + "\n")
PY
fi

# The runtime: every pinned package at its uv.lock version.
timeout --kill-after=10s 300s "$python" -m mbdef cluster check-runtime --out "$logs/runtime.json" \
  || fatal "the installed packages differ from uv.lock (see $logs/runtime.json)"

# The models: copied to node-local disk where the shared disk is slow, then every file checked against the Hub object id that names it.
if (( ${#models[@]} )); then
  if (( stage_models )); then
    for dir in "${model_dirs[@]}"; do
      mkdir -p "$TMPDIR/hub/$dir/blobs"
      find "$hub/$dir/blobs" -type f -print0 | xargs -0 -P 8 -I{} cp {} "$TMPDIR/hub/$dir/blobs/" \
        || fatal "copying $dir to node-local disk failed"
      cp -a "$hub/$dir/snapshots" "$TMPDIR/hub/$dir/"
      if [ -d "$hub/$dir/refs" ]; then cp -a "$hub/$dir/refs" "$TMPDIR/hub/$dir/"; fi
      (cd "$hub/$dir/blobs" && find . -type f -printf '%s %P\n' | sort) > "$logs/$dir-shared.txt"
      (cd "$TMPDIR/hub/$dir/blobs" && find . -type f -printf '%s %P\n' | sort) > "$logs/$dir-local.txt"
      cmp -s "$logs/$dir-shared.txt" "$logs/$dir-local.txt" || fatal "the node-local copy of $dir differs"
    done
    export HF_HUB_CACHE=$TMPDIR/hub
  fi
  model_args=()
  for model in "${models[@]}"; do model_args+=(--model "$model"); done
  timeout --kill-after=10s 1800s "$python" -m mbdef cluster download --offline \
    --cache "$HF_HUB_CACHE" "${model_args[@]}" --out "$logs/models.json" \
    || fatal "a model is missing or differs from the Hub (see $logs/models.json)"
fi

# The vLLM server, when the step needs one: started, waited for, checked, then given to the step.
if (( ${#serve[@]} )); then
  test -x "$CUDA_HOME/bin/nvcc" || fatal "nvcc is missing from $CUDA_HOME (vLLM compiles kernels with it)"
  port=$((20000 + SLURM_JOB_ID % 20000))
  vllm serve "${serve[@]}" --host 127.0.0.1 --port "$port" --data-parallel-rpc-port $((port + 1)) \
    > "$logs/vllm.log" 2>&1 &
  server=$!
  for _ in $(seq 480); do
    curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null && break
    kill -0 "$server" 2>/dev/null || fatal "vLLM exited while starting (see $logs/vllm.log)"
    sleep 5
  done
  "$python" -m mbdef cluster vllm-check --base "http://127.0.0.1:$port" "${serve_check[@]}" \
    --log "$logs/vllm.log" --out "$logs/vllm-health.json" || fatal "the vLLM server failed its check"
  export DSPY_API_BASE="http://127.0.0.1:$port/v1"
fi

# The step.
set +e
bash -c "$step" > "$logs/step.log" 2>&1
rc=$?
set -e
"$python" - "$logs/terminal.json" "$SLURM_JOB_ID" "$site" "$commit" "$step" "$rc" "$start_utc" "$start_s" <<'PY'
import json, sys, time
out, job, site, commit, step, rc, start_utc, start_s = sys.argv[1:]
end = time.time()
record = {"job": job, "site": site, "commit": commit, "step": step, "exit_code": int(rc),
          "start_utc": start_utc, "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end)),
          "seconds": round(end - int(start_s))}
open(out, "w").write(json.dumps(record, indent=2) + "\n")
PY
echo "step exited with $rc after $(( $(date +%s) - start_s )) s; logs in $logs"
exit "$rc"
