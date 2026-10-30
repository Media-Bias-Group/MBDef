# The toolchain: which versions run the pipeline, and how they were checked

| | |
|---|---|
| **Paper step** | All of them: the Python, packages, CUDA and tools every command runs on |
| **Files** | [`pyproject.toml`](../../pyproject.toml) (bounds), [`uv.lock`](../../uv.lock) (exact versions), [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) |
| **Checked by** | the test set on the locked environment, `cluster check-runtime`, and a live vLLM run through the language-model layer (below) |

## The versions

`uv.lock` holds the exact version of every package; `uv sync --locked` installs them. The table lists the ones that decide what the steps compute or how a model is served, the newest version on PyPI when the lock was made, and what holds the lock back where it is older.

| Part | `uv.lock` | Newest when locked | Why not the newest |
|---|---|---|---|
| Python | **3.14** (tested with 3.14.8) | 3.15.0 | vLLM refuses 3.15 (below) |
| CUDA (through torch) | **13.0** | 13.4 | torch's PyPI wheels are built for 13.0, and vLLM pins torch |
| torch | **2.13.0** | 2.14.1 | vLLM 0.32.0 requires `torch==2.13.0` |
| nvidia-cuda-nvcc, -crt, nvvm | **13.0.88** | 13.4.92 | must match torch's CUDA 13.0 headers for FlashInfer's compiler (`pyproject.toml`) |
| vllm | **0.32.0** | 0.32.0 | |
| transformers | **5.19.0** | 5.20.2 | vLLM 0.32.0 requires `<5.20` |
| flashinfer-python | **0.7.0.post1** | 0.7.2 | pinned by vLLM |
| llguidance | **1.7.6** | 1.9.1 | vLLM requires `<1.8` (see [glibc](#an-older-cluster-llguidance-and-glibc) below) |
| xgrammar | **0.2.8** | 0.2.8 | pinned by vLLM |
| numba | **0.65.0** | 0.68.0 | pinned by vLLM |
| numpy | **2.4.6** | 2.5.4 | numba 0.65 requires `<2.5` |
| huggingface-hub | **1.33.0** | 2.2.0 | LiteLLM requires `<2.0` |
| dspy | **3.4.1** | 3.4.1 | the bound stays `<3.5`: 3.5 has not been tried with the language-model layer |
| gepa | 0.1.4 | 0.1.4 | |
| litellm | **1.106.1** | 1.106.1 | |
| optuna | 5.0.0 | 5.0.0 | |
| scipy, scikit-learn | 1.18.1, 1.9.1 | same | |
| matplotlib | **3.11.2** | 3.11.2 | |
| krippendorff | 0.9.0 | 0.9.0 | |
| curl-cffi | 0.16.3 | 0.16.3 | |
| ruff | **0.17.1** (dev group) | 0.17.1 | |
| ty | **0.0.90** (dev group) | 0.0.90 | |
| prek | **0.5.7** (dev group) | 0.5.7 | |
| pytest | 9.1.1 | 9.1.1 | |

The lock covers Linux on x86_64 and aarch64 only (`[tool.uv] environments`): torch has no 3.14 wheels for the other Linux machines. ruff, ty, prek and pytest are in the lock's dev group, so one file pins every tool.

The `tutorials` extra (`uv sync --extra tutorials`) adds what only the notebooks of [tutorials/](../../tutorials/README.md) use: JupyterLab, its kernel and ipywidgets (for progress bars), sentence-transformers (6.2, for MiniLM), langdetect and wordfreq; it is in the lock, but neither the default sync nor any command installs it.

`umap-learn`, `pynndescent`, `igraph`, `leidenalg` and `infomap` are not in the lock, because no code in the repository imports them (an analysis package is added only when the code uses it). Step 5 reads its saved cluster assignments instead of fitting them ([docs/step-5-post-processing/postprocess.md](../step-5-post-processing/postprocess.md#pinned-cluster-map)).

How the table was made: `uv lock --upgrade`, then each package's newest version and its requirements from PyPI's JSON API (for example vLLM 0.32.0's metadata says `torch==2.13.0`, `transformers<5.20.0,>=5.16.1`, `llguidance<1.8.0,>=1.7.0`, `numba==0.65.0`, `flashinfer-python==0.7.0.post1`).

## Python 3.14

The code and its recorded runs were tested with Python 3.14. `requires-python` is `>=3.14`, with no upper bound:

- On a newer Python every command runs, and prints a warning first: `Warning: this code was only tested with Python 3.14; this is Python 3.15.0.` (`TESTED_PYTHON` in [`src/mbdef/__init__.py`](../../src/mbdef/__init__.py)).
- vLLM refuses Python 3.15 and newer, so it carries the marker `python_version < '3.15'` in `pyproject.toml`, and the lock resolves without it there. `embed build`, the one command that runs vLLM in its own process, then stops: `This step runs its model with vLLM, which is not installed (vLLM supports Python 3.14 but not 3.15 or newer). Run it in a Python 3.14 environment: uv sync --python 3.14.` A model server for the other steps can run in its own Python 3.14 environment.
- `cluster check-runtime` records the running Python next to the tested one in its `runtime.json` and warns on a newer one; the packages must match the lock ([docs/setup/cluster.md](cluster.md#the-runtime-cluster-check-runtime)).

vLLM's install page lists Python 3.10 to 3.13, but its wheel (`cp38-abi3`) and its metadata (`<3.15`) allow 3.14, and it serves the pinned models on 3.14 (below).

## Notes on the code and these versions

- **The language-model layer** ([docs/setup/llm.md](llm.md)) is a plain `dspy.LM` on an engine of its own (DSPy's `complete(Request) -> Response`). DSPy's own LiteLLM engine cannot be used as it is, because DSPy hands an engine a canonical request that has no place for `extra_body` (vLLM's `chat_template_kwargs`, the thinking switch of every released call). So the engine keeps `extra_body`, maps the request back to the OpenAI chat form and calls LiteLLM itself, which keeps LiteLLM's response id and provider in the usage log. Worked example: of the 28,126 request receipts of the released TEST and training runs, all map back to their recorded `request_sha256`: 14,874 with a response schema and temperature 0, 6,638 with a schema and no temperature, 6,024 with neither and 590 with temperature but no schema, every one with `max_tokens`, `seed` and `extra_body`. So the recorded-response tests replay unchanged.
- **Errors.** DSPy 3.4 passes an error that is not the server's (the released runs' client-side `KeyError: 'temperature'`, or its replay) on as `LMUnexpectedError`, and its chat format asks again in JSON only after an answer it cannot parse. `llm.stops_the_run` and the chat adapter in `extract/extractor.py` keep the released runs' rule: such an error is an answer that failed, so the chat format asks again in JSON and a window gives no records. The recorded-failure replay test checks it, and live runs follow the same rule.
- **Saved programs.** DSPy writes the saving environment's Python and DSPy versions into a program (the released ones: 3.13 and 3.3.1) and warns on every load when they differ. The frozen programs stay unchanged; the warning is left out of the output, and the training tests compare a trained program apart from those two version strings: the w32k replay gives the released program byte for byte otherwise.
- **GEPA.** DSPy 3.4.1 changed GEPA's default proposer, so a GEPA run with the same seed gives another result than one with DSPy 3.3. The package trains with Bootstrap, which is unchanged.
- **Figures.** The agreement step's figures are compared by the numbers they plot: matplotlib writes its version into an image, so another matplotlib changes the bytes of the same figure.
- **Formatting.** ruff 0.17 formats for Python 3.14, which allows `except A, B:` without parentheses.

## How it was checked

On a workstation with a 24 GB GPU, on the locked environment (Python 3.14.8):

**The tests and checks.** The fast test set (`uv run pytest -q -m "not slow"`), `ruff check`, `ty check` and the pre-commit hook (`prek run --all-files`) run on this environment. `mbdef cluster check-runtime` prints `Python 3.14.8 and 7 packages match uv.lock.`

**A live vLLM run.** With torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.19.0 and FlashInfer 0.7.0.post1, `vllm serve` started the pinned Gemma 12B (revision `1d2c2d7f…`) with the released serve flags, except an 8,192-token context and 60% of the GPU (another local server held 9 GB of it): ready after 140 s cold, 55 s warm. Two calls then went through `make_lm` and a DSPy program, with the settings extraction and the verifier send:

| Call | Settings sent (from its receipt) | Reasoning tokens | Time | Result |
|---|---|---|---|---|
| thinking on | `max_tokens` 8000, `seed` 9, `extra_body` `enable_thinking: true` | 2,090 | 30.8 s | answered in the chat format, first call |
| thinking off | `temperature` 0.0, `max_tokens` 8000, `seed` 9, `extra_body` `enable_thinking: false` | 0 | 0.8 s | answered in the chat format, first call |

So the thinking switch in `extra_body` reaches the server through the engine. The usage log holds each response id (vLLM sends no provider; OpenRouter does). A first try with a 2,048-token cap was cut off (finish reason `length`, 2,047 reasoning tokens) and DSPy asked again in JSON, which answered: the retry path works live too.

## An older cluster: `llguidance` and glibc

On a cluster whose nodes run an older Linux (Rocky Linux 8.10, glibc 2.28):

- For Python 3.14, `llguidance` 1.7.6 is the only locked package without a wheel for glibc 2.28: its x86_64 wheel needs glibc 2.31 (`cp39-abi3-manylinux_2_31`; the `manylinux_2_17` wheel is for the free-threaded build only). Every other package, torch and vLLM included, needs at most 2.28. So `uv sync --locked` builds `llguidance` from its source package there, which needs a Rust compiler (on many clusters `module load rust` provides one).
- The lock's format needs uv 0.13 or newer; an older uv on a login node is updated with `uv self update`.
- So, on such a cluster: `uv self update`, load a Rust compiler, then `uv sync --locked`.
- Nodes with a newer Linux (for example AlmaLinux 10, glibc 2.39) take the wheel directly.

## Faster libraries (RAPIDS)

Not in the lock. No code in the repository runs UMAP or Leiden, and k-means runs only on refits: they belong to Step 5, which keeps its cluster assignments as frozen inputs. Measured on the shape Step 5 clusters (113,494 definition vectors of 768 dimensions), in a separate environment (Python 3.14.8, cuML and cuGraph 26.10 for CUDA 13, umap-learn 0.5.12, scikit-learn 1.9.1, leidenalg 0.12.0 on 24 CPU threads), one run each, with the GPU warmed up first. The vectors are a fixed sample (seed 0) of the corpus embeddings in `$DATA_ROOT/embeddings/qwen3-emb-8b/` (no TEST data), cut to their first 768 dimensions and normalised.

| Task | CPU | GPU (RAPIDS) | Faster | Same result? |
|---|---|---|---|---|
| k-means, 50 clusters, 10 starts | 225.8 s (scikit-learn) | 1.6 s (cuML) | 146× | no: labels agree at adjusted Rand index 0.52; inertia 67,181 vs 67,307 |
| UMAP to 2 dimensions, 15 neighbours, cosine | 21.9 s (umap-learn) | 1.7 s (cuML) | 13× | no (a different layout) |
| Leiden, resolution 1.0, on one 15-neighbour graph | 6.7 s (leidenalg) | 0.17 s (cuGraph) | 40× | no: 69 vs 66 communities, modularity 0.789 vs 0.801 |

So RAPIDS is much faster on this shape, and it gives other clusterings than the CPU libraries. A pinned result made with it would change, so it is not the default anywhere. For Step 5 it suits exploration and refits whose assignments are not the pinned ones.

## How to refresh

1. Raise the bounds in `pyproject.toml` where needed, then `uv lock --upgrade`. vLLM's pins (torch, transformers, FlashInfer, llguidance, numba) decide most of the rest: read them from its PyPI metadata.
2. `uv sync --locked`, then the test set (`uv run pytest -q`, with `-m "not slow"` for the fast part; the slow tests need a GPU and the data root, [docs/setup/data-fetch.md](data-fetch.md)), `ruff check`, `ty check` and `cluster check-runtime`. On a shared machine, run the GPU tests under a lock (`flock`).
3. Start vLLM on the pinned models once and make a call through the language-model layer (the live run above).
4. Update the table. A rebuild can still differ from a stored output for the reasons in [docs/setup/pinned.md](pinned.md).
