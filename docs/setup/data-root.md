# Machine settings (`.env`) and the data root

## `.env`: only what differs between machines

Copy [`.env.example`](../../.env.example) to `.env` in the repository root and fill it in. The file is git-ignored, because it holds keys. Every `mbdef` command reads it once at the start, and a value already set in the environment wins. A command asks only for the keys it uses; if one is missing it stops with, for example:

```text
DATA_ROOT is not set. Copy .env.example to .env and set DATA_ROOT there; .env.example says what it is.
```

| Key | What it is | Read by |
|---|---|---|
| `DATA_ROOT` | The folder that holds the data kept outside git (below) | `data fetch`, `groundtruth build`, `extract train-dataset` and `extract train-filter` (MBDef-Gold), `embed build`, `dataset build`, `evaluate agreement`, `evaluate heldout`, `evaluate paired`, `evaluate grid`, `extract derive`, `closure bundle`, `closure identity evidence`, `postprocess` (its corpus, embeddings and large saved results), `collect` (for its default folder) |
| `HF_TOKEN` | Optional: a Hugging Face key, needed only while the two datasets' repositories are private | `data fetch`, for the files of MBDef and MBDef-Gold |
| `DSPY_API_BASE`, `DSPY_API_KEY` | The language-model endpoint (OpenAI-compatible, for example a local vLLM server) and its key | `extract run`, `extract verify` and `extract train` ([docs/step-3-extraction/extract.md](../step-3-extraction/extract.md)), and `relevance classify` ([docs/step-1-2-papers/relevance.md](../step-1-2-papers/relevance.md)), except with `--replay` |
| `SCIDEF_NLI_DEVICE` | Where the two scoring models (BGE and the entailment model) and the relevance filter's BGE run: `cpu`, `cuda` or `auto` | `evaluate score`, `evaluate heldout` and `evaluate grid`, except with `--replay-scores`; `extract filter`, except with `--replay` |
| `UNPAYWALL_EMAIL` | The contact address Unpaywall asks for in each request | `collect open-access` |
| `OPENALEX_API_KEY` | Optional: a larger daily OpenAlex budget | `collect openalex`, tutorial 05 (`analysis.disciplines`) |
| `CROSSREF_MAILTO` | Optional: the contact address for Crossref's polite pool (`UNPAYWALL_EMAIL` if unset) | `collect crossref`, `collect semantic-scholar` |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional: a Semantic Scholar API key; without it, requests share the public rate limit | `collect semantic-scholar`, `collect semantic-scholar-doi` |
| `CLUSTER_<SITE>_ROOT`, e.g. `CLUSTER_EXAMPLE_ROOT` | The one folder that holds everything of yours on the cluster of site file `cluster/<site>.toml` | `cluster render --site <site>`, the chosen site's keys only ([docs/setup/cluster.md](cluster.md)) |
| `CLUSTER_<SITE>_ACCOUNT`, e.g. `CLUSTER_EXAMPLE_ACCOUNT` | The project account Slurm bills the job to, for a site file that says `needs_account = true` | `cluster render` for such a site |

Method settings (model, entailment model and revision, maximum tokens, thresholds) are not in `.env`. They are command-line flags whose defaults are pinned to the released runs, and every receipt records them, with the commit of the code (`code.commit`; `null` when no git checkout holds the code, as in a copy of the repository without git). Keys such as `DSPY_MODEL` or `SCIDEF_NLI_MODEL` in a `.env` change nothing: no command reads them, which [`tests/test_evaluate_score.py`](../../tests/test_evaluate_score.py) checks.

## The data root

Data that cannot be in git (too big, or naming the annotators) lives in one folder per machine, set as `DATA_ROOT`. Its layout is fixed, so every machine finds a file at the same relative path:

| Folder | What it holds |
|---|---|
| `MBDef-Gold/` | MBDef-Gold, the ground truth, as its Hugging Face repository holds it: the relevant and the non-relevant records (`validated_gt.json`, `validated_gt_non_relevant.json`), the same records as split files (`data/{train,validation,test}.parquet`), the statistics, the manifest, the guide, the report with its figures, and the notes on the earlier ground truths (`history/data-README.md`). Every command that reads the ground truth reads it here unless a flag names another file; `data fetch MBDef-Gold/` downloads it ([docs/setup/data-fetch.md](data-fetch.md#the-two-datasets)) |
| `MBDef/` | MBDef, the released corpus dataset ([docs/dataset/dataset.md](../dataset/dataset.md)), as its Hugging Face repository holds it: its tables under `data/` and the explorer's page data under `explorer/` ([docs/dataset/explorer.md](../dataset/explorer.md)), which `data fetch MBDef/` downloads |
| `groundtruth/` | The ground truth's sources: the curator's export and the raw annotations, and a text snapshot of the annotation's method notes, none of them public ([docs/step-3-extraction/groundtruth.md](../step-3-extraction/groundtruth.md)); the agreement numbers read the same two zips ([docs/step-3-extraction/evaluate.md](../step-3-extraction/evaluate.md)); and under `pseudonymised/` the two zips' publishable copies, which `groundtruth pseudonymise` makes from them and the public deposit holds |
| `relevance/` | The relevance step's inputs and recorded answers ([docs/step-1-2-papers/relevance.md](../step-1-2-papers/relevance.md#inputs)): the 14,862 papers' titles and abstracts (`papers.csv`), the arbiter's labelled examples (`examples.csv`) and the three classifiers' answers (`answers/qwen/`, `answers/gemma/`, `answers/glm/`) |
| `collect/` | The collect step's queue, ledgers, PDF zips and TEI files ([docs/step-1-2-papers/collect.md](../step-1-2-papers/collect.md)); written by its commands, not fetched |
| `embeddings/` | The embedding cache, one folder per model: today `qwen3-emb-8b/`, the Qwen3-Embedding-8B vectors that Step 5 reads ([docs/step-5-post-processing/embed.md](../step-5-post-processing/embed.md)) |
| `selection/` | The extractor selection's cluster job, under `rescore/<date>/` ([docs/step-3-extraction/selection.md](../step-3-extraction/selection.md#the-cluster-job)): the tar of scorings sent to the cluster, and the pair scores, scores and job record that came back; the job writes them on the cluster and they were copied here by hand; `data-root.tsv` lists the released job's files (5 rows, under `rescore/2026-10-20/`) |
| `frozen/` | Frozen inputs over 10 MB, at their repository path (for example `frozen/results/production-20260925/report-20260927/selection/results.json`), and saved outputs of runs that are not in git, at the path the code reads them from (the TEST rows, results and cohort under `frozen/artifacts/experiments/test-eval-20260924/`, and Opus's DEV evaluation, which [the TEST rescoring and the paired comparison with Opus](../step-3-extraction/evaluate.md#the-held-out-test-rescoring) read); and copies of the cluster runs' folders under `frozen/runs/<run>/`, for example the seed run's saved parts, plan, shards, picks and seed manifest, and its derived variants and comparison as stored, under `frozen/runs/production-20260925/` ([`extract derive`](../step-3-extraction/extract.md#the-seed-runs-variants-extract-derive) reads them, and a rebuild is compared with them); the closure's inputs of release v1.1 (its folders `closure-v11/` and `taxonomy-closure-20260928/` and the release folder `definition-corpus-v1.1/`) under `frozen/closure/`, and the seed run's own inputs in `frozen/runs/production-20260925/inputs/` ([docs/step-4-closure/closure.md](../step-4-closure/closure.md#inputs)); Step 5's inputs and its frozen files over 10 MB (the corpus store `frozen/artifacts/experiments/taxonomy-curation-v2/library.sqlite3`, and `frozen/step5-post-processing/<date>/…`, the same paths as in git's `data/frozen/step5-post-processing/`; [docs/step-5-post-processing/postprocess.md](../step-5-post-processing/postprocess.md#inputs)); other frozen inputs are in git under `data/frozen/` |

[`data/data-root.tsv`](../../data/data-root.tsv) lists every file that has a public copy or a command that makes it, with its sha256, size and source. Its first line gives the address of the public deposit that holds the files; it is a placeholder (`https://deposit.invalid/…`) until the deposit is uploaded:

```text
#deposit	https://deposit.invalid/mbdef-data/	(a placeholder: …)
path                                              sha256     bytes      source
frozen/results/production-20260925/report-20260927/selection/results.json  b9ec1d3e…  26742799  https:frozen/results/…/selection/results.json
frozen/runs/production-20260925/corpus/seeds/derived  9579d3ad…  813967253  https:frozen/runs/production-20260925/corpus/seeds/derived.tar.zst#derived
frozen/runs/production-20260925/corpus/seeds/picks.json  eca3194a…  19462  derive:cp data/extractors/picks.json "$DATA_ROOT/…/picks.json"
```

A source is one of three kinds:

| Source | What it means | Rows |
|---|---|---:|
| `hf:<dataset>/<path>` | The file `<path>` of one of the two datasets on Hugging Face, MBDef or MBDef-Gold, at the revision `files.HF_DATASETS` pins, checked by its sha256 | 182 |
| `https:<path>` | The deposit's file `<path>`, stored under a flat name (each `/` becomes `__`) below the deposit's address, checked by its sha256. A folder is one zstd tar archive, `https:<path>.tar.zst#<folder name>`, unpacked with `zstd -d --long=31` | 58 |
| `derive:<command>` | No copy: the command makes the file from files that are public or in git | 1 |

For example, the seed run's derived variants (813,967,253 bytes, 27,093 files) are one folder: the deposit holds them as the file `frozen__runs__production-20260925__corpus__seeds__derived.tar.zst`, whose member `derived` must have the listed folder digest; with a Zenodo record as the deposit its address is `https://zenodo.org/records/<id>/files/frozen__runs__production-20260925__corpus__seeds__derived.tar.zst`. A folder is listed with its folder digest (the sha256 of its `sha256sum` listing, `files.digest`) and its total size.

**Files without a public copy are not in this list.**
They are the papers' full text (the seed and closure TEIs, saved runs that hold windows of it, the identity check's PDF first pages, the corpus store's paragraphs, the TEST cohort's papers), the original annotation exports, the snapshot of the annotation's method notes, and files that name local or cluster folders.
A machine with only the public data has no copy of them, so the commands that read one (for example `extract derive` on the seed run's saved parts) cannot run there; give such a command your own files with its flags.

## Fetch the data

`mbdef data fetch [PREFIX ...]` downloads every missing file of the list, from Hugging Face (the two datasets) or from the public deposit, into `DATA_ROOT` and checks its sha256; a file that is already there is only checked.
Its page, [docs/setup/data-fetch.md](data-fetch.md), has the details, the lines it prints and how to add a file.

```bash
uv run mbdef data fetch MBDef-Gold/     # MBDef-Gold, the ground truth
uv run mbdef data fetch selection/      # only the extractor selection's cluster job
uv run mbdef data fetch                 # everything listed
```
