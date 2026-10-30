# MBDef: The Media Bias Definitions Dataset

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) [![Hugging Face: MBDef](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MBDef-yellow)](https://huggingface.co/datasets/mediabiasgroup/MBDef) [![Hugging Face: MBDef-Gold](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MBDef--Gold-yellow)](https://huggingface.co/datasets/mediabiasgroup/MBDef-Gold) [![Zenodo: MBDef](https://img.shields.io/badge/Zenodo-MBDef-blue)](https://doi.org/10.5281/zenodo.23773908) [![Zenodo: MBDef-Gold](https://img.shields.io/badge/Zenodo-MBDef--Gold-blue)](https://doi.org/10.5281/zenodo.23773944)

The code and the data to replicate and review the creation of the MBDef and MBDef-Gold datasets of definitions related to ***media bias*** quoted from the scientific literature. MBDef holds 34,916 terms with 103,476 definitions from 6,115 papers. We release all the code and data required to replicate the figures and results in [the paper](https://arxiv.org/abs/XXXX.XXXXX), as well as the [DSPy](https://dspy.ai)-optimised prompts used to extract these definitions, the *corpus closure* code, the relevance filters and a dataset visualisation tool, the MBDef Explorer. For details, see the paper.
You can also run the extractor on your own papers, score any extractor against MBDef-Gold with DefScore, train your own extractor, and swap in another model.

![The MBDef Explorer: a map of MBDef's terms, coloured by topic](docs/images/mbdef-explorer-poster.png)

<p align="center">
  <a href="https://media-bias-group.github.io/MBDef/"><img alt="Open the MBDef Explorer" src="https://img.shields.io/badge/Open%20the-MBDef%20Explorer-2f6fdb?style=for-the-badge"></a>
  <a href="https://media-bias-group.github.io/MBDef-ProjectPage/"><img alt="Project page" src="https://img.shields.io/badge/Project-page-24292f?style=for-the-badge"></a>
</p>

## Overview

- **The datasets:** [MBDef](https://huggingface.co/datasets/mediabiasgroup/MBDef) (the definitions extracted from the corpus) and [MBDef-Gold](https://huggingface.co/datasets/mediabiasgroup/MBDef-Gold) (the hand-annotated ground truth: 130 papers, 1,003 records, 1,192 definitions, split into train, validation and test).
- **[Tutorials](tutorials/README.md):** six Jupyter notebooks that explore MBDef and MBDef-Gold: the most-defined terms and their definitions, clusters of definitions by meaning, how media bias connects to neighbouring terms, how far definitions of one term diverge, the research disciplines of the papers, and how often the papers define the bias types of two published taxonomies.
- **[Use MBDef](#use-mbdef):** [extract definitions from your papers](#1-extract-definitions-from-your-papers), [score an extractor with DefScore](#2-score-an-extractor-with-defscore), [train your own extractor](#3-train-your-own-extractor), [swap the model](#4-swap-the-model).
- **[Reproduce the paper](#reproduce-the-paper):** every step, result and number of the paper, with the command that rebuilds it.
- **[Appendix](#appendix)** of the paper, and [docs/README.md](docs/README.md): one page per step. [GLOSSARY.md](GLOSSARY.md) defines the project's words.

## Install

You need [uv](https://docs.astral.sh/uv/) and Python 3.14 (a newer Python runs with a warning; the code was tested with 3.14):

```bash
git clone <repository-url> mbdef
cd mbdef
uv sync --locked          # about 8 GB: the lock includes Torch and vLLM, which the steps that run a model use
cp .env.example .env      # machine settings: the data folder, the model server, the scoring device
```

`.env` holds the settings of your machine ([`.env.example`](.env.example) explains each): `DATA_ROOT`, the folder for the data that is not in git; `DSPY_API_BASE` and `DSPY_API_KEY`, the OpenAI-compatible model server; `SCIDEF_NLI_DEVICE` (`cuda`, `cpu` or `auto`), where the scoring models run.
Some steps need tools that `uv` does not install, for example a running [GROBID](https://grobid.readthedocs.io/) to read PDFs; each step's page says which.

## Use MBDef

The four tasks below run on one machine. A model server needs a GPU; the outputs shown below were made on one 24 GB laptop GPU.

### 1. Extract definitions from your papers

Start an OpenAI-compatible server for the model. The released extractor is Gemma 4 12B with three saved prompt programs ([`data/extractors/`](data/extractors/)):

```bash
uv run vllm serve google/gemma-4-12B-it-qat-w4a16-ct --revision 1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee \
    --reasoning-parser gemma4 --generation-config auto --max-model-len 65536
```

With `DSPY_API_BASE=http://localhost:8000/v1` in `.env`, extract from a folder of GROBID TEIs (`<paper id>.tei.xml`) or of PDFs (`<paper id>.pdf`, converted by a GROBID on `localhost:8070`, for example `docker run --rm --init -p 8070:8070 grobid/grobid:0.9.1-crf`):

```bash
uv run mbdef extract run --teis my-papers/ --out out/run
uv run mbdef extract run --pdfs my-pdfs/ --out out/run      # the same, from PDFs
```

On two papers ("Branding the nation" and "The science of fake news") this took 3.5 minutes and printed the counts of the extractor's two parts (one program on 8,000-character windows; two on 16,000 and 32,000) and of their pool:

```json
{
  "w8k": {"papers": 2, "windows": 6, "failed_windows": 0, "records": 17, "definitions": 36, "candidates": 36, "model_calls": 6},
  "w16k+w32k": {"papers": 2, "windows": 5, "failed_windows": 0, "records": 14, "definitions": 23, "candidates": 23, "model_calls": 5},
  "predictions.jsonl": {"papers": 2, "records": 18, "definitions": 50}
}
```

`out/run/predictions.jsonl` has one definition record per line; every string is copied from the paper:

```json
{"paper_id": "73bfad11b96a69cb882028ead115751adb55252d", "term": "fake news",
 "definitions": [{"text": "We define fake news to be fabricated information that mimics the output of the news media in form, but not in organizational process or intent.", "citation": []}, …],
 "synonyms": [], "description": […], "related_terms": [], "subtypes": [], "supertypes": [{"text": "misinformation", "citation": []}], "examples": […]}
```

The model thinks before it answers and samples with its own settings, so your counts will differ a little from run to run.
For an API instead of a local server, set `DSPY_API_BASE` and `DSPY_API_KEY` to it, pass the model's name with `--model`, and `--thinking unset` if the API refuses vLLM's thinking switch.

**Keep only the records about media bias**, as the paper's extractors do: the frozen relevance filter (a classifier on BGE vectors; no language model) runs on each part, and concatenating the parts' files pools them:

```bash
for part in w8k w16k+w32k; do
  uv run mbdef extract filter --run "out/run/$part" --records union --out "out/filtered/$part"
done
cat out/filtered/*/predictions.jsonl > out/predictions.jsonl
```

This is the paper's "P≥28 Gemma only" extractor. Its "P≥28" extractor also lets a second model check each definition before the filter (`extract verify`), and a whole corpus runs in shards (`extract plan`, `extract derive`): [docs/step-3-extraction/extract.md](docs/step-3-extraction/extract.md).

### 2. Score an extractor with DefScore

DefScore compares two definitions: the mean of their word overlap, their BGE embedding similarity and their entailment both ways, calibrated to 0–1. A predicted definition matches a reference definition at DefScore ≥ 0.5, one to one; precision, recall and F1 count the matches over all definitions, and the full-record score also compares the other fields of matched records ([docs/step-3-extraction/evaluate.md](docs/step-3-extraction/evaluate.md#scoring-extractions-defscore-f1-and-matching)).

Fetch MBDef-Gold into the data root, then score a predictions file on one of its splits (`train`, `validation` or `test`; the reference is the split's relevant records, as in the paper):

```bash
uv run mbdef data fetch MBDef-Gold/
uv run mbdef evaluate score --predictions out/predictions.jsonl --split test --out out/scores
```

Every paper of the split counts, so a paper without predictions adds its definitions to what was missed; `--paper ID` (repeatable) scores on some papers only. The two papers above are test papers; scored on them, with the scoring models on the GPU (19 seconds):

```text
Reference: the test split of validated_gt.json, 2 papers with 10 definitions
out/predictions.jsonl: 8 of 10 reference definitions found, 50 predicted (precision 16.0%, recall 80.0%), full-record score 0.285
  2 of the 2 papers have predictions; F1 26.7%
wrote out/scores
```

`out/scores/scores.json` has the counts of every paper, and `receipt.json` the settings and models.
**Predictions from any tool** need only a paper id, a term and its definitions, one record per line; the other lists (`synonyms`, `description`, `related_terms`, `subtypes`, `supertypes`, `examples`) are optional, and an item is a string or `{"text": …}`:

```json
{"paper_id": "73bfad11b96a69cb882028ead115751adb55252d", "term": "fake news", "definitions": ["fabricated information that mimics the output of the news media in form"]}
```

The paper ids, titles and DOIs of each split are in MBDef-Gold (`validated_gt.json`, or the `paper_id`, `paper_title` and `paper_doi` columns of `data/test.parquet`); the test split has 29 papers with 365 reference definitions. The papers' full texts are not part of MBDef-Gold: get them by their DOIs and name each file by its paper id. Give `--predictions` more than once to score several files in one run.

### 3. Train your own extractor

The released programs were trained with DSPy's BootstrapFewShot: the extractor's instruction plus one worked example, drawn from the first train paper (in a fixed order, seed 9) whose extraction scores above 0 on the full-record score against MBDef-Gold. One program is trained per window size. The 80 train papers are the training set; the 21 validation papers are held out (the optimisers the paper compared and did not select, MIPROv2 and GEPA, choose on them); the test papers are refused. Bootstrap stops at its first success, so the budget is small: the released runs made 9, 5 and 2 model calls ([docs/step-3-extraction/extract.md](docs/step-3-extraction/extract.md#training-the-programs-extract-train)).

With the TEIs of MBDef-Gold's train and validation papers in `gold-teis/` (named by paper id) and a model server as in task 1:

```bash
uv run mbdef extract train-dataset --teis gold-teis/ --out out/train-dataset
for window in w8k w16k w32k; do
  PYTHONHASHSEED=0 uv run mbdef extract train --window "$window" --teis gold-teis/ \
      --dataset out/train-dataset/train-dataset.json --split out/train-dataset/split.json \
      --out "out/programs/$window"
done
uv run mbdef extract run --programs out/programs --teis test-teis/ --out out/run-mine
uv run mbdef evaluate score --predictions out/run-mine/predictions.jsonl --split test --out out/scores-mine
```

A small run, one window size on one train paper (`--window w32k --paper f5c282326822c5c1cca05a49940024480948e09c`), took 1.6 minutes and printed:

```json
{"window": "w32k", "papers": {"train": 1, "dev": 0},
 "papers_tried": [{"paper_id": "f5c282326822c5c1cca05a49940024480948e09c", "score": 0.18573184463499492}],
 "demo": {"paper_id": "f5c282326822c5c1cca05a49940024480948e09c", "window": 0, "windows": 2},
 "model_calls": 2, "failed_calls": 0}
```

Run alone on the two test papers (`extract run --programs out/programs --part w32k`), that program found 4 of their 10 reference definitions with 12 predictions (precision 33.3%, recall 40.0%). Another optimiser, more demos or a stricter acceptance are a few lines in `extract/train.py` ([docs/step-3-extraction/extract.md](docs/step-3-extraction/extract.md#training-the-programs-extract-train)).

### 4. Swap the model

A model released later is one argument: serve it, and pass its name with `--model` (`--revision` is only recorded). The same programs work with any model; retrain them with the same `--model` for the best results (task 3):

```bash
uv run vllm serve Qwen/Qwen3-VL-2B-Instruct --generation-config auto --max-model-len 65536
uv run mbdef extract run --model Qwen/Qwen3-VL-2B-Instruct --thinking unset --format json \
    --max-tokens 8192 --teis my-papers/ --out out/run-qwen
```

`--thinking` is `on` (the released runs), `off`, or `unset` for a model or API without vLLM's thinking switch. `--format json` asks for a JSON object instead of DSPy's field markers: with this small 2-billion-parameter model it left 1 of 11 windows without records, against 6 of 11 in the default format, where the model kept writing until the token limit. Filtered and scored as in tasks 1 and 2, on the same two papers:

```text
out/predictions.jsonl: 8 of 10 reference definitions found, 50 predicted (precision 16.0%, recall 80.0%), full-record score 0.285
out/predictions-qwen.jsonl: 3 of 10 reference definitions found, 23 predicted (precision 13.0%, recall 30.0%), full-record score 0.092
```

The other steps take a model the same way: `extract verify --model` (the verifier), `relevance classify --model` (which papers are about media bias, [docs/step-1-2-papers/relevance.md](docs/step-1-2-papers/relevance.md)), and the corpus run in shards ([docs/step-3-extraction/extract.md](docs/step-3-extraction/extract.md#a-corpus-run-extract-plan-the-jobs-extract-derive)). The relevance filter needs no language model.

## Reproduce the paper

Every result, figure and number of the paper comes from one of the commands below.
`uv run mbdef data fetch` downloads the data they read into the data root (the two datasets from Hugging Face, the saved runs from the public deposit) and checks every file ([docs/setup/data-fetch.md](docs/setup/data-fetch.md)).
The steps that call a language model replay its recorded answers, so a rebuild needs no model and gives the released numbers; each page says what a rerun with a live model changes.

For example, the annotators' agreement (Table 3) in about 3 seconds, the held-out test result of the extractors in about 6 minutes on a GPU, and the paired comparison of our extractor with Opus in a second:

```bash
uv run mbdef data fetch
uv run mbdef evaluate agreement --out out/agreement
uv run mbdef evaluate heldout --out out/heldout
uv run mbdef evaluate paired --out out/paired
```

| Paper step | What the code does | Command | Page |
|---|---|---|---|
| 1, 2 | Decides which of the 14,862 papers are about media bias: two language models read each title and abstract, a third settles their disagreements: **3,869** relevant seed papers | `relevance` | [docs/step-1-2-papers/relevance.md](docs/step-1-2-papers/relevance.md) |
| 1, 2 | The seed list: the 3,869 relevant papers that Step 3 extracted from, and the starting corpus's identifiers | none: data files in `data/seeds/` | [docs/step-1-2-papers/seeds.md](docs/step-1-2-papers/seeds.md) |
| 1 | Finds, downloads and converts the PDFs of the papers to collect | `collect` | [docs/step-1-2-papers/collect.md](docs/step-1-2-papers/collect.md) |
| 3 | Builds the ground truth, MBDef-Gold, from the curator's export; makes publishable copies of the annotation exports | `groundtruth build`, `groundtruth pseudonymise` | [docs/step-3-extraction/groundtruth.md](docs/step-3-extraction/groundtruth.md) |
| 3 | Extracts definition records from papers, verifies and filters them, trains the prompt programs | `extract` | [docs/step-3-extraction/extract.md](docs/step-3-extraction/extract.md) |
| 3 | Scores extractions against the ground truth; the annotators' agreement (Table 3); the held-out test result and the comparison with Opus | `evaluate` | [docs/step-3-extraction/evaluate.md](docs/step-3-extraction/evaluate.md) |
| 3 | Picks the four extractors from the grid of 2,268 scorings | `evaluate grid`, `evaluate select` | [docs/step-3-extraction/selection.md](docs/step-3-extraction/selection.md) |
| 4 | The works the extracted definitions cite, one level deep, and the corpus release | `closure` | [docs/step-4-closure/closure.md](docs/step-4-closure/closure.md) |
| 5 | Decides which corpus terms are about media bias (an ensemble of three scorers) and merges the kept terms into concepts | `postprocess` | [docs/step-5-post-processing/postprocess.md](docs/step-5-post-processing/postprocess.md) |
| | Builds MBDef's tables, its dataset card and the MBDef Explorer | `dataset`, `explorer` | [docs/dataset/dataset.md](docs/dataset/dataset.md), [docs/dataset/explorer.md](docs/dataset/explorer.md) |

Every step runs on one machine; for the steps that need large GPUs, [docs/setup/cluster.md](docs/setup/cluster.md) shows how to run the same command as a job on a Slurm cluster, if you have one.
`uv run mbdef --help` lists every command.

### Where the data lives

- **On Hugging Face:** the two datasets.
  MBDef-Gold is the ground truth (`validated_gt.json`: 130 papers, 1,003 records, 1,192 definitions; the non-relevant records, the report and the statistics beside it), described in [docs/step-3-extraction/groundtruth.md](docs/step-3-extraction/groundtruth.md); its train, validation and test files and its dataset card come from `dataset build-gold` ([docs/dataset/dataset.md](docs/dataset/dataset.md#mbdef-gold-dataset-build-gold)).
  MBDef is the corpus dataset (its five tables; it holds no ground truth), described in [docs/dataset/dataset.md](docs/dataset/dataset.md).
  `uv run mbdef data fetch MBDef-Gold/ MBDef/` downloads both into the data root (below), where every command that reads them finds them.
- **In git, under `data/`:** the inputs of the steps: the ground truth's earlier versions (`ground_truth/`), the saved prompt programs of the released extractor (`extractors/`), the annotation audit and the curation's publishable files (`newAnnotations/`), the seed list (`seeds/`: the 3,869 seed papers with their relevance labels and the 14,862 papers of the starting corpus, [docs/step-1-2-papers/seeds.md](docs/step-1-2-papers/seeds.md)) and the small frozen inputs of the released runs (`frozen/`).
- **Outside git, in the data root:** everything too big for git (the saved runs, the embedding cache, Step 5's large files) and the two datasets.
  It is one folder per machine (`DATA_ROOT` in `.env`) with a fixed layout, listed with a sha256 for every file in [data/data-root.tsv](data/data-root.tsv).
  `uv run mbdef data fetch` downloads it from Hugging Face and the public deposit and checks every file ([docs/setup/data-root.md](docs/setup/data-root.md)).
  The papers' full text and the original annotation exports are not public, so the list leaves them out.

## Appendix

[APPENDIX.md](APPENDIX.md) holds what the paper's appendix leaves out for space: our extractor against Claude Opus, the corpus closure in numbers, and every pair scorer we compared on DefSim, each with the command or file that reproduces it.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) has the tests and the link check.

## Licence

The code is under the [Apache License 2.0](LICENSE).
The data the group made (the annotations, the labels, the decisions about which terms are about media bias, and the tables of the dataset) is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): reuse it, with credit.
The definitions quoted from papers and the papers' metadata (titles, authors, DOIs) stay their authors'; the dataset links each definition to its paper.

## Cite

Citation details will follow with the paper.
