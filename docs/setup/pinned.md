# What is pinned, and why a rebuild can still differ

The paper's outputs were made once, by runs that are over. The code that made them is the package, and each step page says how to rebuild its outputs and compare them with the stored ones. This page says what such a comparison rests on: what is pinned, and the four reasons a rebuild can still differ.

## What is pinned

| What | How it is pinned | Example |
|---|---|---|
| Frozen inputs and outputs, `data/frozen/` (71 files) | their sha256, listed in the run records and manifests that use them; nothing edits them | the held-out TEST plan, `data/frozen/results/test-eval-20260924/plan.json`, whose sha256 `evaluate heldout` checks before it scores anything |
| The ground truth, MBDef-Gold (its files on Hugging Face, among them the fixed texts `build` copies into its release: the guide, the Markdown report, the report's figures and the notes on the earlier ground truths; and from git `data/ground_truth/` and the template `groundtruth_report.tex`) | their sha256 in MBDef-Gold's `validated_gt_manifest.json` | `groundtruth build` writes `validated_gt.json` byte for byte as released ([docs/step-3-extraction/groundtruth.md](../step-3-extraction/groundtruth.md)) |
| The released extractor programs and picks (`data/extractors/`) | their sha256 in the runs' receipts and the selection's audit | all 388 part receipts of the seed run name the three programs' hashes ([data/extractors/README.md](../../data/extractors/README.md)) |
| The files outside git | their sha256 and size in `data/data-root.tsv`; `data fetch` keeps a copy only if both match | 79 rows (20 of the two datasets on Hugging Face, 58 in the public deposit, 1 made by a command); [docs/setup/data-fetch.md](data-fetch.md) |
| The models | a revision per model (`models.PINNED`) | Gemma 4 12B at `1d2c2d7f…`, the extractor of the released picks |
| Python and the packages | `uv.lock`, tested with Python 3.14 | Python 3.14, torch 2.13.0 ([docs/setup/toolchain.md](toolchain.md)) |
| Method settings | named constants as the defaults of each command's flags, written into each receipt; a few are code, not flags: the prompt texts (the judge's instructions, basal's question), what a read-out depends on (the judge's 20 log-probabilities per token and its thinking switch), the kind of classifier `extract train-filter` fits (with its solver and iteration cap), and the identity rules of `collect` | the scoring batch of 32 pairs, as in every released scoring run: another size moves a probability by about 1e-6 ([docs/step-3-extraction/selection.md](../step-3-extraction/selection.md)) |

**Pinned files keep what they recorded.** A file pinned by its hash keeps its bytes, so it also keeps the paths, machines, clusters, accounts, branch names and people's roles that were written into it when it was made: they are records of that run, not settings. The same holds for the fixtures that are byte copies of released outputs (`tests/fixtures/released-outputs.zip`). Of the curation's working files in `data/newAnnotations/`, the repository holds three as copies without the papers' passages or the local PDF paths ([docs/step-3-extraction/groundtruth.md](../step-3-extraction/groundtruth.md)). Such a file can also be out of date: MBDef-Gold's guide names the script that first built the ground truth, while [docs/step-3-extraction/groundtruth.md](../step-3-extraction/groundtruth.md) gives the command now.

## Why a rebuild can still differ

Four causes explain every difference found between a stored output and its rebuild. What can be pinned is pinned; the rest gets a band, and a count never gets one: a count is the recorded number or the check fails.

| Cause | What happens | What is done | Measured example |
|---|---|---|---|
| **Run details written into files** | a time, a folder, the name or hash of the code, a random id in a figure | the comparison leaves out exactly those members, row by row | the ground truth's manifest named `scripts/prepare_validated_gt.py`; a rebuild names `code/groundtruth.py` |
| **Python's hash order** | the order of a set, and what depends on it, changes from one process to the next | every rebuild runs with `PYTHONHASHSEED=0` and records it | the demo that DSPy draws while training a program depends on it ([docs/step-3-extraction/extract.md](../step-3-extraction/extract.md#training-the-programs-extract-train)) |
| **Parallel or GPU rounding** | a GPU adds numbers in an order that depends on the batch, so a probability moves in its sixth decimal; k-means with several threads gives another partition | a band per kind of number: probabilities ±0.00001, the secondary full-record score ±0.001, k-means diagnostics ±0.002 | the held-out TEST's full-record mean of `QWEN-POOL3` is 0.2056 on one GPU and 0.2054 on another; entailment probabilities moved by at most 4.2e-6 in a recheck |
| **Missing or moved inputs** | a file that was on one machine only, or a folder that moved | inputs are frozen in `data/frozen/` or listed with their sha256 in `data/data-root.tsv` | the closure's 538 MB of TEI files came back from a backup with their listed sha256 |

**k-means.** Step 5's clustering cannot be rebuilt to the same partition: two fits with the same seed on the same input gave different partitions (adjusted Rand index 0.986), and the partition changed with the number of threads (1 to 24 threads: 0.60 to 0.99 against the stored partition, none identical). So Step 5 reads its saved partition instead of fitting one ([docs/step-5-post-processing/postprocess.md](../step-5-post-processing/postprocess.md#pinned-cluster-map)).

**One case in numbers.** The extractor selection's secondary score of one TEST paper, `S0957417423021437`, is 0.3816 from the A100 that first scored it and 0.3365 from the H100 that rescored it, because the score pairs each reference definition with one predicted record and two pairings nearly tie. Every count, eligibility reason, audit verdict and pick is the same on both GPUs; the two run means it moves past the band are named exceptions ([docs/step-3-extraction/selection.md](../step-3-extraction/selection.md#the-cluster-job)).

## How it was checked

The stored outputs that the package can make again were rebuilt from their pinned inputs on the locked toolchain ([docs/setup/toolchain.md](toolchain.md)), the scoring models on a GPU and the rest on a CPU, and compared with them under the rules above: of the 139 items compared in one run, none differed beyond its band apart from the two named exceptions of the selection, and Step 5's outputs were compared the same way. Each step page says how its outputs were compared and what came out (for example [docs/step-3-extraction/groundtruth.md](../step-3-extraction/groundtruth.md#how-it-was-checked) and [docs/step-5-post-processing/postprocess.md](../step-5-post-processing/postprocess.md#how-it-was-checked)).

## How to adapt it

- **A new run of a step** writes new files with their own receipts; it does not replace a pinned file. To make it the reference, add its files with their sha256 to the list that pins them (`data/data-root.tsv`, a manifest, or `data/frozen/` with a new name).
- **A new band** belongs next to the number it covers, with a measured example of the difference it allows.
