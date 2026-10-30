# Documentation

The docs describe what the code does: first how to run each step on your own data, then how the paper's results were made and checked. The [README](../README.md) has the four tasks most readers want; this page is the map.

Some released files are pinned by their hash, for example MBDef-Gold's guide on Hugging Face and the README inside the closure's release folder, so their text stays as it was written and can name an older command or folder. The pages here give the current commands; [pinned.md](setup/pinned.md) lists what is pinned, and why a rebuild can differ.

## Use it on your data

| Task | Command | Page |
|---|---|---|
| Extract definition records from your papers (TEIs or PDFs) with the released extractor or another model | `extract run`, then `extract filter` | [extract.md](step-3-extraction/extract.md#run-it-on-your-papers) |
| Score any extractor's records against MBDef-Gold with DefScore | `evaluate score --predictions` | [evaluate.md](step-3-extraction/evaluate.md#score-your-extractor) |
| Train your own extractor's prompt programs on MBDef-Gold | `extract train-dataset`, `extract train` | [extract.md](step-3-extraction/extract.md#training-the-programs-extract-train) |
| Decide which of your papers are about media bias, from title and abstract | `relevance` | [relevance.md](step-1-2-papers/relevance.md#run-it-on-your-papers) |
| Find, download and convert the PDFs of a list of papers | `collect` | [collect.md](step-1-2-papers/collect.md#run-it-on-your-papers) |
| Explore MBDef and MBDef-Gold in Jupyter notebooks: lookups, clusters of definitions, the subtype network, DefScore divergence, research disciplines, taxonomy coverage | none: notebooks, with helpers in `mbdef.analysis` | [tutorials/README.md](../tutorials/README.md) |

## The pages, by paper step

The pages sit in one folder per step of the paper. Each step page has the same six parts: the paper step it belongs to, its command, its inputs, its outputs, how it was checked and how to adapt it. Every page has a worked example next to the numbers.

### Steps 1 and 2: the papers ([`step-1-2-papers/`](step-1-2-papers/))

| Page | What it covers | Command |
|---|---|---|
| [relevance.md](step-1-2-papers/relevance.md) | Which papers of the starting corpus are about media bias (two classifiers and an arbiter) | `mbdef relevance` |
| [seeds.md](step-1-2-papers/seeds.md) | The search, the starting corpus and the 3,869 seed papers | none: data files in `data/seeds/` |
| [collect.md](step-1-2-papers/collect.md) | The full texts of the papers: find, download and convert their PDFs | `mbdef collect` |

### Step 3: extraction and its evaluation ([`step-3-extraction/`](step-3-extraction/))

| Page | What it covers | Command |
|---|---|---|
| [groundtruth.md](step-3-extraction/groundtruth.md) | The ground truth, MBDef-Gold, and the publishable copies of its annotation exports | `mbdef groundtruth build`, `groundtruth pseudonymise` |
| [extract.md](step-3-extraction/extract.md) | Extraction, verifier, relevance filter, training, a corpus run in shards and its variants | `mbdef extract` |
| [evaluate.md](step-3-extraction/evaluate.md) | Scoring, the agreement numbers, the held-out TEST, the paired comparison with Opus | `mbdef evaluate` |
| [selection.md](step-3-extraction/selection.md) | The choice of the four extractors | `evaluate grid`, `evaluate select` |

### Step 4: the corpus closure ([`step-4-closure/`](step-4-closure/))

| Page | What it covers | Command |
|---|---|---|
| [closure.md](step-4-closure/closure.md) | The corpus closure: the works the seed papers cite in their extracted items, one level deep, and the release of the corpus | `mbdef closure` |

### Step 5: post-processing ([`step-5-post-processing/`](step-5-post-processing/))

| Page | What it covers | Command |
|---|---|---|
| [embed.md](step-5-post-processing/embed.md) | The embedding cache of the definition vectors | `mbdef embed build` |
| [postprocess.md](step-5-post-processing/postprocess.md) | Which corpus terms are about media bias, and the merging of the kept terms | `mbdef postprocess` |

### The dataset ([`dataset/`](dataset/))

| Page | What it covers | Command |
|---|---|---|
| [dataset.md](dataset/dataset.md) | MBDef, the released dataset: tables, dataset card, JSON copy; and MBDef-Gold's split files and card | `mbdef dataset` |
| [explorer.md](dataset/explorer.md) | The MBDef Explorer, a web page of the dataset, its page data, and how to publish it on GitHub Pages | `mbdef explorer build`, `explorer data` |

### Setup and shared pieces ([`setup/`](setup/))

| Page | What it is |
|---|---|
| [data-root.md](setup/data-root.md) | The machine settings (`.env`) and the data root that holds what git does not |
| [data-fetch.md](setup/data-fetch.md) | `mbdef data fetch`: fill the data root and check every file |
| [cluster.md](setup/cluster.md) | `mbdef cluster`: run a step's command as a job on a Slurm GPU cluster, if you have one |
| [toolchain.md](setup/toolchain.md) | The locked Python, packages and tools, and how to refresh them |
| [pinned.md](setup/pinned.md) | What is pinned (by hash, version or constant), and why a rebuild can still differ |
| [llm.md](setup/llm.md) | The language-model layer that several steps share (a library, not a command): requests, recorded answers and their replay |

## Elsewhere

- [APPENDIX.md](../APPENDIX.md): the parts of the paper's appendix that did not fit its pages (the comparison with Claude Opus, the corpus closure's numbers, the pair scorers on DefSim), each with what reproduces it.
- [CONTRIBUTING.md](../CONTRIBUTING.md): the checks to run before a change goes in, and the link check of the docs (`mbdef repo check-links`).
- [GLOSSARY.md](../GLOSSARY.md) defines the words of the project (Seed Corpus, Extracted Concept, Media Bias Concept, ...). Use them as written.
