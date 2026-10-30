# Postprocess: `mbdef postprocess`

| | |
|---|---|
| **Paper step** | Step 5 (post-processing): which of the corpus's Extracted Concepts are about media bias, so that the dataset keeps them, and the merging of the kept ones by Label Group |
| **Code** | [`src/mbdef/postprocess/`](../../src/mbdef/postprocess/): `store.py` (the corpus store), `label_match.py` and `label_groups.py` (Label Groups), `paths/` (the relevance paths), `evaluate.py` (the shared evaluation) and `compare.py` (the train + dev comparison), `testrun.py` (the TEST split), `decide.py` (one path's keep or drop), `ensemble.py` (the adopted keep or drop), `basal.py` (the ensemble member basal's items, client and score tables), `judge.py` (the member GT-RAG judge's packages, client and score tables), `term_merging.py` (the merging), `extractor_gate.py` (the extractor's own filter on the ground truth); the parser of every action is in `cli.py` |
| **Commands** | `mbdef postprocess <action>`: `build-store`, `label-groups`, `evaluate`, `decide`, `ensemble`, `merge-terms`, `freeze`, `test`, `extractor-gate`, `basal items`, `basal ask`, `basal table`, `judge package`, `judge ask`, `judge table`; `postprocess <action> --help` lists each action's flags |
| **Checked by** | the recorded files, which the commands rebuild byte for byte ([How it was checked](#how-it-was-checked)); `postprocess build-store --compare` (the corpus store, [below](#the-corpus-store)); the tests `tests/test_postprocess_*.py` |

The repository holds Step 5's code and the files that the commands below read or that the tests compare with: in git under `data/frozen/step5-post-processing/<date>/`, and those over 10 MB at the same path under `$DATA_ROOT/frozen/` (`mbdef data fetch` gets them; [Inputs](#inputs)). The saved TEST result and its read log are not part of this repository ([the TEST split](#the-test-split)).
Every command writes only to the folder its `--out` names and refuses a folder inside the frozen ones.

Two rules hold throughout: the TEST split is compared by hash unless someone asks for a logged read, and what makes a rerun differ is pinned rather than given a tolerance. A path name ending in **@group** is the path in the *Label Group setting*: an Extracted Concept is scored through its whole Label Group (path 1, clustering: the mean over all the group's definitions; the entailment paths: the group's Representative Definition), so every member of a group gets the same score.

## What the step does

The corpus (release v1.1, its P≥28 variant: the extractor's records after the Qwen verifier and the relevance filter, [docs/step-3-extraction/extract.md](../step-3-extraction/extract.md#the-seed-runs-variants-extract-derive), for the seed and the closure papers) holds 67,394 Extracted Concepts: a term as one paper defines it, with 110,399 definitions in all. The files call them terms. Many are not about media bias (for example "deep learning" in a paper that uses it). Step 5 decides, for every Extracted Concept, whether the dataset keeps it, and then merges the kept ones whose labels match. It reads the corpus from the corpus store, one SQLite file that `build-store` makes from release v1.1 ([below](#the-corpus-store)).

1. **Label Groups** (`label-groups`, Step 5.1). Extracted Concepts whose labels match under the Label Match rule (the same words once case, word order, stop words and plurals are set aside) form one Label Group. Two versions of the rule are kept ([below](#two-versions-of-the-label-match-rule)): the first, whose 38,467 groups the frozen train + dev evaluations used, and the adopted one, whose 38,407 groups the term merging uses (their stored names, which `--rule` takes, are `C-20261015` and `C-20261021`).
2. **Relevance paths** (`evaluate`, Step 5.2-5.3). Several ways of scoring an Extracted Concept were compared on the ground truth's train and dev records (1,700 records, 693 relevant), each scored out of fold: a record's score comes from a model that never saw the record's paper. Path 1 (clustering) and path 2 (entailment against a fixed sentence or the Representative Media Bias Definition) are the paper's; the embedding classifier, the nearest-neighbour vote and the keyword rule are comparison rows.
3. **The TEST split** (`test`, Step 5.3). Every frozen method was read once on the 628 test records, and one logged re-analysis compared them all. The rule fixed before the last reads was: a method replaces the embedding classifier only if the 95% interval of its paired test AUC difference lies entirely above 0. One did: an **ensemble** of the classifier, the decision model basal and the retrieval judge (the GT-RAG judge, below), test AUC 0.9334 against the classifier's 0.9228, a difference of +0.0107 (95% interval +0.0007 to +0.0240; 12 methods were compared, and the intervals are not adjusted for that). It is the adopted method.
4. **Keep or drop** (`ensemble`, Step 5.4). The ensemble scores every Extracted Concept of the corpus and keeps those at or above the cut that keeps 95% of the relevant train + dev records (below).
5. **Term merging** (`merge-terms`: merge after the decisions, so the dataset holds no duplicates). The kept Extracted Concepts are merged by Label Group (the adopted Label Match rule): each merged row is a Label Group restricted to its kept members (a Label Group only says that labels match; it is not a Media Bias Concept). Every row keeps its provenance: each member's keys and the normalisation steps that changed its label, the keys that join the row's labels, and flags for joins a person may want to check (word order, spacing, a dropped bracket, a chosen plural).
   Labels kept in some papers and dropped in others are set aside for a look, with the Label Groups where a majority vote would decide differently.

### The ensemble

| Member | What it is | Where its scores come from |
|---|---|---|
| Embedding classifier | Logistic regression on the mean Qwen3-Embedding-8B vector of an Extracted Concept's definitions | runs here: `postprocess evaluate --paths classifier` (train + dev, out of fold) and `postprocess decide --path classifier` (corpus) |
| basal-omni-1.0-ultra, FP8 | An LLM decision model: P(core) + P(satellite) of its answer | frozen score tables, made on H100 and H200 GPUs; `postprocess basal` remakes them from the recorded answers on a CPU, and asks the model again on a GPU ([below](#rerun-basal)) |
| GT-RAG judge (ground-truth retrieval-augmented judge) | Gemma-4-12B (quantisation-aware trained weights) judging each definition, shown the 32 most similar labelled ground-truth definitions (found with Qwen3-Embedding), thinking off, temperature 0; P(yes) + 0.5 P(uncertain), the best of its definitions | frozen score tables, made on H200 GPUs; `postprocess judge` remakes them from the recorded runs on a CPU, and asks the model again on a GPU ([below](#rerun-the-judge)) |

An Extracted Concept of the corpus is scored in three steps ("reference ranks", option (a) of the question where to cut):

1. each member's score is placed on that member's own train + dev scores: its *reference rank* is the share of the 1,700 reference scores below it, ties counted half. The score therefore never depends on the other Extracted Concepts scored with it. (The frozen recipe ranked each batch within itself, option (b); on the corpus, where far more are relevant than in the ground truth, its cut would keep a fixed 56%, so (a) was chosen.)
2. the ensemble score is the weighted mean of the three reference ranks, with the frozen weights (0.931, 0.918 and 0.855, that is 34.4%, 33.9% and 31.6% of their sum; a logistic stacking with C 0.01, fitted on the members' train + dev ranks);
3. it is kept when its ensemble score is at least **0.4150**: the cut that keeps 95% of the relevant train + dev records when each fold is ranked against the other folds' scores. Basal's and the judge's references are their train + dev runs on the same GPU type as their corpus runs (H200).

**Worked example** (`decisions/ensemble.csv`, a seed paper's Extracted Concept *Deep learning*). Its scores are 0.210 (classifier), 0.172 (basal) and 0.005 (judge). Against each member's 1,700 train + dev scores these lie at the reference ranks 0.428, 0.400 and 0.311. The ensemble score is 0.344 × 0.428 + 0.339 × 0.400 + 0.316 × 0.311 = 0.381, below the cut 0.4150, so it is dropped. The classifier alone kept it (0.210 is above its own cut, 0.1848). *framing* in another seed paper scores 0.971, 1.000 and 1.000, ranks 0.955, 0.998 and 0.875, ensemble score 0.944: kept.

**Counts** (`decisions/ensemble.json`, `term-merging/summary.json`): the ensemble keeps 62,066 of the 67,394 Extracted Concepts (92.1%; the classifier alone kept 62,084), with 103,476 of the 110,399 definitions and 6,115 of the 6,241 papers. Of the 38,407 Label Groups of the adopted Label Match rule, 766 are split by the decisions. The term merging leaves 34,916 merged Label Groups (28,210 of them with one Extracted Concept); a majority vote would decide 691 Extracted Concepts the other way (they are set aside).

**Which rule version each count uses.** `ensemble.json` takes its rows and their Label Groups from `20261015/label-groups/terms.jsonl` (the first version of the Label Match rule), so its `label_groups` counts (all kept, all dropped, split: 764 split of 38,467) are by the first version's groups; the decisions themselves do not depend on the groups. The term merging and its `summary.json` use the groups of the adopted version (766 split of 38,407). The two numbers differ only because the adopted version's plural fix moves 64 groups.

**Worked example of a split group** (`term-merging/split-labels.csv`). The Label Group `g00d5e71fe5b3` (key "aggregation based matrix news") holds *Matrix-based news aggregation (MNA)* in one paper, ensemble score 0.409 (below 0.4150: dropped), and *matrix-based news aggregation (MNA)* in another, 0.509 (kept). Its merged row holds only the kept one, and both are set aside in `split-labels.csv` for a person to look at.

### Two versions of the Label Match rule

The first version's plural step tries the "-es" stem before the "-s" stem, so "refugees" becomes "refuge" wherever "refuge" is a word of the labels (a plural audit found these cases). The adopted version fixes this: for a word in "-es" the "-s", "-es" and "-ses → -sis" cuts compete, the one that ends more labels wins (then the one in more labels), and a function word ("them", "not", "us") never does; irregular plurals, the "-ics" words that are plurals and a list of British spellings come from fixed lists. *refugees* is now joined with *refugee*: "refugee" ends 2 labels and occurs in 4, "refuge" ends 1 and occurs in 1 (`20261021/label-groups/labels.jsonl`). Against the first version, 64 groups change (62 merges, 2 splits) and the groups go from 38,467 to 38,407.

Both versions stay, by their stored names (`label_match.RULE_VERSIONS`). The adopted version, `C-20261021`, is the default (`label-groups`, the term merging, the dataset). The first version, `C-20261015`, remakes its frozen Label Groups byte for byte (`label-groups --rule C-20261015`), which the Label Group setting's evaluations, the paths' corpus decisions in `20261015/` and the frozen paths' TEST read used, so `test --read` pins it and `decide --rule C-20261015` takes it. The ensemble does not depend on the groups: it scores each Extracted Concept on its own.

### The corpus store

`postprocess build-store` writes the corpus store: the four tables of release v1.1's extractions that Step 5 reads, in one SQLite file. For each corpus, the 3,869 seed papers first and then the 4,556 closure papers, it reads the four variants of `extract derive` (P≥20, P≥24, P≥28, and P≥28 with Gemma alone; [docs/step-3-extraction/extract.md](../step-3-extraction/extract.md#the-seed-runs-variants-extract-derive)) in the order of the corpus's `derived/DONE.json`.
Each variant's `predictions.json` must have the sha256, the records and the definitions that `DONE.json` lists, or the command stops before it writes anything.

| Table | One row per | Rows from release v1.1 |
|---|---|---|
| `papers` | paper: its id, title, DOI and corpus | 8,425 (3,869 + 4,556) |
| `mentions` | term as one paper names it, over the eight variants: the spellings that give one key in a paper (case, punctuation, a leading article and an acronym in brackets set aside) are one mention, which keeps the first spelling seen and the variant that first had it | 147,461 |
| `occurrences` | distinct text (whitespace collapsed) in one field of one mention (definition, description, synonym, related term, subtype, supertype or example), with the citations of its first item | 524,907 |
| `occurrence_sources` | item of a variant that gave that text: the variant, the record and item number, the text and citations as written | 1,327,091 |

A row's id is a hash of what identifies it, so the same release always gives the same ids. Step 5's view (`corpus.load_view`) takes the seed papers and the closure papers that the definition links mark P≥28, and of their mentions the definitions that a P≥28 variant gave: 67,394 Extracted Concepts with 110,399 definitions.

**Worked example** (the seed paper `pdf-d3fc68cf…`, DOI 10.58256/kq6m6380). Its P≥20 output names one term twice, as *Critical discourse analysis (CDA)* (record 7) and as *Critical discourse analysis* (record 40). Both give the key `critical discourse analysis`, so they are one mention, `m058087013650863a` ("m" and the first 16 hex digits of the SHA-1 of the paper id and the key). It keeps the first spelling, *Critical discourse analysis (CDA)*, the acronym `CDA` and `seeds:P20` as where it was first seen. Its one definition ("CDA approaches discourse as a site where social power is produced and resisted, …") is one occurrence, `o07e59dedd64e13a4`, with five rows in `occurrence_sources`: P≥20 records 7 and 40, P≥24 record 9, P≥28 record 7 and P≥28-single-model record 7. A P≥28 variant gave it, so Step 5's view holds this Extracted Concept with this one definition.

The stored file, `$DATA_ROOT/frozen/artifacts/experiments/taxonomy-curation-v2/library.sqlite3` (1.46 GB), was built by the ingest code that first read the corpus release into a database, the seed papers first and then the closure papers; `build-store` is that ingest reduced to these four tables. The stored file has more tables, which Step 5 does not read and `build-store` does not write: the paragraph of the paper each definition was found in (`contexts`; paper text, which is not public), the list of the eight variants (`sources`; the receipt lists them instead), each record's raw spelling (`mention_sources`), the links between evidence texts (`occurrence_links`), and later curation tables. A rebuilt store is 562 MB.

## Run it

```bash
uv run mbdef data fetch frozen/step5-post-processing          # the frozen files over 10 MB, into DATA_ROOT
uv run mbdef data fetch frozen/closure/definition-corpus-v1.1    # release v1.1, for build-store
uv run mbdef postprocess build-store --out runs/store         # library.sqlite3 (CPU, about 40 s)
uv run mbdef postprocess build-store --out runs/store --overwrite \
  --compare "$DATA_ROOT/frozen/artifacts/experiments/taxonomy-curation-v2/library.sqlite3"   # and compare, row for row
uv run mbdef postprocess ensemble --out runs/ensemble          # ensemble.{csv,json}
uv run mbdef postprocess merge-terms --decisions runs/ensemble/ensemble.csv --out runs/merged
uv run mbdef postprocess test                                  # compares test-eval.json by hash; reads no TEST record
uv run mbdef postprocess label-groups --out runs/label-groups  # the adopted Label Match rule; --rule C-20261015 for the first
uv run mbdef postprocess evaluate --paths classifier,clustering,keyword --out runs/evaluation
uv run mbdef postprocess decide --path classifier --rule C-20261015 --out runs/decisions
```

Every input has a flag whose default is the frozen file of the adopted run. `build-store` takes `--release` (default: release v1.1 in the data root), `--overwrite` and `--compare STORE`, which compares the four tables with those of STORE row for row and exits with 1 if one differs. `label-groups` reads a rebuilt store with `--store runs/store/library.sqlite3`; the other commands read the stored one. `ensemble` takes `--pick`, `--weight-table MEMBER=FILE`, `--reference MEMBER=FILE`, `--corpus-scores MEMBER=FILE` (each repeatable; a member not named keeps its frozen file), `--terms`, `--classifier-decisions`, `--recall` (0.95) and `--sanity LOW HIGH` (0.70 1.00); `merge-terms` takes `--decisions` (default: the adopted ensemble's decisions) and `--label-groups` (default: the Label Groups of the adopted Label Match rule). `evaluate`, `decide` and `test --read` take `--inputs`, the date folder of saved scores they read (default: the frozen folder `20261015/`). Only the entailment paths (`zero-shot`, `representative`) and `extractor-gate` need a GPU, and `basal ask` and `judge ask` need their model's server, which runs on a large GPU ([Rerun basal](#rerun-basal), [Rerun the judge](#rerun-the-judge)).

### Rerun basal

basal (`Remek/basal-omni-1.0-ultra-FP8` on Hugging Face, revision `748350981416ebad4edf3cd1eb22354fde788e6f`, 35.3 GB of weights) is a language model that answers a typed question about a *state*. Here the state is the term and its first 3 definitions, each with its whitespace collapsed, U+FFFD (a character the PDF conversion could not read) removed, and cut to 599 characters + "…" when longer than 600. basal is asked whether the term is *core* (a type of bias), *satellite* (not a type of bias, but a concept needed to describe, detect or explain bias) or *irrelevant*, and returns the probability of each; the score is P(core) + P(satellite). **Worked example** (the recorded corpus run): the first corpus item, `m0000fd9a4a4afc72` ("observer strategies" in a seed paper), got 0.136, 0.583 and 0.281, so its score is 0.136 + 0.583 = 0.719.

`postprocess basal` has three steps. Only `ask` needs the model; `items` and `table` run on a CPU.

```bash
uv run mbdef postprocess basal items --out runs/basal   # items.jsonl (67,394 corpus terms), groundtruth-items.jsonl (1,700 records)
# start basal's server (below); then, from a machine that reaches it:
uv run mbdef postprocess basal ask --items runs/basal/groundtruth-items.jsonl --out runs/basal/groundtruth-answers.jsonl --limit 24  # a quick trial
uv run mbdef postprocess basal ask --items runs/basal/groundtruth-items.jsonl --out runs/basal/groundtruth-answers.jsonl
uv run mbdef postprocess basal ask --items runs/basal/items.jsonl --out runs/basal/answers.jsonl
uv run mbdef postprocess basal table --items runs/basal/groundtruth-items.jsonl --answers runs/basal/groundtruth-answers.jsonl --out runs/basal/reference.csv
uv run mbdef postprocess basal table --items runs/basal/items.jsonl --answers runs/basal/answers.jsonl --out runs/basal/corpus.csv
uv run mbdef postprocess ensemble --reference basal=runs/basal/reference.csv --corpus-scores basal=runs/basal/corpus.csv --out runs/ensemble
```

- `items` writes the corpus's Extracted Concepts (from the corpus store, in mention-id order) and the ground truth's 1,700 train + dev records, the reference the corpus scores are ranked against. It writes no TEST record.
- `ask` sends 32 requests at a time (`--concurrency`), in the order `random.Random(20261021)` shuffles the items to (`--order-seed`; so the server's batches mix items from all over the file), to the server's `/v1/systemone` (`--server`, default `http://127.0.0.1:8000`). It appends each answer to `--out` as it comes back, so the same command run again goes on where a run stopped. A request that fails is tried 5 times; an answer whose probabilities are not the three options summing to 1 is written as an error line and asked again by the next run; the command exits 1 while an item has no answer.
- `table` writes one row per item, in the items' order, from the first answer of each item (two servers may answer the same item), with the method name `--method` (or `--method-of` an existing table). It refuses while an item has no answer.
- `ensemble` then ranks the new corpus scores against the new reference. The weights stay the frozen ones (`pick.json`), fitted on basal's H100 train + dev table (`--weight-table`); new weights need a new weights file and a new TEST read.

**The server.** The recorded runs served basal with the server of the `basal` package (release v2.2.0) on vLLM 0.31.0, one GPU per server: an NVIDIA H200 (141 GB) for the reference and the corpus, an H100 (80 GB) for the train + dev table that the weights were fitted on. On a 48 GB GPU the server needs `--max-len 8192` (the states are under 1,000 tokens). These lines restate that setup; they were not run for this page, which had no such GPU:

```bash
uv venv --python 3.12 basal-env
uv pip install --python basal-env/bin/python "basal @ https://github.com/rkinas/basal/archive/refs/tags/v2.2.0.tar.gz" "vllm==0.31.0"
uvx hf download Remek/basal-omni-1.0-ultra-FP8 --revision 748350981416ebad4edf3cd1eb22354fde788e6f --local-dir basal-fp8
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 \
  basal-env/bin/python -m basal.server --model basal-fp8 --mode vllm --max-len 16384 --gpu-memory 0.9 --host 127.0.0.1 --port 8000
curl -sf http://127.0.0.1:8000/health   # answers once the model is loaded (160 s on the H200)
```

The four `VLLM_…` settings switch off kernels that vLLM compiles the first time it needs them (those builds failed on the recorded machines); vLLM then uses its precompiled FP8 kernels for the same multiplications, and as every request reads probabilities rather than sampling a token, the sampler kernel does not change an answer. On the H200 the 2,328 ground-truth records took 31 s and the 67,394 corpus items 983 s (68.6 items a second); with loading, the whole job took about 20 minutes.

A rerun on other hardware moves the scores a little: on the ground truth, H200 against H100 changed 2.2% of basal's classes and its AUC by +0.00003. **Worked example:** the train record `4d65bab6…:0:19` gets basal's score 0.371 on the H100 (`20261020/decision-models/scores/basal-fp8-h100-r1.csv`) and 0.421 on the H200 (`20261021/corpus-decision-models/scores/basal-fp8-h200-gt-r1.csv`); over the 1,700 train + dev records the median difference is 0.004 and the largest 0.235. `postprocess ensemble` on new scores then gives slightly different decisions; a comparison with the frozen `decisions/ensemble.csv` shows which.

**The recorded tables from the recorded answers.** On a CPU, the commands rebuild the three frozen basal tables byte for byte from the answers the runs recorded (in git under `data/frozen/`, except the corpus answers, which are in the data root: `data fetch frozen/step5-post-processing/20261021/corpus-decision-models`). The recorded ground-truth runs asked all 2,328 records, the 628 TEST records too, for the ensemble's one logged TEST read; their tables are rebuilt from the recorded item list `data/frozen/results/core-satellite-decision-v1/items.jsonl`, whose TEST rows the table copies like any other row (no label is read). `--method-of` takes each table's method name, which names its run.

| Table (under `data/frozen/step5-post-processing/`) | `--items` | `--answers` (the recorded run) | sha256 |
|---|---|---|---|
| `20261021/corpus-decision-models/scores/basal-fp8-h200-corpus.csv` (corpus, 67,394 rows) | `items.jsonl` written by `items` (sha256 `9806c9e9…`, the items the run asked) | `<data root>/frozen/step5-post-processing/20261021/corpus-decision-models/out/basal-fp8-h200-r1-s1of1.jsonl` (H200) | `61ec2a48…` |
| `20261021/corpus-decision-models/scores/basal-fp8-h200-gt-r1.csv` (reference, 2,328 rows) | the recorded item list | `20261021/corpus-decision-models/out/basal-fp8-h200-gt-r1.jsonl` (H200) | `89e4ccb7…` |
| `20261020/decision-models/scores/basal-fp8-h100-r1.csv` (the weights' table, 2,328 rows) | the recorded item list | `20261020/decision-models/verdicts/basal-fp8-h100-r1.jsonl` (H100) | `ad514ef6…` |

For example, the corpus table:

```bash
uv run mbdef postprocess basal table --items runs/basal/items.jsonl \
  --answers <data root>/frozen/step5-post-processing/20261021/corpus-decision-models/out/basal-fp8-h200-r1-s1of1.jsonl \
  --method-of data/frozen/step5-post-processing/20261021/corpus-decision-models/scores/basal-fp8-h200-corpus.csv \
  --out runs/basal/corpus-check.csv   # 67394 rows (0 error lines skipped), sha256 61ec2a48…
```

### Rerun the judge

The GT-RAG judge (`google/gemma-4-12B-it-qat-w4a16-ct`, revision `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee`, served by vLLM) is shown a paper's title, up to 4 of its *statements* (a term and one of its definitions, shown as c1 to c4 so that an id never shows a label) and, for each statement, the 32 most similar train records of the ground truth with their labels (relevant or not), and answers for each statement whether it is a definition and whether it is relevant to a media-bias taxonomy (yes, no or uncertain), as JSON in a fixed schema, thinking off, temperature 0. A statement scores P(yes) + 0.5 P(uncertain) of its relevance answer, read from the log-probabilities of the same request; a term (or a ground-truth record) scores its highest statement. **Worked example** (the recorded corpus run): *observer strategies* (`m0000fd9a4a4afc72`) has 3 definitions, each answered "yes", with P(yes) and P(uncertain) of (0.94620, 0.02858), (0.99994, 0.00003) and (0.99960, 0.00011); they score 0.960, 0.99996 and 0.99965, so the term scores 0.99996.

```bash
uv run mbdef postprocess judge package --out runs/judge   # CPU, about a minute: package.zip (corpus), groundtruth-package.zip (train + dev)
# start the judge's server (below); then, from a machine that reaches it, one run per shard (or all at once):
uv run mbdef postprocess judge ask --package runs/judge/groundtruth-package.zip --out runs/judge/train-dev
uv run mbdef postprocess judge ask --package runs/judge/package.zip --split shard0 --out runs/judge/corpus-shard0   # ... shard7
uv run mbdef postprocess judge table --package runs/judge/groundtruth-package.zip --runs runs/judge/train-dev --out runs/judge/reference.csv
uv run mbdef postprocess judge table --package runs/judge/package.zip --runs runs/judge/corpus-shard* --out runs/judge/corpus.csv
uv run mbdef postprocess ensemble --reference judge=runs/judge/reference.csv --corpus-scores judge=runs/judge/corpus.csv --out runs/ensemble
```

- `package` writes the corpus package (every definition of the corpus view's 67,394 terms: 110,399 statements of 6,241 papers, in 8 shards balanced by statements, 30,016 requests) and the ground truth's (its 1,700 train + dev records: 1,967 statements of 118 papers, shuffled per paper with the seed 20261016, 532 requests; no TEST record), each with a request list. A statement's neighbours are the 32 train records (1,420, from 95 papers) with the highest inner product of the cached Qwen3-Embedding-8B vectors of "term: definition" ([docs/step-5-post-processing/embed.md](embed.md)), a record scoring its best definition, at most one per paper, never from the statement's own paper (for a corpus paper, nor from the train paper with the same title: 21 corpus papers are ground-truth papers). The search is faiss's exact inner product, the corpus 4,096 statements at a time and the ground truth one at a time, as the recorded packages were made: the two ways round the cosines differently in their last bits, which the packages keep.
- `ask` sends each request to the server's `/v1/chat/completions` (`--server`, default `http://127.0.0.1:8000`), 128 at a time (`--concurrency`), with the messages of DSPy's JSON adapter, the answer's JSON schema, seed 9, at most 4,096 tokens and 20 log-probabilities per token; it checks each answer (one finished choice, no repeated key, the schema; the read-out answer equal to the verdict) and appends one line per request to `answers.jsonl`, so the same command run again sends only what has no answer; it exits 1 while a request has no answer, and writes `decisions.json` once every request has one. The recorded runs sent the longest prompts first (that needs the model's tokenizer); `ask` sends them in package order, which changes only which requests share a batch on the GPU.
- `table` writes the corpus table (every statement judged exactly once) or the ground truth's (only train and dev records), in the shared score-table format.

**The server.** The recorded runs used vLLM 0.32.0 (the version `pyproject.toml` pins), one NVIDIA H200 (141 GB) per server: the train + dev run took 591 s on one GPU, each corpus shard about 4,150 s (3,751 requests, 85 million prompt tokens; the longest prompt 28,008 tokens), 8 shards on 4 GPUs. These lines restate that setup; they were not run for this page, which had no such GPU:

```bash
uv run vllm serve google/gemma-4-12B-it-qat-w4a16-ct --revision 1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee \
  --language-model-only --host 127.0.0.1 --port 8000 --enable-prompt-tokens-details --max-model-len 106496 \
  --reasoning-parser gemma4 --max-num-seqs 256 --gpu-memory-utilization 0.90 \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
```

**The recorded tables from the recorded runs.** On a CPU, `table` rebuilds both frozen judge tables byte for byte from the recorded runs' `decisions.json` (in the data root, `frozen/step5-post-processing/20261020/judge-sweep/runs/`: the train + dev run `s1-qwen8b-k32-think0` and the 8 corpus shards, 2.26 GB with the requests and raw answers), with the packages `package` writes; and `package` gives the recorded corpus package byte for byte:

| Output | From | sha256 |
|---|---|---|
| `package.zip`'s `inputs.json` | `package` (the corpus store, the ground truth, the embedding cache) | `4c58203f…`, the recorded corpus package |
| `requests.jsonl` | `package` | `ab3d6efe…` |
| `20261020/judge-sweep/corpus/20261021/judge-corpus-scores.csv` (corpus, 67,394 rows) | `table` on the 8 corpus shards | `50646363…` |
| `20261020/judge-sweep/stage1/20261021/scores/s1-qwen8b-k32-think0-prob.csv` (train + dev, 1,700 rows) | `table` on the train + dev run | `c6245bcb…` |

The requests themselves come back too: for every one of the 532 train + dev and 30,016 corpus requests, the hash of the model name, messages, settings and answer schema that the recorded runs logged equals the hash of the request `ask` sends (`tests/test_postprocess_judge.py`).

## Inputs

All `<date>/…` paths below are under `data/frozen/step5-post-processing/` in git; "(data root)" marks a file over 10 MB, which is at `$DATA_ROOT/frozen/step5-post-processing/<date>/…` instead. `data fetch` gets those (`data/data-root.tsv` lists each with its sha256 and where it comes from).

| Input | Where | What it is |
|---|---|---|
| The ground truth | MBDef-Gold's `$DATA_ROOT/MBDef-Gold/validated_gt.json` and `validated_gt_non_relevant.json` (`data fetch MBDef-Gold/`; `postprocess judge package` takes another folder with `--ground-truth-folder`) | the relevant and the non-relevant records, with each paper's split |
| The vote tallies | `20261015/gt-label-quality/records.csv` | how many annotators called each record relevant |
| The corpus store | `$DATA_ROOT/frozen/artifacts/experiments/taxonomy-curation-v2/library.sqlite3` | every extracted term and definition of release v1.1 (`build-store` makes the four tables Step 5 reads again, [above](#the-corpus-store)) |
| Release v1.1's extractions | `$DATA_ROOT/frozen/closure/definition-corpus-v1.1/{seeds,closure}/derived/` (`DONE.json` and `<variant>/predictions.json`) | what `build-store` reads (it reads no paper text) |
| The P≥28 view | `$DATA_ROOT/frozen/closure/definition-corpus-v1.1/links/definition-links.csv` (in release v1.1's folder) | which closure papers belong to the view |
| The embeddings | `$DATA_ROOT/embeddings/qwen3-emb-8b/` | the Qwen3-Embedding-8B vector of every definition ([docs/step-5-post-processing/embed.md](embed.md)) |
| The Label Groups of the first Label Match rule (`C-20261015`) | `20261015/label-groups/` (data root) | `terms.jsonl` (each Extracted Concept and its group; the ensemble's row order) and `groups.jsonl` (each group with its definitions) |
| The Label Groups of the adopted Label Match rule (`C-20261021`) | `20261021/label-groups/` (`terms`, `groups`, `labels.jsonl` in the data root; `summary.json`, `run.json` in git) | the groups the term merging uses, with each label's provenance |
| The Representative Definitions | `20261015/group-representatives/representatives.jsonl` (data root) | each Label Group's Representative Definition, which the entailment paths' Label Group setting scores (a frozen input; the command that chose them is not part of this repository) |
| Path 1's cluster map | `20261015/clusters/points.npz` | each corpus definition's cluster, saved with the frozen path ([below](#pinned-cluster-map)) |
| The extractor's filter | `data/frozen/results/semantic-title-relevance-v2.zip` (through `extract/relevance.py`); its scores on the ground truth in `20261015/extractor-gate/gate-scores.csv` | the relevance filter of the extraction run, replayed on the ground truth (`extractor-gate`) |
| The frozen paths and the TEST result | `20261015/test-evaluation/` | `frozen-paths.json`; the TEST result `test-eval.json` and the read log `test-reads.json` are not part of this repository |
| The paths' frozen train + dev scores | `20261015/evaluation/scores/<path>.csv` (classifier, clustering, the entailment paths and their Label Group setting) | the out-of-fold scores each path's corpus cut is set on (`decide`), and the classifier's train + dev table of the ensemble |
| The classifier's corpus decisions | `20261015/decisions/classifier.csv` | made by `decide`; the ensemble reads its scores |
| The frozen weights | `20261020/ensembles/pick.json` | the members, the combiner, the weights |
| basal's scores | `20261021/corpus-decision-models/scores/` (corpus and H200 train + dev), `20261020/decision-models/scores/basal-fp8-h100-r1.csv` (H100, the weights) | see [Rerun basal](#rerun-basal) |
| basal's recorded answers | `20261021/corpus-decision-models/out/basal-fp8-h200-r1-s1of1.jsonl` (corpus, data root), `20261021/corpus-decision-models/out/basal-fp8-h200-gt-r1.jsonl`, `20261020/decision-models/verdicts/basal-fp8-h100-r1.jsonl` | one line per item `{id, p: {core, satellite, irrelevant}, choice, model, repeat}`: what `postprocess basal table` rebuilds the three tables from |
| The recorded runs' ground-truth items | `data/frozen/results/core-satellite-decision-v1/items.jsonl` | the 2,328 items `{id, split, state}` the two ground-truth runs asked, TEST records included |
| The judge's recorded runs | `20261020/judge-sweep/runs/` (data root, 2.26 GB): `s1-qwen8b-k32-think0/` (train + dev) and `corpus/s1-qwen8b-k32-think0-shard0/` to `shard7/` | each run's `decisions.json` (every statement's verdict with its relevance probabilities), `requests-inputs.json`, `requests.jsonl` (the logged hash of every request), `raw-responses.jsonl`, `config.json`: what `postprocess judge table` rebuilds the two judge tables from |
| The judge's scores | `20261020/judge-sweep/stage1/20261021/scores/s1-qwen8b-k32-think0-prob.csv` (train + dev), `20261020/judge-sweep/corpus/20261021/judge-corpus-scores.csv` (corpus) | see the table above |

[`data/data-root.tsv`](../../data/data-root.tsv) lists every file under `DATA_ROOT` with its sha256 and source. The models of the entailment paths (and BGE-base for the extractor's filter) are read from the Hugging Face cache at their pinned revisions, the constants `NLI_MODEL`, `NLI_REVISION`, `BGE_MODEL` and `BGE_REVISION` in [`evaluate/metric.py`](../../src/mbdef/evaluate/metric.py) (the entailment model is the one of the extraction score DefScore, the paper's pair score of word overlap, embedding similarity and entailment). Step 5 scores entailment in batches of 128 pairs (`nli.BATCH_SIZE`), as every Step 5 run did; the batch size changes only float noise.

## Outputs

Each command writes into the folder `--out` names:

| File | Written by | What it is |
|---|---|---|
| `library.sqlite3` (562 MB for release v1.1), `receipt.json` | `build-store` | the corpus store's four tables; the receipt lists every file read with its sha256, the row counts and, with `--compare`, the result of the comparison |
| `ensemble.csv` (16.9 MB for the corpus) | `ensemble` | one row per corpus term: its members' scores and reference ranks, the ensemble score, and keep (`True` or `False`) |
| `ensemble.json` | `ensemble` | the cut, the weights, the inputs' sha256, the counts kept (terms, definitions, papers, seed and closure papers, Label Groups all kept, all dropped or split), the same counts for the within-corpus variant (b) and for the classifier alone, and the sanity rule |
| `terms.jsonl`, `groups.jsonl` (with `joined_by`), `labels.jsonl` (each label's keys and normalisation steps; the adopted version only), `summary.json`, `run.json` | `label-groups` | the Label Groups |
| `concepts.jsonl` (41.2 MB for the corpus) | `merge-terms` | one row per merged Label Group: its labels and how often each is used, `joined_by`, `merge_flags`, and its kept Extracted Concepts with their keys, normalisation steps and definitions |
| `split-labels.csv` | `merge-terms` | set aside: every term of a Label Group that the decisions split (some kept, some dropped), with its scores |
| `majority-disagreements.csv` | `merge-terms` | set aside: the terms a majority vote of their group would decide the other way |
| `summary.json` | `merge-terms` | counts before and after, the merge flags, and the check of the Label Match rule rerun on the kept labels alone |
| `scores/<path>.csv`, `metrics.json`, `curves.json` | `evaluate` | the train + dev comparison |
| `<path>.csv`, `<path>.json` | `decide` | one path's keep or drop of every corpus term, and its cut and counts |
| `frozen-paths.json` | `freeze` | the frozen paths' settings and evidence |
| `test-eval.json`, `test-scores-<path>.csv`, `test-reads.json` | `test --read` | a TEST read: its result, the scores and the read log (the earlier reads plus this one) |
| `gate-scores.csv`, `gate.json` | `extractor-gate` | the extractor's filter on the ground truth (train + dev; every split with `--read-test`) |
| `items.jsonl`, `groundtruth-items.jsonl`, `receipt.json` | `basal items` | what basal is asked about: one `{id, split, state}` per corpus term (22.8 MB for the corpus) and per train + dev record; the receipt records `--max-definitions`, `--max-chars` and the inputs' sha256 |
| the answers file (`--out`) | `basal ask` | one line per answer, appended as it comes back (14.9 MB for the corpus) |
| the score table (`--out`) | `basal table` | `method, unit, item_id, split, score`: P(core) + P(satellite) of each item |
| `package.zip`, `requests.jsonl`, `groundtruth-package.zip`, `groundtruth-requests.jsonl`, `receipt.json` | `judge package` | the request packages (one `inputs.json` each: the papers and their statements, the train records, each statement's 32 neighbours; 51 MB zipped for the corpus) and one line per request; the receipt records `--neighbours`, `--batch` and the inputs' sha256 |
| `answers.jsonl`, `decisions.json`, `generation-complete.json`, `receipt.json` | `judge ask` | one line per request as it comes back; every statement's verdicts once all are answered; the receipt (written then) records the model and its settings |
| the score table (`--out`) | `judge table` | each term's (or record's) highest P(yes) + 0.5 P(uncertain) |

The stored outputs of the adopted run are the frozen files of `20261021/` (`decisions/`, `label-groups/`, `term-merging/`). `ensemble` writes nothing when the share of kept corpus terms leaves 70% to 100%, a rule fixed before the judge's corpus scores existed: the classifier alone keeps 92.1%.

## What a rerun pins

Three things can make a rerun of Step 5 differ from the stored files although nothing changed: the CPU thread count of k-means, the order Python gives a set of strings, and GPU rounding. Each is pinned, rather than given a tolerance, except the last, which no setting removes ([docs/setup/pinned.md](../setup/pinned.md)).

### Pinned cluster map

Path 1 (clustering) groups the corpus definitions into 100 clusters with k-means, then scores a term by how relevant the training records in its clusters were. scikit-learn's k-means does not end in the same clusters from run to run when it uses four or more CPU threads: 30 runs of the frozen path (3 at each of ten thread counts from 1 to 24) gave train + dev AUCs from 0.841 to 0.860. So a rerun would not reproduce the stored numbers.

**What is pinned.** Every command reads path 1's clusters from the saved map, `20261015/clusters/points.npz` (each corpus definition's cluster, by its id), instead of fitting k-means again. Only the PCA, which gives the same result every time, is fitted. Ties in the neighbour vote go to the smallest cluster number. `--refit-clusters` fits a new k-means instead, as on new data.

**Why.** A pinned map gives the same clustering numbers on every machine and thread count, so a rerun can be compared exactly. **The cost:** the stored path 1 numbers came from k-means runs whose clusters were not saved, so they cannot all come back. The saved map reproduces the term setting's train + dev AUC (0.8534) exactly; the Label Group setting's (0.8406) came from a second clustering in the same run that was never saved, and gives 0.8394 with the map. The ranking of the methods does not change: the classifier is about 0.06 above path 1 in every draw. Path 1's stored corpus decisions came from such a fit too: with the map, path 1 keeps 60,113 corpus terms where the stored decisions kept 60,131 (610 terms decided the other way), and its Label Group setting 61,717 where they kept 61,716 (one term).

### Hash order

Python orders a set of strings by their hashes, which change from process to process unless `PYTHONHASHSEED` is fixed. Run a rerun with `PYTHONHASHSEED=0`, and an order that comes from such a set is the same in every rerun, on every machine.

### GPU rounding

The entailment paths score on the GPU. The exact scorer (fp32, batches of 128 pairs, as every Step 5 run) gives probabilities that move by float noise between GPUs and library versions. **Worked example** (a 24 GB GPU, against the stored decisions): the zero-shot path's 67,394 corpus scores differ from the stored ones by at most 3.2e-5 (only 2 by more than 1e-5; *facilitator*: 0.187985 stored, 0.188017 now), and every keep or drop is the same, because the cut (0.00114) lies far from those scores. Such probabilities are compared within the band for numbers a GPU recomputes (1e-5) and each keep or drop exactly ([docs/setup/pinned.md](../setup/pinned.md)); six stored scores lie beyond it (2 of zero-shot, 3 of representative, 1 of its Label Group setting; largest 9.3e-5).

## The TEST split

The TEST split is held out: every read weakens it, so reads are rare, deliberate and logged in a read log, `20261015/test-evaluation/test-reads.json` (21 entries, each with its reason). Recomputing anything from it, even AUCs from saved TEST scores, counts as a read; so does loading a TEST record, since a record's id names its label (`<paper>:<1 relevant | 0 not>:<index>`).

- **By default it is compared by hash only.** `mbdef postprocess test` checks that the saved result `test-eval.json` still has its sha256 (`84e3b3db…`) and reads nothing else.
- **A fresh read** needs `postprocess test --read --reason "…" --out FOLDER`. Before anything of TEST is read, it prints a warning with the paths and the reason, asks the person at the terminal to type `read the test split`, and refuses without a terminal. The read is logged with its reason in `FOLDER/test-reads.json`, which starts as a copy of the frozen log. Every read goes through `testrun.log_test_read`.
- **The extractor's filter** (`extractor-gate`) scores only the train and dev records by default; `--read-test --reason` scores the TEST records too, through the same guard.
- `ensemble` and `merge-terms` read no TEST record: the ensemble's cut and weights come from train + dev only.
- **In this repository** `test-eval.json` and `test-reads.json` are not included: the read log names who asked for each read and the result's notes quote the requests. So `postprocess test` says that there is nothing to compare, and `postprocess test --read` and `extractor-gate --read-test` stop before anything of TEST is read, because without the log nobody can tell how often the split was read. The TEST numbers above are those of the logged reads. `frozen-paths.json` is here: `decide`, `evaluate` and the other commands build the frozen paths from it.

## How it was checked

- **The adopted run** (CPU only, about 30 s): `label-groups` on the corpus store remakes the Label Groups of the adopted Label Match rule, `ensemble` remakes `decisions/ensemble.csv` from the frozen member scores, and `merge-terms` remakes the term merging; each comes back byte for byte with the frozen files of `20261021/`, with the same cut and counts (the slow tests `test_the_frozen_corpus_decisions_come_back_byte_for_byte` and `test_the_frozen_term_merging_comes_back`).
- **The frozen paths** (needs a GPU): the Label Groups of the first Label Match rule, the train + dev evaluation of the seven headline paths and their corpus decisions in `20261015/` (classifier, path 1 and its Label Group setting, the two entailment paths and their Label Group settings) were remade from the frozen inputs, each cut on its saved train + dev scores. The Label Groups came back byte for byte (38,467 groups over 67,394 Extracted Concepts), the classifier's decisions byte for byte, every train + dev AUC at its four decimals, and every other keep or drop and score the same (the entailment paths' scores within 1e-5), except the differences [What a rerun pins](#what-a-rerun-pins) names.
- **The corpus store** (CPU, about 40 s): `build-store --compare` on release v1.1 gives the four tables of the stored `library.sqlite3` row for row, in the order they were written (8,425, 147,461, 524,907 and 1,327,091 rows), and `label-groups --store` on the rebuilt store writes the Label Groups of the adopted Label Match rule byte for byte (67,394 Extracted Concepts, 110,399 definitions, 38,407 groups). `tests/test_postprocess_store.py` holds a miniature release (`tests/fixtures/store-mini/`, synthetic text) and the rows that the original ingest code wrote from it: `build-store` must write the same rows in the same order.
- **The tests** `tests/test_postprocess_*.py`. `test_postprocess_ensemble.py` and `test_postprocess_term_merging.py` run the package on a miniature input whose expected files are the outputs of the original implementation, and the package must write the same files byte for byte. `test_postprocess_commands.py` runs the commands: the TEST guard refuses a read without a terminal before anything is read, a read without the read log is refused, and no command writes into a frozen folder. `test_postprocess_label_match.py` holds the rule's cases, those of the adopted version and the answers that the first version keeps. Two slow tests compare with the real files of the adopted run.
- **basal** (`tests/test_postprocess_basal.py`, CPU only). The test fixtures hold the outputs of the original implementation on a made-up corpus store, with a stand-in server (`tests/mini_basal.py`) that answers with the probabilities of the first nine recorded corpus answers; `postprocess basal` must write the same items, send the same request bodies in the same order, write the same answer lines (an error line for a malformed answer, then only the missing item asked again) and the same table, byte for byte. On the real files: the ground-truth items must equal the train + dev lines of the recorded item list, and the three frozen basal tables must come back byte for byte from the recorded answers (the corpus table in a slow test, which also checks that `items` gives the corpus items the run asked, sha256 `9806c9e9…`).
- **The judge** (`tests/test_postprocess_judge.py`, CPU only). The test fixtures hold the outputs of the original implementation on a made-up ground truth, corpus store and embedding cache, with a stand-in server (`tests/mini_judge.py`); `postprocess judge` must write the same packages, send requests with the same logged hash (model, messages, settings, answer schema), gather the same `decisions.json` and write the same tables, byte for byte. Three slow tests do it on the real files: the recorded corpus package, the hashes of all 30,548 recorded requests, and the two frozen judge tables from the recorded runs.

## How to adapt it

- **New corpus or extractor:** `build-store --release FOLDER` reads any folder with the layout that `extract derive` writes (`seeds/derived/` and `closure/derived/`, each with `DONE.json` and its variants); then pass `--store` and `--links` to `label-groups`; the paths read the corpus through `corpus.load_view`. A new corpus needs its definitions in the embedding cache ([docs/step-5-post-processing/embed.md](embed.md)), a new cluster map (`--refit-clusters` fits a new k-means instead of reading `points.npz`), and new corpus scores of basal and the judge (the commands above), given to `ensemble` with `--corpus-scores MEMBER=FILE`.
- **Another member or another hardware:** a member is a score table in the shared format (`method, unit, item_id, split, score`); `--weight-table MEMBER=FILE` names the train + dev table the weights were fitted on, `--reference MEMBER=FILE` the table each member's corpus scores are ranked against (from the same GPU type as the corpus run).
  New weights need a new weights file (`--pick`) and a new logged TEST read; new references only move the cut.
- **Another cut:** `ensemble --recall 0.9` sets the share of relevant train + dev records the cut keeps; `decide --recall 0.9` does the same for one path.
- **Other settings of the two model members:** basal's state, `basal items --max-definitions 1 --max-chars 300` (one definition per term, cut at 299 characters + "…"); the judge's neighbours and batch, `judge package --neighbours 16 --batch 2` (and `judge ask --batch 2`); its model and settings, `judge ask --model … --temperature … --max-tokens … --seed …`. Each run's receipt records them. basal's question (`QUESTION` in `basal.py`) and the judge's instructions (`INSTRUCTIONS` in `judge.py`) are text, not settings: a new question is an edit there. New scores need a new reference and new weights (the item above).
- **Another path:** a class with `fit` and `score` in `paths/`, and one more branch in `paths.make`; `evaluate` and `decide` then take it by name.

After a change, rerun `evaluate`, `ensemble` and `merge-terms` into new folders; TEST is read again only with `--read` and a reason.

## Not in this step

- **The probes, sweeps and the ensemble search** that chose the paths, their settings and the ensemble: none of them is needed to rerun the decisions from the frozen scores, and they are not part of this repository.
- **The 3D picture of path 1's clusters, and the choice of every Label Group's Representative Definition**: the repository keeps their outputs that later steps read (`clusters/points.npz`, `representatives.jsonl`), not the commands that made them.
- **Job files for the members on a cluster**: [Rerun basal](#rerun-basal) and [Rerun the judge](#rerun-the-judge) give the servers and the commands instead.
- **Protocol B** (keep or drop on the extractor's own candidates) and the human check of unmatched candidates: explorations whose inputs and results are not part of this repository.
