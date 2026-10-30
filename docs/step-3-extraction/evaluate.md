# Evaluate: `mbdef evaluate`

| | |
|---|---|
| **Paper step** | Step 3 (information extraction): how well the annotators of its ground truth agreed (appendix, Table 3) and what the curator did; how many reference definitions an extractor finds (the DefScore matcher), the held-out TEST result and the paired comparison with Opus. The choice of the four extractors from the grid of 2,268 scorings is the next page, [docs/step-3-extraction/selection.md](selection.md) |
| **Code** | [`src/mbdef/evaluate/`](../../src/mbdef/evaluate/): `agreement.py` (Table 3), `metric.py` (DefScore, matching, full-record score), `score.py` (the general scorer), `heldout.py` (the TEST rescoring), `paired.py` (our extractor against Opus); [`src/mbdef/stats.py`](../../src/mbdef/stats.py) (the statistics). `grid.py` and `selection.py` are the extractor selection ([docs/step-3-extraction/selection.md](selection.md)) |
| **Commands** | `mbdef evaluate agreement --out FOLDER`, `evaluate score --predictions FILE --split test --out FOLDER` (or `--scorings LIST`), `evaluate heldout --out FOLDER` and `evaluate paired --out FOLDER`; `evaluate grid` and `evaluate select` are in [docs/step-3-extraction/selection.md](selection.md) |
| **Checked by** | the tests [`tests/test_evaluate.py`](../../tests/test_evaluate.py), [`test_evaluate_score.py`](../../tests/test_evaluate_score.py), [`test_evaluate_heldout.py`](../../tests/test_evaluate_heldout.py), [`test_evaluate_paired.py`](../../tests/test_evaluate_paired.py) (the last two compare the commands with what the original implementation wrote on the tests' miniature inputs, stored in `tests/fixtures/released-outputs.zip`); the GPU test [`test_evaluate_score_gpu.py`](../../tests/test_evaluate_score_gpu.py), run by hand; reruns compared with the released outputs (below; why a rerun can differ: [docs/setup/pinned.md](../setup/pinned.md)) |

## Score your extractor

Write your extractor's records as a predictions file, one record per line (`{"paper_id", "term", "definitions": [...]}`, optionally the other lists; `extract run`, `extract verify` and `extract filter` write one, [docs/step-3-extraction/extract.md](extract.md)), fetch MBDef-Gold (`uv run mbdef data fetch MBDef-Gold/`) and score it on a split:

```bash
uv run mbdef evaluate score --predictions predictions.jsonl --split test --out out/scores
```

It prints how many reference definitions the predictions found, precision, recall, F1 and the full-record score, and writes `scores.json` and a receipt. [The command](#mbdef-evaluate-score) has the details; the rest of this page is how the paper's numbers were made.

## What this page covers

The first part of this page is the agreement numbers; the second, from [Scoring extractions](#scoring-extractions-defscore-f1-and-matching), is the scoring, the held-out TEST rescoring and the paired comparison with Opus. The grid of 2,268 setups that is scored with the same scorer, and the pick of the four extractors, are on [their own page](selection.md).

## What the agreement step does

Each of the 2,488 records of V3 was judged in Label Studio by two annotators (the *first round*). They answered two questions: **Q1** "Is it a definition?" and **Q2** "Is the concept relevant for the Media Bias Taxonomy task?". A third annotator answered both questions again on every record where the first two disagreed (the *third round*), and a curator then reviewed and edited the records. The command:

1. reads every answer from the raw Label Studio exports and joins it to its V3 record;
2. counts, per question, how the first two annotators answered each record, and computes how far they agreed beyond chance (Krippendorff's alpha), with the numbers around it (below);
3. counts what the curator did, from the curator's export;
4. draws two figures and writes a receipt.

**Table 3, recomputed** (the first two annotators of each record):

| | N | Yes/Yes | No/No | Disagree | % agree | alpha |
|---|---:|---:|---:|---:|---:|---:|
| Q1 Is a definition? | 2,486 | 2,095 | 59 | 332 | 86.6 | 0.19 |
| Q2 Relevant for the taxonomy? | 2,488 | 737 | 1,050 | 701 | 71.8 | 0.43 |

The paper prints 86.7 for Q1's agreement; the exact value is 86.645%, so it rounds to 86.6. Q1 has two records fewer because in two records one of the two first-round answers to Q1 is missing (for example record `167f1d5e…:18`, *miss rate*, where account 3 answered only Q2); `agreement.json` lists both under `n_difference`.

**A worked example: alpha for Q1.** Alpha compares how often the annotators disagree with how often they would disagree by luck, if each said *Yes* as often as the annotators really did:

- 2,486 records give 4,972 answers: 2 × 2,095 + 332 = 4,522 *Yes* and 2 × 59 + 332 = 450 *No*.
- Observed disagreement: 332 of 2,486 records, 0.1335.
- Disagreement expected by luck: 2 × 4,522 × 450 / (4,972 × 4,971) = 0.1647. So by luck alone two annotators would agree on 83.5% of the records, because 91% of all answers are *Yes*.
- alpha = 1 − 0.1335 / 0.1647 = **0.1890**. Equivalently, the annotators reached 86.6%: 3.1 of the 16.5 points that were possible above luck.

The same arithmetic, line by line, is in `agreement.json` under `arithmetic`. Q2's answers are mixed (44% *Yes*), so luck gives only 50.8% and the same kind of disagreement costs less: alpha = 0.4275.

The numbers around the table, and why each is there:

| Part of `agreement.json` | What it says | Q1 | Q2 |
|---|---|---|---|
| `table` | Every two-rater number from the three counts: chance agreement, Scott's pi, Gwet's AC1, PABAK, positive and negative agreement | AC1 0.840 | AC1 0.445 |
| `rater_order` | Cohen's kappa needs a "rater A"; the two people change from batch to batch, so kappa moves with that arbitrary choice (1,000 random orderings), alpha does not | kappa 0.1888–0.1905 | 0.4274–0.4281 |
| `uncertainty` | 95% intervals of alpha, resampling whole papers (150) or whole batches (321), 10,000 times | papers 0.116–0.260 | 0.345–0.502 |
| `third_round` | 931 records were disputed on Q1 or Q2 (102 on both); the third answer, and alpha if the third round were pooled in (not reported) | 332 disputed | 701 disputed |
| `final_labels` | The majority of the 2–3 answers against the curator's released labels, and where each record ended up | | |
| `annotators` | Per account and per pair of accounts: answers given, disagreements, alpha | | |
| `paper_claims` | The paper's claims around the table, recomputed (11 first-round accounts, 47% from two organisers, …) | | |
| `cross_checks` | The same counts from two earlier outputs, from the code that first made them (`Corpus.statistics` of the ground-truth step) and from the `krippendorff` package | same counts, alpha 0.18897 | same counts, alpha 0.42755 |

`curation.json` counts the curator's work: 338 records reviewed one by one (334 of them kept), 2,150 accepted under the curator's policy, 174 kept records changed against V3, and the assistant's proposals (372 records; the curator took the assistant's text in 133 fields, kept V3 in 40 and wrote 44 by hand).

## Run it

The raw exports and the curator's export are not in git, because they may name the annotators. They live in the data root ([docs/setup/data-root.md](../setup/data-root.md)), together with pseudonymised copies of the two exports that name no one and give the same numbers ([docs/step-3-extraction/groundtruth.md](groundtruth.md#publishable-copies-groundtruth-pseudonymise); read by default where the originals are not there, or passed with `--annotations` and `--export`). With `DATA_ROOT` set in `.env`:

```bash
uv run mbdef data fetch groundtruth/   # once: the ground truth's sources
uv run mbdef evaluate agreement --out /tmp/agreement
```

Where the data root holds only the pseudonymised copies (the originals are not public), `agreement` reads them by default and prints the same table; `agreement.json` then names the copies as its inputs.

It takes about 3 s and prints the table, then the comparison with the paper's printed table:

```text
Q1 2486 2095 59 332 86.65 0.189
Q2 2488 737 1050 701 71.82 0.4275
{
 "Q1": {
  "N": [2486, 2486, true],
  ...
```

Each comparison is `[printed, recomputed, equal at the printed precision]`; only Q1's "% agree" is `false` (86.7 printed, 86.6452 recomputed). The command writes only into `--out`, which must be new or empty.

## Inputs

| Input | Where | sha256 | What it is |
|---|---|---|---|
| `Annotations.zip` | `$DATA_ROOT/groundtruth/newAnnotations/` | `09192c66…` | The raw Label Studio exports: 13 weekly, 2 completion and 4 third-round files |
| `validated-v3.zip` | `$DATA_ROOT/groundtruth/` | `27ac4f29…` | The curator's export: the released labels (`relevance-labels.json`) and the curation records |
| `newAnnotations/Annotations.zip`, `validated-v3.zip` | `$DATA_ROOT/groundtruth/pseudonymised/` | `764c356e…`, `db6bc11b…` | The pseudonymised copies of the two exports, which give the same numbers (below) |
| `ground_truth/v3.json` | `data/` | | The 2,488 records that were annotated |
| `frozen/labelStudioXML.md` | `data/` | | The labelling form: it shows that Q1 was not a required answer |
| `newAnnotations/annotation-audit.json` | `data/` | | An earlier copy of the table, for the cross-check |
| `validated_gt_stats.json` | `$DATA_ROOT/MBDef-Gold/` | | MBDef-Gold's statistics (`data fetch MBDef-Gold/`), another earlier copy of the table, for the cross-check |

The flags `--annotations`, `--export` and `--data` point at other copies of these inputs.

## Outputs

| File in `--out` | What it is |
|---|---|
| `agreement.json` | Table 3 and every number around it (above) |
| `curation.json` | What the curator did |
| `fig1_agreement_tables.png`, `.svg` | The two 2×2 tables of the first two answers |
| `fig2_observed_vs_chance.png`, `.svg` | Observed against chance agreement: why alpha is low for Q1 |
| `receipt.json` | The pinned settings, the sha256 of every input and output, the package versions, the code commit and the time |

Both JSON files are written with indent 1 and without escaping non-ASCII text, as the released ones are. The SVG files carry the time they were drawn; their element ids are fixed, so two runs differ only in that time.

## The statistics module

[`stats.py`](../../src/mbdef/stats.py) holds the one copy of each statistic: `pair_counts`, `alpha_from_counts` and `two_rater_numbers` (two raters, from the three counts), `krippendorff_alpha` (any number of raters, missing answers allowed), `cohen_kappa`, and `cluster_bootstrap`. The ground-truth step's annotation statistics use the same functions. Every random procedure takes its seed, its number of replicates and its resampling unit as parameters, because each reported interval must be remade with the ones it was made with. The pinned ones are named constants in `evaluate/agreement.py` and are written into the receipt:

| Setting | Value | Used for |
|---|---|---|
| `BOOTSTRAP_REPLICATES`, `BOOTSTRAP_SEEDS` | 10,000; numpy `default_rng(1)` by paper, `default_rng(2)` by batch | the intervals in `uncertainty` |
| `KAPPA_ORDERINGS`, `KAPPA_SEED` | 1,000; Python `random.Random(20261016)` | the kappa range in `rater_order` |
| `LARGE_PAIR` | 100 records | which annotator pairs get an alpha range |
| `ORGANISERS` | accounts 2 and 3 | the organisers' share of the judgments |
| `JSON_FORMAT` | indent 1, non-ASCII kept | both JSON files |

## How it was checked

- **Against the original implementation.** On made-up answers, each function of `stats.py` returned exactly what the original agreement code returned (alpha, the two-rater numbers, kappa, and the bootstrap with the same seed), and on the miniature annotation round the table equalled that of the original `Corpus.statistics`.
- **A rerun against the released outputs.** Both PNG figures are byte-identical; `agreement.json` and `curation.json` are identical except for the fields that name the code and the input paths (`generated_by` and `inputs`; `export`); the SVG figures are identical once their date and random ids are removed; the five counts are exact (2,488 records, 2,486 Q1 pairs, 2,488 Q2 pairs, 931 disputed records, 372 records with an assistant proposal).
- **From the pseudonymised copies.** The command ran on the copies (`--annotations` and `--export`) and on the originals, on a CPU. Both printed the same Table 3 (Q1 2,486 pairs, 2,095 / 59 / 332, 86.65%, alpha 0.189; Q2 2,488 pairs, 737 / 1,050 / 701, 71.82%, alpha 0.4275), the released table. `agreement.json` is the same except `inputs`, which names the files read; `curation.json` is the same except `export`, its file name; both PNG figures are byte-identical. The copies hold every answer and every account number, and lose only a draft's name and e-mail, local paths and paper passages, none of which the numbers read. A test does the same on the miniature round.
- **The tests** run the command on a miniature annotation round made up for the test ([`tests/mini_agreement.py`](../../tests/mini_agreement.py): 9 records, 4 made-up annotators, a missing answer, a repeated submission and a third round) whose Table 3 is worked out by hand in the file: Q1 8 pairs, 5/1/2, alpha 0.3750; Q2 9 pairs, 6/1/2, alpha 0.3929; 4 records disputed.

## How to adapt it

- **Another annotation round of the same kind** (Label Studio exports with `definition_is_definition_<i>` and `term_relevant_<i>` controls, a third round in `Annotation_Disagreements/`, and a curator's export with the same members): pass `--annotations`, `--export` and `--data`. The table, the third round, the intervals and the figures follow the data.
- **The paper's claims** are pinned: the printed table (`PAPER`), the organisers' accounts (`ORGANISERS`) and the claims in the keys of `paper_claims`. Change them for another paper.
- **Other questions or answer values**: `Q` and `CONTROL` name the two questions; the statistics in `stats.py` count *Yes* and *No*.
- **Other intervals**: change `BOOTSTRAP_REPLICATES` or `BOOTSTRAP_SEEDS`, or call `stats.cluster_bootstrap` with other clusters (the resampling unit).
- **After a new round**, the ground truth changes, so the ground-truth step and everything after it in Step 3 must be rerun.

## Scoring extractions: DefScore, F1 and matching

An extractor writes definition records for a paper; the ground truth has the paper's reference records.
The scorer counts how many reference **definitions** the extractor found:

1. **Pair score (DefScore).** Every reference definition of a paper is compared with every predicted one. DefScore is the mean of three similarities, calibrated and clipped to [0, 1]: `clip(1.40284 × (token F1 + BGE cosine + entailment) / 3 − 0.29737, 0, 1)`. Token F1 compares the lower-case ASCII words; BGE cosine is the cosine of the two texts' `BAAI/bge-base-en-v1.5` vectors; entailment is the mean of `tasksource/ModernBERT-large-nli`'s entailment probability in both directions.
2. **Matching.** The one-to-one assignment that accepts the most pairs with DefScore ≥ 0.5, ties broken by the total score. An accepted pair is a match (tp); the other predictions are fp, the other reference definitions fn.
3. **Pooling.** tp, fp and fn are added over the papers of the reference. Precision = tp / predicted definitions; recall = tp / all reference definitions (a paper without output keeps its reference definitions in the denominator).
4. **The secondary score** (full-record, stored as `M3-full-v1`): records are aligned one-to-one by their best definition pair; then inside the aligned pairs the term must be the same (after normalising), each of the seven text fields is matched one-to-one at DefScore ≥ 0.5, and the citations count by overlap. Each field gets an F1; the paper's score is the mean F1 over the fields that have something on at least one side (a field with something on one side only scores 0), and a scoring's score is the mean over its papers.

**A worked example** (TEST paper `73bfad11…`, in the scorer tests' sample [`tests/fixtures/score-sample/`](../../tests/fixtures/score-sample/)). The reference definition *"fabricated information that mimics the output of the news media in form, but not in organizational process or intent"* and Opus's *"We define fake news to be fabricated information that mimics the output of the news media in form, …"* have token F1 0.8636, BGE cosine 0.8761 and entailment (0.3992 + 0.9757) / 2 = 0.6875.
Their mean is 0.8091, so DefScore = 1.40284 × 0.8091 − 0.29737 = **0.8376**, a match. The paper has 6 reference definitions and Opus (row `Rdb54680`) predicted 8: 5 pairs match one-to-one (tp 5, fp 3, fn 1). With the second paper of the sample (`4243aa9b…`: tp 2 of 4, 19 predicted), Opus finds 7 of 10 definitions with 27 predictions: precision 25.9%, recall 70.0%. These are the released TEST run's own per-paper numbers.

`metric.py` is the matcher and `score.score` the scorer of the released scoring runs (the held-out TEST rescoring and the grid share it). The text normalisation (`groundtruth.ascii_words`) and the dataset views (`groundtruth.dataset_views`) are in `groundtruth.py`.

### The pinned settings

| Setting | Value | Where |
|---|---|---|
| `DEFSCORE_SLOPE`, `DEFSCORE_INTERCEPT` | 1.4028410596332237, −0.2973741431607951 (the fit `calibrated_base_equal_mean`, in [`data/frozen/artifacts/defsim_benchmark.json`](../../data/frozen/artifacts/defsim_benchmark.json)) | `metric.py` |
| `MATCH_THRESHOLD` | 0.5, the cutoff of a match (`--threshold`) | `metric.py` |
| `SOFT_THRESHOLD` | 0.25, the soft many-to-one credit that training optimises (`definition_set_details`) | `metric.py` |
| `BGE_MODEL`, `BGE_REVISION` | `BAAI/bge-base-en-v1.5` at `a5beb1e3…`; 512 tokens, batches of 32 texts | `metric.py` |
| `NLI_MODEL`, `NLI_REVISION` | `tasksource/ModernBERT-large-nli` at `ca476cb9…`; 2,048 tokens per pair | `metric.py` |
| `NLI_BATCH_SIZE` | 32 pairs per batch, as the released scoring runs (`--nli-batch-size`; another size moves probabilities by about 1e-6) | `metric.py` |

The models run on `SCIDEF_NLI_DEVICE` from `.env` (`cuda`, `cpu` or `auto`). Every receipt records all of the above. Step 5 (`postprocess/nli.py`) calls the entailment model as `NLIScorer(batch_size=128).directional(pairs)`, with these model and revision constants; the slow test `test_step_5_can_still_call_the_entailment_scorer_as_before` keeps that call working.

**Where `DEFSCORE_SLOPE` and `DEFSCORE_INTERCEPT` come from.** They are a straight-line fit on SciDef's DefSim benchmark (60 definition pairs that people rated from 1 to 5, public): for each pair, x is the mean of the three similarities and y the rating mapped to [0, 1] as (rating − 1) / 4; the slope is cov(x, y) / var(x) over the 60 pairs and the intercept mean(y) − slope × mean(x), which gave 1.40284 and −0.29737. The record [`defsim_benchmark.json`](../../data/frozen/artifacts/defsim_benchmark.json) keeps these two numbers, the same fit on five paper-disjoint folds (fold 0, fitted on the other 40 pairs: slope 1.43836, intercept −0.30451; the five slopes lie between 1.375 and 1.438), and how well the calibrated DefScore agrees with the ratings on pairs it was not fitted on (Pearson 0.985 over the 60 held-out predictions). It does not keep the 60 pairs' three similarities, so the fit cannot be redone from the record alone: a refit means scoring DefSim's 60 pairs again with the two models and fitting the line. The fit is not a command here; the two constants are pinned, and every scoring receipt records them.

**Recorded pair scores.** `--record-scores FILE` writes every BGE similarity and entailment probability the models computed; `--replay-scores FILE` answers every one from such a file and loads no model. A pair that is not in the file stops the scoring with `NotRecorded`, naming the pair, so a changed input cannot pass unnoticed. The tests replay the sample [`tests/fixtures/score-sample/`](../../tests/fixtures/score-sample/) this way; the slow GPU test runs the real models on it by hand and checks every score within ±1e-5 of the recorded one:

```bash
uv run pytest -m slow tests/test_evaluate_score_gpu.py   # on a shared machine, under a lock (flock)
```

### `mbdef evaluate score`

The general scorer. It scores **predictions files** against one split of MBDef-Gold, or a list of **scorings**, each the predictions of one setup on its own reference.

**Predictions files** (`--predictions FILE`, repeatable; each file is one scoring, named by its path). One JSON object per line, one record per line:

```json
{"paper_id": "73bfad11b96a69cb882028ead115751adb55252d", "term": "fake news", "definitions": ["fabricated information that mimics the output of the news media in form"]}
```

`paper_id`, `term` and `definitions` are required; `synonyms`, `description`, `related_terms`, `subtypes`, `supertypes` and `examples` are optional lists, and every item is a string or `{"text": ..., "citation": [...]}`. Lines of the same paper are pooled and records with the same term (ignoring case) merged (`records.read_predictions`), so `cat a.jsonl b.jsonl` is the pool of two extractors. A line that is not a record stops the command with its line number.
The reference is the relevant records of one split of MBDef-Gold's `validated_gt.json` (`--split`: `train` 80 papers with 712 definitions, `validation` 21 with 115, `test` 29 with 365, the default; `--gold FILE` for another copy). Every paper of the split counts: a paper without predictions keeps its definitions in the recall's denominator. `--paper ID` (repeatable) makes the reference only those papers of the split. A paper of the file outside the reference is counted (`papers_outside_reference`) and not scored.
The tests check that the Opus row of the sample below, written as one predictions file, gives the per-paper counts of its scoring ([`tests/test_evaluate_score.py`](../../tests/test_evaluate_score.py)); the README shows a live run.

**Scorings** (`--scorings LIST`):

```json
[
 {"id": "Rdb54680", "reference": "reference.json", "papers": "rows/Rdb54680/papers"},
 {"id": "QWEN-POOL3", "reference": "reference.json", "papers": "rows/QWEN-POOL3/papers"}
]
```

`reference` is a file with a `papers` list (each `paper_id` and `definition_records`), and `papers` a folder of `{"paper_id", "definition_records"}` files, both relative to the list's folder. On the scorer tests' sample:

```bash
uv run mbdef evaluate score --scorings tests/fixtures/score-sample/scorings.json \
  --replay-scores tests/fixtures/score-sample/pair-scores.json --out /tmp/scores
```

```text
Rdb54680: 7 of 10 reference definitions found, 27 predicted (precision 25.9%, recall 70.0%), full-record score 0.122
QWEN-POOL3: 10 of 10 reference definitions found, 46 predicted (precision 21.7%, recall 100.0%), full-record score 0.273
wrote /tmp/scores
```

It writes `scores.json` (per scoring: the reference's sha256 and counts, the pooled tp/fp/fn, precision, recall, F1 and full-record score, and every paper's counts) and `receipt.json`. **If any scoring fails** (an unreadable or malformed paper file, a paper outside the reference, a model error, a pair that was not recorded), the command stops with `Scoring <id> failed: …` and writes nothing. Code that builds its scorings in memory calls `score.score(gold, predicted, pair_scorer)` directly.

### The held-out TEST rescoring

`mbdef evaluate heldout` rescores the held-out TEST evaluation: 12 frozen rows (older extractor candidates, Opus, Kimi, GPT-5.6 Sol, SOOFI and Gemma) on the 29 held-out TEST papers, whose reference has 365 definitions. It scores the saved rows with the general scorer, then applies the frozen plan's rules (verbatim in `results.md`): a row is **eligible** if its run is complete, every paper has output and at most 5% of its extraction windows and verifier batches failed; the **pick** is the eligible row with the highest precision among rows with recall ≥ 80% (292 of 365), preferring a self-hosted Qwen row within 2 matches, and if no row reaches 80%, the highest recall among rows with precision ≥ 15%.

| Input | Where | sha256 |
|---|---|---|
| The plan (`plan.json`), whose hash the command checks: the neutral copy (below) | `data/frozen/results/test-eval-20260924/` | `bd80acc8…` |
| The reference (`inputs/test-reference.json`) and its readiness record (`inputs/READY.json`) | `data/frozen/results/test-eval-20260924/` | `1d99ec81…`, `f1d3aba3…` |
| The cohort (`test-sources.json`: the 29 paper ids, with the papers' full text, so it is not public; give your own with `--sources`) | `$DATA_ROOT/frozen/artifacts/experiments/test-eval-20260924/final/inputs/` | `2dc80739…` |
| The saved rows (`<row>/status.json`, `<row>/papers/*.json`; 860 files, 19.6 MB) | `$DATA_ROOT/frozen/artifacts/experiments/test-eval-20260924/collected/rows/` | folder `52c7404f…` |

**The plan is a neutral copy.** The original plan (sha256 `7cb89ade…`, 4,814 bytes) named a person in its `authorization` field; the plan here is a copy that rewords only that field (`bd80acc8…`, 4,810 bytes). The rules the command reads (`pick_rule`, `eligibility`, the rows, the cohort) are the same bytes.
The readiness record and the grid audit keep their bytes, so they cite the original's sha256. `results.json` carries the copy's hash as `plan_sha256`, the one field in which a rerun differs from the stored results.

```bash
uv run mbdef data fetch frozen/artifacts/   # once: the copy of the TEST run (the cohort is not public)
uv run mbdef evaluate heldout --out /tmp/heldout
```

It takes about 6 minutes on a 24 GB GPU and prints one line per row, then the pick:

| Row | Found / 365 | Predicted | Precision | Eligible |
|---|---:|---:|---:|---|
| R0bea6bc (Qwen) | 252 | 678 | 37.2% | yes |
| R6e24b3d (Qwen) | 246 | 602 | 40.9% | yes |
| Ra1da1ae (Qwen) | 248 | 638 | 38.9% | yes |
| QWEN-POOL3 (Qwen, three passes pooled) | **291** | 1,173 | 24.8% | yes |
| Rdb54680 (Opus + Qwen verifier) | **290** | 1,002 | 28.9% | yes |
| R80dd4ae (Opus) | 260 | 725 | 35.9% | yes |
| R9ec8007 (Opus) | 26 | 52 | 50.0% | no: 78 of 93 extraction windows failed |
| R13f6cd1 (Kimi) | 235 | 511 | 46.0% | yes |
| Rde76462 (GPT-5.6 Sol) | 245 | 594 | 41.2% | no: the run is not complete |
| R906aabe (SOOFI) | 193 | 1,186 | 16.3% | yes |
| R8ce2900 (Gemma 12B) | 259 | 1,244 | 20.8% | yes |
| R74933e3 (Gemma 12B) | 166 | 293 | 56.7% | yes |

No eligible row reaches 292, so the fallback picks **QWEN-POOL3**: 291 of 365 (79.7% recall), the highest recall among the eligible rows with precision ≥ 15%. Every count and the pick are those of the released run. The outputs are `results.json` and `results.md` in the format of the stored results, and `receipt.json`.

### Our extractor against Opus, paper by paper

`mbdef evaluate paired` compares our P≥28 extractor (`gemma12b/TR-bootstrap-8k+16k32k/chat-default-think/filtered-ordinary-qv`, from the selection) with Opus paper by paper, from saved per-paper counts only: on TEST the row `Rdb54680` above, on DEV its parent run `R59b5d1d`.

| Input | Where under `$DATA_ROOT` | sha256 |
|---|---|---|
| The selection's `results.json` (our per-paper counts) | `frozen/results/production-20260925/report-20260927/selection/` | `b9ec1d3e…` |
| The TEST rescoring's `results.json` (Opus on TEST) | `frozen/artifacts/experiments/test-eval-20260924/collected/` | `dfc608a0…` |
| Opus's DEV `evaluation.json` | `frozen/artifacts/experiments/matched-winners-v1/refinement/R59b5d1d/definitions-strict/` | `1078b504…` |

```bash
uv run mbdef evaluate paired --out /tmp/paired    # about 1 s, no GPU
```

```text
test: papers=29 ours=(294, 961, 365) opus=(290, 1002, 365)
  found diff (ours-opus) +4  95% CI (-27, 34)
  recall diff pp +1.1  95% CI (-7.9, +9.3)
  precision diff pp +1.7  95% CI (-3.2, +5.8)
  papers where ours finds more / fewer / same: 13 / 7 / 9
```

(and the same for DEV: 95 against 100 of 115, −4.3 points [−14.1, +3.4]; then the permutation p-values, TEST 0.860 and DEV 0.431 for recall). Ours finds 4 more of the 365 TEST definitions than Opus: +4 / 365 = +1.1 points of recall. The interval resamples whole papers with both systems together (paired), 20,000 times, with Python's `random.Random(20261013)`, and takes the sorted draws at index `int(0.025 × 20000)` and `int(0.975 × 20000) − 1`; the permutation test swaps the two systems within each paper 20,000 times with `random.Random(20261014)`. These are `stats.paired_bootstrap`, `stats.index_interval`, `stats.paired_permutation_p` (which take the generator and the resampling unit as parameters) and `stats.sign_test`. `output.txt` is byte for byte the stored output (`7192db74…`); `paired.json` holds the same numbers as data.

### How the scoring was checked

- **Against the original implementation.** With the same recorded pair scores, `score.score` returned exactly what the original scorer returned (every count and float of both rows of the sample); the set matching, the full-record score (with and without partial records), the one-to-one matching on 150 random score matrices full of ties, token F1, the text normalisation and the dataset views equalled the original code's; the DefScore numbers equal the fit record. What the original TEST scorer wrote on the same rows (eligible, incomplete, broken, unplanned and missing rows), and its answers to the rule cases, are stored (`tests/fixtures/released-outputs.zip`); [`tests/test_evaluate_heldout.py`](../../tests/test_evaluate_heldout.py) compares the command's results with them, field by field, apart from the run details. The paired comparison with Opus is checked the same way ([`test_evaluate_paired.py`](../../tests/test_evaluate_paired.py)), and the grid scorer on [the selection page](selection.md).
- **Reruns against the released outputs** ([docs/setup/pinned.md](../setup/pinned.md) says why a rerun can differ). `evaluate heldout`: all 12 rows have the same tp, fp, fn and predicted counts for every one of the 29 papers, the same eligibility and the same pick as the released run; the full-record means of R0bea6bc, Ra1da1ae and QWEN-POOL3 are 0.0002 lower (one paper's record alignment has two near-tied solutions), inside a ±0.001 band. `evaluate paired`: `output.txt` byte-identical.
- **The tests** replay recorded pair scores, so they need no model weights and no GPU.

### How to adapt the scoring

- **Other predictions or another reference:** a predictions file with `--predictions` (another copy of MBDef-Gold with `--gold`), or a scorings list with `--scorings`.
- **Another cutoff:** `--threshold`. **Another entailment batch size:** `--nli-batch-size`.
- **Other models or another calibration** change every number: change `BGE_MODEL`, `NLI_MODEL`, their revisions and `DEFSCORE_SLOPE`/`DEFSCORE_INTERCEPT` together, refit the calibration, and record new pair scores for the scorer tests' sample (`tests/fixtures/make_score_sample.py`).
- **Another TEST plan:** `heldout.py` holds the frozen rules of the TEST plan (its hash is checked); copy its eligibility and pick functions for a new plan rather than changing them.

## The extractor selection

Scoring the grid of 2,268 setups and picking the four extractors of the paper (`evaluate grid` and `evaluate select`, with the cluster job that does the scoring) has its own page: [docs/step-3-extraction/selection.md](selection.md).

## Not in this step

Step 5's evaluation is on its own page ([docs/step-5-post-processing/postprocess.md](../step-5-post-processing/postprocess.md)). The breakdowns of the TEST results and the page that presented them are not part of this repository.
