# Extractor selection: `mbdef evaluate grid` and `evaluate select`

| | |
|---|---|
| **Paper step** | Step 3 (information extraction): the choice of the four extractors from the grid of 2,268 scorings |
| **Code** | [`src/mbdef/evaluate/grid.py`](../../src/mbdef/evaluate/grid.py) (scores every setup with the general scorer, [`score.py`](../../src/mbdef/evaluate/score.py), [docs/step-3-extraction/evaluate.md](evaluate.md#scoring-extractions-defscore-f1-and-matching)), [`selection.py`](../../src/mbdef/evaluate/selection.py) (the audit, the pick rule and the cluster job's command) |
| **Commands** | `mbdef evaluate grid --out FOLDER`, with `--write-scorings` for the cluster job; `mbdef evaluate select --scores FILE --out FOLDER`; `mbdef cluster render` for the job ([docs/setup/cluster.md](../setup/cluster.md)) |
| **Checked by** | the tests [`test_evaluate_grid.py`](../../tests/test_evaluate_grid.py), [`test_evaluate_selection.py`](../../tests/test_evaluate_selection.py) and [`test_selection_job.py`](../../tests/test_selection_job.py); a replay compared with the stored results (below; why a rerun can differ: [docs/setup/pinned.md](../setup/pinned.md)) |

## Compare your own extractors

Score each setup's predictions file on the same split and compare the lines: `evaluate score --predictions a.jsonl --predictions b.jsonl --split validation` ([docs/step-3-extraction/evaluate.md](evaluate.md#score-your-extractor)). The rule that picked the paper's extractors from the scored grid is `evaluate select`, below.

## Where it sits

The step before it is [extraction](extract.md) (it made the runs that are scored here), the step after it is the seed run's variants (`extract derive`, [docs/step-3-extraction/extract.md](extract.md#the-seed-runs-variants-extract-derive)), which pools the picked extractors' saved parts.

## What the step does

The paper's extractors were chosen from a **grid**: two models, Qwen 3.6 35B and Gemma 4 12B, read the 21 DEV and 29 TEST papers in many ways, and each way is a **setup**. A setup is a model, one prompt program or a pool of them, one output format or a pool of them, and one post-processing stage (all extracted records, or only those a verifier kept, or only those a relevance filter kept). Per model and split there are 567 setups, so **2,268 scorings** in all (567 × 2 models × 2 splits); 1,188 more scorings of single runs are scored beside them and are candidates for the picks too. `evaluate grid` scores every one; `evaluate select` applies the frozen rule that picks the four extractors of the paper.

**A worked example.** The setup behind the P≥28 pick is `gemma12b/TR-bootstrap-8k+16k32k/chat-default-think/filtered-ordinary-qv`: Gemma 4 12B; the two prompt programs that DSPy's Bootstrap optimiser trained for 8k and for 16k/32k windows, pooled; the chat format with thinking on; and the records that survived the relevance filter and Qwen's verification. For each of the 29 TEST papers the two runs' records at that stage are put one after the other and merged (the same term with the same text counts once). The result has 961 definitions, of which 294 match one of the 365 reference definitions: recall 294 / 365 = 80.5%, precision 294 / 961 = 30.6%. The two runs made 596 + 491 = 1,087 model calls. On the 21 DEV papers the same setup finds 95 of 115 (436 predicted).

## Inputs

| Input | Where | sha256 |
|---|---|---|
| The grid's plan and its amendments 1-9, whose hashes the commands check | `data/frozen/results/grid-20260925/` | `plan.json` `5604ef10…`, amendment 8 `08733f9d…` |
| The audit: a verdict per cohort (one model run on one split) | `data/frozen/results/grid-20260925/audit/audit.json` | `7cdcba40…` |
| The DEV and TEST references (115 and 365 definitions) | `data/frozen/results/qwen-replication-sweep-20260923/dev-reference.json`, `data/frozen/results/test-eval-20260924/inputs/test-reference.json` | `8e5696cc…`, `1d99ec81…` |
| The grid's saved runs (3.30 GB) | `$DATA_ROOT/frozen/runs/grid-20260925/cohorts/` | folder `1c2fd568…` |
| The trained-prompt runs (2.14 GB) | `$DATA_ROOT/frozen/runs/training-20260925/eval/cohorts/` | folder `c349dd05…` |
| The TEST evaluation's results (Opus's calls, the replicate note) | `$DATA_ROOT/frozen/artifacts/experiments/test-eval-20260924/collected/results.json` | `dfc608a0…` |
| The stored scoring run and selection, which a rerun is compared with | `$DATA_ROOT/frozen/runs/grid-20260925/score/runs/20260925T190230Z/` and `$DATA_ROOT/frozen/results/production-20260925/report-20260927/selection/` | folder `342ce4b4…`; the selection's `results.json` `b9ec1d3e…` |

## Run it

```bash
uv run mbdef data fetch frozen/runs/ frozen/results/ frozen/artifacts/   # once
uv run mbdef evaluate grid --out /tmp/grid       # the models run on SCIDEF_NLI_DEVICE (.env)
uv run mbdef evaluate select --scores /tmp/grid/scores.json --out /tmp/selection
```

`evaluate grid` writes `scores.json` (every setup's matches, predicted definitions, per-paper counts, full-record score, eligibility and model calls, before any audit or pick) and `receipt.json`. A setup that cannot be scored stops the command and nothing is written. **Eligibility** is the plan's failure policy: every pooled run complete, every paper with an output, at most 5% of extraction windows and verifier batches failed. Setups of one model on one split run in one process, so a definition pair that many setups share is scored once: the whole grid asks for about 653,000 distinct entailment pairs. `--gpus` and `--workers` spread the four processes (TEST and DEV, Qwen and Gemma) over GPUs.

`evaluate select` applies the rule of the plan's amendment 8 to those scores:

1. **The audit.** An independent audit judged each cohort. A setup is excluded if any of its cohorts has a defect, and the audit's view of whether a cohort's failures were few enough replaces the cohort's own.
   1,107 of the 1,134 TEST setups are eligible and pass.
2. **The floors.** For each precision floor, 20%, 24% and 28%: among the eligible, audit-passing TEST setups with at least 292 of the 365 reference definitions found (80% recall) and at least that precision, take the one that finds the most. Setups within 2 matches of it are tied; the tie goes to the higher precision, then to fewer model calls.
3. **The fourth output** (amendment 9): the single-model variant of the P≥28 pick, the same Gemma generations with the relevance filter but without Qwen's verifier. No rule picks it.

With the stored scores and the corrected audit (`--replaces`, below) it printed:

```text
P>=20: gemma12b/TR-bootstrap-8k+16k32k/chat-default-think+json-default-think/filtered-ordinary-noverify: TEST 317/1550 (86.8% recall, 20.5% precision), DEV 100/716
P>=24: gemma12b/TR-bootstrap-8k+16k32k/chat-default-think/verified-self: TEST 310/1200 (84.9% recall, 25.8% precision), DEV 101/592
P>=28: gemma12b/TR-bootstrap-8k+16k32k/chat-default-think/filtered-ordinary-qv: TEST 294/961 (80.5% recall, 30.6% precision), DEV 95/436
P>=28-single-model (amendment 9): gemma12b/TR-bootstrap-8k+16k32k/chat-default-think/filtered-ordinary-noverify: TEST 294/1036 (80.5% recall, 28.4% precision), DEV 95/468
```

**Reading the P≥28 line.** 6 of the setups reach 292 matches and 28% precision. The most any finds is 294, and all six are within 2 of it, so the tie goes to precision: 294 / 961 = 30.6% is the highest, and its 1,087 calls are the fewest among the equal ones. For P≥20, 205 setups qualify, the most any finds is 317, and 3 are within 2 of it. For P≥24, 62 qualify and 2 are within 2 of the 312 best. The result also records the cheapest setup for each recall level, the table of all setups, Opus's own reference row (`Rdb54680`: 290 of 365 on TEST, 100 of 115 on DEV) and the superseded rules of amendments 1 and 5, as the stored `results.md` does.

**Correcting an audit** (`--replaces`). The first audit (`806075de…`) wrongly excluded all 48 trained-prompt cohorts, so the stored scores were selected again with the corrected audit (`7cdcba40…`).
`--replaces RESULTS --reason TEXT` first proves that RESULTS's own audit, applied to the scores, gives RESULTS's picks, writes that replay into `replaced/`, and records the correction in a `reselect` block. The command is the rule's own code in both cases.

## Outputs

| File in `--out` | Written by | What it is |
|---|---|---|
| `scores.json` | `evaluate grid` | Every setup's matches, predicted definitions, per-paper counts, full-record score, eligibility and model calls, before any audit or pick |
| `results.json`, `results.md` | `evaluate select` | The picks, the cheapest setup for each recall level, the table of all setups, Opus's own reference row and the superseded rules, in the format of the stored results |
| `replaced/` | `evaluate select --replaces` | The replay of the results that were corrected (below) |
| `receipt.json` | both | The settings, the hashes of every input and output, the code commit and the time |
| `<split>-<model>.json`, one folder of paper files per set of predicted records, the two references | `evaluate grid --write-scorings` | The scorings lists for the cluster job (below), and what they name |

## The cluster job

The scoring needs the entailment and embedding models on a GPU, so the rescoring of all 2,268 setups runs as **one cluster job** on one H100 GPU, at the normal QOS (the rule of [docs/setup/cluster.md](../setup/cluster.md)). It is the one cluster job of this step. It ran once and took 28 min 30 s ([the released job](#the-released-job), below). The job does not need the 5.4 GB of saved runs: the scorings are written on a workstation first, and the job scores them with the general scorer ([`evaluate score`](evaluate.md#mbdef-evaluate-score)).

**1. Write the scorings** (on a workstation, at the commit the job will run; about 1 minute, no GPU):

```bash
uv run mbdef evaluate grid --write-scorings --out $DATA_ROOT/selection/rescore/2026-10-20/scorings
tar -C $DATA_ROOT/selection/rescore/2026-10-20 -cf - scorings \
  | zstd -19 --long=31 -T0 -o $DATA_ROOT/selection/rescore/2026-10-20/scorings.tar.zst
```

For each of the four families (TEST and DEV, Qwen and Gemma) this writes a scorings list `<split>-<model>.json` and one folder of paper files per distinct set of predicted records, plus copies of the two references and a receipt: **2,789 distinct record sets (837 and 558 on TEST for Gemma and Qwen, 836 and 558 on DEV), 69,736 files, 1.55 GB**. Sets that several setups share are written once, named by the hash of their records. The tar is 7,944,521 bytes (7.9 MB, sha256 `1b17f315…`); it is what travels to the cluster. A second run writes the same files (checked, apart from `receipt.json`).

**2. The job's one command**, which `cluster render` puts into the job script (`selection.JOB_STEP`):

```bash
STEP=$(cat <<'END'
O=$DATA_ROOT/selection/rescore/2026-10-20; S=$TMPDIR/scorings; mkdir -p $O/pair-scores && tar -I 'zstd --long=31' -xf $O/scorings.tar.zst -C $TMPDIR && printf '%s\n' test-qwen test-gemma12b dev-qwen dev-gemma12b | xargs -P 4 -I{} mbdef evaluate score --scorings $S/{}.json --record-scores $O/pair-scores/{}.json --out $O/scores/{} && (cd $O/pair-scores && sha256sum *.json > ../pair-scores.sha256)
END
)
uv run mbdef cluster render --site example --gpus 1 --time 02:00:00 \
    --model nli --model bge --job-name selection-rescore --step "$STEP" --out selection.sbatch
```

(`--site example` renders with `cluster/example.toml`; for a real job, write your cluster's site file and pass its name. The cluster folder, `CLUSTER_<SITE>_ROOT`, is set in your `.env`; [docs/setup/cluster.md](../setup/cluster.md#inputs-the-site-file-and-your-env).) The released job was rendered for an H100 partition of a cluster, with the GPU type and node-local disk in its `--gres`.

It unpacks the tar onto the node's own disk (1.4 s), scores the four lists in four processes at once on the one GPU (each process loads the two models and keeps its own cache of pair scores, so a pair that its 600 to 840 record sets share is scored once), writes every pair score to `pair-scores/<split>-<model>.json`, and writes those files' hashes. If any of the four fails, the job fails and `pair-scores.sha256` is not written. Render fills in the commit it was run at; the job refuses to run on any other.

**3. What is on the cluster before the job** (set up on the login node, which has network; compute nodes have none): `$ROOT/repo` checked out at the commit of the script and clean; `$ROOT/venv` from `uv sync --locked` at that commit; the two models, `mbdef cluster download --model nli --model bge --cache $ROOT/hf-cache/hub --out $ROOT/logs/models-<date>.json` (1.6 GB and 0.4 GB, no login needed); and `$ROOT/data-root` (the cluster's data root, [docs/setup/cluster.md](../setup/cluster.md)) holding `selection/rescore/2026-10-20/scorings.tar.zst`, whose sha256 is checked against the workstation's. Nothing else: the job reads no saved run and no reference outside the tar.

**4. After the job**, its outputs under `$ROOT/data-root/selection/rescore/2026-10-20/` on the cluster are `pair-scores/` (four files, 38.6 MB in all), `pair-scores.sha256` and `scores/<split>-<model>/` (each family's `scores.json` and receipt, 341 MB); the job's own record is `$ROOT/logs/job-<id>/` (`terminal.json` with the seconds it took, the GPU and package checks, `step.log`). They are copied into the workstation's data root, checked, backed up and listed in `data/data-root.tsv` (`cluster` is the login node's name in your SSH configuration):

```bash
rsync -az cluster:$ROOT/data-root/selection/rescore/2026-10-20/{pair-scores,scores,pair-scores.sha256} \
  $DATA_ROOT/selection/rescore/2026-10-20/
rsync -az cluster:$ROOT/logs/job-<id>/ cluster:$ROOT/logs/slurm-<id>.out \
  cluster:$ROOT/data-root/selection/rescore/2026-10-20/selection.sbatch $DATA_ROOT/selection/rescore/2026-10-20/job/
(cd $DATA_ROOT/selection/rescore/2026-10-20/pair-scores && sha256sum -c ../pair-scores.sha256)
```

`-z` matters: without compression the copy from the cluster ran at about 100 KB/s, so the 341 MB of `scores/` would have taken about an hour; with it, all of it took 3 minutes.

**What the job costs.** The numbers in the table are from a workstation's 24 GB GPU, measured before the job to plan it; the job's own are [below](#the-released-job). The work is 654,208 distinct entailment pairs, as the job recorded them: 302,299 on TEST for Qwen, 235,896 on TEST for Gemma, 70,560 and 45,453 on DEV for Qwen and Gemma (a count with a stand-in scorer had planned 301,543 and 235,822 for TEST), and 2,789 sets whose matching runs on one CPU core each.

| | Measured |
|---|---|
| Entailment model, fp32, one process, batch size 16, **32**, 64, 128 (37,593 pairs of DEV Gemma) | 360, **356**, 344, 315 pairs per second |
| 14 TEST Qwen sets, 21 TEST Gemma sets (205,343 and 189,736 pairs) | 389 and 390 pairs per second; matching 1.5 and 1.1 s per set |
| The job's command on the two DEV families (45,453 and 70,560 pairs, two processes on one GPU) | 6 min 59 s, GPU busy 80% of the time on average |
| Replaying those pair scores on the CPU (`evaluate grid --replay-scores`, 2 workers) | 129 s |

Why one GPU, four processes and batches of 32: the entailment model is compute-bound in fp32, so a larger batch is not faster (batch 128 is 11% slower than 32, because the shorter pairs of a batch are padded to its longest). The batch stays at **32**, as in every released scoring run, because another size moves a probability by about 1e-6, which could move a pair across the 0.5 cutoff and change a count that a rerun must reproduce exactly. TF32 or bf16 would be faster and would move probabilities by about 1e-3, so they are not used. Four processes keep the GPU busy while each one does its matching on its own CPU core. A second GPU would only shorten the GPU part and not the matching of the two TEST families, which is the longest chain (about 21 minutes of matching for Qwen and 15 to 25 for Gemma on the workstation). Splitting those two families over more processes would cut the walltime by roughly 15 to 20 minutes and needs a merge of the recorded pair scores; it is not done for one job.

**Expected time.** On the workstation's GPU the job would take about 35 to 45 minutes (the longest process, TEST Qwen: about 21 minutes of matching and 13 of entailment scoring, a little more when four processes share the GPU). An H100 scores the entailment part faster, and a cluster node's CPU cores may be slower than the workstation's, so the plan was **40 to 60 minutes, requested as 2 hours**. It took 28 min 30 s ([the released job](#the-released-job)).

**A rerun does not need the job again.** With the job's pair scores in the data root, all the code reruns on the CPU without the models: `evaluate grid --replay-scores $DATA_ROOT/selection/rescore/2026-10-20/pair-scores`, then `evaluate select --replaces` the stored scoring run with the corrected audit. Against both stored results files and the scoring run, only run details differ (times, how the work was spread, where the models ran, the paths of inputs, the names of the code): every count, eligibility reason, audit verdict, call count and pick is identical, and the secondary full-record score of each setup is within ±0.001 (per paper it is compared only through those means, and two single-run means are exceptions, below). A code change that asks for a pair score the job did not record stops the rerun with `no recorded …`; then the job must run again.
Without the job's pair scores in the data root (`data fetch selection/`), `evaluate grid --replay-scores` stops and says where they come from.

**Rehearsal on DEV.** Before the job, its command ran on the workstation's GPU for the two DEV families (the real tar, the real models), and `evaluate grid --replay-scores` then rebuilt all 1,134 DEV setups and 594 single cohorts from the recorded pair scores without a missing pair. Against the stored scoring run, matches, predicted definitions, reference count and model calls are identical for all 1,134 setups, and so are all 23,814 per-paper rows (tp, predicted, gold). The full-record score differs by at most 2.65e-4 (9 setups differ by more than 1e-6; the band is ±0.001, and [docs/setup/pinned.md](../setup/pinned.md) says why such differences arise).

### The released job

The rescoring ran once, as **Slurm job 3527672**. Its script (sha256 of `selection.sbatch` `fe27424e…`) asked for an H100 partition, `--qos=normal`, one H100 80GB, 8 CPUs, 100 GB, `--time 02:00:00`. Its input was the scorings tar `1b17f315…` (7,944,521 bytes), the same file on the workstation and on the cluster.

| | From the job's record (`job/` in the data root) |
|---|---|
| Slurm | COMPLETED, exit code 0, 08:26:32 to 08:55:02 UTC: **28 min 30 s** |
| The job's own record (`terminal.json`) | exit code 0, 1,708 s from the job's start to the end of the step |
| Checks before the step (GPU, packages, the models copied to node-local disk and checked) | passed, 40 s: Python 3.13.10 and 7 packages as in `uv.lock`; both models match their Hub object ids |
| DEV Qwen, DEV Gemma (70,560 and 45,453 pairs) | done 9 min 17 s and 9 min 22 s after the step began |
| TEST Qwen, TEST Gemma (302,299 and 235,896 pairs) | done 26 min 51 s and 27 min 45 s after the step began |
| Pair scores | four files, 38,608,063 bytes; all four pass `sha256sum -c` against the job's `pair-scores.sha256`, on the cluster and after the copy |

So the H100 job was faster than the workstation plan (35 to 45 minutes on its GPU): TEST Gemma, not TEST Qwen, was the last of the four to finish.

**The same pairs on two GPUs.** On DEV, the job's pair scores (H100) and the rehearsal's (the workstation's 24 GB GPU, above) cover exactly the same pairs: 23,143 and 35,868 similarities, 45,453 and 70,560 entailments. The entailment probabilities differ by at most 5.4e-5 and the BGE similarities by at most 3.6e-7, and no pair is on the other side of the 0.5 cutoff on one GPU than on the other.

**Where the outputs are.** The pair scores that a replay reads come from this job; the replay runs on whatever commit is checked out (its receipt names that commit). The job's files are in the data root under `selection/rescore/2026-10-20/` (`pair-scores/`, `pair-scores.sha256`, `scores/`, `job/` with the job's record, its Slurm log and the script), each listed in `data/data-root.tsv`, from which `uv run mbdef data fetch selection/` fetches them into an empty data root (an archive of 22,096,493 bytes, sha256 `6203c5e1…`, in about 9 s).

**What a replay found.** Replaying the job's pair scores on a workstation's CPU gives every count, eligibility reason, audit verdict, model-call count and pick of the stored results exactly (P≥28: 294 of 365 found, 961 predicted). Only the secondary full-record score moves. Of the 3,456 setup and single-run means, 73 move by more than 1e-6 and two by more than 0.001; of the 86,400 per-paper scores, 59 in 5 papers move by more than 0.001, with each paper's matches, predicted and reference definitions unchanged. The largest case: on the TEST paper `S0957417423021437`, Qwen's single run `B-minimal-16k32k/json-default` scores 0.3816 on the first run's A100 and 0.3365 from the H100's pair scores, because the score pairs each reference definition with one predicted record and two pairings nearly tie; that paper's change, spread over 29 papers, moves the run's mean from 0.17275 to 0.17119 (-0.00155, in its `combined` and `filtered-title-noverify` stages), while its 268 matches of 1,092 stay. It is the GPU, not the code: on the DEV paper `261a5f3b…` the workstation's 24 GB GPU gives exactly the A100's 0.2135 and the H100 0.2270, and the workstation's own DEV rehearsal moved 9 other per-paper scores by more than 0.001, so a rerun on another GPU would not remove it. So per-paper full-record scores are compared only through each setup's mean, as for the held-out TEST rescoring, and the two means that miss the ±0.001 band by 0.00155 are the known exceptions, each with its stored and its new value.

## How it was checked

- **The pick rule alone** ([`tests/test_evaluate_selection.py`](../../tests/test_evaluate_selection.py)): a miniature table of scored setups whose numbers make each part of the rule decide one pick: the 292-match floor, the 2-match tie, the precision tie-break, the fewer-calls tie-break and the audit exclusion.
- **Against the original implementation** ([`tests/test_evaluate_grid.py`](../../tests/test_evaluate_grid.py)): on a miniature grid (`tests/mini_grid.py`) with a made-up pair scorer, `evaluate grid` and `evaluate select` give what the original grid scorer and selector gave (stored in `tests/fixtures/released-outputs.zip`), apart from the run's time, the seconds each scoring took and the name of the merge function; replaying recorded pair scores equals the recorded run; a pair that was not recorded stops the scoring; and the general scorer scores the written scorings as the grid does.
- **Against the stored selection**: `evaluate select --replaces` on the stored scores writes `results.json` and `results.md` byte for byte as stored, apart from the run's time, the input paths and the code's name, and its replay with the first audit equals the scoring run's own files. It prints the four picks above: **P≥28 finds 294 of the 365 TEST definitions.**
- **The job** ([`tests/test_selection_job.py`](../../tests/test_selection_job.py)): the job script renders for the example site with `--qos=normal` and the two pinned models, passes `bash -n` and a stand-in `sbatch` is never called; the step, run under bash with a stand-in scorer, unpacks the scorings, calls the scorer once per family with the paths above, and fails if one family fails; and this page contains the step word for word.

## How to adapt it

- **Another grid:** the shape (models, prompt programs, formats, stages, pools) is the named constants at the top of `grid.py`; the reference and cohort folders are flags of `evaluate grid`.
- **Another rule:** the floors, the 292-match minimum and the 2-match tie are flags of `evaluate select` (`--precision-floors 20 24 28`, `--match-floor 292`, `--tie-matches 2`); their defaults are the named constants `FINAL_PRECISION_FLOORS`, `FINAL_MATCH_FLOOR` and `FINAL_TIE_MATCHES` in `selection.py`, and the receipt records the values used. For example `--precision-floors 20 30` gives two picks, `P>=20` and `P>=30`. The plan's hashes are checked, so a new plan needs its own `PLAN_SHA256`.
- **Another audit:** `--audit FILE` (and `--extra-audit` for cohorts it lacks).
- **Another cluster or GPU count:** render with another `--site` or `--gpus`; the step does not name either.

## Not in this step

- **The page that presented the selection** only formatted `results.json`; it is not part of this repository.
- **The grid's own runs** (the extraction, verification and filtering) are not remade here; their saved outputs are the inputs above.
