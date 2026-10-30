# Frozen inputs of the gated outputs

These files are inputs of outputs the baseline gate checks (spec #42, layer 2), so they must stay readable on `dev`
(list (a) of #46, CG-4). They sit here at their original repository paths: `data/frozen/results/grid-20260925/plan.json`
was `results/grid-20260925/plan.json` in the cleanup base `cleanup-base-2026-10-19`. They are not to be edited.

- `results/test-eval-20260924/inputs/` and `results/qwen-replication-sweep-20260923/dev-reference.json`: the TEST and
  DEV reference definitions and the TEST readiness record.
- `results/grid-20260925/plan.json`, `plan-amendment-1.json` to `-9.json` and `audit/audit.json`: the 25 Sep grid plan,
  whose hashes the scorer checks, and its audit.
- `results/semantic-relevance-v1.zip`, `semantic-title-relevance-v2.zip`, `semantic-title-blocks-v3.zip`: the frozen
  relevance filters.
- `results/training-20260925/plan.json`, `plan-amendment-1.json` and `runner/{train-dataset,split,job-plan}.json`: the
  prompt training of 25 Sep (IS-Q6 on spec #42). The plans are the ones the released program manifests in
  `data/extractors/` name by sha256; the dataset (the 80 TRAIN and 21 DEV papers with their references) and the split
  are what `taxonomy_pipeline extract train` reads (docs/extract.md), and the job plan pinned both. Training is not
  gated; these are kept so it can be rerun as released.
- `artifacts/defsim_benchmark.json`: the fit record of the M3 settings (the two numbers are in `src/metrics.py`).
- `labelStudioXML.md`, the TEST analysis inputs, the report inputs of the selection and the closure audit inputs.
- `docs/closure-loss-audit.md`: the version at commit `e1be09f` that the closure bundle rebuild reads (CG-6);
  `docs/closure-loss-audit.md` on `dev` is the later version.

- `step5-post-processing/<date>/`: Step 5's frozen inputs and the stored outputs the gate compares with (the pinned
  cluster map, the vote tallies, the extractor filter's saved scores, the frozen paths, the TEST result and its read
  log, the classifier's scores, the ensemble's pick and its members' score tables, and the 21 Oct Label Group, decision
  and term-merging files). They were under `results/step5-post-processing/` while Step 5 was open and moved here, with
  the same `<date>/…` paths, when Step 5 was closed (22 Oct 2026). Those over 10 MB are in the data root under
  `frozen/step5-post-processing/` (`data/data-root.tsv`); docs/postprocess.md lists them all. A release leaves out
  the TEST result and its read log (`20261015/test-evaluation/test-eval.json` and `test-reads.json`); they stay on
  `dev`, where the gate reads them.

The selection's `results.json` (26.7 MB), a frozen input over 10 MB, is not in git on `dev`: `data/data-root.tsv`
gives its path under `DATA_ROOT`, its sha256 and where to get it.
