"""`mbdef postprocess decide --path <name>`: one path's keep or drop for every corpus Extracted Concept.

The cut keeps 95% of the relevant records in the path's train + dev out-of-fold scores (`<inputs>/evaluation/scores/<path>.csv`, recomputed when missing). The entailment paths score the corpus with the exact reference scorer by default; with --fast they use the fast scorer, rescoring with the reference every pair within --band of the path's cut, and record an audit of the fast scores against the reference on a random sample. Writes `<out>/<path>.csv` (one row per Extracted Concept: mention_id, paper_id, corpus, label, score, keep) and `<out>/<path>.json` (counts, cut, provenance).

Worked example (the frozen classifier): its train + dev out-of-fold scores give the cut 0.1848, which keeps 62,084 of the 67,394 Extracted Concepts; "observer strategies" in a seed paper scores 0.785, so it is kept.
"""

import csv
import json
from pathlib import Path

import numpy as np

from . import groundtruth
from .compare import make_in_setting, provenance
from .corpus import load_view
from .evaluate import ScoreTable, cross_validate, paper_folds, recall_cut
from .frozen import frozen_kwargs
from .label_match import RULE_VERSION, label_groups
from .paths.clustering import PINNED
from .testrun import corpus_decisions

RECALL = 0.95  # the cut keeps this share of the relevant train + dev records
BAND = 0.015  # with --fast: pairs this close to the cut are rescored with the reference


def decide_path(
    name,
    *,
    inputs,
    out,
    recall=RECALL,
    fast=False,
    band=BAND,
    pinned=PINNED,
    rule=RULE_VERSION,
):
    """Write `out`/<name>.csv and .json (module docstring) and return the counts. `inputs` is the date folder of the saved out-of-fold scores and the Representative Definitions; `rule` is the version of the Label Match rule of the Label Groups (the "@group" setting and the counts of split groups)."""
    inputs, out = Path(inputs), Path(out)
    train = groundtruth.load(splits=("train", "dev"))
    corpus = load_view().concepts
    shared = {}
    oof_file = inputs / "evaluation/scores" / f"{name.replace(':', '-')}.csv"
    if oof_file.exists():
        oof = ScoreTable.read_csv(oof_file)
    else:
        oof = cross_validate(
            make_in_setting(
                name, {}, corpus, inputs, frozen_kwargs(name), pinned, rule
            ),
            train,
            folds=paper_folds(train),
            corpus=corpus,
        )
    y = np.array([r.label for r in train])
    cut = recall_cut(y, np.array([oof.scores[r.id] for r in train]), recall)
    nli_path = name.startswith(("representative", "zero-shot"))
    if nli_path and fast:
        from .nli import BandedNLI

        shared["nli"] = BandedNLI(threshold=cut, band=band)
    rows, counts = corpus_decisions(
        make_in_setting(
            name, shared, corpus, inputs, frozen_kwargs(name), pinned, rule
        ),
        train,
        corpus,
        cut,
        groups=label_groups([c.label for c in corpus], rule=rule),
    )
    out.mkdir(parents=True, exist_ok=True)
    file = name.replace(":", "-")
    with (out / f"{file}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = dict(
        path=name,
        cut=cut,
        target_recall=recall,
        cut_from="train + dev out-of-fold scores",
        counts=counts,
        nli=None
        if not nli_path
        else (
            f"fast fp16, exact rescoring within {band:g} of the cut"
            if fast
            else "exact reference"
        ),
        provenance=provenance(),
    )
    scorer = shared.get("nli")
    if nli_path and fast and scorer is not None:
        report["rescored_pairs"] = scorer.last_rescored
        report["audit"] = scorer.audit(scorer.last_pairs)
    (out / f"{file}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(path=name, cut=round(cut, 4), **counts)))
    return 0
