"""The extractor's own relevance filter, replayed on the ground truth: which records would have reached the corpus?

Before an extracted term entered the corpus, the extraction run kept it only if a frozen logistic regression on BGE-base embeddings of "<paper title>\\n<term>\\n<definition 1>\\n..." gave it a probability of at least 0.2372 (the cut that kept 95% of the relevant training records out of fold). The regression was trained on 1,171 train records of 80 train papers (592 relevant, 579 not); it never saw a dev or test paper. A ground-truth record that scores below the cut would never have been in the corpus, so the records at or above it are the corpus-like ground truth.

The filter is the package's one copy, `extract/relevance.py` (`title`, the study semantic-title-relevance-v2): its encoder and its frozen classifier, checked against their sha256 when loaded. Training records get the filter's own out-of-fold probability (saved with the filter), all others the final model's. Nothing is refitted.

By default only the train and dev records are scored. The stored scores (`20261015/extractor-gate/gate-scores.csv`) also hold the TEST records: scoring those again is a read of the TEST split (a record's id even names its label), so it needs `--read-test --reason "..."` and goes through the TEST split's guard (`testrun.log_test_read`: a loud warning, confirmation at the prompt, and an entry in the read log). The read log is not part of this repository, so here `--read-test` stops before anything of TEST is read (`testrun.start_from_frozen`).

Worked example: the dev record "016f7a07…:1:1" scores 0.386, above 0.2372, so it would have reached the corpus.
"""

import json
import zipfile
from pathlib import Path

import numpy as np

from mbdef.extract import relevance

from . import FROZEN, REPO, groundtruth
from .evaluate import ScoreTable

FILTER = "title"  # its name in extract/relevance.py (FILTERS)
STUDY = REPO / relevance.ZIPS / f"{relevance.FILTERS[FILTER]['study']}.zip"
NAME = "extractor relevance filter"
PATH = "extractor-filter"  # its name among the frozen paths of the test run (a comparison row)
SUBSET = "passes the extractor's relevance filter (corpus-like)"


def study_file(name):
    """The bytes of one file of the filter's study (its saved out-of-fold and dev probabilities)."""
    with zipfile.ZipFile(STUDY) as z:
        return z.read(name)


def cutoff():
    return float(relevance.FILTERS[FILTER]["cutoff"])


def titles():
    """{paper id: title} from the ground-truth files' paper metadata (the filter's own title source)."""
    out = {}
    for path, _ in groundtruth.paths():
        for p in json.loads(path.read_text())["papers"]:
            if (
                out.setdefault(p["paper_id"], p["metadata"]["title"])
                != p["metadata"]["title"]
            ):
                raise ValueError(f"conflicting titles for {p['paper_id']}")
    return out


def feature(title, term, definitions):
    """The filter's input text: the paper's title, the term and each definition, one per line."""
    return relevance.title_feature(
        title, {"term": term, "definitions": [{"text": d} for d in definitions]}
    )


class Filter:
    """The frozen `title` filter of extract/relevance.py, on texts given all at once (32 per batch, in order)."""

    def __init__(self, device="auto"):
        self.loaded = relevance.load_filter(FILTER, REPO / relevance.ZIPS)
        self.encoder = relevance.Encoder(device=device)

    def probabilities(self, texts):
        """P(relevant) per feature text: BGE's CLS vector, L2-normalised, fp32, no truncation (as in the extraction run)."""
        return self.loaded["classifier"].predict_proba(self.encoder.vectors(texts))[
            :, 1
        ]


def training_oof(records):
    """{record id: out-of-fold probability} for the ground-truth records the filter was trained on; each must be the same text the filter saw."""
    oof = json.loads(study_file("oof.json"))
    by_id, title = {r.id: r for r in records}, titles()
    out = {}
    for o in oof:
        rid = f"{o['paper_id']}:{o['relevance']}:{o['record_index']}"
        if rid in by_id:
            r = by_id[rid]
            if feature(title[r.paper_id], r.term, r.definitions) != o["feature_text"]:
                raise ValueError(
                    f"filter training record {rid} is not the ground-truth record ({o['term']!r})"
                )
            out[rid] = float(o["probability"])
    return out


def gate_scores(records, filt):
    """ScoreTable of the filter's probability for every record (out of fold for its own training records)."""
    title = titles()
    p = filt.probabilities(
        [feature(title[r.paper_id], r.term, r.definitions) for r in records]
    )
    scores = {r.id: float(v) for r, v in zip(records, p, strict=True)}
    scores.update(training_oof(records))
    return ScoreTable(NAME, "term", scores, {r.id: r.split for r in records})


def passing(table, cut=None):
    """Ids of the records the filter would have let into the corpus."""
    cut = cutoff() if cut is None else cut
    return {i for i, s in table.scores.items() if s >= cut}


def reproduction(filt):
    """Largest difference between this replay and the probabilities the extraction run saved for its dev records."""
    saved = json.loads(study_file("decisions.json"))
    mine = filt.probabilities([d["feature_text"] for d in saved])
    return dict(
        records=len(saved),
        max_abs_diff=float(
            np.max(np.abs(mine - np.array([float(d["probability"]) for d in saved])))
        ),
    )


def gate_command(out, device="auto", read_test=False, reason=None):
    """`postprocess extractor-gate`: score the ground truth's train + dev records (with `read_test`, every split, after the TEST guard) with the extractor's relevance filter into `out`: gate-scores.csv and gate.json."""
    from . import groundtruth
    from .testrun import log_test_read, start_from_frozen

    out = Path(out)
    if read_test:
        start_from_frozen(out, REPO / FROZEN / "20261015/test-evaluation")
        log_test_read(out, [PATH], reason)
        records = groundtruth.load()
    else:
        records = groundtruth.load(splits=("train", "dev"))
    filt = Filter(device)
    table = gate_scores(records, filt)
    out.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "gate-scores.csv")
    cut, kept = cutoff(), passing(table)
    counts = {}
    for r in records:
        c = counts.setdefault(r.split, {"relevant": [0, 0], "not relevant": [0, 0]})
        c["relevant" if r.label else "not relevant"][r.id in kept] += 1
    summary = dict(
        filter=str(STUDY.relative_to(REPO)),
        cutoff=cut,
        model=relevance.ENCODER[0],
        revision=relevance.ENCODER[1],
        splits=sorted(counts),
        out_of_fold_training_records=len(training_oof(records)),
        reproduction=reproduction(filt),
        passing={
            split: {k: dict(dropped=v[0], kept=v[1]) for k, v in c.items()}
            for split, c in counts.items()
        },
    )
    (out / "gate.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0
