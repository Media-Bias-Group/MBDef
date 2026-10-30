"""`mbdef postprocess ensemble`: Step 5.4, keep or drop every Extracted Concept of the corpus.

The adopted method combines three members' scores of an Extracted Concept (a term as one paper defines it): the embedding classifier, the decision model basal-omni-1.0-ultra (FP8) and the GT-RAG judge (Gemma-4-12B, retrieval of 32 labelled ground-truth definitions, thinking off). Only the classifier runs here; basal's and the judge's scores were made on H200 GPUs and are read from their frozen score tables (docs/step-5-post-processing/postprocess.md gives the commands that made them).

How an Extracted Concept of the corpus is scored ("reference ranks", option (a) of the question where to cut):

  1. each member's score x is placed on that member's own train + dev scores (1,700 records), its reference:
     r = (#{reference < x} + 0.5 * #{reference == x}) / 1,700, so the score never depends on the other concepts scored;
  2. score = sum_m w_m r_m / sum_m |w_m|, with the frozen weights w of `pick.json` (a logistic stacking, C 0.01, fitted on the members' train + dev scores ranked among themselves);
  3. keep it when score >= the cut that keeps 95% of the relevant train + dev records, set out of fold on the same scale: each fold's records are placed on the other folds' scores.

Worked example (the adopted run): weights 0.931, 0.918 and 0.855 (34.4%, 33.9% and 31.6% of their sum). The Extracted Concept "Deep learning" of a seed paper has the scores 0.210, 0.172 and 0.005, at the reference ranks 0.428, 0.400 and 0.311, so it scores 0.344 * 0.428 + 0.339 * 0.400 + 0.316 * 0.311 = 0.381; the cut is 0.4150, so it is dropped (the classifier alone kept it). 62,066 of the 67,394 Extracted Concepts are kept.

Writes decisions/ensemble.csv (one row per Extracted Concept: the members' scores and reference ranks, the score, keep) and decisions/ensemble.json (cut, weights, counts, the inputs' sha256; for comparison the counts of the same weights on ranks within the corpus, the frozen recipe's variant (b), and of the classifier alone). Nothing is written when the share kept leaves 70% to 100% (a check fixed before the judge's corpus scores existed; the classifier alone keeps 92.1%).
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from mbdef.files import file_sha256

from . import frozen_file, groundtruth
from .compare import shown
from .evaluate import ScoreTable, f1_cut, paper_folds, recall_cut
from .label_match import group_counts
from .provenance import code_version
from .term_merging import truthy

MEMBERS = ("classifier", "basal", "judge")
RECALL = 0.95  # the cut keeps this share of the relevant train + dev records
# the share of corpus Extracted Concepts kept must lie in it, or nothing is written
SANITY = (0.70, 1.00)

# The inputs of the adopted run, each a frozen file `<date>/...` (`--pick`, `--weight-table`, ... override them). The frozen choice of members, combiner and weights:
PICK = "20261020/ensembles/pick.json"
# The members' train + dev score tables the frozen weights were fitted on (the pick's `train_dev_table`s)
WEIGHT_TABLES = {
    "classifier": "20261015/evaluation/scores/classifier.csv",
    "basal": "20261020/decision-models/scores/basal-fp8-h100-r1.csv",
    "judge": "20261020/judge-sweep/stage1/20261021/scores/s1-qwen8b-k32-think0-prob.csv",
}
# Each member's reference: its train + dev scores from the same GPU type as its corpus scores
REFERENCES = {
    "classifier": "20261015/evaluation/scores/classifier.csv",
    "basal": "20261021/corpus-decision-models/scores/basal-fp8-h200-gt-r1.csv",
    "judge": "20261020/judge-sweep/stage1/20261021/scores/s1-qwen8b-k32-think0-prob.csv",
}
# its corpus scores and its own keep
CLASSIFIER_DECISIONS = "20261015/decisions/classifier.csv"
CORPUS_SCORES = {
    "classifier": CLASSIFIER_DECISIONS,
    "basal": "20261021/corpus-decision-models/scores/basal-fp8-h200-corpus.csv",
    "judge": "20261020/judge-sweep/corpus/20261021/judge-corpus-scores.csv",
}
# every corpus Extracted Concept with its Label Group (Label Match C-20261015): the rows and their order (over 10 MB, so in the data root)
TERMS = "20261015/label-groups/terms.jsonl"


def default_inputs():
    """The inputs of the adopted run, as `decide_ensemble` takes them."""
    return dict(
        pick=frozen_file(PICK),
        weight_tables={m: frozen_file(p) for m, p in WEIGHT_TABLES.items()},
        references={m: frozen_file(p) for m, p in REFERENCES.items()},
        corpus_scores={m: frozen_file(p) for m, p in CORPUS_SCORES.items()},
        terms=frozen_file(TERMS),
        classifier_decisions=frozen_file(CLASSIFIER_DECISIONS),
    )


def norm_rank(v):
    """rank / n within the batch, rank 1 = the lowest score, ties at their average rank."""
    v = np.asarray(v, dtype=float)
    return rankdata(v, method="average") / len(v)


def ref_rank(x, ref):
    """The share of the reference scores below each x, ties counted half: (#{ref < x} + 0.5 #{ref == x}) / len(ref)."""
    ref = np.sort(np.asarray(ref, dtype=float))
    x = np.asarray(x, dtype=float)
    lo, hi = np.searchsorted(ref, x, "left"), np.searchsorted(ref, x, "right")
    return (lo + 0.5 * (hi - lo)) / len(ref)


def weighted(ranks, weights):
    """sum_m w_m r_m / sum_m |w_m|: the stacked score on the 0-1 scale of the ranks (one column per member)."""
    weights = np.asarray(weights, dtype=float)
    return ranks @ weights / np.abs(weights).sum()


def table_scores(path, records):
    """A member's train + dev scores, in the order of `records` (a score table in the shared format)."""
    scores = ScoreTable.read_csv(path).scores
    return np.array([scores[r.id] for r in records], dtype=float)


def refit_weights(tables, labels, C):
    """Stacking as frozen: L2 logistic regression with C on each member's within-batch ranks of all records."""
    from sklearn.linear_model import LogisticRegression

    ranks = np.column_stack([norm_rank(v) for v in tables])
    fit = LogisticRegression(C=C, max_iter=5000).fit(ranks, labels)
    return fit.coef_[0].copy(), float(fit.intercept_[0])


def reference_cuts(references, labels, fold, weights, recall=RECALL):
    """The 95%-recall and best-F1 cuts on the reference scale, out of fold: each fold's records are ranked against the other folds' scores of each member, combined with the frozen weights; the cuts are set on all records."""
    oof = np.empty(len(labels))
    for f in sorted(set(fold.tolist())):
        held, rest = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        oof[held] = weighted(
            np.column_stack([ref_rank(v[held], v[rest]) for v in references]), weights
        )
    return recall_cut(labels, oof, recall), f1_cut(labels, oof)


def read_corpus_scores(path):
    """{mention id: score} from a CSV keyed by mention_id or item_id, column `score`; and what was read."""
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no scores")
    key = "mention_id" if "mention_id" in rows[0] else "item_id"
    if len({r[key] for r in rows}) != len(rows):
        raise ValueError(
            f"{path}: more than one row per {key}; one score per term is expected"
        )
    return {r[key]: float(r["score"]) for r in rows}, dict(
        path=str(path), column="score", sha256=file_sha256(Path(path)), rows=len(rows)
    )


def decide_ensemble(
    *,
    pick,
    weight_tables,
    references,
    corpus_scores,
    terms,
    classifier_decisions,
    out,
    records=None,
    recall=RECALL,
    sanity=SANITY,
):
    """Score and decide every corpus Extracted Concept (module docstring); write `out`/ensemble.csv and .json.

    `records` are the train + dev ground-truth records the references are read for (default: the ground truth's)."""
    frozen = json.loads(Path(pick).read_text())
    members = frozen["members"]
    if frozen["scale"] != "weighted mean" or tuple(members) != MEMBERS:
        raise ValueError(f"{pick}: not the frozen stacking pick of {MEMBERS}")
    weights = np.array([frozen["weights"][m] for m in members])
    if records is None:
        records = groundtruth.load(splits=("train", "dev"))
    labels = np.array([r.label for r in records])
    folds = paper_folds(records)
    fold = np.array([folds[r.id] for r in records])
    refs = [table_scores(references[m], records) for m in members]
    cut, best_f1 = reference_cuts(refs, labels, fold, weights, recall)
    refit, intercept = refit_weights(
        [table_scores(weight_tables[m], records) for m in members],
        labels,
        frozen["C"],
    )

    concepts = [json.loads(line) for line in Path(terms).open()]
    mids = [c["mention_id"] for c in concepts]
    raw, used = {}, {}
    for m in members:
        scores, used[m] = read_corpus_scores(corpus_scores[m])
        missing = sum(i not in scores for i in mids)
        if missing:
            raise ValueError(
                f"{m}: {missing} of {len(mids)} corpus terms have no score"
            )
        raw[m] = np.array([scores[i] for i in mids], dtype=float)
        if not np.isfinite(raw[m]).all():
            raise ValueError(f"{m}: non-finite corpus score")
    ranks = {m: ref_rank(raw[m], ref) for m, ref in zip(members, refs, strict=True)}
    score = weighted(np.column_stack([ranks[m] for m in members]), weights)
    within = weighted(np.column_stack([norm_rank(raw[m]) for m in members]), weights)
    with Path(classifier_decisions).open(newline="") as f:
        clf = {r["mention_id"]: truthy(r["keep"]) for r in csv.DictReader(f)}
    keep_clf = np.array([clf[i] for i in mids])
    n_defs = np.array([c["n_definitions"] for c in concepts])
    paper = np.array([c["paper_id"] for c in concepts])
    seeds = np.array([c["corpus"] == "seeds" for c in concepts])

    def counts(kept):
        return dict(
            terms=int(kept.sum()),
            terms_share=float(kept.mean()),
            definitions=int(n_defs[kept].sum()),
            definitions_share=float(n_defs[kept].sum() / n_defs.sum()),
            papers=len(set(paper[kept].tolist())),
            of=dict(
                terms=len(mids),
                definitions=int(n_defs.sum()),
                papers=len(set(paper.tolist())),
            ),
        )

    keep = score >= cut
    share = float(keep.mean())
    low, high = sanity
    check = dict(
        rule=f"the share of corpus terms (a) keeps at its {recall:.0%}-recall cut must lie in [{low:.2f}, {high:.2f}]; "
        "otherwise no decisions file",
        share_a=share,
        share_classifier=float(keep_clf.mean()),
        passed=bool(low <= share <= high),
    )
    if not check["passed"]:
        raise SystemExit(
            f"sanity rule failed: (a) keeps {share:.1%} of corpus terms, outside {low:.0%} to {high:.0%}; "
            "no decisions file written"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ensemble.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mention_id",
                "paper_id",
                "corpus",
                "label",
                *[f"{m}_score" for m in members],
                *[f"{m}_ref_rank" for m in members],
                "score",
                "keep",
            ]
        )
        for j, c in enumerate(concepts):
            w.writerow(
                [
                    c["mention_id"],
                    c["paper_id"],
                    c["corpus"],
                    c["term"],
                    *[repr(float(raw[m][j])) for m in members],
                    *[repr(float(ranks[m][j])) for m in members],
                    repr(float(score[j])),
                    bool(keep[j]),
                ]
            )
    kept = counts(keep)
    report = dict(
        path="ensemble",
        method=f"ensemble: {' + '.join(members)} | {frozen['combiner']}, reference-scale ranks (option a)",
        cut=cut,
        target_recall=recall,
        cut_from="train + dev out-of-fold scores on the reference scale (frozen v2 weights)",
        best_f1_cut=best_f1,
        weights=frozen["weights"],
        score="sum_m w_m r_m / sum_m |w_m|, r_m = (#{ref < x} + 0.5 #{ref = x}) / n_ref",
        references={m: shown(references[m]) for m in members},
        inputs={m: {**used[m], "path": shown(used[m]["path"])} for m in members},
        counts=dict(
            terms=len(mids),
            kept=int(keep.sum()),
            excluded=int((~keep).sum()),
            seeds=dict(terms=int(seeds.sum()), kept=int((keep & seeds).sum())),
            closure=dict(terms=int((~seeds).sum()), kept=int((keep & ~seeds).sum())),
            label_groups=group_counts([c["group_id"] for c in concepts], keep),
            definitions=dict(total=kept["of"]["definitions"], kept=kept["definitions"]),
            papers=dict(total=kept["of"]["papers"], kept=kept["papers"]),
        ),
        comparison={
            f"(a) reference ranks, {recall:.0%}-recall cut": counts(keep),
            "(a) reference ranks, best-F1 cut": counts(score >= best_f1),
            "(b) within-corpus ranks, frozen 95%-recall cut": counts(
                within >= frozen["cuts"]["cut"]
            ),
            "(b) within-corpus ranks, frozen best-F1 cut": counts(
                within >= frozen["cuts"]["f1_cut"]
            ),
            "classifier alone (its frozen decisions, cut 0.1848)": counts(keep_clf),
        },
        sanity=check,
        nli=None,
        pick=shown(pick),
        pick_sha256=file_sha256(Path(pick)),
        weights_refit=dict(
            weights=dict(zip(members, refit.tolist(), strict=True)),
            intercept=intercept,
            max_abs_difference=float(np.abs(refit - weights).max()),
            note="the stacking refitted on the members' train + dev tables; the decisions use the frozen weights",
        ),
        provenance=code_version(),
    )
    (out / "ensemble.json").write_text(json.dumps(report, indent=1))
    return report


def ensemble_command(out, **inputs):
    """`postprocess ensemble`: decide every corpus Extracted Concept into `out` and say what was kept."""
    r = decide_ensemble(**inputs, out=out)
    k = r["counts"]
    print(
        f"cut {r['cut']:.4f}: kept {k['kept']:,} of {k['terms']:,} Extracted Concepts ({r['sanity']['share_a']:.1%}; "
        f"the classifier alone {r['sanity']['share_classifier']:.1%}), {k['definitions']['kept']:,} of "
        f"{k['definitions']['total']:,} definitions, {k['papers']['kept']:,} of {k['papers']['total']:,} papers; "
        f"wrote {out}"
    )
    return 0
