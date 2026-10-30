"""Step 5.3: one evaluation for every relevance path.

A path's output is a ScoreTable (one score per item, higher = more Relevant). `evaluate` turns score tables and the ground-truth Records into the comparison table.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from . import SEED


@dataclass(frozen=True)
class ScoreTable:
    method: str
    unit: str  # "term" or "label group"
    scores: dict[str, float] = field(default_factory=dict)  # item id -> score
    splits: dict[str, str] = field(
        default_factory=dict
    )  # item id -> train | dev | test | corpus, when known

    def write_csv(self, path):
        """The shared score-table format: columns method, unit, item_id, split, score (one row per item)."""
        with Path(path).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "unit", "item_id", "split", "score"])
            for item, v in self.scores.items():
                w.writerow(
                    [
                        self.method,
                        self.unit,
                        item,
                        self.splits.get(item, ""),
                        repr(float(v)),
                    ]
                )

    @classmethod
    def read_csv(cls, path):
        with Path(path).open(newline="") as f:
            rows = list(csv.DictReader(f))
        methods, units = {r["method"] for r in rows}, {r["unit"] for r in rows}
        if len(methods) != 1 or len(units) != 1:
            raise ValueError(
                f"{path}: one method and one unit per score table, got {methods} / {units}"
            )
        return cls(
            methods.pop(),
            units.pop(),
            {r["item_id"]: float(r["score"]) for r in rows},
            {r["item_id"]: r["split"] for r in rows if r.get("split")},
        )


def auc(y, s):
    """Chance that a random relevant item scores above a random non-relevant one; ties count half (Mann-Whitney).
    NaN when one of the two classes is empty."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ranks = rankdata(s)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    """Sum over score thresholds (tied scores together) of precision times the recall gained there; NaN without a relevant item."""
    from sklearn.metrics import average_precision_score

    if not np.asarray(y).sum():
        return float("nan")
    return float(average_precision_score(y, s))


def at_k(y, s, k):
    """(precision@k, recall@k) as expected values when the cut at rank k falls inside a block of tied scores."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    v = np.sort(s)[::-1][k - 1]
    above, tied = s > v, s == v
    hits = y[above].sum() + (k - above.sum()) * y[tied].mean()
    return float(hits / k), float(hits / y.sum())


def curves(y, s):
    """ROC points (FPR, TPR) from (0, 0) and precision-recall points, one per distinct score, keeping every item at or above it (tied scores enter together)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    roc, pr = [dict(threshold=None, fpr=0.0, tpr=0.0)], []
    for t in sorted(set(s.tolist()), reverse=True):
        keep = s >= t
        tp, fp = int((keep & (y == 1)).sum()), int((keep & (y == 0)).sum())
        roc.append(
            dict(threshold=t, fpr=fp / n0 if n0 else 0.0, tpr=tp / n1 if n1 else 0.0)
        )
        pr.append(
            dict(threshold=t, recall=tp / n1 if n1 else 0.0, precision=tp / (tp + fp))
        )
    return dict(roc=roc, pr=pr)


def paper_folds(records, k=5, seed=SEED):
    """{record id: fold}: stratified by label, a paper never split across folds (StratifiedGroupKFold, shuffled)."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.array([r.label for r in records])
    groups = np.array([r.paper_id for r in records])
    out = {}
    for f, (_, te) in enumerate(
        StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y, groups
        )
    ):
        for i in te:
            out[records[i].id] = f
    return out


def cross_validate(path, records, *, folds, corpus=None, unit="term"):
    """Out-of-fold scores: for each fold, fit the path on the other folds and score the held-out one.
    Only train and dev records are allowed: the test split is read once, by the test run, never here."""
    if any(r.split == "test" for r in records):
        raise ValueError(
            "cross-validation got test-split records; the test split is evaluated only once, separately"
        )
    scores = {}
    for f in sorted(set(folds[r.id] for r in records)):
        train = [r for r in records if folds[r.id] != f]
        held = [r for r in records if folds[r.id] == f]
        path.fit(train, corpus)
        scores.update(
            {r.id: float(v) for r, v in zip(held, path.score(held), strict=True)}
        )
    return ScoreTable(
        method=path.name,
        unit=unit,
        scores=scores,
        splits={r.id: r.split for r in records},
    )


def recall_cut(y, s, recall):
    """Largest score threshold that keeps at least `recall` of the relevant items (score >= cut is kept)."""
    pos = np.sort(np.asarray(s, dtype=float)[np.asarray(y) == 1])[::-1]
    return float(pos[int(np.ceil(recall * len(pos))) - 1])


def f1_cut(y, s):
    """Score threshold with the highest F1 (2PR / (P + R)) when every item scoring at or above it is kept; on a tie of F1 the higher threshold (fewer kept) wins."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    last = np.r_[
        np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1
    ]  # last index of each block of tied scores
    tp = np.cumsum(ys)[last]
    f1 = 2 * tp / (last + 1 + y.sum())  # 2TP / (kept + relevant)
    return float(ss[last[int(np.argmax(f1))]])


def decisions(y, s, keep):
    """The confusion matrix of keep/drop decisions (kept = predicted Relevant) and every measure read off it."""
    y, keep = np.asarray(y), np.asarray(keep, dtype=bool)
    tp, fp = int((keep & (y == 1)).sum()), int((keep & (y == 0)).sum())
    fn, tn = int((~keep & (y == 1)).sum()), int((~keep & (y == 0)).sum())
    n = tp + fp + fn + tn
    ratio = lambda a, b: a / b if b else 0.0  # noqa: E731
    recall, precision = ratio(tp, tp + fn), ratio(tp, tp + fp)
    specificity, npv = ratio(tn, tn + fp), ratio(tn, tn + fn)
    den = float(np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    accuracy = ratio(tp + tn, n)
    chance = ratio((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), n * n)
    return dict(
        recall=recall,
        precision=precision,
        f1=ratio(2 * tp, 2 * tp + fp + fn),
        f2=ratio(5 * tp, 5 * tp + 4 * fn + fp),
        accuracy=accuracy,
        balanced_accuracy=(recall + specificity) / 2,
        specificity=specificity,
        npv=npv,
        fpr=ratio(fp, fp + tn),
        mcc=(tp * tn - fp * fn) / den if den else 0.0,
        kappa=ratio(accuracy - chance, 1 - chance),
        youden_j=recall + specificity - 1,
        excluded_share=float(1 - keep.mean()),
        kept=tp + fp,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def cv_keep(y, s, fold, fit_cut):
    """Honest operating point inside cross-validation: each held-out fold is cut at the threshold `fit_cut(y, s)` sets on the other folds."""
    keep = np.zeros(len(y), dtype=bool)
    for f in np.unique(fold):
        other = fold != f
        keep[~other] = s[~other] >= fit_cut(y[other], s[other])
    return keep


def cv_cut_decisions(y, s, fold, recall):
    """Keep/drop of every record at the cut that keeps `recall` of the relevant records of the other folds."""
    return cv_keep(y, s, fold, lambda a, b: recall_cut(a, b, recall))


def precision_if_relevant_share(y, s, recall, shares, cut=None):
    """Precision a collection with another share of relevant items would see at a cut:
    P = TPR*pi / (TPR*pi + FPR*(1 - pi)), with TPR and FPR read off the given records. The cut is the given one (fixed on other data, e.g. train + dev for the test split) or, when none is given, this ranking's own recall cut."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    cut = recall_cut(y, s, recall) if cut is None else cut
    tpr, fpr = float((s[y == 1] >= cut).mean()), float((s[y == 0] >= cut).mean())
    return {
        pi: (
            tpr * pi / (tpr * pi + fpr * (1 - pi))
            if tpr * pi + fpr * (1 - pi) > 0
            else None
        )
        for pi in shares
    }


def paper_resamples(records, draws, seed):
    """Index arrays of bootstrap resamples: papers drawn with replacement, all of a drawn paper's records kept.
    Every method is scored on the same resamples, so differences between methods are paired."""
    rng = np.random.default_rng(seed)
    papers = np.array([r.paper_id for r in records])
    uniq = np.unique(papers)
    by = {p: np.flatnonzero(papers == p) for p in uniq}
    y = np.array([r.label for r in records])
    out = []
    for _ in range(draws):
        idx = np.concatenate([by[p] for p in rng.choice(uniq, len(uniq))])
        if 0 < y[idx].sum() < len(idx):
            out.append(idx)
    return out


def evaluate(
    tables,
    records,
    *,
    folds=None,
    recall=0.95,
    ks=(100, 300),
    prevalences=(0.01, 0.10),
    draws=2000,
    seed=SEED,
    cuts=None,
    f1_cuts=None,
    subsets=None,
    recalls=(0.80, 0.90, 0.95),
):
    """Comparison table: {method: metrics} for score tables over the given ground-truth records.

    Two operating points get the full confusion-matrix measures (see `decisions`): "cut", the threshold keeping `recall` of the relevant records, and "f1_cut", the threshold with the highest F1. With `folds` both are set on the other folds; `cuts` / `f1_cuts` ({method: threshold}, fixed on other data) override them. Subsets get the same measures at the same thresholds."""
    y = np.array([r.label for r in records])
    S = {
        name: np.array([t.scores[r.id] for r in records], dtype=float)
        for name, t in tables.items()
    }
    resamples = paper_resamples(records, draws, seed)
    boot = {
        name: np.array([auc(y[i], s[i]) for i in resamples]) for name, s in S.items()
    }
    boot_ap = {
        name: np.array([average_precision(y[i], s[i]) for i in resamples])
        for name, s in S.items()
    }
    unan = np.array([bool(r.unanimous) for r in records])
    fold = None if folds is None else np.array([folds[r.id] for r in records])
    out = {}
    for name, s in S.items():
        at = {k: at_k(y, s, k) for k in ks if k <= len(s)}
        out[name] = dict(
            auc=auc(y, s),
            auc_ci95=np.percentile(boot[name], [2.5, 97.5]).tolist(),
            ap=average_precision(y, s),
            ap_ci95=np.percentile(boot_ap[name], [2.5, 97.5]).tolist(),
            ap_baseline=float(y.mean()),
            precision_at={k: p for k, (p, _) in at.items()},
            recall_at={k: r for k, (_, r) in at.items()},
            precision_at_recall={
                q: float(y[s >= recall_cut(y, s, q)].mean()) for q in recalls
            },
            precision_if_relevant_share=precision_if_relevant_share(
                y, s, recall, prevalences, (cuts or {}).get(name)
            ),
            delta_auc_vs={
                o: dict(
                    mean=float((boot[name] - boot[o]).mean()),
                    ci95=np.percentile(boot[name] - boot[o], [2.5, 97.5]).tolist(),
                )
                for o in S
                if o != name
            },
            records=len(records),
            relevant=int(y.sum()),
            papers=len({r.paper_id for r in records}),
            bootstrap=dict(draws=len(resamples), seed=seed, unit="paper"),
        )
        keeps = {}
        if fold is not None:
            keeps["cut"] = (
                cv_cut_decisions(y, s, fold, recall),
                dict(target_recall=recall, set_on="other folds"),
            )
            keeps["f1_cut"] = (
                cv_keep(y, s, fold, f1_cut),
                dict(target="highest F1", set_on="other folds"),
            )
        if cuts and name in cuts:
            keeps["cut"] = (
                s >= cuts[name],
                dict(cut=cuts[name], target_recall=recall, set_on="given"),
            )
        if f1_cuts and name in f1_cuts:
            keeps["f1_cut"] = (
                s >= f1_cuts[name],
                dict(cut=f1_cuts[name], target="highest F1", set_on="given"),
            )
        for point, (keep, info) in keeps.items():
            out[name][point] = dict(**info, **decisions(y, s, keep))
        subs = {}
        for label, ids in (subsets or {}).items():
            sub = np.array([r.id in ids for r in records])
            subs[label] = dict(
                records=int(sub.sum()),
                relevant=int(y[sub].sum()),
                auc=auc(y[sub], s[sub]),
                ap=average_precision(y[sub], s[sub]),
                ap_baseline=float(y[sub].mean()),
                **{
                    point: decisions(y[sub], s[sub], keep[sub])
                    for point, (keep, _) in keeps.items()
                },
            )
        if subs:
            out[name]["subsets"] = subs
        if unan.any() and 0 < y[unan].sum() < unan.sum():
            out[name]["unanimous"] = dict(
                auc=auc(y[unan], s[unan]),
                ap=average_precision(y[unan], s[unan]),
                records=int(unan.sum()),
            )
    return out
