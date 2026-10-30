"""`mbdef postprocess evaluate`: cross-validate relevance paths on train + dev and write the comparison.

Path 1 (clustering) reads the pinned cluster map (paths/clustering.py, PINNED) unless --refit-clusters.

Outputs in the folder `--out` names:
  scores/<path>.csv   out-of-fold score table per path (shared score-table format)
  metrics.json        the comparison table (evaluate.evaluate) with folds, seeds and provenance
  curves.json         ROC and precision-recall points of every method (evaluate.curves)
"""

import datetime as dt
import json
from pathlib import Path

from mbdef import repo

from . import REPO, SEED, groundtruth, stored
from .evaluate import ScoreTable, cross_validate, curves, evaluate, paper_folds
from .frozen import frozen_kwargs
from .label_match import RULE_VERSION
from .paths import make
from .paths.clustering import PINNED


def provenance():
    try:
        commit = repo.git_or_none(REPO, "rev-parse", "HEAD")
        status = repo.git_or_none(
            REPO, "status", "--porcelain", "src/mbdef/postprocess"
        )
    except OSError:
        commit = None
    if (
        commit is None
    ):  # no git, or not a checkout (the repository unpacked as plain files)
        commit, dirty = None, None
    else:
        commit, dirty = commit.decode().strip(), bool((status or b"").strip())
    return dict(
        commit=commit,
        postprocess_dirty=dirty,
        run_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )


def shown(path):
    """A path as a receipt shows it: relative to the repository when inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def representatives(base):
    """{word-set key: representative text}: the Representative Definition of every Label Group of Label Match C-20261015, which the Label Group setting of the entailment paths scores (a frozen input)."""
    f = stored(base, "group-representatives/representatives.jsonl")
    if not f.exists():
        raise SystemExit(
            f"{f} is missing: get it with `mbdef data fetch frozen/step5-post-processing`"
        )
    with f.open() as fh:
        return {r["key"]: r["representative"]["text"] for r in map(json.loads, fh)}


def make_in_setting(
    name, shared, corpus, base, kwargs=None, pinned=PINNED, rule=RULE_VERSION
):
    """A path by name; with the suffix "@group" wrapped in the Label Group setting (path 1: all of a group's definitions; path 2: the group's Representative Definition), its groups made with Label Match version `rule`. Path 1 reads the cluster map `pinned` (None: fit a new k-means)."""
    if name.partition("@")[0].partition(":")[0] == "clustering":
        kwargs = {**(kwargs or {}), "pinned": pinned}
    if not name.endswith("@group"):
        return make(name, shared, kwargs)
    from .group_setting import GroupSetting

    inner = make(name[: -len("@group")], shared, kwargs)
    if name.startswith(("representative", "zero-shot")):
        return GroupSetting(
            inner,
            corpus,
            member="representative",
            representatives=representatives(base),
            rule=rule,
        )
    return GroupSetting(inner, corpus, member="all", rule=rule)


def evaluate_paths(
    names, *, inputs, out, extra=(), draws=2000, pinned=PINNED, rule=RULE_VERSION
):
    """Cross-validate the paths `names` on train + dev into `out` (see the module docstring); `inputs` is the date folder whose saved representatives and extractor gate scores the Label Group setting and the comparison read."""
    records = groundtruth.load(splits=("train", "dev"))
    folds = paper_folds(records, k=5, seed=SEED)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    tables, shared = {}, {}
    corpus = None
    if any(n.partition(":")[0] == "clustering" or n.endswith("@group") for n in names):
        from .corpus import load_view

        corpus = load_view().concepts
    for name in names:
        path = make_in_setting(
            name, shared, corpus, inputs, frozen_kwargs(name), pinned, rule
        )  # frozen settings, if frozen
        table = cross_validate(
            path,
            records,
            folds=folds,
            corpus=corpus,
            unit="label group" if name.endswith("@group") else "term",
        )
        table.write_csv(out / "scores" / f"{name.replace(':', '-')}.csv")
        tables[table.method] = table
    for f in [
        *sorted((out / "scores").glob("*.csv")),
        *extra,
    ]:  # every table of this run folder is compared together
        t = ScoreTable.read_csv(f)
        tables.setdefault(t.method, t)
    subsets = {}
    if any(t.unit == "label group" for t in tables.values()):
        from .corpus import load_view
        from .group_setting import GroupSetting

        probe = GroupSetting(
            make("keyword"),
            corpus if corpus is not None else load_view().concepts,
            rule=rule,
        )
        subsets["matched to a corpus Label Group"] = {
            r.id for r in records if probe.group_of(r) is not None
        }
    gate = stored(inputs, "extractor-gate/gate-scores.csv")
    if gate.exists():  # the extractor's own relevance filter: a comparison row and the corpus-like subset
        from .extractor_gate import SUBSET, passing

        g = ScoreTable.read_csv(gate)
        tables.setdefault(g.method, g)
        subsets[SUBSET] = passing(g) & {r.id for r in records}
    metrics = evaluate(
        tables, records, folds=folds, draws=draws, seed=SEED, subsets=subsets
    )
    subset_evaluations = {}
    for (
        label,
        ids,
    ) in (
        subsets.items()
    ):  # each subset also as its own evaluation, cut on the other folds of the subset
        sub = [r for r in records if r.id in ids]
        if len({r.label for r in sub}) == 2:
            subset_evaluations[label] = evaluate(
                tables,
                sub,
                folds={r.id: folds[r.id] for r in sub},
                draws=draws,
                seed=SEED,
            )
    report = dict(
        split="train + dev, out-of-fold (test untouched)",
        folds=dict(k=5, seed=SEED, grouped_by="paper"),
        clusters=None if pinned is None else shown(pinned),
        provenance=provenance(),
        methods=metrics,
        subset_evaluations=subset_evaluations,
    )
    (out / "metrics.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str)
    )
    y = [r.label for r in records]
    (out / "curves.json").write_text(
        json.dumps(
            dict(
                split="train + dev, out-of-fold",
                unit="records kept at or above each score",
                methods={
                    n: curves(y, [t.scores[r.id] for r in records])
                    for n, t in tables.items()
                },
            ),
            separators=(",", ":"),
        )
    )
    for n, m in metrics.items():
        print(
            f"{m['auc']:.3f} [{m['auc_ci95'][0]:.3f}, {m['auc_ci95'][1]:.3f}]  AP {m['ap']:.3f}  {n}"
        )
    return report
