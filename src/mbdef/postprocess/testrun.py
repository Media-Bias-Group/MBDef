"""Step 5.3, final: the one-time evaluation of frozen paths on the test split.

Each frozen path is fitted on all train + dev records (and the corpus), scores the test records, and is cut at two thresholds set on its train + dev out-of-fold scores: the one that keeps `recall` of the relevant records, and the one with the highest F1. A re-read can summarise the test scores an earlier read saved instead of refitting.

The TEST split is compared by hash only by default (a check never reads TEST): `mbdef postprocess test` checks that the saved result, test-eval.json, still has its sha256 and reads nothing else. Recomputing anything from the TEST split, even AUCs from saved TEST scores, counts as a read. A read happens only when someone asks for it with --read: it warns, needs a reason, asks for confirmation at the prompt, and is logged in test-reads.json with its reason (`log_test_read`, which every read goes through).
"""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from . import FROZEN, REPO, SEED
from .evaluate import (
    ScoreTable,
    cross_validate,
    evaluate,
    f1_cut,
    paper_folds,
    recall_cut,
)
from .frozen import load_frozen
from .label_match import group_counts

READS = "test-reads.json"
FROZEN_RULE = "C-20261015"  # Label Match version of the frozen @group paths (label_match.RULE_VERSIONS)
# The saved TEST result, compared by hash only; it is not part of this repository
TEST_EVAL = REPO / FROZEN / "20261015/test-evaluation/test-eval.json"
TEST_EVAL_SHA256 = "84e3b3dbc65d6c7cce1c666cf8023a93b23587a1616aba407bf98ff8321a9547"
CONFIRM = "read the test split"  # what the person types to confirm a read


def compare_by_hash(path=TEST_EVAL, expected=TEST_EVAL_SHA256):
    """(same, sha256 found): the saved TEST result against its pinned hash. Reads no TEST record."""
    from mbdef.files import file_sha256

    found = file_sha256(Path(path))
    return found == expected, found


def ask_at_prompt(paths, reason, reads):
    """Warn loudly and ask the person at the terminal to confirm a read of the TEST split; refuse otherwise."""
    if not reason:
        raise SystemExit(
            "A read of the TEST split needs a reason (--reason), which the read log keeps."
        )
    last = f"; the last of {len(reads)} reads was on {reads[-1]['at']}" if reads else ""
    print(
        "\n".join(
            [
                "",
                "!" * 78,
                "!! WARNING: this reads the held-out TEST split. Every read is logged and weakens the split.",
                f"!! Paths: {', '.join(paths)}",
                f"!! Reason: {reason}{last}.",
                "!" * 78,
            ]
        ),
        file=sys.stderr,
    )
    if not sys.stdin.isatty():
        raise SystemExit(
            "Refused: a read of the TEST split needs confirmation at an interactive prompt."
        )
    if input(f'Type "{CONFIRM}" to go on: ').strip() != CONFIRM:
        raise SystemExit("Not confirmed; nothing of the TEST split was read.")


def log_test_read(out_dir, paths, reason, confirm=ask_at_prompt):
    """Confirm a read of the test split (`confirm(paths, reason, earlier reads)`, by default at the prompt), then append it to `out_dir/test-reads.json`; every read after the first needs a reason."""
    out_dir = Path(out_dir)
    log_file = out_dir / READS
    reads = json.loads(log_file.read_text()) if log_file.exists() else []
    if reads and reason is None:
        raise SystemExit(
            f"The test split was already read on {reads[-1]['at']}; pass a reason (--reason) to "
            "read it again."
        )
    confirm(list(paths), reason, reads)
    reads.append(
        dict(
            at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            paths=list(paths),
            reason=reason,
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(reads, indent=1))
    return reads


def run_test(
    paths,
    out_dir,
    *,
    train,
    test,
    factory,
    corpus=None,
    recall=0.95,
    draws=2000,
    seed=SEED,
    force_reason=None,
    oof_tables=None,
    test_tables=None,
    subsets=None,
    confirm=ask_at_prompt,
):
    """`oof_tables` / `test_tables` ({path: ScoreTable}): saved train + dev out-of-fold scores and saved test scores of the same frozen paths; given, they are used instead of refitting. `subsets` ({label: test record ids}) get the same measures at the same cuts. `confirm` asks before the read (see `log_test_read`)."""
    out_dir = Path(out_dir)
    frozen = load_frozen(out_dir)
    missing = [p for p in paths if p not in frozen]
    if missing:
        raise ValueError(
            f"paths not frozen (no entry in {out_dir / 'frozen-paths.json'}): {missing}"
        )
    if any(r.split == "test" for r in train) or any(r.split != "test" for r in test):
        raise ValueError(
            "train must hold only train/dev records and test only test records"
        )
    reads = log_test_read(out_dir, paths, force_reason, confirm)

    y_train = np.array([r.label for r in train])
    tables, cuts, f1_cuts = {}, {}, {}
    for name in paths:
        settings = frozen[name]["settings"]
        if oof_tables and name in oof_tables:
            oof = oof_tables[name]
        else:
            oof = cross_validate(
                factory(name, settings), train, folds=paper_folds(train), corpus=corpus
            )
        s = np.array([oof.scores[r.id] for r in train])
        cuts[name], f1_cuts[name] = recall_cut(y_train, s, recall), f1_cut(y_train, s)
        if test_tables and name in test_tables:
            tables[name] = test_tables[name]
            continue
        # built from the frozen record, not from code defaults
        path = factory(name, settings)
        path.fit(train, corpus)
        tables[name] = ScoreTable(
            path.name,
            "label group" if name.endswith("@group") else "term",
            {r.id: float(v) for r, v in zip(test, path.score(test), strict=True)},
            {r.id: r.split for r in test},
        )
    # the train + dev cuts, never test-derived
    methods = evaluate(
        tables,
        test,
        draws=draws,
        seed=seed,
        recall=recall,
        cuts=cuts,
        f1_cuts=f1_cuts,
        subsets=subsets,
    )
    for name in tables:
        for point in ("cut", "f1_cut"):
            methods[name][point]["set_on"] = "train + dev out-of-fold scores"
        methods[name]["frozen"] = frozen[name]
    # each subset with both classes also as its own evaluation (bootstrap intervals, paired differences), same cuts
    subset_evaluations = {}
    for label, ids in (subsets or {}).items():
        sub = [r for r in test if r.id in ids]
        if len({r.label for r in sub}) == 2:
            subset_evaluations[label] = evaluate(
                tables,
                sub,
                draws=draws,
                seed=seed,
                recall=recall,
                cuts=cuts,
                f1_cuts=f1_cuts,
            )
    for ev in subset_evaluations.values():
        for name in ev:
            for point in ("cut", "f1_cut"):
                ev[name][point]["set_on"] = "train + dev out-of-fold scores"
    report = dict(
        split="test, read once",
        records=len(test),
        methods=methods,
        subset_evaluations=subset_evaluations,
        test_reads=reads,
    )
    (out_dir / "test-eval.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str)
    )
    for name, t in tables.items():
        t.write_csv(out_dir / f"test-scores-{name.replace(':', '-')}.csv")
    return report


def corpus_decisions(path, train, corpus, cut, groups=None):
    """Step 5.4: fit the path on all train + dev records, score every corpus Extracted Concept and keep those with a score at or above the cut. Returns (rows, counts); with `groups` (the Label Group of each corpus concept) the counts also say how many Label Groups are kept whole, dropped whole, or split by the decisions."""
    path.fit(train, corpus)
    scores = path.score(corpus)
    rows = [
        dict(
            mention_id=c.mention_id,
            paper_id=c.paper_id,
            corpus=c.corpus,
            label=c.label,
            score=float(s),
            keep=bool(s >= cut),
        )
        for c, s in zip(corpus, scores, strict=True)
    ]
    counts = dict(
        terms=len(rows),
        kept=sum(r["keep"] for r in rows),
        excluded=sum(not r["keep"] for r in rows),
    )
    for part in ("seeds", "closure"):
        sub = [r for r in rows if r["corpus"] == part]
        counts[part] = dict(terms=len(sub), kept=sum(r["keep"] for r in sub))
    if groups is not None:
        counts["label_groups"] = group_counts(groups, [r["keep"] for r in rows])
    return rows, counts


def compare_test():
    """`postprocess test`: the saved TEST result against its pinned hash; reads nothing of the TEST split."""
    from .compare import shown

    if not TEST_EVAL.exists():
        raise SystemExit(
            f"{shown(TEST_EVAL)}, the saved result of the TEST evaluation, is not part of this repository, "
            "so there is nothing to compare here."
        )
    same, found = compare_by_hash()
    print(
        f"{shown(TEST_EVAL)}: {'same' if same else 'CHANGED'}, sha256 {found}"
        + ("" if same else f", pinned {TEST_EVAL_SHA256}")
        + ". Compared by hash only; nothing of the TEST split was read (pass --read to read it again)."
    )
    return 0 if same else 1


def start_from_frozen(out, inputs):
    """Give the new read's folder `out` the frozen record of paths and the read log of `inputs` (the frozen test-evaluation folder by default), unless `out` already holds its own (from `postprocess freeze --out`).
    Without a read log in either folder it refuses: nobody could tell how often the TEST split was read."""
    import shutil

    out, inputs = Path(out), Path(inputs)
    if not (out / READS).exists() and not (inputs / READS).exists():
        raise SystemExit(
            f"The read log of the TEST split ({inputs / READS}) is not part of this repository. Without it "
            "nobody can tell how often the TEST split was read, so this command does not read the TEST split; "
            'the TEST numbers stay those of the logged reads (docs/step-5-post-processing/postprocess.md, "The TEST split").'
        )
    out.mkdir(parents=True, exist_ok=True)
    for name in ("frozen-paths.json", READS):
        if not (out / name).exists() and (Path(inputs) / name).exists():
            shutil.copyfile(Path(inputs) / name, out / name)


def read_test(out, inputs, names=None, reason=None, from_saved=False):
    """`postprocess test --read`: the frozen paths on the TEST split once more, into `out` (warns, needs a reason, asks at the prompt and is logged). `inputs` is the date folder of the saved out-of-fold scores, the frozen record, the saved TEST scores (with `from_saved`) and the extractor filter's scores."""
    from . import groundtruth
    from .compare import make_in_setting
    from .corpus import load_view

    out, inputs = Path(out), Path(inputs)
    start_from_frozen(out, inputs / "test-evaluation")
    names = names or list(load_frozen(out))
    # The guard comes first: the saved TEST scores and even the record ids (they name the label) are TEST reads.
    earlier = json.loads((out / READS).read_text()) if (out / READS).exists() else []
    if earlier and reason is None:
        raise SystemExit(
            f"The test split was already read on {earlier[-1]['at']}; pass a reason (--reason) to "
            "read it again."
        )
    ask_at_prompt(names, reason, earlier)
    oof, saved, subsets = {}, {}, {}
    for n in names:
        f = inputs / "evaluation/scores" / f"{n.replace(':', '-')}.csv"
        if f.exists():
            oof[n] = ScoreTable.read_csv(f)
        f = inputs / "test-evaluation" / f"test-scores-{n.replace(':', '-')}.csv"
        if from_saved and f.exists():
            saved[n] = ScoreTable.read_csv(f)
    gate = inputs / "extractor-gate/gate-scores.csv"
    if gate.exists():  # the extractor's own relevance filter: a comparison row and the corpus-like subset
        from .extractor_gate import PATH, SUBSET, passing

        g = ScoreTable.read_csv(gate)
        oof[PATH] = saved[PATH] = g
        subsets[SUBSET] = {i for i in passing(g) if g.splits.get(i) == "test"}
    corpus = load_view().concepts if any(n not in saved for n in names) else None
    shared = {}
    report = run_test(
        names,
        out,
        train=groundtruth.load(splits=("train", "dev")),
        test=groundtruth.load(splits=("test",)),
        factory=lambda n, settings: make_in_setting(
            n,
            shared,
            corpus,
            inputs,
            settings.get("kwargs", {}),
            rule=FROZEN_RULE,  # the paths were frozen with the Label Groups of this rule version
        ),
        corpus=corpus,
        oof_tables=oof,
        test_tables=saved,
        subsets=subsets,
        force_reason=reason,
        # confirmed above, before anything of TEST was read
        confirm=lambda *confirmed: None,
    )
    for n, m in report["methods"].items():
        c = m["cut"]
        print(
            f"{m['auc']:.3f} [{m['auc_ci95'][0]:.3f}, {m['auc_ci95'][1]:.3f}]  cut: recall {c['recall']:.3f} "
            f"precision {c['precision']:.3f} excluded {c['excluded_share']:.3f}  {n}"
        )
    return 0
