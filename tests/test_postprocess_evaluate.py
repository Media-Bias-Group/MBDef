"""Seam B: the shared Step 5.3 evaluation, driven only through its public interface on tiny fixtures.

Expected values are worked by hand in the comments, never recomputed the way the code does.
"""

import pytest
from cli_helpers import real_mbdef_gold

from mbdef.postprocess.evaluate import ScoreTable, evaluate
from mbdef.postprocess.groundtruth import Record


def rec(i, label, paper="p1", split="train", votes=(2, 0)):
    return Record(
        id=f"r{i}",
        paper_id=paper,
        split=split,
        term=f"t{i}",
        label=label,
        definitions=(f"d{i}",),
        votes=votes,
    )


# labels and scores of a 6-record fixture: positives score 0.9, 0.4, 0.8; negatives 0.4, 0.1, 0.7
LABELS = [1, 1, 0, 0, 1, 0]
SCORES = [0.9, 0.4, 0.4, 0.1, 0.8, 0.7]


def six():
    records = [rec(i, y, paper=f"p{i % 3}") for i, y in enumerate(LABELS)]
    table = ScoreTable(
        method="m",
        unit="term",
        scores={r.id: s for r, s in zip(records, SCORES, strict=True)},
    )
    return records, table


def test_auc_counts_ties_as_half_a_win():
    # 9 positive-negative pairs: 0.9 beats 3, 0.8 beats 3, 0.4 beats 0.1 and ties 0.4 -> 7.5 / 9
    records, table = six()
    report = evaluate({"m": table}, records, draws=50)
    assert report["m"]["auc"] == pytest.approx(7.5 / 9)


def test_average_precision_sums_precision_at_each_recall_step():
    # thresholds 0.9 (P 1, R 1/3), 0.8 (P 1, R 2/3), 0.7 (P 2/3, R 2/3), 0.4 (P 3/5, R 1): 1/3 + 1/3 + 1/3 * 3/5
    records, table = six()
    m = evaluate({"m": table}, records, draws=50)["m"]
    assert m["ap"] == pytest.approx(2 / 3 + 1 / 5)
    assert m["ap_baseline"] == pytest.approx(0.5)


def test_precision_and_recall_at_k_average_over_tied_scores():
    # top 4: 0.9 (rel), 0.8 (rel), 0.7 (not), then one of the tied 0.4 pair (one rel, one not) -> 2.5 expected hits
    records, table = six()
    m = evaluate({"m": table}, records, ks=(4,), draws=50)["m"]
    assert m["precision_at"][4] == pytest.approx(2.5 / 4)
    assert m["recall_at"][4] == pytest.approx(2.5 / 3)


def test_order_of_tied_records_does_not_change_precision_at_k():
    records, table = six()
    flipped = list(reversed(records))
    a = evaluate({"m": table}, records, ks=(4,), draws=50)["m"]["precision_at"][4]
    b = evaluate({"m": table}, flipped, ks=(4,), draws=50)["m"]["precision_at"][4]
    assert a == b


class SpyPath:
    """A path that remembers what it was fitted on and scores each item by its fold-free index."""

    name = "spy"

    def __init__(self):
        self.fitted_on, self.scored = [], []

    def fit(self, train, corpus=None):
        self.fitted_on.append({r.id for r in train})

    def score(self, items):
        self.scored.append({r.id for r in items})
        return [int(r.id[1:]) / 100 for r in items]


def twenty(split="train"):
    # 10 papers with one relevant and one non-relevant record each
    return [
        rec(2 * p + j, 1 - j, paper=f"p{p}", split=split)
        for p in range(10)
        for j in range(2)
    ]


def test_cross_validation_never_fits_on_the_records_it_scores_and_keeps_papers_together():
    from mbdef.postprocess.evaluate import cross_validate, paper_folds

    records = twenty()
    folds = paper_folds(records, k=5, seed=1)
    for p in {r.paper_id for r in records}:
        assert len({folds[r.id] for r in records if r.paper_id == p}) == 1
    spy = SpyPath()
    table = cross_validate(spy, records, folds=folds)
    assert set(table.scores) == {r.id for r in records}
    for fitted, scored in zip(spy.fitted_on, spy.scored, strict=True):
        assert not fitted & scored
    assert len(spy.fitted_on) == 5


def test_cross_validation_refuses_test_records():
    from mbdef.postprocess.evaluate import cross_validate, paper_folds

    records = twenty() + [rec(99, 1, paper="px", split="test")]
    with pytest.raises(ValueError, match="test"):
        cross_validate(SpyPath(), records, folds=paper_folds(records[:-1], k=5, seed=1))


def test_operating_cut_is_set_on_the_other_folds_and_applied_to_the_held_out_one():
    # fold 0: a(rel .9) b(rel .3) c(not .5); fold 1: d(rel .8) e(rel .6) f(not .7); target recall 0.5 fold 0 is cut at the best fold-1 positive (.8): keeps a; fold 1 at the best fold-0 positive (.9): keeps nothing -> kept {a}: recall 1/4, precision 1, excluded 5/6, MCC (1*2 - 0) / sqrt(1*4*2*5)
    labels = dict(a=1, b=1, c=0, d=1, e=1, f=0)
    scores = dict(a=0.9, b=0.3, c=0.5, d=0.8, e=0.6, f=0.7)
    records = [
        Record(id=k, paper_id=k, split="train", term=k, label=v, definitions=("x",))
        for k, v in labels.items()
    ]
    folds = dict(a=0, b=0, c=0, d=1, e=1, f=1)
    m = evaluate(
        {"m": ScoreTable("m", "term", scores)},
        records,
        folds=folds,
        recall=0.5,
        ks=(),
        draws=50,
    )["m"]
    cut = m["cut"]
    assert cut["recall"] == pytest.approx(0.25)
    assert cut["precision"] == pytest.approx(1.0)
    assert cut["excluded_share"] == pytest.approx(5 / 6)
    assert cut["mcc"] == pytest.approx(2 / 40**0.5)


def test_precision_at_an_assumed_corpus_relevance_rate():
    # recall 95% of 3 positives keeps all three -> cut 0.4: TPR 1, FPR 2/3 (negatives .4 and .7 kept) with 1% relevant: 0.01 / (0.01 + 2/3 * 0.99) = 0.01 / 0.67
    records, table = six()
    m = evaluate({"m": table}, records, prevalences=(0.01,), ks=(), draws=50)["m"]
    assert m["precision_if_relevant_share"][0.01] == pytest.approx(0.01 / 0.67)


def test_paired_bootstrap_is_reproducible_and_compares_methods_on_the_same_resampled_papers():
    records = twenty()
    a = ScoreTable(
        "a",
        "term",
        {r.id: (0.9 if r.label else 0.2) - int(r.id[1:]) / 100 for r in records},
    )
    same = ScoreTable("same", "term", dict(a.scores))
    one = evaluate({"a": a, "same": same}, records, ks=(), draws=200, seed=3)
    two = evaluate({"a": a, "same": same}, records, ks=(), draws=200, seed=3)
    assert one["a"]["auc_ci95"] == two["a"]["auc_ci95"]
    lo, hi = one["a"]["auc_ci95"]
    assert lo <= one["a"]["auc"] <= hi
    # identical scores on identical resamples: the paired difference is exactly zero in every draw
    assert one["same"]["delta_auc_vs"]["a"]["ci95"] == [0.0, 0.0]


def test_unanimous_column_uses_only_records_without_a_dissenting_vote():
    # the two split-vote records are the only mistakes of this ranking, so unanimous-only AUC is perfect
    records = [
        rec(0, 1, votes=(2, 0)),
        rec(1, 1, votes=(2, 1)),
        rec(2, 0, votes=(0, 2)),
        rec(3, 0, votes=(1, 2)),
    ]
    table = ScoreTable("m", "term", {"r0": 0.9, "r1": 0.1, "r2": 0.2, "r3": 0.8})
    m = evaluate({"m": table}, records, ks=(), draws=50)["m"]
    assert m["unanimous"]["auc"] == pytest.approx(1.0)
    assert m["unanimous"]["records"] == 2


def test_a_score_table_handed_in_as_a_file_is_evaluated_like_one_made_in_process(
    tmp_path,
):
    records, table = six()
    table = ScoreTable(
        "m", "term", table.scores, splits={r.id: r.split for r in records}
    )
    path = tmp_path / "handed-in-scores.csv"
    table.write_csv(path)
    assert path.read_text().splitlines()[0] == "method,unit,item_id,split,score"
    back = ScoreTable.read_csv(path)
    assert (back.method, back.unit) == ("m", "term")
    assert back.splits == {r.id: "train" for r in records}
    assert evaluate({"m": back}, records, draws=50) == evaluate(
        {"m": table}, records, draws=50
    )


GOLD, needs_gold = real_mbdef_gold()  # MBDef-Gold in the data root


@needs_gold
def test_ground_truth_loads_the_fixed_paper_splits_with_non_relevant_records_as_negatives(
    monkeypatch,
):
    # The counts of MBDef-Gold: train 592/828, dev 101/179 relevant/non-relevant.
    # The TEST split is not loaded: its labels are read only through the guarded `postprocess test --read`.
    from mbdef.postprocess.groundtruth import load

    monkeypatch.setenv("DATA_ROOT", str(GOLD.parent))

    records = load(splits=("train", "dev"))
    by = {}
    for r in records:
        by.setdefault(r.split, [0, 0])[1 - r.label] += 1
    assert by == {"train": [592, 828], "dev": [101, 179]}
    assert sum(1 for r in records if r.unanimous is False) == 536
    assert all(r.definitions for r in records)


class ConstantByTerm:
    """A frozen-able path: scores 0.9 for terms starting with 'rel', 0.1 otherwise; remembers what it was fitted on."""

    name = "constant"

    def __init__(self):
        self.fitted = []

    def fit(self, train, corpus=None):
        self.fitted = [r.split for r in train]

    def score(self, items):
        return [0.9 if r.term.startswith("rel") else 0.1 for r in items]


def confirmed(paths, reason, reads):
    """Stands in for the person who confirms a TEST read at the prompt."""


def split_fixture():
    train = [
        Record(f"a{i}", f"p{i}", "train", "rel" if i % 2 else "x", i % 2, ("d",))
        for i in range(12)
    ]
    test = [
        Record(f"t{i}", f"q{i}", "test", "rel" if i % 2 else "x", i % 2, ("d",))
        for i in range(6)
    ]
    return train, test


def test_test_run_refuses_a_path_that_was_not_frozen(tmp_path):
    from mbdef.postprocess.testrun import run_test

    train, test = split_fixture()
    with pytest.raises(ValueError, match="not frozen"):
        run_test(
            ["constant"],
            tmp_path,
            train=train,
            test=test,
            factory=lambda name, settings: ConstantByTerm(),
        )


def test_test_split_is_read_once_unless_a_reason_is_given(tmp_path):
    from mbdef.postprocess.frozen import freeze
    from mbdef.postprocess.testrun import run_test

    train, test = split_fixture()
    freeze(
        tmp_path,
        "constant",
        settings={"kwargs": {"cut_word": "rel"}},
        evidence={"note": "fixture"},
    )
    path, given = ConstantByTerm(), []
    factory = lambda name, settings: given.append(settings) or path  # noqa: E731
    result = run_test(
        ["constant"],
        tmp_path,
        train=train,
        test=test,
        factory=factory,
        confirm=confirmed,
    )
    assert given and all(
        g == {"kwargs": {"cut_word": "rel"}} for g in given
    )  # every path is built from its frozen record
    assert set(path.fitted) == {"train"}  # fitted on train + dev only
    assert result["methods"]["constant"]["auc"] == pytest.approx(1.0)
    with pytest.raises(SystemExit, match="already read"):
        run_test(
            ["constant"],
            tmp_path,
            train=train,
            test=test,
            factory=lambda name, settings: ConstantByTerm(),
            confirm=confirmed,
        )
    again = run_test(
        ["constant"],
        tmp_path,
        train=train,
        test=test,
        factory=lambda name, settings: ConstantByTerm(),
        force_reason="rerun after fixing the report",
        confirm=confirmed,
    )
    assert [r["reason"] for r in again["test_reads"]] == [
        None,
        "rerun after fixing the report",
    ]


def test_a_reread_from_saved_test_scores_fits_nothing_and_reports_both_cuts_set_on_train_and_dev(
    tmp_path,
):
    # saved test scores: t1 .9 (rel), t3 .6 (rel), t5 .2 (rel), t0 .7, t2 .1, t4 .05 (not); out-of-fold train scores:
    # 5 positives at .8, positive a11 and negatives a8 a10 at .5, the rest .3 -> 95%-recall cut .5 (all 6 positives); F1 10/11 at .8 beats 12/14 at .5 -> F1 cut .8. Recall cut keeps t1 t3 t0, F1 cut keeps t1
    from mbdef.postprocess.frozen import freeze
    from mbdef.postprocess.testrun import run_test

    train, test = split_fixture()
    freeze(tmp_path, "constant", settings={}, evidence={"note": "fixture"})
    oof = ScoreTable(
        "constant",
        "term",
        {
            r.id: (0.8 if i in (1, 3, 5, 7, 9) else 0.5 if i in (8, 10, 11) else 0.3)
            for i, r in enumerate(train)
        },
    )
    saved = ScoreTable(
        "constant", "term", dict(t0=0.7, t1=0.9, t2=0.1, t3=0.6, t4=0.05, t5=0.2)
    )

    def factory(name, settings):
        raise AssertionError(
            "a re-read from saved scores must not build, fit or score a path"
        )

    m = run_test(
        ["constant"],
        tmp_path,
        train=train,
        test=test,
        factory=factory,
        oof_tables={"constant": oof},
        test_tables={"constant": saved},
        subsets={"odd": {"t1", "t3"}},
        confirm=confirmed,
    )["methods"]["constant"]
    assert (m["cut"]["cut"], m["cut"]["tp"], m["cut"]["fp"]) == (0.5, 2, 1)
    assert (m["f1_cut"]["cut"], m["f1_cut"]["tp"], m["f1_cut"]["fp"]) == (0.8, 1, 0)
    assert (
        m["cut"]["set_on"] == m["f1_cut"]["set_on"] == "train + dev out-of-fold scores"
    )
    assert m["subsets"]["odd"]["cut"]["tp"] == 2
    whole = run_test(
        ["constant"],
        tmp_path,
        train=train,
        test=test,
        factory=factory,
        oof_tables={"constant": oof},
        test_tables={"constant": saved},
        subsets={"odd": {"t1", "t3", "t0"}},
        force_reason="again",
        confirm=confirmed,
    )
    sub = whole["subset_evaluations"]["odd"][
        "constant"
    ]  # the subset as its own evaluation, same train + dev cuts
    assert (sub["records"], sub["cut"]["cut"], sub["cut"]["tp"], sub["cut"]["fp"]) == (
        3,
        0.5,
        2,
        1,
    )
    assert "auc_ci95" in sub


def test_corpus_decisions_keep_every_term_at_or_above_the_cut_and_count_them_by_corpus_part():
    from mbdef.postprocess.corpus import Definition, ExtractedConcept
    from mbdef.postprocess.testrun import corpus_decisions

    class ByLabel:
        name = "by label"

        def fit(self, train, corpus=None):
            self.fitted = len(train)

        def score(self, items):
            return [
                {"framing": 0.9, "media bias": 0.5, "lstm": 0.1}[i.label] for i in items
            ]

    corpus = [
        ExtractedConcept("m1", "p1", "seeds", "framing", (Definition("o1", "x"),)),
        ExtractedConcept("m2", "p2", "closure", "media bias", (Definition("o2", "y"),)),
        ExtractedConcept("m3", "p3", "closure", "lstm", (Definition("o3", "z"),)),
    ]
    train, _ = split_fixture()
    rows, counts = corpus_decisions(ByLabel(), train, corpus, cut=0.5)
    assert [(r["mention_id"], r["keep"]) for r in rows] == [
        ("m1", True),
        ("m2", True),
        ("m3", False),
    ]
    assert counts == dict(
        terms=3,
        kept=2,
        excluded=1,
        seeds=dict(terms=1, kept=1),
        closure=dict(terms=2, kept=1),
    )


def test_precision_at_an_assumed_relevance_rate_uses_a_cut_fixed_elsewhere_when_given():
    # cut 0.6 fixed on other data: positives .9 .8 kept (TPR 2/3), negative .7 kept (FPR 1/3) with 1% relevant: (2/3 * .01) / (2/3 * .01 + 1/3 * .99) = 2 / 101
    records, table = six()
    m = evaluate(
        {"m": table}, records, prevalences=(0.01,), ks=(), draws=50, cuts={"m": 0.6}
    )["m"]
    assert m["precision_if_relevant_share"][0.01] == pytest.approx(2 / 101)


def test_subset_metrics_score_only_the_listed_records():
    # on the subset {r0 (rel .9), r2 (not .4), r3 (not .1)} the ranking is perfect
    records, table = six()
    m = evaluate(
        {"m": table}, records, ks=(), draws=50, subsets={"matched": {"r0", "r2", "r3"}}
    )["m"]
    sub = m["subsets"]["matched"]
    assert (sub["records"], sub["relevant"], sub["auc"], sub["ap"]) == (3, 1, 1.0, 1.0)


def test_subset_decisions_count_only_the_listed_records_at_the_same_cut():
    # cut 0.75 on {r0 (rel .9), r2 (not .4), r3 (not .1)}: r0 kept, both negatives dropped -> every decision right
    records, table = six()
    m = evaluate(
        {"m": table},
        records,
        ks=(),
        draws=50,
        cuts={"m": 0.75},
        subsets={"matched": {"r0", "r2", "r3"}},
    )["m"]
    cut = m["subsets"]["matched"]["cut"]
    assert (cut["tp"], cut["fp"], cut["fn"], cut["tn"]) == (1, 0, 0, 2)
    assert cut["accuracy"] == pytest.approx(1.0)


def test_decisions_at_a_given_cut_report_the_whole_confusion_matrix():
    # cut 0.75 keeps r0 (rel .9) and r4 (rel .8): TP 2, FP 0, FN 1 (r1 .4), TN 3 (.4, .1, .7) precision 1, recall 2/3, F1 2PR/(P+R) = 0.8, F2 5PR/(4P+R) = 5/7, accuracy 5/6, specificity 1, NPV 3/4, balanced accuracy (2/3 + 1)/2 = 5/6, MCC (2*3 - 0)/sqrt(2*3*3*4) = 6/sqrt(72), kappa: observed 5/6, chance (2/6)(3/6) + (4/6)(3/6) = 1/2 -> (5/6 - 1/2)/(1/2) = 2/3, Youden J 2/3 + 1 - 1
    records, table = six()
    cut = evaluate({"m": table}, records, ks=(), draws=50, cuts={"m": 0.75})["m"]["cut"]
    assert cut["cut"] == 0.75
    assert (cut["tp"], cut["fp"], cut["fn"], cut["tn"]) == (2, 0, 1, 3)
    expected = dict(
        precision=1.0,
        recall=2 / 3,
        f1=0.8,
        f2=5 / 7,
        accuracy=5 / 6,
        specificity=1.0,
        npv=3 / 4,
        balanced_accuracy=5 / 6,
        fpr=0.0,
        mcc=6 / 72**0.5,
        kappa=2 / 3,
        youden_j=2 / 3,
        excluded_share=4 / 6,
    )
    assert {k: cut[k] for k in expected} == pytest.approx(expected)


def test_f1_cut_is_the_threshold_with_the_highest_f1():
    # keep >= .9: F1 .5; >= .8: P 1, R 2/3, F1 .8; >= .7: 2/3; >= .4: P 3/5, R 1, F1 .75; >= .1: 2/3 -> 0.8
    from mbdef.postprocess.evaluate import f1_cut

    assert f1_cut(LABELS, SCORES) == 0.8


def test_f1_operating_cut_is_set_on_the_other_folds_too():
    # fold 0: a(rel .9) b(rel .3) c(not .5); fold 1: d(rel .8) e(rel .6) f(not .7) best F1 on fold 1 is at .6 (P 2/3, R 1) -> fold 0 keeps a; best F1 on fold 0 is at .3 -> fold 1 keeps d, e, f -> TP 3 (a, d, e), FP 1 (f), FN 1 (b), TN 1 (c)
    labels = dict(a=1, b=1, c=0, d=1, e=1, f=0)
    scores = dict(a=0.9, b=0.3, c=0.5, d=0.8, e=0.6, f=0.7)
    records = [
        Record(id=k, paper_id=k, split="train", term=k, label=v, definitions=("x",))
        for k, v in labels.items()
    ]
    folds = dict(a=0, b=0, c=0, d=1, e=1, f=1)
    m = evaluate(
        {"m": ScoreTable("m", "term", scores)}, records, folds=folds, ks=(), draws=50
    )["m"]
    f1 = m["f1_cut"]
    assert (f1["tp"], f1["fp"], f1["fn"], f1["tn"]) == (3, 1, 1, 1)
    assert f1["f1"] == pytest.approx(0.75)


def test_precision_at_a_recall_reads_the_ranking_at_its_own_cut():
    # recall 0.6 needs 2 of 3 positives -> cut .8 keeps r0, r4: precision 1; recall 0.8 needs all 3 -> cut .4 keeps 5: 3/5
    records, table = six()
    m = evaluate({"m": table}, records, ks=(), draws=50, recalls=(0.6, 0.8))["m"]
    assert m["precision_at_recall"] == pytest.approx({0.6: 1.0, 0.8: 0.6})


def test_corpus_decisions_count_label_groups_kept_dropped_or_split():
    from mbdef.postprocess.corpus import Definition, ExtractedConcept
    from mbdef.postprocess.testrun import corpus_decisions

    class ByLabel:
        name = "by label"

        def fit(self, train, corpus=None):
            pass

        def score(self, items):
            return [
                {"framing": 0.9, "framings": 0.2, "lstm": 0.1, "media bias": 0.8}[
                    i.label
                ]
                for i in items
            ]

    c = lambda m, label: ExtractedConcept(  # noqa: E731
        m, f"p{m}", "seeds", label, (Definition(f"o{m}", "x"),)
    )
    corpus = [
        c("m1", "framing"),
        c("m2", "framings"),
        c("m3", "lstm"),
        c("m4", "media bias"),
    ]
    train, _ = split_fixture()
    _, counts = corpus_decisions(ByLabel(), train, corpus, cut=0.5, groups=[0, 0, 1, 2])
    # group 0 (framing, framings) is split, group 1 (lstm) dropped, group 2 (media bias) kept
    assert counts["label_groups"] == dict(groups=3, all_kept=1, all_dropped=1, split=1)


def test_commands_take_a_paths_settings_from_the_frozen_record_and_fall_back_to_defaults_for_other_variants(
    tmp_path,
):
    from mbdef.postprocess.frozen import freeze, frozen_kwargs

    freeze(
        tmp_path,
        "clustering",
        settings={"kwargs": {"dims": 20, "k": 100}},
        evidence={"note": "fixture"},
    )
    assert frozen_kwargs("clustering", tmp_path) == {"dims": 20, "k": 100}
    assert (
        frozen_kwargs("zero-shot:2", tmp_path) is None
    )  # an exploratory variant that was never frozen


def test_curves_give_roc_and_precision_recall_points_at_every_distinct_score():
    # y 1 0 1 0, scores .9 .8 .7 .1: keep >= .9 -> TPR 1/2, FPR 0, P 1; >= .8 -> 1/2, 1/2, P 1/2; >= .7 -> 1, 1/2, P 2/3; >= .1 -> 1, 1, P 1/2. The ROC curve starts at (0, 0)
    from mbdef.postprocess.evaluate import curves

    c = curves([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert [(p["fpr"], p["tpr"]) for p in c["roc"]] == pytest.approx(
        [(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 1), (1, 1)]
    )
    assert [(p["recall"], p["precision"]) for p in c["pr"]] == pytest.approx(
        [(0.5, 1), (0.5, 0.5), (1, 2 / 3), (1, 0.5)]
    )
    assert [p["threshold"] for p in c["pr"]] == [0.9, 0.8, 0.7, 0.1]


def test_a_test_read_is_refused_without_a_reason_a_terminal_or_the_typed_confirmation(
    tmp_path, monkeypatch
):
    """A fresh TEST read warns, needs a reason and a typed confirmation at the prompt, and nothing is logged or read when any of them is missing."""
    import sys

    from mbdef.postprocess import testrun
    from mbdef.postprocess.frozen import freeze

    train, test = split_fixture()
    freeze(tmp_path, "constant", settings={}, evidence={"note": "fixture"})

    def factory(name, settings):
        raise AssertionError("nothing may be built before the read is confirmed")

    read = lambda reason: testrun.run_test(  # noqa: E731
        ["constant"],
        tmp_path,
        train=train,
        test=test,  # noqa: E731
        factory=factory,
        force_reason=reason,
    )
    with pytest.raises(SystemExit, match="needs a reason"):
        read(None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="interactive prompt"):
        read("a reason")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    with pytest.raises(SystemExit, match="Not confirmed"):
        read("a reason")
    assert not (tmp_path / testrun.READS).exists()  # no read was logged


def test_a_confirmed_test_read_is_logged_with_its_reason(tmp_path, monkeypatch, capsys):
    import sys

    from mbdef.postprocess import testrun

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: testrun.CONFIRM)
    reads = testrun.log_test_read(tmp_path, ["classifier"], "a fixture read")
    assert [(r["paths"], r["reason"]) for r in reads] == [
        (["classifier"], "a fixture read")
    ]
    assert "WARNING" in capsys.readouterr().err


def test_test_eval_is_compared_by_hash_only(tmp_path):
    import hashlib

    from mbdef.postprocess.testrun import compare_by_hash

    saved = tmp_path / "test-eval.json"
    saved.write_text('{"records": 628}')
    right = hashlib.sha256(saved.read_bytes()).hexdigest()
    assert compare_by_hash(saved, right) == (True, right)
    assert compare_by_hash(saved, "0" * 64) == (False, right)
