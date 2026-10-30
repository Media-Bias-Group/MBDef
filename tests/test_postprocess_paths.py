"""Seams C and D (and the keyword comparison row): relevance paths driven through the path interface on fixtures."""

import pytest

from mbdef.postprocess.groundtruth import Record


def rec(i, term, defs, label=1, paper="p", split="train"):
    return Record(
        id=f"r{i}",
        paper_id=paper,
        split=split,
        term=term,
        label=label,
        definitions=tuple(defs),
    )


def test_keyword_rule_flags_terms_or_definitions_that_mention_bias_media_news_journalism_or_press():
    from mbdef.postprocess.paths.keyword import KeywordRule

    items = [
        rec(0, "framing bias", ["a slant"]),
        rec(1, "LSTM", ["a recurrent neural network"]),
        rec(2, "news values", ["criteria"]),
        rec(3, "gatekeeping", ["how journalists select stories"]),
        rec(4, "agenda setting", ["the press shapes salience"]),
        rec(5, "biased", ["x"]),
        rec(6, "rebias", ["x"]),
    ]
    path = KeywordRule()
    path.fit(items[:2])
    assert list(path.score(items)) == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]


class FakeEmbeddings:
    """Embedding provider stub: a fixed vector per "term: definition" text."""

    def __init__(self, table):
        self.table = table

    def vectors(self, keys):
        import numpy as np

        return np.array([self.table[k] for k in keys], dtype=float)


def three_clusters():
    """Corpus definitions in three far-apart blobs A (x=+10), B (x=-10), C (y=+10); training records in A and B only.
    Training: two relevant and one non-relevant definition in A, one relevant in B -> base rate p0 = 3/4."""
    import numpy as np

    rng = np.random.default_rng(0)
    centres = dict(A=(10.0, 0.0), B=(-10.0, 0.0), C=(0.0, 10.0))
    table, corpus = {}, []
    for c, (x, y) in centres.items():
        for i in range(6):
            key = f"corpus {c}{i}: text"
            table[key] = (x + rng.normal(0, 0.3), y + rng.normal(0, 0.3))
            corpus.append(rec(100 + len(corpus), f"corpus {c}{i}", ["text"], label=0))
    train = []
    for i, (c, y) in enumerate([("A", 1), ("A", 1), ("A", 0), ("B", 1)]):
        table[f"train{i}: text"] = (centres[c][0] + 0.1, centres[c][1] - 0.1)
        train.append(rec(i, f"train{i}", ["text"], label=y))
    table["query: in A"] = (9.8, 0.2)
    table["query: in C"] = (0.1, 9.9)
    table["query: in B"] = (-9.9, 0.1)
    return FakeEmbeddings(table), corpus, train


@pytest.mark.parametrize(
    "a, expected",
    [
        # a = 0: A 2/3, B 1, C has no training definition -> base rate 3/4; the query's score is the mean of A and C
        (0.0, (2 / 3 + 0.75) / 2),
        # a = 2: A (2 + 2*.75) / (3 + 2) = 0.7, C (0 + 1.5) / (0 + 2) = 0.75 -> mean 0.725
        (2.0, (0.7 + 0.75) / 2),
    ],
)
def test_cluster_share_is_the_smoothed_relevant_share_averaged_over_a_terms_definitions(
    a, expected
):
    from mbdef.postprocess.paths.clustering import ClusterShare

    emb, corpus, train = three_clusters()
    path = ClusterShare(dims=2, k=3, a=a, neighbours=3, embeddings=emb)
    path.fit(train, corpus)
    query = rec(50, "query", ["in A", "in C"])
    assert path.score([query])[0] == pytest.approx(expected)


def test_a_cluster_without_labelled_training_definitions_scores_the_base_rate():
    from mbdef.postprocess.paths.clustering import ClusterShare

    emb, corpus, train = three_clusters()
    path = ClusterShare(dims=2, k=3, a=5.0, neighbours=3, embeddings=emb)
    path.fit(train, corpus)
    assert path.score([rec(51, "query", ["in C"])])[0] == pytest.approx(0.75)
    # B: (1 + 5 * .75) / (1 + 5)
    assert path.score([rec(52, "query", ["in B"])])[0] == pytest.approx(4.75 / 6)


def labelled_points():
    """Training definitions: three relevant near (1, 0), three non-relevant near (0, 1)."""
    table, train = {}, []
    for i, (v, y) in enumerate(
        [
            ((1, 0.05), 1),
            ((1, -0.05), 1),
            ((0.95, 0), 1),
            ((0.05, 1), 0),
            ((-0.05, 1), 0),
            ((0, 0.95), 0),
        ]
    ):
        table[f"t{i}: text"] = v
        train.append(rec(i, f"t{i}", ["text"], label=y, paper=f"p{i}"))
    table["q: near relevant"] = (1, 0.01)
    table["q: between"] = (0.7, 0.71)
    return FakeEmbeddings(table), train


def test_neighbour_vote_is_the_relevant_share_among_the_nearest_labelled_definitions():
    from mbdef.postprocess.paths.neighbours import NeighbourVote

    emb, train = labelled_points()
    path = NeighbourVote(neighbours=3, embeddings=emb)
    path.fit(train)
    near, between = path.score(
        [rec(10, "q", ["near relevant"]), rec(11, "q", ["between"])]
    )
    assert near == pytest.approx(1.0)
    # cosine from (0.7, 0.71): t3 (0.05, 1) 0.746, t0 (1, 0.05) 0.737, t5 (0, 0.95) 0.712 -> 1 relevant of 3
    assert between == pytest.approx(1 / 3)


def test_embedding_classifier_ranks_relevant_above_non_relevant_on_separable_data():
    from mbdef.postprocess.paths.neighbours import EmbeddingClassifier

    emb, train = labelled_points()
    path = EmbeddingClassifier(embeddings=emb, inner_folds=3)
    path.fit(train)
    s = path.score(train)
    assert min(s[:3]) > max(s[3:])


class FakeNLI:
    """NLI provider stub: P(entailment) for (premise, hypothesis) from a fixed table (0 when absent); records calls."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def entail(self, pairs):
        import numpy as np

        self.calls += list(pairs)
        return np.array([self.table.get(p, 0.0) for p in pairs])


REP = "media bias is slanted news"
P = {("d1", REP): 0.1, (REP, "d1"): 0.6, ("d2", REP): 0.3, (REP, "d2"): 0.2}


@pytest.mark.parametrize(
    "direction, expected",
    [
        ("definition_to_representative", 0.3),  # max(d1 0.1, d2 0.3)
        ("representative_to_definition", 0.6),  # max(d1 0.6, d2 0.2)
        ("larger_of_both", 0.6),  # max(max(.1, .6), max(.3, .2))
        ("mean_of_both", 0.35),  # max((.1 + .6) / 2, (.3 + .2) / 2)
    ],
)
def test_nli_against_the_representative_scores_a_term_by_its_best_definition(
    direction, expected
):
    from mbdef.postprocess.paths.nli import RepresentativeNLI

    path = RepresentativeNLI(REP, direction=direction, nli=FakeNLI(P))
    path.fit([])
    assert path.score([rec(0, "framing", ["d1", "d2"])])[0] == pytest.approx(expected)


def test_zero_shot_nli_puts_term_and_definition_in_the_premise_and_the_term_in_the_hypothesis():
    from mbdef.postprocess.paths.nli import ZeroShotNLI

    nli = FakeNLI(
        {
            ("framing: d1", "framing is a form of media bias."): 0.4,
            ("framing: d2", "framing is a form of media bias."): 0.7,
        }
    )
    path = ZeroShotNLI("{term} is a form of media bias.", nli=nli)
    assert path.score([rec(0, "framing", ["d1", "d2"])])[0] == pytest.approx(0.7)
    assert ("framing: d1", "framing is a form of media bias.") in nli.calls


def test_representative_definition_has_the_highest_coverage_with_mean_probability_breaking_ties():
    from mbdef.postprocess.nli import coverage, representative

    # a entails b and c (coverage 1); b entails a only (1/2); c entails a and b but with lower probabilities (1)
    table = {
        ("a", "b"): 0.9,
        ("a", "c"): 0.8,
        ("b", "a"): 0.7,
        ("c", "a"): 0.6,
        ("c", "b"): 0.55,
    }
    cov, mean_p = coverage(["a", "b", "c"], FakeNLI(table))
    assert list(cov) == pytest.approx([1.0, 0.5, 1.0])
    assert (
        representative(["a", "b", "c"], FakeNLI(table)) == 0
    )  # a and c tie on coverage; a has the higher mean P


class MeanLength:
    """Stub path scoring an item as the mean over its definitions of len(text) / 10; records (term, texts) asked about."""

    name = "mean length"

    def __init__(self):
        self.seen = []

    def fit(self, train, corpus=None):
        pass

    def score(self, items):
        from mbdef.postprocess.paths.base import term, texts

        self.seen += [(term(i), tuple(texts(i))) for i in items]
        return [sum(len(t) for t in texts(i)) / len(texts(i)) / 10 for i in items]


def corpus_fixture():
    from mbdef.postprocess.corpus import Definition, ExtractedConcept

    c = lambda m, label, defs: ExtractedConcept(  # noqa: E731
        m,
        f"p{m}",
        "seeds",
        label,
        tuple(Definition(f"o{m}{i}", t) for i, t in enumerate(defs)),
    )
    return [
        c(1, "Media bias", ["aa", "aaaa"]),
        c(2, "media biases", ["aaaaaa"]),
        c(3, "framing", ["fr"]),
    ]


def test_label_group_setting_scores_a_matching_record_by_the_mean_over_all_its_groups_definitions():
    from mbdef.postprocess.group_setting import GroupSetting

    setting = GroupSetting(MeanLength(), corpus_fixture(), member="all")
    # group {Media bias, media biases}: definitions of length 2, 4, 6 -> mean 0.4. "bias in the media" matches it by the word-set key, "media-bias" by the spacing key; "agenda setting" has no group and keeps its own score (length 1)
    s = setting.score(
        [
            rec(0, "bias in the media", ["x"]),
            rec(1, "media-bias", ["yy", "zz"]),
            rec(2, "agenda setting", ["w"]),
        ]
    )
    assert s == pytest.approx([0.4, 0.4, 0.1])
    assert setting.matched == 2


def test_label_group_setting_can_use_only_the_groups_representative_definition_under_its_own_label():
    from mbdef.postprocess.group_setting import GroupSetting

    path = MeanLength()
    setting = GroupSetting(
        path,
        corpus_fixture(),
        member="representative",
        representatives={"bias media": "aaaaaa"},
    )
    assert setting.score([rec(0, "Media Bias", ["x", "y"])]) == pytest.approx([0.6])
    assert (
        "media biases",
        ("aaaaaa",),
    ) in path.seen  # scored under the label of the concept it comes from


def test_label_group_setting_rejects_a_representative_for_an_unknown_group():
    from mbdef.postprocess.group_setting import GroupSetting

    with pytest.raises(ValueError, match="no corpus Label Group"):
        GroupSetting(
            MeanLength(),
            corpus_fixture(),
            member="representative",
            representatives={"media bias": "x"},
        )


def test_fast_nli_rescores_exactly_the_pairs_near_the_decision_cut_it_is_given():
    import numpy as np

    from mbdef.postprocess.nli import BandedNLI

    class Fixed:
        def __init__(self, values):
            self.values = values

        def entail(self, pairs):
            return np.array([self.values[p] for p in pairs])

    pairs = [("a", "r"), ("b", "r"), ("c", "r")]
    fast = Fixed({("a", "r"): 0.0060, ("b", "r"): 0.0300, ("c", "r"): 0.0071})
    exact = FakeNLI({("a", "r"): 0.0066, ("c", "r"): 0.0059})
    nli = BandedNLI(threshold=0.0063, band=0.002, reference=exact, fast=fast)
    # a and c lie within 0.002 of the cut 0.0063 and are rescored; b is far from it
    assert list(nli.entail(pairs)) == pytest.approx([0.0066, 0.0300, 0.0059])
    assert exact.calls == [("a", "r"), ("c", "r")]


def test_clustering_can_place_a_definition_by_nearest_cluster_centre_as_k_means_itself_would():
    from mbdef.postprocess.paths.clustering import ClusterShare

    emb, corpus, train = three_clusters()
    path = ClusterShare(dims=2, k=3, a=2.0, assignment="centre", embeddings=emb)
    path.fit(train, corpus)
    # same blobs as before: a definition in A gets A's share 0.7, in C the base rate 0.75
    assert path.score([rec(50, "query", ["in A", "in C"])])[0] == pytest.approx(
        (0.7 + 0.75) / 2
    )


class RecordingEmbeddings(FakeEmbeddings):
    """FakeEmbeddings that also remembers every text it was asked for."""

    def __init__(self, table):
        super().__init__(table)
        self.asked = []

    def vectors(self, keys):
        self.asked += list(keys)
        return super().vectors(keys)


def test_clustering_fit_never_sees_test_records_and_scoring_them_changes_nothing_it_learned():
    import numpy as np

    from mbdef.postprocess.paths.clustering import ClusterShare

    emb, corpus, train = three_clusters()
    emb = RecordingEmbeddings(emb.table)
    test = [
        rec(50, "query", ["in A"], split="test"),
        rec(51, "query", ["in C"], split="test"),
    ]
    path = ClusterShare(dims=2, k=3, a=2.0, neighbours=3, embeddings=emb)
    path.fit(train, corpus)
    asked_in_fit, shares = list(emb.asked), np.array(path.share)
    path.score(test)
    assert not {"query: in A", "query: in C"} & set(
        asked_in_fit
    )  # fit asked only for corpus and training texts
    assert np.array_equal(
        path.share, shares
    )  # scoring test records leaves the learned shares as they were


def test_representative_ties_on_coverage_and_mean_probability_go_to_more_papers_then_the_smallest_id():
    from mbdef.postprocess.nli import representative

    # a and b entail each other at 0.9: coverage 1 and mean P 0.9 each, a full tie on the first two rules
    table = {("a", "b"): 0.9, ("b", "a"): 0.9}
    assert (
        representative(["a", "b"], FakeNLI(table), papers=[1, 3], ids=["o1", "o2"]) == 1
    )  # b is cited by more papers
    assert (
        representative(["a", "b"], FakeNLI(table), papers=[2, 2], ids=["o2", "o1"]) == 1
    )  # same papers: b has id o1


def test_label_group_setting_scores_a_record_matching_a_single_definition_group_by_that_one_definition():
    from mbdef.postprocess.group_setting import GroupSetting

    # "framing" is a group of one concept with one definition "fr" (length 2 -> 0.2), in both member settings
    for member in ("all", "representative"):
        setting = GroupSetting(MeanLength(), corpus_fixture(), member=member)
        assert setting.score(
            [rec(0, "Framing", ["a much longer text"])]
        ) == pytest.approx([0.2]), member


def test_fast_nli_audit_reports_the_largest_difference_and_the_flips_at_the_threshold():
    from mbdef.postprocess.nli import BandedNLI

    class Fixed:
        def __init__(self, values):
            self.values = values

        def entail(self, pairs):
            return [self.values[p] for p in pairs]

    pairs = [("x", "y"), ("x", "z"), ("y", "z")]
    fast = Fixed({("x", "y"): 0.49, ("x", "z"): 0.80, ("y", "z"): 0.20})
    ref = Fixed({("x", "y"): 0.51, ("x", "z"): 0.80, ("y", "z"): 0.25})
    # |dP| 0.02, 0, 0.05 -> largest 0.05; (x, y) is below 0.5 fast and above it exactly -> 1 flip
    a = BandedNLI(threshold=0.5, reference=ref, fast=fast).audit(pairs)
    assert (a["pairs"], a["flips"]) == (3, 1) and a["max_abs_diff"] == pytest.approx(
        0.05
    )
