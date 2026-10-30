import pytest

from mbdef.postprocess.label_match import (
    RULE_VERSION,
    RULE_VERSIONS,
    STEPS,
    label_groups,
    label_keys,
    step_savings,
    trace,
    vocabulary,
)

SAME, APART = True, False

# The 40 real label pairs of the comparison that settled the Label Match rule, each pinned to the agreed answer: "to frame" stays apart from "frame", a bracketed translation stays apart, and bare acronyms are never expanded.
EDGE_CASES = [
    ("word order", "media bias", "bias in the media", SAME),
    ("word order", "sentiment analysis", "analysis of sentiment", SAME),
    ("word order", "framing theory", "theory of framing", SAME),
    ("word order", "source credibility", "credibility of sources", SAME),
    ("stop words", "satire or parody", "satire and parody", SAME),
    ("stop words", "agenda setting theory", "agenda setting as a theory", SAME),
    ("stop word in a compound", "in-group bias", "group bias", APART),
    ("stop word in a compound", "in/security", "security", APART),
    ("stop word in a compound", "in-topic", "topic", APART),
    ("verb vs noun", "to frame", "frame", APART),
    ("plural of a non-final word", "echo chambers effect", "echo chamber effect", SAME),
    (
        "plural of a non-final word",
        "uses and gratifications theory",
        "uses and gratification theory",
        SAME,
    ),
    ("hyphen / spacing", "mis-information", "misinformation", SAME),
    ("hyphen / spacing", "hatespeech", "hate speech", SAME),
    ("hyphen / spacing", "click bait", "clickbait", SAME),
    ("hyphen / spacing", "gate-keeping", "gatekeeping", SAME),
    ("underscore", "appeal_to_authority", "appeal to authority", SAME),
    ("quotes / apostrophes", "'hard' news", "hard news", SAME),
    ("quotes / apostrophes", "liar's dividend", "liars dividend", SAME),
    ("prefix in brackets", "(mis)information", "misinformation", SAME),
    ("prefix in brackets", "(hyper)partisan", "hyperpartisan", SAME),
    ("acronym after long form", "Sentiment Analysis (SA)", "sentiment analysis", SAME),
    ("acronym before long form", "FN (fake news)", "fake news", SAME),
    ("acronym before long form", "fp (false positive)", "false positive", SAME),
    ("one- or two-letter tag", "Accuracy (A)", "accuracy", SAME),
    ("one- or two-letter tag", "propaganda (p)", "propaganda", SAME),
    ("one- or two-letter tag", "Arab(s)", "Arab", SAME),
    (
        "alternative name in brackets",
        "misinformation (or fake news)",
        "misinformation",
        SAME,
    ),
    ("qualifier in brackets", "Affect (negative)", "Affect (positive)", APART),
    ("qualifier in brackets", "Affect (negative)", "affect", APART),
    (
        "qualifier in brackets",
        "misinformation (genuine information shared with the intent to harm)",
        "misinformation",
        APART,
    ),
    ("translation in brackets", "hate speech (discurso de odio)", "hate speech", APART),
    ("British / American", "polarisation", "polarization", SAME),
    ("British / American", "rumour", "rumor", SAME),
    ("British / American", "politicised", "politicized", SAME),
    ("leading article", "a priori bias", "priori bias", APART),
    ("leading article", "the framing effect", "framing effect", SAME),
    ("'X' vs 'X bias'", "anchoring", "anchoring bias", APART),
    ("'X' vs 'X bias'", "annotation", "annotation bias", APART),
    ("bare acronym", "AI", "artificial intelligence", APART),
]


def test_edge_case_table_has_the_40_compared_pairs():
    assert len(EDGE_CASES) == 40
    assert len({(a, b) for _, a, b, _ in EDGE_CASES}) == 40


@pytest.mark.parametrize(
    ("topic", "a", "b", "same"),
    EDGE_CASES,
    ids=[f"{a} | {b}" for _, a, b, _ in EDGE_CASES],
)
def test_edge_case_pair_is_one_label_group_only_when_agreed(topic, a, b, same):
    first, second = label_groups([a, b])
    assert (first == second) is same, topic


# Worked examples: (label, vocabulary the label is read with, word-set key, spacing key). The vocabulary decides which plurals and British spellings are rewritten, so each example names the words that must be attested.
KEYS = [
    ("bias in the media", [], "bias media", "biasinthemedia"),
    ("click-bait", [], "bait click", "clickbait"),
    ("FN (fake news)", [], "fake news", "fakenews"),
    ("Sentiment Analysis (SA)", [], "analysis sentiment", "sentimentanalysis"),
    ("Affect (negative)", [], "affect negative", "affectnegative"),
    ("in-group bias", [], "bias group in", "ingroupbias"),
    ("a priori bias", [], "a bias priori", "aprioribias"),
    ("the framing effect", [], "effect framing", "framingeffect"),
    (
        "echo chambers effect",
        ["echo chamber"],
        "chamber echo effect",
        "echochambereffect",
    ),
    ("polarisation", ["polarization"], "polarization", "polarization"),
    ("liar's dividend", [], "dividend liars", "liarsdividend"),
    ("(mis)information", [], "misinformation", "misinformation"),
]


def test_groups_chain_across_both_keys():
    labels = [
        "Appeal to Authority",  # word-set "appeal authority", spacing "appealtoauthority"
        "Appeal_to_Authority",  # word-set "appeal authority to" (compound keeps "to"), spacing "appealtoauthority"
        "authority appeal",  # word-set "appeal authority", spacing "authorityappeal"
        "hate speech",  # word-set "hate speech", spacing "hatespeech"
        "hatespeech",  # word-set "hatespeech", spacing "hatespeech"
        "speech of hate",  # word-set "hate speech", spacing "speechofhate"
        "hate",
        "speech",
        "media bias",
        "bias in the media",
        "fake news",
        "FN (fake news)",
        "news, fake",  # word-set "fake news", spacing "newsfake"
        "AI",
        "artificial intelligence",
    ]
    groups = label_groups(labels)
    assert groups == [0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 6, 7]
    # "Appeal_to_Authority" and "authority appeal" share neither key: they meet only through "Appeal to Authority".
    vocab = vocabulary(labels)
    underscore, inverted = (
        label_keys("Appeal_to_Authority", vocab),
        label_keys("authority appeal", vocab),
    )
    assert underscore[0] != inverted[0] and underscore[1] != inverted[1]
    # Same for "hatespeech" and "speech of hate", joined through "hate speech".
    joined, spread = (
        label_keys("hatespeech", vocab),
        label_keys("speech of hate", vocab),
    )
    assert joined[0] != spread[0] and joined[1] != spread[1]


def test_switching_one_step_off_counts_the_groups_it_saves():
    # One pair per step; "liar's dividend" also needs the plural step ("liars" -> "liar").
    labels = [
        "Echo Chamber",
        "echo chamber",  # case
        "filter bubbles",
        "filter bubble",  # plural
        "fake news (FN)",
        "fake news",  # brackets
        "satire or parody",
        "satire and parody",  # stop words
        "bias, selection",
        "selection bias",  # word order
        "click bait",
        "clickbait",  # hyphens and spacing
        "polarisation",
        "polarization",  # British / American
        "the framing effect",
        "framing effect",  # leading article
        "liar's dividend",
        "liar dividend",  # quotes, plural
    ]
    assert label_groups(labels) == [i // 2 for i in range(len(labels))]
    assert step_savings(labels) == dict(
        case=1,
        plural=2,
        brackets=1,
        stop=1,
        order=1,
        join=1,
        spelling=1,
        article=1,
        quotes=1,
    )
    apart = label_groups(labels, off="plural")
    assert apart[2] != apart[3] and apart[16] != apart[17] and apart[0] == apart[1]


def test_every_step_has_a_plain_name_and_unknown_steps_are_refused():
    assert set(STEPS) == {
        "case",
        "plural",
        "brackets",
        "stop",
        "order",
        "join",
        "spelling",
        "article",
        "quotes",
    }
    assert all(STEPS.values())
    with pytest.raises(ValueError, match="plurals"):
        label_groups(["framing"], off="plurals")


def test_groups_of_no_labels_and_of_repeated_labels():
    assert label_groups([]) == []
    assert label_groups(["framing", "Framing", "framing"]) == [0, 0, 0]


@pytest.mark.parametrize(
    ("label", "attested", "word_set", "spacing"), KEYS, ids=[k[0] for k in KEYS]
)
def test_label_keys_worked_examples(label, attested, word_set, spacing):
    vocab = vocabulary([label, *attested])
    assert label_keys(label, vocab) == (word_set, spacing)


# The plural audit behind Label Match C-20261021: plurals and spellings that Label Match C-20261015 got wrong or missed, each pinned with the words the corpus attests ("refuge" must not win over "refugee", "civics" is a field, "pilling" is not "piling").
PLURALS_20261021 = [
    (["refugees", "refugee", "refuge"], SAME, (0, 1)),
    (["themes", "theme", "them"], SAME, (0, 1)),
    (["uses and gratifications", "use and gratification", "us"], SAME, (0, 1)),
    (["journalistic roles", "journalistic role", "rol"], SAME, (0, 1)),
    (
        ["rationales", "rationale", "rational choice", "rational actor", "a rationale"],
        SAME,
        (0, 1),
    ),
    (
        ["redes sociales", "red social", "social media", "social bot", "sociale"],
        SAME,
        (0, 1),
    ),
    (["crises", "crisis", "political crisis", "crise"], SAME, (0, 1)),
    (
        ["knowledge bases", "knowledge base", "data base", "user base", "legal basis"],
        SAME,
        (0, 1),
    ),
    (["hostile media phenomena", "hostile media phenomenon"], SAME, (0, 1)),
    (["infodemics", "infodemic"], SAME, (0, 1)),
    (["civics", "civic"], APART, (0, 1)),
    (["epistemics", "epistemic"], APART, (0, 1)),
    (["judgement", "judgment"], SAME, (0, 1)),
    (["labelling theory", "labeling theory"], SAME, (0, 1)),
    (["red-pilling", "piling"], APART, (0, 1)),
    (["new media", "news media"], APART, (0, 1)),
]


@pytest.mark.parametrize(
    ("labels", "same", "pair"),
    PLURALS_20261021,
    ids=[" | ".join(x[0][:2]) for x in PLURALS_20261021],
)
def test_plurals_and_spellings_of_20261021(labels, same, pair):
    groups = label_groups(labels)
    assert (groups[pair[0]] == groups[pair[1]]) is same


def test_trace_names_each_step_and_agrees_with_the_keys():
    labels = [
        "Refugees",
        "refugee",
        "refuge",
        "the Framing Effects",
        "framing effect",
        "FN (fake news)",
        "fake news",
    ]
    vocab = vocabulary(labels)
    t = trace("Refugees", vocab)
    assert (
        (t["word_set_key"], t["spacing_key"])
        == label_keys("Refugees", vocab)
        == ("refugee", "refugee")
    )
    assert t["steps"][0] == "case-folded" and t["steps"][1].startswith(
        "plural 'refugees' -> 'refugee' (-s, ends 1 labels"
    )
    assert trace("the Framing Effects", vocab)["steps"] == [
        "case-folded",
        "leading article 'the' dropped",
        "plural 'effects' -> 'effect' (-s)",
    ]
    assert trace("FN (fake news)", vocab)["steps"] == [
        "'FN' dropped: an acronym of its bracket '(fake news)', which is kept as the label"
    ]


# Rule C-20261015 stays as it is (the frozen paths' Label Groups use it): the first attested of "-ies", "-yses", "-es" and "-s", no fixed lists. The cases Label Match C-20261021 changed come out the C-20261015 way.
@pytest.mark.parametrize(
    ("labels", "pair", "same"),
    [
        (
            ["refugees", "refugee", "refuge"],
            (0, 2),
            SAME,
        ),  # "-es" first: "refugees" -> "refuge"
        (["refugees", "refugee", "refuge"], (0, 1), APART),
        (["themes", "theme", "them"], (0, 2), SAME),
        (
            ["judgement", "judgment"],
            (0, 1),
            APART,
        ),  # no word list for British spellings
        (
            ["hostile media phenomena", "hostile media phenomenon"],
            (0, 1),
            APART,
        ),  # no irregular plurals
        (["infodemics", "infodemic"], (0, 1), APART),  # "-ics" is never cut
    ],
)
def test_rule_c_20261015_keeps_its_answers(labels, pair, same):
    groups = label_groups(labels, rule="C-20261015")
    assert (groups[pair[0]] == groups[pair[1]]) is same


def test_the_default_rule_is_c_20261021_and_unknown_versions_are_refused():
    assert RULE_VERSION == "C-20261021"
    assert RULE_VERSIONS == ("C-20261015", "C-20261021")
    assert vocabulary(["a"]).rule == RULE_VERSION
    with pytest.raises(ValueError, match="unknown rule version"):
        vocabulary(["a"], rule="C-19990101")
