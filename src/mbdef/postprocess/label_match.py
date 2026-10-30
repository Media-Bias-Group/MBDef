"""Label Match (Step 5.1): when two labels name the same thing up to spelling.

Every label gets two keys, and Label Groups are the connected components over both:

  word-set key  content words, singular, sorted       "bias in the media" -> "bias media"
  spacing key   all words, singular, in order, joined  "click-bait"        -> "clickbait"

So "media bias" joins "bias in the media" through the word-set key and "click bait" joins "clickbait" through the spacing key. The steps, in order:

  1. Unicode NFKC.
  2. Brackets, read on the original case. Dropped: an acronym of the other part ("FN (fake news)", "Sentiment Analysis (SA)"), a 1-2 character tag ("Accuracy (A)", "Arab(s)"), an abbreviation of a word of the label, or an alternative name introduced by "or / i.e. / aka" ("misinformation (or fake news)"). Anything else is kept as a qualifier, so "Affect (negative)" is not "affect". A prefix in brackets, "(mis)information", just loses its brackets.
  3. Case-folding; quotes and apostrophes removed without a space ("liar's" -> "liars"); hyphen, slash, dash and underscore become spaces, but their words stay marked as one compound; other punctuation becomes a space.
  4. A leading article is dropped, except in "a priori" (and "a posteriori", "a fortiori").
  5. Every word made singular when the singular is attested in the vocabulary (words such as news, media, bias, politics are never touched). For a word in "-es" the "-s", "-es" and "-ses -> -sis" cuts compete ("refugees" -> "refugee", "classes" -> "class", "crises" -> "crisis"): the one that ends more labels wins (then the one in more labels), and a function word ("them", "not", "us") never does. Irregular plurals ("phenomena", "criteria", "men") and the "-ics" words that are plurals ("topics", "heuristics", "publics"; not "civics" or "ethics") come from fixed lists.
  6. British spellings mapped to the American form when that form is attested (-isation, -ised, -ising, -ise, -our, and a fixed list: "judgement", "labelling", "modelling", "defence", "centre", ...).
  7. Stop words dropped only between content words and never inside a compound: "in-group bias" keeps "in", "bias in the media" loses "in" and "the".

Bare acronyms ("AI") are never expanded. The vocabulary is every word of the labels being grouped after steps 1-4, with the number of distinct labels it occurs in. `trace` lists the steps that changed one label, for provenance.

Two versions of the rule are kept, by name (`RULE_VERSIONS`). C-20261021 (the default) is the one described above. C-20261015 made the Label Groups that the frozen @group evaluations use: its step 5 takes the first attested of "-ies -> -y", "-yses -> -ysis", "-es" and "-s" (so "refugees" became "refuge" when "refuge" was a word of the labels), with no fixed lists, and its step 6 has no word list.
A vocabulary carries its version, so `vocabulary(labels, rule)` picks it for every key made with it.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "by",
    "with",
    "as",
    "at",
    "from",
    "into",
    "within",
    "or",
}
# Words ending in -s that are not plurals.
KEEP = {
    "news",
    "media",
    "bias",
    "series",
    "species",
    "means",
    "politics",
    "economics",
    "ethics",
    "analytics",
    "statistics",
    "linguistics",
    "physics",
    "mathematics",
    "dynamics",
    "semantics",
    "pragmatics",
    "metrics",
    "logistics",
    "optics",
    "genetics",
    "semiotics",
    "rhetorics",
    "demographics",
    "graphics",
    "robotics",
    "informatics",
    "lens",
    "chaos",
    "ethos",
    "pathos",
    "logos",
    "status",
    "consensus",
    "corpus",
    "virus",
    "campus",
    "focus",
    "bonus",
    "census",
    "thesis",
    "crisis",
    "analysis",
    "basis",
    "emphasis",
    "synthesis",
    "hypothesis",
    "diagnosis",
    "axis",
    "gas",
    "alias",
    "canvas",
    "atlas",
    "arts",
    "humanities",
    "always",
    "perhaps",
    "towards",
    "afterwards",
    "whereas",
    "across",
    "various",
    "previous",
    "serious",
    "famous",
    "nervous",
    "anxious",
    "obvious",
}
# Words that are never the singular of an "-es" plural ("themes" is not "them" + "es").
NOT_SINGULAR = STOP | {
    "them",
    "not",
    "us",
    "his",
    "its",
    "was",
    "has",
    "these",
    "those",
}
# Plurals the suffix rules cannot read (the plural audit behind C-20261021). "media" and "data" are deliberately absent.
IRREGULAR = {
    "phenomena": "phenomenon",
    "criteria": "criterion",
    "corpora": "corpus",
    "matrices": "matrix",
    "indices": "index",
    "appendices": "appendix",
    "stimuli": "stimulus",
    "curricula": "curriculum",
    "men": "man",
    "women": "woman",
    "children": "child",
}
# "-ics" words that are plurals of an "-ic" noun; every other "-ics" word (civics, ethics, epistemics, linguistics, forensics, ...) names a field and stays as it is.
ICS_PLURALS = {
    "topics",
    "subtopics",
    "tactics",
    "heuristics",
    "characteristics",
    "publics",
    "counterpublics",
    "logics",
    "infodemics",
    "epidemics",
    "pandemics",
    "panics",
    "skeptics",
    "sceptics",
    "critics",
    "clinics",
    "comics",
    "generics",
    "republics",
    "relics",
    "mimics",
}
# British spellings outside the suffix rules, word by word (a suffix rule would also turn "pilling" into "piling").
UK_US = {
    "centre": "center",
    "centres": "centers",
    "theatre": "theater",
    "theatres": "theaters",
    "defence": "defense",
    "offence": "offense",
    "offences": "offenses",
    "pretence": "pretense",
    "licence": "license",
    "judgement": "judgment",
    "judgements": "judgments",
    "labelling": "labeling",
    "labelled": "labeled",
    "labellers": "labelers",
    "labeller": "labeler",
    "modelling": "modeling",
    "modelled": "modeled",
    "signalling": "signaling",
    "channelling": "channeling",
    "travelling": "traveling",
    "counselling": "counseling",
    "cancelled": "canceled",
    "programme": "program",
    "programmes": "programs",
    "catalogue": "catalog",
    "analogue": "analog",
    "dialogue": "dialog",
    "dialogues": "dialogs",
}
RULE = "Label Match"  # the rule's name in the files it writes
# The rule version: C-20261021 differs from C-20261015 in that "-es" plurals choose between the "-s" and "-es" cut by how many labels each ends, then occurs in. Give a new version whenever a key can change.
RULE_VERSION = "C-20261021"
RULE_VERSIONS = ("C-20261015", "C-20261021")
LATIN = ("priori", "posteriori", "fortiori")
# The steps that can be switched off one at a time to count what each saves (`step_savings`), with their plain names.
STEPS = {
    "case": "case-folding",
    "plural": "plural → singular",
    "brackets": "brackets (acronyms, tags, alternative names)",
    "stop": "stop words",
    "order": "word order",
    "join": "hyphens, slashes and spacing (spacing key)",
    "spelling": "British → American spelling",
    "article": "leading article",
    "quotes": "quotes and apostrophes",
}

_BRACKET = re.compile(
    r"^(?P<a>.*?)\s*[(\[]\s*(?P<b>[^()\[\]]{1,120})\s*[)\]]\s*(?P<rest>.*)$"
)
_PREFIX = re.compile(r"\((\w{1,12})\)(?=\w)")
_ALT = re.compile(
    r"^(or|i\.?\s?e\.?|e\.?\s?g\.?|aka|a\.k\.a\.|also|also known as|known as|hereafter|henceforth)\b",
    re.I,
)
_QUOTES = re.compile(r"[\"'‘’“”`´]")
_JOIN = re.compile(r"[\-/–—_]")
_PUNCT = re.compile(r"[^\w\s\x00]")
_SPELL = (
    ("isations", "izations"),
    ("isation", "ization"),
    ("ised", "ized"),
    ("ising", "izing"),
    ("ises", "izes"),
    ("ise", "ize"),
    ("ours", "ors"),
    ("our", "or"),
)


def _is_acronym(short: str, long_form: str) -> bool:
    s = re.sub(r"[^A-Za-z0-9]", "", short)
    if not 2 <= len(s) <= 12 or " " in short.strip():
        return False
    initials = "".join(w[0] for w in re.split(r"[\s\-/]+", long_form) if w).casefold()
    if s.casefold() == initials:
        return True  # "fp (false positive)": exact initials, any case
    if sum(c.isupper() for c in s) < max(2, len(s) // 2):
        return False
    it = iter(initials + " " + long_form.casefold())
    return all(c in it for c in s.casefold() if c.isalpha())


def _split_brackets_why(text: str) -> tuple[str, str | None, list[str]]:
    """(label without its bracket, the bracket content kept as a qualifier or None, what was done and why)."""
    why = [f"prefix in brackets opened: '({p})'" for p in _PREFIX.findall(text)]
    text = _PREFIX.sub(r"\1", text)
    m = _BRACKET.match(text)
    if not m or not m["a"].strip():
        return text, None, why
    a, b, rest = m["a"].strip(" ,;:-"), m["b"].strip(), m["rest"].strip()
    main = f"{a} {rest}".strip()
    if _is_acronym(b, a):
        return main, None, [*why, f"bracket '({b})' dropped: an acronym of '{a}'"]
    if not rest and _is_acronym(a, b):
        return (
            b,
            None,
            [
                *why,
                f"'{a}' dropped: an acronym of its bracket '({b})', which is kept as the label",
            ],
        )
    tag = re.sub(r"\W", "", b).casefold()
    if len(tag) <= 2:
        return main, None, [*why, f"bracket '({b})' dropped: a tag of 1-2 characters"]
    if _ALT.match(b):
        return main, None, [*why, f"bracket '({b})' dropped: an alternative name"]
    if len(tag) <= 5 and any(w.casefold().startswith(tag) for w in a.split()):
        return (
            main,
            None,
            [*why, f"bracket '({b})' dropped: an abbreviation of a word of the label"],
        )
    return main, b, [*why, f"bracket '({b})' kept as a qualifier"]


def _split_brackets(text: str) -> tuple[str, str | None]:
    """(label without its bracket, the bracket content kept as a qualifier or None)."""
    main, qualifier, _ = _split_brackets_why(text)
    return main, qualifier


def _words(label: str, off: str | None = None) -> list[tuple[str, bool]]:
    """Words after steps 1-4, each with True when it came from a compound (or is the "a" of "a priori")."""
    text = unicodedata.normalize("NFKC", label or "")
    if off != "brackets":
        main, qualifier = _split_brackets(text)
        text = f"{main} {qualifier}" if qualifier else main
    if off != "case":
        text = text.casefold()
    if off != "quotes":
        text = _QUOTES.sub("", text)
    words = []
    for chunk in text.split():
        if off == "join":
            words += [
                (p, False) for p in _PUNCT.sub(" ", chunk.replace("\x00", " ")).split()
            ]
            continue
        parts = (
            _PUNCT.sub(" ", _JOIN.sub("\x00", chunk)).replace("\x00", " \x00 ").split()
        )
        compound = "\x00" in parts
        words += [(p, compound) for p in parts if p != "\x00"]
    latin = {
        i
        for i in range(len(words) - 1)
        if words[i][0] == "a" and words[i + 1][0] in LATIN
    }
    words = [(w, c or i in latin) for i, (w, c) in enumerate(words)]
    if (
        off != "article"
        and len(words) > 1
        and words[0][0] in ("the", "a", "an")
        and 0 not in latin
    ):
        words = words[1:]
    return words


def _singular_20261015(w: str, vocab: Mapping[str, int]) -> tuple[str, str | None]:
    """Rule C-20261015's plural step: the first attested of "-ies -> -y", "-yses -> -ysis", "-es" and "-s"."""
    if (
        len(w) < 4
        or w in KEEP
        or not w.endswith("s")
        or w.endswith(("ss", "us", "is", "ics"))
    ):
        return w, None
    candidates = (
        ([(w[:-3] + "y", "-ies -> -y")] if w.endswith("ies") else [])
        + ([(w[:-4] + "ysis", "-yses -> -ysis")] if w.endswith("yses") else [])
        + ([(w[:-2], "-es")] if w.endswith("es") else [])
        + [(w[:-1], "-s")]
    )
    return next(((c, how) for c, how in candidates if c in vocab), (w, None))


def _version(vocab: Mapping[str, int]) -> str:
    return getattr(vocab, "rule", RULE_VERSION)


def _singular_why(w: str, vocab: Mapping[str, int]) -> tuple[str, str | None]:
    """(singular, how it was chosen) of one word; the reason is None when the word is kept."""
    if _version(vocab) == "C-20261015":
        return _singular_20261015(w, vocab)
    if w in IRREGULAR and IRREGULAR[w] in vocab:
        return IRREGULAR[w], "irregular plural"
    if w in ICS_PLURALS and w[:-1] in vocab:
        return w[:-1], "-s (an -ics plural)"
    if (
        len(w) < 4
        or w in KEEP
        or not w.endswith("s")
        or w.endswith(("ss", "us", "is", "ics"))
    ):
        return w, None
    for suffix, repl in (("ies", "y"), ("yses", "ysis")):
        if w.endswith(suffix) and (c := w[: -len(suffix)] + repl) in vocab:
            return c, f"-{suffix} -> -{repl}"
    if w.endswith("es"):
        options = ((w[:-1], "-s"), (w[:-2], "-es")) + (
            ((w[:-2] + "is", "-ses -> -sis"),) if w.endswith("ses") else ()
        )
        cut = [(c, how) for c, how in options if c in vocab and c not in NOT_SINGULAR]
        if not cut:
            return w, None
        final = getattr(vocab, "final", {})
        c, how = max(cut, key=lambda x: (final.get(x[0], 0), vocab[x[0]], x[1] == "-s"))
        if len(cut) > 1:
            how += (
                f", ends {final.get(c, 0)} labels and occurs in {vocab[c]}, against "
                + "; ".join(
                    f"{final.get(o, 0)} and {vocab[o]} for '{o}'"
                    for o, _ in cut
                    if o != c
                )
            )
        return c, how
    return (w[:-1], "-s") if w[:-1] in vocab else (w, None)


def _singular(w: str, vocab: Mapping[str, int]) -> str:
    return _singular_why(w, vocab)[0]


def _american(w: str, vocab: Mapping[str, int]) -> str:
    if w in UK_US and _version(vocab) != "C-20261015":
        return UK_US[w] if UK_US[w] in vocab else w
    if len(w) < 6:
        return w
    for uk, us in _SPELL:
        if w.endswith(uk):
            c = w[: -len(uk)] + us
            return c if c in vocab else w
    return w


class Vocabulary(Counter):
    """Word -> number of distinct labels it occurs in; `final` counts the labels it ends (where a head noun stands); `rule` is the version of the Label Match rule that keys made with this vocabulary follow."""

    final: Counter
    rule: str


def vocabulary(labels: Iterable[str], rule: str = RULE_VERSION) -> Vocabulary:
    """Every word of the labels after steps 1-4: the words a plural or British spelling may be rewritten to, with how often each is used, which settles "-s" against "-es" ("rationales": "rationale" ends 5 labels, "rational" 4)."""
    words = [[w for w, _ in _words(label)] for label in set(labels)]
    vocab = Vocabulary(w for ws in words for w in set(ws))
    vocab.final = Counter(ws[-1] for ws in words if ws)
    if rule not in RULE_VERSIONS:
        raise ValueError(
            f"unknown rule version {rule!r}; versions are {', '.join(RULE_VERSIONS)}"
        )
    vocab.rule = rule
    return vocab


def label_keys(
    label: str, vocab: Mapping[str, int], off: str | None = None
) -> tuple[str, str]:
    """(word-set key, spacing key) of one label; `off` names one step of `STEPS` to leave out."""
    if off is not None and off not in STEPS:
        raise ValueError(f"unknown step {off!r}; steps are {', '.join(STEPS)}")
    words = _words(label, off)
    if off != "plural":
        words = [(_singular(w, vocab), c) for w, c in words]
    if off != "spelling":
        words = [(_american(w, vocab), c) for w, c in words]
    last = len(words) - 1
    content = [
        w
        for i, (w, c) in enumerate(words)
        if off == "stop" or c or w not in STOP or i in (0, last)
    ] or [w for w, _ in words]
    word_set = " ".join(content if off == "order" else sorted(content))
    return word_set, "" if off == "join" else "".join(w for w, _ in words)


def trace(label: str, vocab: Mapping[str, int]) -> dict:
    """Both keys of one label and every step that changed it, in plain words: the provenance of its Label Group.

    The keys are those of `label_keys`; the steps repeat `_words` and `label_keys`, recording what each one did.
    """
    steps = []
    text = unicodedata.normalize("NFKC", label or "")
    if text != (label or ""):
        steps.append("Unicode normalised (NFKC)")
    main, qualifier, why = _split_brackets_why(text)
    steps += why
    text = f"{main} {qualifier}" if qualifier else main
    if text.casefold() != text:
        steps.append("case-folded")
    text = text.casefold()
    if _QUOTES.search(text):
        steps.append("quotes and apostrophes removed")
    text = _QUOTES.sub("", text)
    if _JOIN.search(text):
        steps.append(
            "hyphen, slash, dash or underscore: the parts are one compound, joined in the spacing key"
        )
    if _PUNCT.search(_JOIN.sub(" ", text)):
        steps.append("other punctuation removed")
    words = _words(label)
    full = [
        w
        for chunk in text.split()
        for w in _PUNCT.sub(" ", _JOIN.sub(" ", chunk)).split()
    ]
    if len(full) > len(words):
        steps.append(f"leading article '{full[0]}' dropped")
    out = []
    for w, c in words:
        s, how = _singular_why(w, vocab)
        if how:
            steps.append(f"plural '{w}' -> '{s}' ({how})")
        a = _american(s, vocab)
        if a != s:
            steps.append(f"British spelling '{s}' -> '{a}'")
        out.append((a, c))
    last = len(out) - 1
    content = [
        w for i, (w, c) in enumerate(out) if c or w not in STOP or i in (0, last)
    ] or [w for w, _ in out]
    dropped = (
        [
            w
            for i, (w, c) in enumerate(out)
            if not (c or w not in STOP or i in (0, last))
        ]
        if len(content) < len(out)
        else []
    )
    if dropped:
        steps.append(
            "stop words left out of the word-set key: "
            + ", ".join(f"'{w}'" for w in dropped)
        )
    keys = (" ".join(sorted(content)), "".join(w for w, _ in out))
    if keys != label_keys(label, vocab):
        raise AssertionError(f"trace and label_keys disagree on {label!r}")
    return dict(
        label=label,
        words=[w for w, _ in out],
        word_set_key=keys[0],
        spacing_key=keys[1],
        steps=steps,
    )


def label_groups(
    labels: Sequence[str], off: str | None = None, rule: str = RULE_VERSION
) -> list[int]:
    """Label Group of every label: 0, 1, 2, ... numbered in order of each group's first label.

    Two labels are in one group when they share either key, directly or through a chain of other labels. The vocabulary is read from `labels` themselves, always with every step on; `off` leaves one step of `STEPS` out of the keys.
    """
    vocab = vocabulary(labels, rule)
    keys = [label_keys(label, vocab, off) for label in labels]
    parent = list(range(len(labels)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for which in (0, 1):
        first: dict[str, int] = {}
        for i, k in enumerate(keys):
            if k[which]:
                a, b = find(first.setdefault(k[which], i)), find(i)
                parent[max(a, b)] = min(a, b)
    number: dict[int, int] = {}
    return [number.setdefault(find(i), len(number)) for i in range(len(labels))]


def step_savings(labels: Sequence[str], rule: str = RULE_VERSION) -> dict[str, int]:
    """How many Label Groups each step saves: the groups with that one step switched off, minus the groups with all on.

    Steps overlap, so the savings do not add up to the total saved over exact labels.
    """
    distinct = sorted(
        set(labels)
    )  # equal labels always share a group, so counting distinct labels is enough
    groups = len(set(label_groups(distinct, rule=rule)))
    return {
        step: len(set(label_groups(distinct, off=step, rule=rule))) - groups
        for step in STEPS
    }


def group_counts(groups: Sequence, keep: Sequence) -> dict[str, int]:
    """How keep or drop decisions fall on Label Groups: `groups` is each decided item's group, `keep` its decision.
    Counts the groups, those whose items are all kept, all dropped, and those the decisions split."""
    by: dict = {}
    for g, k in zip(groups, keep, strict=True):
        by.setdefault(g, set()).add(bool(k))
    return dict(
        groups=len(by),
        all_kept=sum(v == {True} for v in by.values()),
        all_dropped=sum(v == {False} for v in by.values()),
        split=sum(len(v) == 2 for v in by.values()),
    )
