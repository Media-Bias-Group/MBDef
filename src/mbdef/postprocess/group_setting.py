"""The Label Group setting: every term whose label Label-Matches a corpus Label Group gets that group's score, as in Step 5.4 where all terms of a group share one decision. Terms without a corpus group keep their own score.

member="all"             the group's score is the mean over all its corpus definitions: the definition-weighted mean of
                         its member terms' scores (for path 1, whose term score is a mean over definitions);
member="representative"  the group is scored by its Representative Definition only, under the label of the term it
                         comes from (path 2, as the method's description says).

`rule` is the version of the Label Match rule the groups are made with (label_match.RULE_VERSIONS); the frozen train + dev evaluations use C-20261015.
"""

from collections import defaultdict
from dataclasses import dataclass

from .label_match import RULE_VERSION, label_groups, label_keys, vocabulary
from .paths.base import term, texts


@dataclass(frozen=True)
class _GroupItem:
    term: str
    definitions: tuple[str, ...]


class GroupSetting:
    def __init__(
        self, path, corpus, member="all", representatives=None, rule=RULE_VERSION
    ):
        if member not in ("all", "representative"):
            raise ValueError("member must be 'all' or 'representative'")
        self.path, self.member = path, member
        self.name = f"{path.name} · Label Group"
        labels = [c.label for c in corpus]
        self.vocab = vocabulary(labels, rule)
        groups = label_groups(labels, rule=rule)
        self.members = defaultdict(list)
        self.by_word_set, self.by_spacing = {}, {}
        for c, g in zip(corpus, groups, strict=True):
            self.members[g].append(c)
            k0, k1 = label_keys(c.label, self.vocab)
            self.by_word_set.setdefault(k0, g)
            if k1:
                self.by_spacing.setdefault(k1, g)
        self.representative = {}
        for key, text in (
            representatives or {}
        ).items():  # keyed by a member's word-set key, e.g. "bias media"
            if key not in self.by_word_set:
                raise ValueError(
                    f"representative given for {key!r}, which is no corpus Label Group's word-set key"
                )
            self.representative[self.by_word_set[key]] = text
        self.matched = 0

    def group_of(self, item):
        k0, k1 = label_keys(term(item), self.vocab)
        if k0 in self.by_word_set:
            return self.by_word_set[k0]
        return self.by_spacing.get(k1) if k1 else None

    def _representative_item(self, g):
        concepts = self.members[g]
        text = self.representative.get(
            g, texts(concepts[0])[0]
        )  # without one, the group has one distinct text
        owner = next((c for c in concepts if text in texts(c)), concepts[0])
        return _GroupItem(owner.label, (text,))

    def _group_scores(self, distinct):
        if not distinct:
            return {}
        if self.member == "representative":
            return dict(
                zip(
                    distinct,
                    self.path.score([self._representative_item(g) for g in distinct]),
                    strict=True,
                )
            )
        concepts = [c for g in distinct for c in self.members[g]]
        scores = iter(self.path.score(concepts))
        out = {}
        for g in distinct:
            weighted = [(next(scores), len(texts(c))) for c in self.members[g]]
            out[g] = sum(s * n for s, n in weighted) / sum(n for _, n in weighted)
        return out

    def fit(self, train, corpus=None):
        self.path.fit(train, corpus)

    def score(self, items):
        groups = [self.group_of(i) for i in items]
        self.matched = sum(g is not None for g in groups)
        distinct = sorted({g for g in groups if g is not None})
        group_score = self._group_scores(distinct)
        alone = [i for i, g in zip(items, groups, strict=True) if g is None]
        own = iter(self.path.score(alone)) if alone else iter(())
        return [group_score[g] if g is not None else next(own) for g in groups]
