"""Comparison row: the keyword rule. Fits nothing; scores 1 when the term or a definition mentions bias, media, news, journalism or press (whole words), else 0."""

import re

from .base import term, texts

PATTERN = re.compile(r"\b(bias|biased|media|news|journalis\w*|press)\b", re.I)


class KeywordRule:
    name = "keyword rule"

    def fit(self, train, corpus=None):
        pass

    def score(self, items):
        return [
            float(bool(PATTERN.search(term(item) + " " + " ".join(texts(item)))))
            for item in items
        ]
