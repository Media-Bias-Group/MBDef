"""Step 5.2 path 2: relevance from NLI entailment (no fitting; a term scores its best definition).

RepresentativeNLI  each definition against the Representative Media Bias Definition, in one of four combinations of
                   the two directions; frozen: representative => definition (best of four on train+dev).
ZeroShotNLI        premise "<term>: <definition>", hypothesis a fixed sentence with {term} filled in; frozen:
                   "This text is about media bias." (best of five predeclared hypotheses on train+dev).
"""

import numpy as np

from .base import term, texts

DIRECTIONS = (
    "definition_to_representative",
    "representative_to_definition",
    "larger_of_both",
    "mean_of_both",
)
REPRESENTATIVE = (
    "Media bias refers to the bias produced when journalists report about an event in a prejudiced "
    "manner or with a slanted viewpoint"
)
HYPOTHESES = (
    "This text is about media bias.",
    "{term} is a type of media bias or a concept used to study media bias.",
    "{term} is a form of media bias.",
    "{term} is relevant to research on media bias.",
    "{term} is related to news media, journalism or media bias.",
)


def _best_per_item(items, values):
    out, k = [], 0
    for item in items:
        n = len(texts(item))
        out.append(float(np.max(values[k : k + n])))
        k += n
    return out


class RepresentativeNLI:
    def __init__(
        self,
        representative=REPRESENTATIVE,
        direction="representative_to_definition",
        nli=None,
    ):
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        self.representative, self.direction, self.nli = representative, direction, nli
        self.name = (
            f"NLI against the representative definition: {direction.replace('_', ' ')}"
        )

    def fit(self, train, corpus=None):
        pass

    def score(self, items):
        if self.nli is None:
            from ..nli import ReferenceNLI

            self.nli = ReferenceNLI()
        defs = [t for item in items for t in texts(item)]
        r = self.representative
        if self.direction == "definition_to_representative":
            values = np.asarray(self.nli.entail([(d, r) for d in defs]), float)
        elif self.direction == "representative_to_definition":
            values = np.asarray(self.nli.entail([(r, d) for d in defs]), float)
        else:
            fwd = np.asarray(self.nli.entail([(d, r) for d in defs]), float)
            bwd = np.asarray(self.nli.entail([(r, d) for d in defs]), float)
            values = (
                np.maximum(fwd, bwd)
                if self.direction == "larger_of_both"
                else (fwd + bwd) / 2
            )
        return _best_per_item(items, values)


class ZeroShotNLI:
    def __init__(self, hypothesis=HYPOTHESES[0], nli=None):
        self.hypothesis, self.nli = hypothesis, nli
        self.name = f'NLI zero-shot: "{hypothesis}"'

    def fit(self, train, corpus=None):
        pass

    def score(self, items):
        if self.nli is None:
            from ..nli import ReferenceNLI

            self.nli = ReferenceNLI()
        pairs = [
            (f"{term(item)}: {t}", self.hypothesis.format(term=term(item)))
            for item in items
            for t in texts(item)
        ]
        return _best_per_item(items, np.asarray(self.nli.entail(pairs), float))
