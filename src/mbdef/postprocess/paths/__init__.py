"""Relevance paths: anything with fit(train, corpus) and score(items), by the short name the command line uses.

A name may carry one parameter after a colon: "representative:mean_of_both", "zero-shot:2" (index into HYPOTHESES).
The entailment paths share one entailment scorer per run (`shared["nli"]`); the paths that read embeddings take `shared["embeddings"]` when it is given.

    keyword         KeywordRule           the term or a definition says bias, media, news, journalism or press
    clustering      ClusterShare          path 1: how relevant the training records in the term's clusters were
    neighbours      NeighbourVote         share relevant among the nearest labelled definitions
    classifier      EmbeddingClassifier   logistic regression on the mean definition vector
    representative  RepresentativeNLI     path 2: entailment against the Representative Media Bias Definition
    zero-shot       ZeroShotNLI           path 2: entailment of a fixed sentence about media bias

A new path is a class with `fit` and `score` and one more branch in `make`.
"""

from .clustering import ClusterShare
from .keyword import KeywordRule
from .neighbours import EmbeddingClassifier, NeighbourVote
from .nli import HYPOTHESES, RepresentativeNLI, ZeroShotNLI


def make(name, shared=None, kwargs=None):
    """The path called `name`; `kwargs` (e.g. from the frozen record) override the class defaults."""
    base, _, param = name.partition(":")
    kwargs = dict(kwargs or {})
    if base in ("clustering", "neighbours", "classifier"):
        if shared and "embeddings" in shared:
            kwargs.setdefault("embeddings", shared["embeddings"])
        if base == "clustering":
            return ClusterShare(**kwargs)
        if base == "neighbours":
            return NeighbourVote(**kwargs)
        return EmbeddingClassifier(**kwargs)
    if base in ("representative", "zero-shot"):
        shared = shared if shared is not None else {}
        if "nli" not in shared:
            from ..nli import ReferenceNLI

            shared["nli"] = ReferenceNLI()
        kwargs["nli"] = shared["nli"]
        if base == "representative":
            if param:
                kwargs["direction"] = param
            return RepresentativeNLI(**kwargs)
        if param:
            kwargs["hypothesis"] = HYPOTHESES[int(param)]
        return ZeroShotNLI(**kwargs)
    if base == "keyword":
        return KeywordRule(**kwargs)
    raise SystemExit(
        f"unknown path {name!r}: keyword, clustering, neighbours, classifier, representative or zero-shot"
    )
