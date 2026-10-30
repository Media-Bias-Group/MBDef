"""Comparison rows that learn directly from labelled neighbours (no clusters):

NeighbourVote        share of relevant records among the `neighbours` nearest labelled training definitions
                     (cosine, full embedding space); an item scores the mean over its definitions.
EmbeddingClassifier  logistic regression on an item's mean definition vector (L2-normalised), C chosen by inner
                     paper-grouped cross-validation on log-loss.
"""

import numpy as np

from mbdef.embed import definition_text

from .. import SEED
from ..embeddings import EmbeddingCache
from .base import term, texts

CS = (0.1, 0.3, 1.0, 3.16, 10.0, 31.6, 100.0)


def _definition_matrix(embeddings, items):
    owner, keys = [], []
    for i, item in enumerate(items):
        for t in texts(item):
            owner.append(i)
            keys.append(definition_text(term(item), t))
    X = embeddings.vectors(keys)
    return np.array(owner, dtype=np.int64), X / np.linalg.norm(X, axis=1, keepdims=True)


class NeighbourVote:
    name = (
        "nearest neighbours: share relevant among the 15 nearest labelled definitions"
    )

    def __init__(self, neighbours=15, embeddings=None):
        self.neighbours, self.embeddings = neighbours, embeddings

    def fit(self, train, corpus=None):
        self.embeddings = self.embeddings or EmbeddingCache()
        owner, self.X = _definition_matrix(self.embeddings, train)
        self.y = np.array([r.label for r in train], dtype=float)[owner]

    def score(self, items):
        owner, Q = _definition_matrix(self.embeddings, items)
        n = min(self.neighbours, len(self.X))
        votes = np.empty(len(Q))
        for s in range(0, len(Q), 1024):
            near = np.argpartition(-(Q[s : s + 1024] @ self.X.T), n - 1, axis=1)[:, :n]
            votes[s : s + 1024] = self.y[near].mean(axis=1)
        return (
            np.bincount(owner, weights=votes, minlength=len(items))
            / np.bincount(owner, minlength=len(items))
        ).tolist()


class EmbeddingClassifier:
    name = "embedding classifier: logistic regression on the mean definition vector"

    def __init__(self, embeddings=None, inner_folds=5, seed=SEED):
        self.embeddings, self.inner_folds, self.seed = embeddings, inner_folds, seed

    def _items(self, items):
        owner, D = _definition_matrix(self.embeddings, items)
        X = np.zeros((len(items), D.shape[1]))
        np.add.at(X, owner, D)
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def fit(self, train, corpus=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold

        self.embeddings = self.embeddings or EmbeddingCache()
        X, y = self._items(train), np.array([r.label for r in train])
        groups = np.array([r.paper_id for r in train])
        splits = list(
            StratifiedGroupKFold(
                self.inner_folds, shuffle=True, random_state=self.seed
            ).split(X, y, groups)
        )
        loss = []
        for c in CS:
            ll = []
            for tr, te in splits:
                p = np.clip(
                    LogisticRegression(C=c, max_iter=3000)
                    .fit(X[tr], y[tr])
                    .predict_proba(X[te])[:, 1],
                    1e-6,
                    1 - 1e-6,
                )
                ll.append(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
            loss.append(np.mean(ll))
        self.C = CS[int(np.argmin(loss))]
        self.model = LogisticRegression(C=self.C, max_iter=3000).fit(X, y)

    def score(self, items):
        return self.model.predict_proba(self._items(items))[:, 1].tolist()
