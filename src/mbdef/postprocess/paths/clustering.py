"""Step 5.2 path 1: relevance from clusters of corpus definitions.

The corpus geometry is fitted on the corpus alone (no labels): definition vectors -> PCA to `dims` -> k-means with `k` clusters. A ground-truth definition is placed in a cluster either by the majority of its `neighbours` nearest corpus definitions (cosine in the PCA space; assignment="neighbours", the frozen setting) or by the nearest cluster centre in the PCA space, k-means' own rule (assignment="centre"). `fit` learns each cluster's smoothed share of relevant training definitions, (relevant + a*p0) / (n + a), with p0 the training records' relevant share; `score` gives an item the mean share over its definitions' clusters.

Frozen settings (nested CV on train+dev): dims 20, k 100, a 2, neighbours 15.

Pinned cluster map (the rule: pin what makes a rerun differ): scikit-learn's k-means does not end in the same clustering from run to run when it uses four or more CPU threads (30 runs of the frozen path gave train + dev AUC 0.841-0.860).
So a rerun reads the saved map, `pinned` = the frozen clusters/points.npz (each corpus Definition's cluster, by its id), instead of fitting k-means again; only the PCA, which is deterministic, is fitted. `pinned=None` fits a new k-means, as on new data.
"""

import numpy as np

from mbdef.embed import definition_text

from .. import FROZEN, REPO, SEED
from ..embeddings import EmbeddingCache
from .base import term, texts

# The saved cluster map: def_id and cluster of every corpus Definition (a frozen input; see docs/step-5-post-processing/postprocess.md, "Pinned cluster map")
PINNED = REPO / FROZEN / "20261015/clusters/points.npz"


def pinned_labels(path, corpus):
    """The saved cluster of every corpus Definition, in corpus Definition order; every Definition must be in the map."""
    z = np.load(path)
    saved = dict(zip(z["def_id"].tolist(), z["cluster"].tolist(), strict=True))
    ids = [d.def_id for c in corpus for d in c.definitions]
    missing = [i for i in ids if i not in saved]
    if missing or len(ids) != len(saved):
        raise ValueError(
            f"the cluster map {path} holds {len(saved):,} Definitions, the corpus {len(ids):,} "
            f"({len(missing):,} not in the map): pass pinned=None to fit a new k-means"
        )
    return np.array([saved[i] for i in ids], dtype=np.int64)


class ClusterShare:
    name = "clustering: cluster share"

    def __init__(
        self,
        dims=20,
        k=100,
        a=2.0,
        neighbours=15,
        assignment="neighbours",
        embeddings=None,
        seed=SEED,
        pinned=None,
    ):
        if assignment not in ("neighbours", "centre"):
            raise ValueError("assignment must be 'neighbours' or 'centre'")
        if pinned is not None and assignment == "centre":
            raise ValueError(
                "a pinned cluster map has no k-means centres: use assignment='neighbours'"
            )
        self.dims, self.k, self.a, self.neighbours, self.seed = (
            dims,
            k,
            a,
            neighbours,
            seed,
        )
        self.assignment = assignment
        self.embeddings = embeddings
        self.pinned = pinned  # a saved cluster map (points.npz), or None to fit k-means
        self._geometry = (
            None  # (corpus id, pca, unit corpus coordinates, corpus cluster labels)
        )

    def _keys(self, items):
        owner, keys = [], []
        for i, item in enumerate(items):
            for t in texts(item):
                owner.append(i)
                keys.append(definition_text(term(item), t))
        return np.array(owner, dtype=np.int64), keys

    def _fit_geometry(self, corpus):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        if self._geometry is not None and self._geometry[0] is corpus:
            return
        if self.embeddings is None:
            self.embeddings = EmbeddingCache()
        _, keys = self._keys(corpus)
        X = self.embeddings.vectors(keys)
        pca = PCA(
            self.dims,
            svd_solver="randomized" if self.dims < min(X.shape) else "full",
            random_state=self.seed,
        ).fit(X)
        Z = pca.transform(X)
        unit = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        if self.pinned is not None:
            self._geometry, self._kmeans = (
                (corpus, pca, unit, pinned_labels(self.pinned, corpus)),
                None,
            )
            return
        km = KMeans(self.k, n_init=3, random_state=self.seed).fit(Z)
        self._geometry = (corpus, pca, unit, km.labels_)
        self._kmeans = km

    def assign(self, items):
        """(owner index per definition, cluster per definition) for the items' definitions."""
        owner, keys = self._keys(items)
        if self._geometry is None or self.embeddings is None:
            raise ValueError("fit the path on the corpus before assigning clusters")
        _, pca, unit, labels = self._geometry
        Q = pca.transform(self.embeddings.vectors(keys))
        if self.assignment == "centre":
            if self._kmeans is None:
                raise ValueError("a pinned cluster map has no k-means centres")
            return owner, self._kmeans.predict(Q).astype(np.int64)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        n = min(self.neighbours, len(unit))
        clusters = np.empty(len(Q), dtype=np.int64)
        for s in range(0, len(Q), 1024):
            sim = Q[s : s + 1024] @ unit.T
            near = np.argpartition(-sim, n - 1, axis=1)[:, :n]
            clusters[s : s + 1024] = [
                np.bincount(labels[row], minlength=self.k).argmax() for row in near
            ]
        return owner, clusters

    def corpus_geometry(self):
        """(unit-length PCA coordinates, cluster label) of every corpus definition, in corpus definition order."""
        if self._geometry is None:
            raise ValueError("fit the path on the corpus first")
        return self._geometry[2], self._geometry[3]

    def fit(self, train, corpus=None):
        if corpus is None and self._geometry is None:
            raise ValueError("clustering needs the corpus to fit its geometry")
        if corpus is not None:
            self._fit_geometry(corpus)
        y = np.array([r.label for r in train], dtype=float)
        owner, clusters = self.assign(train)
        self.p0 = float(y.mean())
        rel = np.bincount(clusters, weights=y[owner], minlength=self.k)
        n = np.bincount(clusters, minlength=self.k).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.share = np.where(
                n + self.a > 0, (rel + self.a * self.p0) / (n + self.a), self.p0
            )

    def score(self, items):
        owner, clusters = self.assign(items)
        sums = np.bincount(owner, weights=self.share[clusters], minlength=len(items))
        counts = np.bincount(owner, minlength=len(items))
        return np.where(counts > 0, sums / np.maximum(counts, 1), self.p0).tolist()
