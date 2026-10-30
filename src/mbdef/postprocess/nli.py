"""Entailment (NLI) for Step 5, with the model of the extraction score DefScore (the paper's pair score of word overlap, embedding similarity and entailment): tasksource/ModernBERT-large-nli at its pinned revision.

Providers expose `entail(pairs) -> P(entailment)` with pair[0] the premise:
  ReferenceNLI  the exact fp32 scorer of the metric (mbdef.evaluate.metric.NLIScorer, at its pinned model
                and revision NLI_MODEL and NLI_REVISION), in batches of BATCH_SIZE pairs; every evaluation score uses it.
  BandedNLI     the fast scorer (nli_fast.CompiledNLI: fp16, token-budget batches, torch.compile; 5.6x faster), with
                every pair whose fast score lies within `band` of the decision threshold (0.5 for Coverage, the path's
                cut for keep or drop decisions) rescored with the reference, so every decision at that threshold
                matches the reference (plain fast scoring flipped 2,934 of 92,969 pairs within 0.01 of 0.5; the
                largest fast-against-exact difference seen was 0.0128). `audit(pairs)` compares a random sample with the
                reference.

Coverage of a Definition inside a set = share of the set's other texts it entails (P >= threshold, it is the premise).
The Representative Definition of a set has the highest Coverage; ties go to the higher mean P, then to the text used by more distinct papers, then to the smallest Definition id."""

import re

import numpy as np

from . import SEED

# Pairs per batch of the exact scorer: every saved Step 5 entailment score was made with 128 pairs per batch. The batch changes only float noise (how the pairs are padded), but keeping it lets a rerun reproduce the saved probabilities to within 1e-5 (the rule: pin what makes a rerun differ, rather than widen the tolerance).
BATCH_SIZE = 128


class ReferenceNLI:
    def __init__(self, batch_size=BATCH_SIZE):
        from mbdef.evaluate.metric import NLIScorer

        self.scorer = NLIScorer(batch_size=batch_size)

    def entail(self, pairs):
        return np.asarray(self.scorer.directional(list(pairs)), dtype=float)


class BandedNLI:
    def __init__(
        self, threshold=0.5, band=0.01, reference=None, device="cuda", fast=None
    ):
        """`threshold` is the decision point the scores feed: 0.5 for Coverage, a path's cut for keep/drop decisions."""
        if fast is None:
            from .nli_fast import CompiledNLI

            fast = CompiledNLI(device=device)
        self.fast, self.threshold, self.band = fast, threshold, band
        self._reference = reference
        self.last_rescored, self.last_pairs = 0, []

    @property
    def reference(self):
        if self._reference is None:
            self._reference = ReferenceNLI()
        return self._reference

    def entail(self, pairs):
        pairs = list(pairs)
        p = np.asarray(self.fast.entail(pairs), dtype=float)
        near = np.flatnonzero(np.abs(p - self.threshold) < self.band)
        if len(near):
            p[near] = self.reference.entail([pairs[i] for i in near])
        self.last_rescored, self.last_pairs = int(len(near)), pairs
        return p

    def audit(self, pairs, n=2000, seed=SEED):
        """Largest |dP| and decision flips at the threshold of the fast scores (before band rescoring) on a sample."""
        pairs = list(pairs)
        rng = np.random.default_rng(seed)
        sample = [
            pairs[i] for i in rng.choice(len(pairs), min(n, len(pairs)), replace=False)
        ]
        f, r = (
            np.asarray(self.fast.entail(sample), float),
            np.asarray(self.reference.entail(sample), float),
        )
        return dict(
            pairs=len(sample),
            max_abs_diff=float(np.abs(f - r).max()),
            flips=int(((f >= self.threshold) != (r >= self.threshold)).sum()),
        )


def dedup_key(text):
    """Duplicates inside a set are judged ignoring case, punctuation and whitespace."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def coverage(texts, nli, threshold=0.5):
    """(coverage, mean P) per text against all other texts of the set (exhaustive, both orders of every pair)."""
    n = len(texts)
    if n < 2:
        return np.ones(n), np.ones(n)
    pairs = [(texts[i], texts[j]) for i in range(n) for j in range(n) if i != j]
    p = np.asarray(nli.entail(pairs), dtype=float).reshape(n, n - 1)
    return (p >= threshold).mean(axis=1), p.mean(axis=1)


def representative(texts, nli, papers=None, ids=None, threshold=0.5):
    """Index of the Representative Definition among `texts` (see module docstring for the tie order)."""
    cov, mean_p = coverage(texts, nli, threshold)
    papers = papers or [0] * len(texts)
    ids = ids or list(range(len(texts)))
    return min(
        range(len(texts)), key=lambda i: (-cov[i], -mean_p[i], -papers[i], ids[i])
    )
