"""Evaluation statistics: the one copy of each agreement statistic the package computes.

Plain functions over plain values. Each random procedure takes its seed, its number of replicates and its resampling unit as parameters, because every reported interval was made with particular ones and must be remade with the same; the callers pin them as named constants and record them.

- `pair_counts`, `alpha_from_counts` and `two_rater_numbers`: agreement of two raters on a Yes/No question, from the three counts Yes/Yes, No/No and disagree.
- `krippendorff_alpha`: Krippendorff's alpha for nominal answers in general (any number of raters per unit, missing answers allowed), from the coincidence matrix.
- `cohen_kappa`: Cohen's kappa, which depends on which rater counts as "A".
- `cluster_bootstrap`: percentile intervals for observed agreement and alpha, resampling whole clusters of pairs (for example papers or batches).
- `paired_bootstrap`, `index_interval`, `paired_permutation_p` and `sign_test`: two systems compared on the same units (for example papers), as the paired comparison with Opus does; Python's `random`.

numpy is imported inside `cluster_bootstrap` only, so the rest of the command line runs without it.
"""

import itertools
from collections import Counter, defaultdict
from math import comb


def pair_counts(pairs) -> tuple[int, int, int]:
    """(Yes/Yes, No/No, disagree) of (answer, answer) pairs."""
    return (
        sum(a == b == "Yes" for a, b in pairs),
        sum(a == b == "No" for a, b in pairs),
        sum(a != b for a, b in pairs),
    )


def alpha_from_counts(yy: int, nn: int, dis: int) -> float:
    """Krippendorff's alpha of two raters from the three counts; nan if only one answer is used.

    With n = yy + nn + dis records there are m = 2n answers: 2yy + dis Yes and 2nn + dis No. Observed disagreement is dis / n; expected disagreement is 2 Yes No / (m (m - 1)).
    """
    n = yy + nn + dis
    m = 2 * n
    yes, no = 2 * yy + dis, 2 * nn + dis
    d_e = 2 * yes * no / (m * (m - 1))
    return 1 - (dis / n) / d_e if d_e else float("nan")


def two_rater_numbers(yy: int, nn: int, dis: int) -> dict:
    """Every two-rater number from the three counts. Order-free: only the pooled answers count."""
    n = yy + nn + dis
    m = 2 * n
    yes, no = 2 * yy + dis, 2 * nn + dis
    po = (yy + nn) / n
    d_o = dis / n
    d_e = 2 * yes * no / (m * (m - 1))
    p_yes = yes / m
    pe_pi = p_yes**2 + (1 - p_yes) ** 2
    pe_ac1 = 2 * p_yes * (1 - p_yes)
    return {
        "N": n,
        "yes_yes": yy,
        "no_no": nn,
        "disagree": dis,
        "answers_total": m,
        "yes_answers": yes,
        "no_answers": no,
        "share_yes_answers": p_yes,
        "observed_agreement": po,
        "krippendorff": {
            "observed_disagreement_Do": d_o,
            "expected_disagreement_De": d_e,
            "chance_agreement_1_minus_De": 1 - d_e,
            "alpha": 1 - d_o / d_e,
        },
        "scott_pi": {"chance_agreement": pe_pi, "pi": (po - pe_pi) / (1 - pe_pi)},
        "gwet_ac1": {"chance_agreement": pe_ac1, "ac1": (po - pe_ac1) / (1 - pe_ac1)},
        "pabak": 2 * po - 1,
        "positive_agreement": 2 * yy / (2 * yy + dis) if yy + dis else None,
        "negative_agreement": 2 * nn / (2 * nn + dis) if nn + dis else None,
        "records_with_at_least_one_yes": yy + dis,
        "records_with_at_least_one_no": nn + dis,
    }


def krippendorff_alpha(units) -> dict:
    """Krippendorff's alpha (nominal) for units of any number of answers; None marks a missing one.

    A unit with fewer than two answers cannot be paired and is left out (Krippendorff 2011).
    """
    o: defaultdict[tuple, float] = defaultdict(float)  # the coincidence matrix
    for values in units:
        values = [v for v in values if v is not None]
        m = len(values)
        if m < 2:
            continue
        for i, j in itertools.permutations(range(m), 2):
            o[(values[i], values[j])] += 1 / (m - 1)
    n_c: defaultdict[str, float] = defaultdict(float)
    for (c, _k), w in o.items():
        n_c[c] += w
    n = sum(n_c.values())
    d_o = sum(w for (c, k), w in o.items() if c != k) / n
    d_e = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k) / (n * (n - 1))
    return {
        "alpha": 1 - d_o / d_e if d_e else None,
        "pairable_values": n,
        "value_counts": dict(n_c),
        "observed_disagreement": d_o,
        "expected_disagreement": d_e,
    }


def cohen_kappa(pairs) -> tuple[float, float]:
    """Cohen's kappa of (rater A, rater B) Yes/No pairs, and its chance agreement."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    pe = sum(pa[c] * pb[c] for c in ("Yes", "No")) / n**2
    return (po - pe) / (1 - pe), pe


def cluster_bootstrap(pairs_by_cluster: dict, replicates: int, seed: int) -> dict:
    """95% percentile intervals of observed agreement and alpha, resampling whole clusters.

    The resampling unit is the cluster: the keys of `pairs_by_cluster` (sorted), each holding its (answer, answer) pairs. Each replicate draws as many clusters as there are, with replacement, from numpy's `default_rng(seed)`, in one (replicates x clusters) draw; the interval is numpy's (linear) 2.5th and 97.5th percentile of the replicates.
    """
    import numpy as np

    names = sorted(pairs_by_cluster)
    counts = np.array(
        [pair_counts(pairs_by_cluster[c]) for c in names], dtype=float
    )  # clusters x (yy, nn, dis)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(names), size=(replicates, len(names)))
    total = counts[idx].sum(axis=1)
    yy, nn, dis = total[:, 0], total[:, 1], total[:, 2]
    n = yy + nn + dis
    m = 2 * n
    d_e = 2 * (2 * yy + dis) * (2 * nn + dis) / (m * (m - 1))
    alpha = 1 - (dis / n) / d_e
    po = (yy + nn) / n

    def interval(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {
        "clusters": len(names),
        "replicates": replicates,
        "seed": seed,
        "observed_agreement_95ci": interval(po),
        "alpha_95ci": interval(alpha),
    }


def paired_bootstrap(units: list, statistic, replicates: int, rng) -> list:
    """`statistic(sample)` for each of `replicates` resamples of `units` with replacement.

    The resampling unit is one element of `units` (for a paired comparison, one paper with both systems' counts). Each resample draws `len(units)` units one at a time with `rng.choice(units)`, in order, so a given `random.Random` gives the same draws as the paper's paired comparison with Opus; the caller passes the generator, which goes on from one call to the next.
    """
    return [statistic([rng.choice(units) for _ in units]) for _ in range(replicates)]


def index_interval(values: list, low: float = 0.025, high: float = 0.975) -> tuple:
    """(v[int(low n)], v[int(high n) - 1]) of the n sorted values: the 95% interval by index, as the paper's paired comparison with Opus takes it (no interpolation)."""
    v = sorted(values)
    return v[int(low * len(v))], v[int(high * len(v)) - 1]


def paired_permutation_p(a: dict, b: dict, units: list, statistic, swaps: int, rng):
    """Two-sided permutation p-values of each component of `statistic(a, b)`.

    Under "no difference", which system gave a unit's result is exchangeable: each swap keeps a unit's two values when `rng.random() < 0.5` and exchanges them otherwise, unit by unit in the order of `units`. A swap counts if a component's absolute value reaches the observed one (less 1e-12); p = (count + 1) / (swaps + 1).
    """
    observed = statistic(a, b)
    hits = [0] * len(observed)
    for _ in range(swaps):
        first, second = {}, {}
        for unit in units:
            x, y = (a[unit], b[unit]) if rng.random() < 0.5 else (b[unit], a[unit])
            first[unit], second[unit] = x, y
        value = statistic(first, second)
        for k in range(len(observed)):
            hits[k] += abs(value[k]) >= abs(observed[k]) - 1e-12
    return tuple((h + 1) / (swaps + 1) for h in hits)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided sign test on the units where the systems differ (ties are left out)."""
    k, n = min(wins, losses), wins + losses
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2**n)
