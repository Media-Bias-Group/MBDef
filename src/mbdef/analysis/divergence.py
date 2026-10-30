"""The definitional divergence of tutorial 04: how similar the definitions of one term are across papers, by DefScore, on MBDef-Gold's relevant records (tutorials/README.md).

`definitions` lists the definitions of the relevant records of all three splits; `build_pairs` pairs them: same term across papers, same term within a paper, and a seeded sample of pairs of different terms from different papers (the last two are reference distributions, not negative labels). `run` scores the pairs with DefScore (`evaluate/metric.py`, its pinned models and calibration) and `figure` draws the cross-paper scores of the ten terms in the most papers.
DefScore similarity does not prove equivalence or contradiction.
"""

import hashlib
import importlib.metadata
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from mbdef.analysis import SOURCES
from mbdef.analysis.plot_style import (
    BAR_ALPHA,
    BLUE,
    EMPHASIS_FONT_SCALE,
    FIGURE_WIDTH,
    FONT_FAMILY,
    FONT_SIZE,
    INK,
    SMALL_FONT_SCALE,
)
from mbdef.evaluate import metric

BATCH = (
    512  # pairs scored between two checkpoint writes (the models batch them further)
)


def term_key(term):
    """A term as the pairs match it: case, spacing and outer punctuation set aside (no other merging)."""
    return " ".join(term.casefold().split()).strip(" .,;:\"'")


def definitions(gold: pl.DataFrame) -> list[dict]:
    """One row per non-empty definition of a relevant record, papers in their first order in GOLD."""
    papers = defaultdict(list)
    for record in gold.filter(pl.col("relevant")).iter_rows(named=True):
        papers[record["paper_id"]].append(record)
    rows = []
    for paper_id, records in papers.items():
        for record_index, record in enumerate(records):
            for item_index, item in enumerate(record["definitions"]):
                text = (item["text"] or "").strip()
                if text:
                    rows.append(
                        {
                            "paper_id": paper_id,
                            "term": record["term"],
                            "term_key": term_key(record["term"]),
                            "text": text,
                            "record_index": record_index,
                            "item_index": item_index,
                            "record_id": record["record_id"],
                            "split": record["split"],
                            "title": record["paper_title"],
                            "cited": bool(item["citations"]),
                        }
                    )
    return rows


def build_pairs(rows, baseline_size):
    """{group: [(i, j)]} of row indices: cross_paper and same_paper (same term key), different_term (a uniform sample of BASELINE_SIZE pairs, seed 0)."""
    by_term = defaultdict(list)
    for i, row in enumerate(rows):
        by_term[row["term_key"]].append(i)
    groups = {"cross_paper": [], "same_paper": [], "different_term": []}
    for members in by_term.values():
        for a, b in combinations(members, 2):
            group = (
                "same_paper"
                if rows[a]["paper_id"] == rows[b]["paper_id"]
                else "cross_paper"
            )
            groups[group].append((a, b))
    # Uniform reservoir sample of eligible unordered pairs, with a fixed seed.
    rng, eligible, sample = random.Random(0), 0, []
    for a, b in combinations(range(len(rows)), 2):
        if (
            rows[a]["term_key"] == rows[b]["term_key"]
            or rows[a]["paper_id"] == rows[b]["paper_id"]
        ):
            continue
        eligible += 1
        if len(sample) < baseline_size:
            sample.append((a, b))
        else:
            position = rng.randrange(eligible)
            if position < baseline_size:
                sample[position] = (a, b)
    groups["different_term"] = sorted(sample)
    return groups


def summary(values, terms=None):
    """Mean, median, quartiles and share below 0.5; with TERMS, a whole-term bootstrap interval of the mean and the term-balanced mean."""
    values = np.asarray(values)
    if not len(values):
        return {"n": 0}
    result = {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "share_below_0_5": float(np.mean(values < 0.5)),
    }
    if terms is not None:
        unique, inverse = np.unique(terms, return_inverse=True)
        sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse)
        draws = np.random.default_rng(0).integers(
            0, len(unique), size=(5000, len(unique))
        )
        means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        result["mean_ci95_term_bootstrap"] = np.quantile(means, [0.025, 0.975]).tolist()
        result["bootstrap_terms"] = len(unique)
        result["term_balanced_mean"] = float(np.mean(sums / counts))
    return result


def run(
    gold: pl.DataFrame, out, baseline_pairs=2000, device="auto", font_size=FONT_SIZE
):
    """Score every pair of GOLD's definitions and write report.json, pairs.jsonl and the figure to OUT.

    Scores are kept in OUT/score_cache.json as they are made, so an interrupted run resumes; the cache is used only for the same definitions, models and settings.
    """
    if font_size <= 0 or baseline_pairs < 1:
        raise ValueError("Font size and baseline pair count must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = definitions(gold)
    groups = build_pairs(rows, baseline_pairs)
    if not groups["cross_paper"]:
        raise ValueError("No same-term cross-paper definition pairs")
    specification = {
        "input_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        "bge_model": metric.BGE_MODEL,
        "bge_revision": metric.BGE_REVISION,
        "nli_model": metric.NLI_MODEL,
        "nli_revision": metric.NLI_REVISION,
        "slope": metric.DEFSCORE_SLOPE,
        "intercept": metric.DEFSCORE_INTERCEPT,
        "baseline_pairs": baseline_pairs,
        "pair_sampling": "reservoir-seed-0",
        "preprocessing": "casefold/whitespace/outer punctuation for ground-truth terms; definition text unchanged",
    }
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    checkpoint = out / "score_cache.json"
    cache = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    scored = cache.get("scores", {}) if cache.get("key") == key else {}
    all_pairs = [pair for members in groups.values() for pair in members]
    missing = [(a, b) for a, b in all_pairs if f"{a}:{b}" not in scored]
    print(
        f"{len(rows)} definitions; {len(all_pairs)} pairs; {len(missing)} to score",
        flush=True,
    )
    if missing:
        similarity, entailment = metric.models(device)
    for start in range(0, len(missing), BATCH):
        batch = missing[start : start + BATCH]
        values = metric.defscore_pair_scores(
            [(rows[a]["text"], rows[b]["text"]) for a, b in batch],
            similarity,
            entailment,
        )
        scored.update(
            {f"{a}:{b}": value for (a, b), value in zip(batch, values, strict=True)}
        )
        temp = checkpoint.with_suffix(".tmp")
        temp.write_text(json.dumps({"key": key, "scores": scored}))
        temp.replace(checkpoint)
        print(
            f"Scored {min(start + BATCH, len(missing))}/{len(missing)} missing pairs",
            flush=True,
        )
    pairs = []
    summaries = {}
    for group, members in groups.items():
        for a, b in members:
            pairs.append(
                {
                    "group": group,
                    "term_key": rows[a]["term_key"],
                    "a": rows[a],
                    "b": rows[b],
                    "defscore": scored[f"{a}:{b}"],
                }
            )
        summaries[group] = summary(
            [scored[f"{a}:{b}"] for a, b in members],
            [rows[a]["term_key"] for a, b in members]
            if group != "different_term"
            else None,
        )
    cross_paper = defaultdict(list)
    for pair in pairs:
        if pair["group"] == "cross_paper":
            cross_paper[pair["term_key"]].append(pair)
    terms = []
    for term, members in sorted(cross_paper.items()):
        papers = {p[side]["paper_id"] for p in members for side in ("a", "b")}
        if len(papers) >= 4:
            terms.append(
                {
                    "term": term,
                    "n_papers": len(papers),
                    **summary([p["defscore"] for p in members]),
                }
            )
    relevant = gold.filter(pl.col("relevant"))
    report = {
        "software_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "huggingface-hub",
                "numpy",
                "matplotlib",
            )
        },
        "method": specification,
        "definitions": len(rows),
        "papers": len({r["paper_id"] for r in rows}),
        "groups": summaries,
        "terms": terms,
        "sources": SOURCES,
        "selection": {
            "relevant_only": True,
            "splits": sorted(gold["split"].unique()),
            "records": dict(gold.group_by("split").len().sort("split").iter_rows()),
            "relevant_records": dict(
                relevant.group_by("split").len().sort("split").iter_rows()
            ),
        },
        "paper_metadata": {
            r["paper_id"]: {
                "title": r["paper_title"],
                "doi": r["paper_doi"],
                "publication_year": r["paper_year"],
            }
            for r in relevant.unique(
                "paper_id", keep="first", maintain_order=True
            ).iter_rows(named=True)
        },
        "limitations": [
            "Relevant MBDef-Gold records from all three splits; no additional language or length filtering.",
            "Whole-term bootstrap does not fully capture dependence from papers shared across terms.",
            "DefScore similarity is not proof of equivalence or contradiction; different-term pairs are not negative gold labels.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with (out / "pairs.jsonl").open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    figure(report, pairs, out, font_size)
    return report


def figure(report, pairs, out, font_size=FONT_SIZE):
    """Boxplots of the cross-paper DefScores of the ten terms in the most papers (at least four), as SVG and PNG, with their audit (boxplots.json)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    plt.rcParams.update(
        {"svg.fonttype": "path", "font.family": FONT_FAMILY, "font.size": font_size}
    )
    terms = sorted(report["terms"], key=lambda t: (-t["n_papers"], t["term"]))[:10]
    values = [
        [
            p["defscore"]
            for p in pairs
            if p["group"] == "cross_paper" and p["term_key"] == t["term"]
        ]
        for t in terms
    ]
    if any(len(v) != t["n"] for v, t in zip(values, terms, strict=True)):
        raise ValueError("Boxplot passage-pair counts disagree with report")
    fig, box_ax = plt.subplots(figsize=(FIGURE_WIDTH * 0.60, 7.2), layout="constrained")
    box_ax.boxplot(
        values,
        vert=False,
        patch_artist=True,
        widths=0.6,
        whis=1.5,
        showmeans=True,
        boxprops={"facecolor": BLUE, "edgecolor": BLUE, "alpha": BAR_ALPHA},
        medianprops={"color": INK, "linewidth": 2.2},
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": INK,
            "markersize": 8,
        },
        whiskerprops={"color": BLUE, "linewidth": 1.5},
        capprops={"color": BLUE, "linewidth": 1.5},
        flierprops={
            "marker": "o",
            "markersize": 10,
            "markerfacecolor": BLUE,
            "markeredgecolor": BLUE,
            "alpha": 0.4,
        },
    )
    box_ax.set_yticks(
        range(1, len(terms) + 1),
        [f"{t['term']} ({t['n_papers']})" for t in terms],
        fontsize=font_size,
    )
    box_ax.invert_yaxis()
    box_ax.set_xlim(-0.02, 1.02)
    box_ax.set_xlabel(
        "Cross-Paper DefScore Definition Similarity",
        fontsize=font_size * EMPHASIS_FONT_SCALE,
        labelpad=12,
    )
    box_ax.tick_params(axis="x", labelsize=font_size * SMALL_FONT_SCALE)
    box_ax.tick_params(axis="y", length=0)
    box_ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("svg", "png"):
        fig.savefig(
            out / f"definitional-divergence.{suffix}",
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(fig)
    audit = {
        "selection": "Ten terms represented in the most distinct papers; at least four papers; alphabetical tie-breaking.",
        "comparison": "Same-term cross-paper definition-pair calibrated DefScore scores.",
        "boxes": "Interquartile range; black median; white diamond mean; 1.5-IQR whiskers; dots are outliers.",
        "terms": terms,
    }
    (out / "boxplots.json").write_text(json.dumps(audit, indent=2))
