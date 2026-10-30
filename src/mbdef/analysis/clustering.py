"""The definition clusters of tutorial 02: English definitions of media bias and its subtypes, embedded with MiniLM and clustered with PCA and HDBSCAN (tutorials/README.md).

`run` selects the passages (`select`), screens them for English (`english_only`), embeds them, clusters them, and compares the clusters with a TF-IDF/SVD clustering of the same passages and with other HDBSCAN settings. `make_figure` draws the coverage of the ten largest clusters.
The cluster labels are aids for reading, not validated taxonomy entries. sentence-transformers, langdetect and wordfreq are in the `tutorials` extra.
"""

import csv
import hashlib
import importlib.metadata
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize as unit_length

from mbdef.analysis import SOURCES
from mbdef.analysis.plot_style import (
    FIGURE_WIDTH,
    FONT_FAMILY,
    FONT_SIZE,
    INK,
    SMALL_FONT_SCALE,
    similarity_cmap,
)
from mbdef.analysis.queries import find_concept, hierarchy

ROOT_TERM = "media bias"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def clean(text, citations):
    text = unicodedata.normalize("NFKC", text)
    for citation in sorted(set(citations), key=len, reverse=True):
        if len(citation) >= 5 and re.search(r"\d{4}", citation):
            text = text.replace("(" + citation + ")", " ").replace(citation, " ")
    return " ".join(text.split()).strip()


def select(data, root, depth, min_words):
    root_id = find_concept(data, root)
    edges, _ = hierarchy(data)
    children = defaultdict(set)
    for child, parent in edges:
        children[parent].add(child)
    depths, frontier = {root_id: 0}, {root_id}
    for level in range(1, depth + 1):
        frontier = {
            child for parent in frontier for child in children[parent]
        } - depths.keys()
        depths.update({cid: level for cid in frontier})
    labels = dict(data["concepts"].select("concept_id", "label").iter_rows())
    corpus = dict(data["instances"].select("mention_id", "corpus").iter_rows())
    documents, counters = {}, Counter()
    for definition in data["definitions"].iter_rows(named=True):
        cid = definition["concept_id"]
        if cid not in depths:
            continue
        counters["selected_passage_occurrences"] += 1
        text = clean(definition["text"], definition["citations"])
        if len(text.split()) < min_words:
            counters["short_passage_occurrences_excluded"] += 1
            continue
        key = text.casefold()
        entry = documents.setdefault(
            key, {"text": text, "terms": set(), "paper_ids": set(), "provenance": []}
        )
        term = labels[cid]
        entry["terms"].add(term)
        entry["paper_ids"].add(definition["paper_id"])
        entry["provenance"].append(
            {
                **definition,
                "canonical_term": term,
                "corpus": corpus[definition["mention_id"]],
            }
        )
    docs = []
    for key, entry in sorted(documents.items()):
        entry["document_id"] = hashlib.sha256(key.encode()).hexdigest()[:20]
        entry["terms"] = sorted(entry["terms"])
        entry["paper_ids"] = sorted(entry["paper_ids"])
        docs.append(entry)
    counters["unique_passages"] = len(docs)
    counters["duplicates_collapsed"] = (
        counters["selected_passage_occurrences"]
        - counters["short_passage_occurrences_excluded"]
        - len(docs)
    )
    return docs, depths, dict(counters)


def clustering(features, min_size, min_samples):
    model = HDBSCAN(
        min_cluster_size=min_size,
        min_samples=min_samples,
        cluster_selection_method="leaf",
        metric="euclidean",
        n_jobs=4,
        copy=True,
    )
    labels = model.fit_predict(features)
    # Stable presentation IDs, ordered largest cluster first.
    counts = Counter(int(c) for c in labels if c >= 0)
    order = sorted(counts, key=lambda c: (-counts[c], c))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping.get(int(c), -1) for c in labels]), model.probabilities_


def english_only(docs):
    """Heuristic English screening, with excluded material saved for audit.

    Full passages: seeded language detection, English probability >=0.80.
    Short term labels: all alphabetic tokens must occur in English word-frequency
    data (Zipf >=1); short labels are unreliable for standalone language detection.
    This conservative vocabulary rule can exclude uncommon English terminology.
    """
    from langdetect import DetectorFactory, LangDetectException, detect_langs
    from wordfreq import zipf_frequency

    DetectorFactory.seed = 42
    excluded, retained, partial_exclusions = [], [], []
    term_ok = {}
    for doc in docs:
        for term in doc["terms"]:
            tokens = re.findall(
                r"[a-z]+",
                unicodedata.normalize("NFKD", term)
                .encode("ascii", "ignore")
                .decode()
                .lower(),
            )
            term_ok[term] = bool(tokens) and all(
                zipf_frequency(token, "en") >= 1 for token in tokens
            )
        try:
            languages = detect_langs(doc["text"])
            probability = next(
                (float(item.prob) for item in languages if item.lang == "en"), 0.0
            )
            language = languages[0].lang
        except LangDetectException:
            language, probability = "unknown", 0.0
        allowed = {t for t in doc["terms"] if term_ok[t]}
        if probability < 0.80 or not allowed:
            excluded.append(
                {
                    **doc,
                    "detected_language": language,
                    "english_probability": probability,
                    "reason": "passage_language"
                    if probability < 0.80
                    else "term_vocabulary",
                }
            )
            continue
        occurrence = [r for r in doc["provenance"] if r["canonical_term"] in allowed]
        removed = [r for r in doc["provenance"] if r["canonical_term"] not in allowed]
        if removed:
            partial_exclusions.append(
                {
                    "document_id": doc["document_id"],
                    "removed_provenance": removed,
                    "reason": "term_vocabulary; passage retained through other English terms",
                }
            )
        retained.append(
            {
                **doc,
                "terms": sorted(allowed),
                "provenance": occurrence,
                "paper_ids": sorted({r["paper_id"] for r in occurrence}),
                "english_probability": probability,
            }
        )
    return (
        retained,
        excluded,
        sorted(t for t, ok in term_ok.items() if not ok),
        partial_exclusions,
    )


def metrics(features, labels):
    assigned = labels >= 0
    clusters = len(set(labels[assigned]))
    result = {
        "clusters": clusters,
        "assigned_fraction": float(assigned.mean()),
        "unassigned": int((~assigned).sum()),
    }
    result["silhouette_assigned"] = (
        float(
            silhouette_score(
                features[assigned],
                labels[assigned],
                sample_size=min(2000, int(assigned.sum())),
                random_state=42,
            )
        )
        if 1 < clusters < assigned.sum()
        else None
    )
    return result


def run(
    data,
    out,
    root=ROOT_TERM,
    depth=2,
    min_words=8,
    min_cluster_size=20,
    min_samples=5,
    model=MODEL,
):
    """Cluster the English definitions of ROOT and its subtypes down to DEPTH steps; write the report, the assignments, the selected passages, the language exclusions and an embedding cache to OUT.

    Passages shorter than MIN_WORDS words (after citations are removed) are left out; identical passages are clustered once.
    """
    args = SimpleNamespace(
        root=root,
        depth=depth,
        min_words=min_words,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        model=model,
    )
    if min(depth, min_words, min_cluster_size, min_samples) < 1:
        raise ValueError("Depth and filtering/clustering parameters must be positive.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    concept_labels = dict(data["concepts"].select("concept_id", "label").iter_rows())
    docs, depths, selection_counts = select(data, args.root, args.depth, args.min_words)
    docs, language_excluded, excluded_terms, partial_exclusions = english_only(docs)
    selection_counts["unique_passages_before_language_filter"] = selection_counts[
        "unique_passages"
    ]
    selection_counts["unique_passages"] = len(docs)
    selection_counts["language_or_term_exclusions"] = len(language_excluded)
    (out / "language_exclusions.json").write_text(
        json.dumps(
            {
                "excluded_terms": excluded_terms,
                "passages": language_excluded,
                "partial_provenance_exclusions": partial_exclusions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if len(docs) <= max(args.min_cluster_size, 3):
        raise ValueError("Too few eligible passages for clustering.")
    texts = [d["text"] for d in docs]
    print(
        f"Selected {len(docs):,} unique passages from {len(depths)} candidate terms.",
        flush=True,
    )
    key = hashlib.sha256((args.model + "\n" + "\n".join(texts)).encode()).hexdigest()
    cache = out / "sentence_embeddings.npz"
    if cache.exists() and str(np.load(cache)["key"]) == key:
        archive = np.load(cache)
        embeddings = archive["embeddings"]
        truncation = int(archive["truncated_passages"])
        revision = (
            str(archive["model_revision"])
            if "model_revision" in archive
            else "not recorded"
        )
        print("Loaded cached sentence embeddings.", flush=True)
    else:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        model = SentenceTransformer(
            args.model,
            revision=MODEL_REVISION if args.model == MODEL else None,
            device="cpu",
        )
        revision = str(getattr(model[0].auto_model.config, "_commit_hash", None))  # ty: ignore[unresolved-attribute]
        lengths = model.tokenizer(
            texts, truncation=False, add_special_tokens=True, return_length=True
        )["length"]
        truncation = sum(n > model.max_seq_length for n in lengths)
        embeddings = model.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        np.savez_compressed(
            cache,
            key=key,
            embeddings=embeddings,
            truncated_passages=truncation,
            model_revision=revision,
        )
    dimensions = min(50, len(docs) - 1, embeddings.shape[1])
    pca = PCA(n_components=dimensions, random_state=42)
    semantic = unit_length(pca.fit_transform(embeddings))
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=30000,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(
        n_components=min(dimensions, tfidf.shape[1] - 1), random_state=42
    )
    lexical = unit_length(svd.fit_transform(tfidf))
    labels, probabilities = clustering(
        semantic, args.min_cluster_size, args.min_samples
    )
    baseline, _ = clustering(lexical, args.min_cluster_size, args.min_samples)
    print("Clustering finished; running sensitivity checks.", flush=True)
    features = vectorizer.get_feature_names_out()
    summaries = []
    for c in sorted(set(labels) - {-1}):
        indices = np.flatnonzero(labels == c)
        mean = np.asarray(tfidf[indices].mean(axis=0)).ravel()
        keywords = [str(features[i]) for i in mean.argsort()[::-1][:8]]
        centroid = embeddings[indices].mean(axis=0)
        similarity = (
            embeddings[indices] @ centroid / max(np.linalg.norm(centroid), 1e-12)
        )
        representative = indices[np.argsort(similarity)[::-1][:5]]
        boundary = indices[np.argsort(probabilities[indices])[:3]]
        terms = Counter(t for i in indices for t in docs[i]["terms"])
        papers = {p for i in indices for p in docs[i]["paper_ids"]}
        summaries.append(
            {
                "cluster": int(c),
                "label": f"C{c + 1:02d}",
                "automatic_keywords": keywords,
                "passages": len(indices),
                "distinct_terms": len(terms),
                "distinct_papers": len(papers),
                "top_terms": terms.most_common(12),
                "representatives": [docs[i] for i in representative],
                "boundary_examples": [docs[i] for i in boundary],
            }
        )
    sensitivity = []
    for size, samples in sorted(
        {(15, 5), (20, 5), (30, 5), (20, 10), (args.min_cluster_size, args.min_samples)}
    ):
        alternative, _ = clustering(semantic, size, samples)
        shared = (labels >= 0) & (alternative >= 0)
        sensitivity.append(
            {
                "min_cluster_size": size,
                "min_samples": samples,
                **metrics(semantic, alternative),
                "ari_all_including_noise": float(
                    adjusted_rand_score(labels, alternative)
                ),
                "jointly_assigned_fraction": float(shared.mean()),
                "ari_jointly_assigned": float(
                    adjusted_rand_score(labels[shared], alternative[shared])
                )
                if shared.sum() > 1
                else None,
            }
        )
    jointly = (labels >= 0) & (baseline >= 0)
    report = {
        "sources": SOURCES,
        "selection": {
            "root": args.root,
            "depth": args.depth,
            "min_words": args.min_words,
            "candidate_concept_depths": depths,
            "candidate_labels": {cid: concept_labels[cid] for cid in depths},
            "language_filter": "langdetect English probability >=0.80; all term tokens English Zipf frequency >=1",
            "excluded_terms": excluded_terms,
            **selection_counts,
        },
        "model": args.model,
        "model_revision": revision,
        "embedding_dimensions": int(embeddings.shape[1]),
        "pca_dimensions": dimensions,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "truncated_passages": truncation,
        "clustering": {
            "method": "sklearn HDBSCAN",
            "selection": "leaf",
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "features": "PCA-50 sentence embeddings, then L2 normalization",
        },
        "semantic_metrics": metrics(semantic, labels),
        "tfidf_metrics": metrics(lexical, baseline),
        "baseline_comparison": {
            "ari_all_including_noise": float(adjusted_rand_score(labels, baseline)),
            "jointly_assigned_fraction": float(jointly.mean()),
            "ari_jointly_assigned": float(
                adjusted_rand_score(labels[jointly], baseline[jointly])
            )
            if jointly.sum() > 1
            else None,
        },
        "sensitivity": sensitivity,
        "clusters": summaries,
        "versions": {
            p: importlib.metadata.version(p)
            for p in [
                "sentence-transformers",
                "transformers",
                "torch",
                "scikit-learn",
                "numpy",
                "langdetect",
                "wordfreq",
            ]
        },
    }
    (out / "cluster_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "selected_definitions.json").write_text(
        json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "cluster_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "document_id",
                "text",
                "terms",
                "paper_ids",
                "semantic_cluster",
                "membership_strength",
                "tfidf_cluster",
            ]
        )
        writer.writerows(
            [
                d["document_id"],
                d["text"],
                json.dumps(d["terms"]),
                json.dumps(d["paper_ids"]),
                int(labels[i]),
                float(probabilities[i]),
                int(baseline[i]),
            ]
            for i, d in enumerate(docs)
        )
    print(f"Saved clustering data to {out}", flush=True)


def make_figure(data_dir, output_dir=None, font_size=FONT_SIZE):
    """Draw the coverage of the ten largest clusters from the files `run` wrote to DATA_DIR (passages, terms and papers per cluster); write it to OUTPUT_DIR (default DATA_DIR) as SVG and PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from matplotlib.ticker import MaxNLocator

    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir
    report = json.loads((data_dir / "cluster_report.json").read_text())
    with (data_dir / "cluster_assignments.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = np.array([int(r["semantic_cluster"]) for r in rows])
    summaries = report["clusters"]
    term_counts = defaultdict(Counter)
    for row in rows:
        cluster = int(row["semantic_cluster"])
        if cluster >= 0:
            # Each normalized term counts once per unique definition passage.
            term_counts[cluster].update(set(json.loads(row["terms"])))
    for summary in summaries:
        ranked = sorted(
            term_counts[summary["cluster"]].items(),
            key=lambda item: (-item[1], item[0]),
        )
        summary["display_terms"] = [[term, count] for term, count in ranked[:2]]

    if len(rows) != report["selection"]["unique_passages"]:
        raise ValueError(
            "Assignment table and cluster report disagree on passage count."
        )
    if set(labels) - {-1} != {s["cluster"] for s in summaries}:
        raise ValueError("Assignment table and cluster report disagree on cluster IDs.")
    for summary in summaries:
        if np.count_nonzero(labels == summary["cluster"]) != summary["passages"]:
            raise ValueError(f"Incorrect passage count for {summary['label']}")
    plt.rcParams.update({"svg.fonttype": "path", "font.family": FONT_FAMILY})
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH, 7.2),
        layout="constrained",
        gridspec_kw={"wspace": 0.18},
    )
    coverage = sorted(summaries, key=lambda s: (-s["passages"], s["cluster"]))[:10]
    blue_scale = similarity_cmap()
    row_colors = blue_scale(np.linspace(0.95, 0.30, len(coverage)))
    y = np.arange(len(coverage))
    names = [" / ".join(term for term, count in s["display_terms"]) for s in coverage]
    for ax, field, title in zip(
        axes,
        ["passages", "distinct_terms", "distinct_papers"],
        ["Unique Definition Passages", "Distinct Terms", "Distinct Source Papers"],
        strict=False,
    ):
        for i, summary in enumerate(coverage):
            value = summary[field]
            ax.barh(
                i, value, height=0.72, color=row_colors[i], edgecolor=INK, linewidth=0.4
            )
            rgb = np.asarray(row_colors[i][:3])
            linear = np.where(
                rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
            )
            luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
            # Pick white or dark text for contrast against each blue shade.
            color = (
                "white"
                if 1.05 / (luminance + 0.05) > (luminance + 0.05) / 0.0831
                else INK
            )
            text = ax.text(
                value / 2,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=font_size * SMALL_FONT_SCALE,
                color=color,
            )
            text.set_gid(f"coverage-count:{summary['label']}:{field}:{value}")
        ax.set_ylim(len(coverage) - 0.35, -0.65)
        ax.set_yticks(
            y,
            names if ax is axes[0] else [""] * len(names),
            fontsize=font_size * (6.25 / 6.5),
        )
        ax.set_xlabel(title, fontsize=font_size, labelpad=12)
        ax.tick_params(axis="x", labelsize=font_size * SMALL_FONT_SCALE)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.spines[["top", "right"]].set_visible(False)
    stem = output_dir / "definition-cluster"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    report["figure_encoding"] = {
        "labels": "Two most frequent defined terms per cluster, counting unique definition passages; ties resolved alphabetically.",
        "palette": "Sequential shades of shared blue",
        "cluster_colors": {
            summary["label"]: to_hex(color)
            for summary, color in zip(coverage, row_colors, strict=False)
        },
        "note": "Darker to lighter in descending definition-passage rank; each row has the same shade across all three coverage measures. No hatching.",
    }
    (data_dir / "cluster_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        f"Saved {stem.with_suffix('.svg')} ({len(rows):,} passages; {len(summaries)} clusters)."
    )
