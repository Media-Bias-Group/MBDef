"""The taxonomy coverage of tutorial 06: how many MBDef papers define each bias type of two published taxonomies (Spinde et al. 2025, Rodrigo-Ginés et al. 2024), seed and closure papers apart (tutorials/README.md).

`concepts.csv` (tutorials/assets/taxonomy-coverage/) lists each concept, its level, whether each taxonomy lists it (`full`), covers it under another name (`half`) or not (`none`), and its search terms. A definition counts for a concept when a search term has the Label Match key (`postprocess.label_match`, the rule that made MBDef's merged terms) of its term, or of a synonym its own paper gives that term. Concepts flagged `needs_manual_check` count a definition only if its text names news, media, journalism, journalists or the press. A paper counts once per concept, however many of its definitions match. These are automatic matches, not a semantic review; `manual_exclusions.csv` (record_id, concept, reason; a blank concept excludes the definition everywhere) removes the ones a reader rejects.
"""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import median
from textwrap import fill
from typing import Literal

from mbdef.analysis import SOURCES
from mbdef.analysis.plot_style import (
    BAR_ALPHA,
    BLUE,
    EMPHASIS_FONT_SCALE,
    FIGURE_WIDTH,
    FONT_FAMILY,
    FONT_SIZE,
    GRAY,
    INK,
    SMALL_FONT_SCALE,
)
from mbdef.postprocess import label_match

TAXONOMIES = ("spinde", "rodrigo")
CITATION_KEYS = (
    "Spinde2025",
    "rodrigo2024systematic",
)  # the bibliography keys of the LaTeX overlay
CONTEXT = re.compile(r"\b(?:news|media|journalism|journalists?|press)\b", re.I)
EVIDENCE_FIELDS = (
    "concept",
    "record_id",
    "paper_id",
    "source",
    "term",
    "term_key",
    "synonyms",
    "matched_search_terms",
    "match_via",
    "status",
    "definition_text",
)
# Taxonomy papers whose presence in MBDef and MBDef-Gold `check_taxonomy_sources` reports.
TAXONOMY_SOURCES = [
    {
        "bibkey": "rodrigo2024systematic",
        "doi": "10.1016/j.eswa.2023.121641",
        "title": "A systematic review on media bias detection: What is media bias, how it is expressed, and how to detect it",
    },
    {
        "bibkey": "menzner2026table",
        "doi": "10.48550/arxiv.2601.05358",
        "title": "The Table of Media Bias Elements: A sentence-level taxonomy of media bias types and propaganda techniques",
    },
]


def keys(text, vocab):
    """The Label Match keys of TEXT: its word-set key and, if it has one, its spacing key."""
    word_set, spacing = label_match.label_keys(text, vocab)
    return {"w" + word_set} | ({"s" + spacing} if spacing else set())


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_concepts(path, vocab):
    """The rows of concepts.csv, checked; each gets `queries`: {search term: its keys}."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"group", "concept", *TAXONOMIES, "search_terms", "needs_manual_check"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} needs the columns {', '.join(sorted(required))}")
    if len({row["concept"] for row in rows}) != len(rows):
        raise ValueError(f"{path} lists a concept twice")
    for row in rows:
        for key in TAXONOMIES:
            if row["group"] == "Reference" and not row[key].strip():
                row[key] = "none"
        row.setdefault("half_note", "")
        if any(row[key] not in {"full", "half", "none"} for key in TAXONOMIES):
            raise ValueError("Taxonomy memberships must be full, half, or none")
        if row["needs_manual_check"].lower() not in {"true", "false", "yes", "no"}:
            raise ValueError("needs_manual_check must be yes/no or true/false")
        row["needs_manual_check"] = row["needs_manual_check"].lower() in {"true", "yes"}
        terms = sorted(
            {term.strip() for term in row["search_terms"].split(";") if term.strip()}
        )
        if not terms:
            raise ValueError(f"No search terms for {row['concept']}")
        row["queries"] = {term: keys(term, vocab) for term in terms}
    return rows


def read_exclusions(path, def_ids, concepts):
    """{(record_id, concept)} of the exclusions file; concept "" excludes the record for every concept."""
    if path is None or not Path(path).exists():
        return set()
    valid_concepts = {row["concept"] for row in concepts}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "record_id" not in reader.fieldnames:
            raise ValueError(f"{path} needs a record_id column")
        rows = list(reader)
    exclusions = set()
    for row in rows:
        record_id, concept = (
            row["record_id"].strip(),
            (row.get("concept") or "").strip(),
        )
        if record_id not in def_ids or (concept and concept not in valid_concepts):
            raise ValueError(
                f"Unknown exclusion record or concept: {record_id}, {concept}"
            )
        exclusions.add((record_id, concept))
    return exclusions


def audit(concepts, data, exclusions, vocab):
    """Match every MBDef definition (DATA: `analysis.load()`) against the concepts' search terms.

    Returns (one result row per concept, ordered by group then papers; one evidence row per matched definition and concept; one match-log row per search term)."""
    query_index = defaultdict(set)
    for index, concept in enumerate(concepts):
        for query, query_keys in concept["queries"].items():
            for key in query_keys:
                query_index[key].add((index, query))
    instances = {
        row["mention_id"]: row
        for row in data["instances"]
        .select("mention_id", "word_set_key", "spacing_key", "synonyms")
        .iter_rows(named=True)
    }
    corpus = dict(data["papers"].select("paper_id", "corpus").iter_rows())
    synonym_keys = {}
    support = [defaultdict(set) for _ in concepts]
    evidence, log = [], defaultdict(list)
    for record in (
        data["definitions"]
        .select("def_id", "mention_id", "paper_id", "term", "text")
        .iter_rows(named=True)
    ):
        mention = record["mention_id"]
        instance = instances[mention]
        synonyms = sorted(
            {
                item["text"].strip()
                for item in instance["synonyms"]
                if item["text"].strip()
            }
        )
        if mention not in synonym_keys:
            synonym_keys[mention] = set().union(
                *(keys(text, vocab) for text in synonyms)
            )
        term_keys = {"w" + instance["word_set_key"]} | (
            {"s" + instance["spacing_key"]} if instance["spacing_key"] else set()
        )
        hits = defaultdict(dict)
        # A term match overrides a synonym match of the same search term.
        for via, found in (("synonym", synonym_keys[mention]), ("term", term_keys)):
            for key in found:
                for index, query in query_index.get(key, ()):
                    hits[index][query] = via
        source = "seed" if corpus[record["paper_id"]] == "seeds" else "closure"
        for index, queries in hits.items():
            concept = concepts[index]
            excluded = (record["def_id"], "") in exclusions or (
                record["def_id"],
                concept["concept"],
            ) in exclusions
            context_pass = not concept["needs_manual_check"] or bool(
                CONTEXT.search(record["text"])
            )
            status = (
                "manual_exclusion"
                if excluded
                else ("included" if context_pass else "filtered_context")
            )
            row = dict(
                zip(
                    EVIDENCE_FIELDS,
                    [
                        concept["concept"],
                        record["def_id"],
                        record["paper_id"],
                        source,
                        record["term"],
                        instance["word_set_key"],
                        "; ".join(synonyms),
                        "; ".join(sorted(queries)),
                        "; ".join(sorted(set(queries.values()))),
                        status,
                        record["text"],
                    ],
                    strict=True,
                )
            )
            evidence.append(row)
            if status == "included":
                support[index][source].add(record["paper_id"])
            for query in queries:
                log[index, query].append(row)
    results, matches = [], []
    for index, concept in enumerate(concepts):
        seed, closure = len(support[index]["seed"]), len(support[index]["closure"])
        results.append(
            {key: value for key, value in concept.items() if key != "queries"}
            | {
                "seed": seed,
                "closure": closure,
                "total": seed + closure,
                "listed_by": sum(concept[key] != "none" for key in TAXONOMIES),
                "full_by": sum(concept[key] == "full" for key in TAXONOMIES),
            }
        )
        for query in concept["queries"]:
            rows = log[index, query]
            included = [row for row in rows if row["status"] == "included"]
            matches.append(
                {
                    "concept": concept["concept"],
                    "search_term": query,
                    "candidate_records": len(rows),
                    "included_records": len(included),
                    "filtered_context": sum(
                        row["status"] == "filtered_context" for row in rows
                    ),
                    "manual_exclusions": sum(
                        row["status"] == "manual_exclusion" for row in rows
                    ),
                    "distinct_papers": len({row["paper_id"] for row in included}),
                    "seed": len(
                        {row["paper_id"] for row in included if row["source"] == "seed"}
                    ),
                    "closure": len(
                        {
                            row["paper_id"]
                            for row in included
                            if row["source"] == "closure"
                        }
                    ),
                }
            )
    ordered = []
    for group in dict.fromkeys(row["group"] for row in results):
        members = [row for row in results if row["group"] == group]
        if group != "Reference":
            members.sort(key=lambda row: -row["total"])
        ordered.extend(members)
    return ordered, evidence, matches


def select_display_rows(rows):
    """The concepts both taxonomies cover (half counts as covered)."""
    return [
        row for row in rows if row["group"] != "Reference" and row["listed_by"] >= 2
    ]


def inclusion_exports(out, results, evidence, papers):
    """included_papers.csv (one row per counted concept and paper) and counting_diagnostics.csv; fails if the evidence does not add up to the counts."""
    by_concept, grouped = defaultdict(list), defaultdict(list)
    for row in evidence:
        by_concept[row["concept"]].append(row)
        if row["status"] == "included":
            grouped[row["concept"], row["paper_id"]].append(row)
    included = []
    for (concept, pid), matches in sorted(grouped.items()):
        if len({row["source"] for row in matches}) != 1:
            raise ValueError("One paper has conflicting source assignments")
        included.append(
            {
                "concept": concept,
                "paper_id": pid,
                "source": matches[0]["source"],
                "title": papers[pid]["title"],
                "doi": papers[pid]["doi"],
                "accepted_records": len(matches),
                "record_ids": "; ".join(sorted({row["record_id"] for row in matches})),
                "match_via": "term"
                if any("term" in row["match_via"].split("; ") for row in matches)
                else "synonym_only",
                "matched_search_terms": "; ".join(
                    sorted(
                        {
                            query
                            for row in matches
                            for query in row["matched_search_terms"].split("; ")
                        }
                    )
                ),
            }
        )
    write_csv(
        out / "included_papers.csv",
        included,
        list(included[0]) if included else ["concept"],
    )
    diagnostics = []
    for result in results:
        accepted = [row for row in included if row["concept"] == result["concept"]]
        split = Counter(row["source"] for row in accepted)
        if (split["seed"], split["closure"]) != (result["seed"], result["closure"]):
            raise ValueError("Evidence paper counts disagree with reported counts")
        candidates = by_concept[result["concept"]]
        diagnostics.append(
            {
                "concept": result["concept"],
                "included_papers": len(accepted),
                "term_matched_papers": sum(
                    row["match_via"] == "term" for row in accepted
                ),
                "synonym_only_papers": sum(
                    row["match_via"] == "synonym_only" for row in accepted
                ),
                "candidate_papers": len({row["paper_id"] for row in candidates}),
                "context_filtered_records": sum(
                    row["status"] == "filtered_context" for row in candidates
                ),
                "manually_excluded_records": sum(
                    row["status"] == "manual_exclusion" for row in candidates
                ),
                "context_required": result["needs_manual_check"],
            }
        )
    write_csv(out / "counting_diagnostics.csv", diagnostics, list(diagnostics[0]))


def _same_source(doi, title, target):
    def plain_doi(value):
        return re.sub(
            r"^https?://(?:dx\.)?doi\.org/", "", (value or "").strip().casefold()
        )

    def plain_title(value):
        return re.sub(r"[^a-z0-9]+", "", (value or "").casefold())

    return plain_doi(doi) == plain_doi(target["doi"]) or plain_title(
        title
    ) == plain_title(target["title"])


def check_taxonomy_sources(papers, gold, out):
    """Whether the taxonomy papers (TAXONOMY_SOURCES) are in MBDef's papers or in MBDef-Gold (GOLD: `analysis.load_gold()`); taxonomy_source_membership.csv/.json and their MBDef-Gold definitions. Nothing is added to the counts."""
    gold_rows = gold.to_dicts()
    summary, definitions = [], []
    for target in TAXONOMY_SOURCES:
        corpus = [
            row
            for row in papers.values()
            if _same_source(row["doi"], row["title"], target)
        ]
        annotations = [
            row
            for row in gold_rows
            if _same_source(row["paper_doi"], row["paper_title"], target)
        ]
        summary.append(
            target
            | {
                "main_corpus_papers": len(corpus),
                "main_corpus_paper_ids": "; ".join(
                    sorted(row["paper_id"] for row in corpus)
                ),
                "ground_truth_splits": "; ".join(
                    sorted({row["split"] for row in annotations})
                ),
                "ground_truth_records": len(annotations),
                "ground_truth_relevant_records": sum(
                    row["relevant"] for row in annotations
                ),
                "ground_truth_paper_ids": "; ".join(
                    sorted({row["paper_id"] for row in annotations})
                ),
            }
        )
        definitions += [
            {
                "bibkey": target["bibkey"],
                "split": row["split"],
                "record_id": row["record_id"],
                "paper_id": row["paper_id"],
                "term": row["term"],
                "annotation_relevant": row["relevant"],
                "definition_text": definition["text"],
            }
            for row in annotations
            for definition in row["definitions"]
        ]
    write_csv(out / "taxonomy_source_membership.csv", summary, list(summary[0]))
    write_csv(
        out / "taxonomy_source_ground_truth_definitions.csv",
        definitions,
        [
            "bibkey",
            "split",
            "record_id",
            "paper_id",
            "term",
            "annotation_relevant",
            "definition_text",
        ],
    )
    report = {
        "sources": summary,
        "main_corpus_size": len(papers),
        "method": "Normalized DOI or normalized full-title equality; MBDef's paper table and all three MBDef-Gold splits checked separately. No corpus records added.",
    }
    (out / "taxonomy_source_membership.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


def run(data, gold, out, concepts, exclusions=None, citation_keys=CITATION_KEYS):
    """Count the concepts of the CONCEPTS file in MBDef (DATA: `analysis.load()`; GOLD: `analysis.load_gold()`) and write the counts, their evidence, the figure and its LaTeX wrapper into OUT; returns report.json's content.

    EXCLUSIONS defaults to manual_exclusions.csv next to CONCEPTS, if it exists."""
    out, concepts = Path(out), Path(concepts)
    # shortcut: MBDef's keys came from the vocabulary of all corpus mentions, which it does not ship; its own terms give the same keys for all but 93 of 62,066, so search terms and synonyms get the released keys up to rare plurals. Upgrade if MBDef ships its vocabulary.
    vocab = label_match.vocabulary(data["instances"]["term"].to_list())
    concept_rows = read_concepts(concepts, vocab)
    exclusions_path = (
        Path(exclusions) if exclusions else concepts.with_name("manual_exclusions.csv")
    )
    if exclusions is not None and not exclusions_path.exists():
        raise FileNotFoundError(f"Exclusions file not found: {exclusions_path}")
    excluded = read_exclusions(
        exclusions_path, set(data["definitions"]["def_id"]), concept_rows
    )
    rows, evidence, match_log = audit(concept_rows, data, excluded, vocab)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "coverage_counts.csv", rows, list(rows[0]))
    write_csv(out / "match_log.csv", match_log, list(match_log[0]))
    write_csv(out / "matched_records.csv", evidence, EVIDENCE_FIELDS)
    # Remove review files of concepts no longer flagged in concepts.csv.
    for path in out.glob("manual_check_*.csv"):
        path.unlink()
    for concept in concept_rows:
        if concept["needs_manual_check"]:
            slug = re.sub(r"[^a-z0-9]+", "_", concept["concept"].lower()).strip("_")
            write_csv(
                out / f"manual_check_{slug}.csv",
                [row for row in evidence if row["concept"] == concept["concept"]],
                EVIDENCE_FIELDS,
            )
    displayed = select_display_rows(rows)
    summary = [
        f"Displayed taxonomy concepts: {len(displayed)}",
        f"All taxonomy concepts matched: {sum(row['group'] != 'Reference' for row in rows)}",
        f"Concepts with >=1 distinct paper: {sum(row['total'] >= 1 for row in displayed)}",
        f"Concepts with >=5 distinct papers: {sum(row['total'] >= 5 for row in displayed)}",
        "",
        "Median papers by taxonomy membership (full or half counts as listed):",
    ]
    for count in range(len(TAXONOMIES), 0, -1):
        totals = [row["total"] for row in displayed if row["listed_by"] == count]
        summary.append(
            f"  Listed by {count}: {median(totals) if totals else 'n/a'} (n={len(totals)})"
        )
    summary += [
        "",
        "Counts use automatic Label Match term/synonym matching; flagged records await human review.",
        f"Manual exclusions supplied: {len(excluded)}",
        "Records filtered for context must contain news, media, journalism, journalist(s), or press in the definition text.",
    ]
    (out / "coverage_summary.txt").write_text("\n".join(summary) + "\n")
    configuration = [concepts, *([exclusions_path] if exclusions_path.exists() else [])]
    report = {
        "sources": SOURCES,
        "configuration_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in configuration
        },
        "rows": rows,
        "displayed_concepts": [row["concept"] for row in displayed],
        "displayed_taxonomies": list(TAXONOMIES),
        "half_meaning": "Covered under a different name",
        "context_regex": CONTEXT.pattern,
        "matching": f"Label Match keys ({label_match.RULE_VERSION}) of the search terms against each definition's term and its paper's synonyms of that term",
        "counting_unit": "Distinct paper_id per taxonomy concept, split by corpus.",
        "review_status": "Automatic matches; manual review pending. Taxonomy memberships supplied by figure specification.",
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    caption = (
        "Coverage of bias types covered by at least two existing taxonomies in MBDef. "
        "Dots indicate coverage by Spinde and Rodrigo-Ginés (half: covered under a different name); "
        "Concepts follow the supplied two-taxonomy comparison table. "
        "Bars count distinct papers in the released main corpus, excluding the separate ground-truth tables, "
        "with at least one definition record matching a specified normalized "
        "term or a synonym attached to that term-in-paper instance, after configured context screening "
        "and manual exclusions. Grey and blue identify seed and closure papers. "
        "Labels show total (seed, closure) paper counts. Bar heights use a linear scale, "
        "so grey and blue segment heights represent their source counts and proportions. "
        "Gatekeeping and framing candidates are screened for media-context keywords; "
        "automatic matches remain subject to semantic review."
    )
    (out / "caption.tex").write_text("\\caption{" + caption + "}\n")
    papers = {
        row["paper_id"]: row
        for row in data["papers"]
        .select("paper_id", "title", "doi")
        .iter_rows(named=True)
    }
    check_taxonomy_sources(papers, gold, out)
    inclusion_exports(out, rows, evidence, papers)
    make_figure(displayed, out, citation_keys)
    return report


def make_figure(rows, out, citation_keys=CITATION_KEYS):
    """Bars of papers per concept (seed grey, closure blue) over the taxonomies' membership dots: taxonomy_coverage.png, _preview.svg, the LaTeX SVG and its wrapper in OUT."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator

    # Compensate for the wider canvas so manuscript labels match the shared style.
    font_size = FONT_SIZE * 1.4375 * 1.15
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": font_size,
            "svg.fonttype": "path",
            "axes.linewidth": 0.4,
        }
    )
    positions, groups, previous, x = [], [], None, 0.0
    for row in rows:
        if row["group"] != previous:
            if previous is not None:
                x += 0.65
            groups.append([row["group"], x, x])
            previous = row["group"]
        positions.append(x)
        groups[-1][2] = x
        x += 1
    # Preserve 7-point type while accommodating rotated concept labels.
    fig = plt.figure(figsize=(FIGURE_WIDTH * 1.4375, 16), facecolor="white")
    bar_ax = fig.add_axes((0.10, 0.63, 0.88, 0.25))
    dot_ax = fig.add_axes((0.10, 0.42, 0.88, 0.17), sharex=bar_ax)
    bar_ax.set_xlim(-0.7, positions[-1] + 0.7)
    maximum = max(row["total"] for row in rows)
    # A linear scale is required for segment heights to represent source proportions.
    bar_ax.set_yscale("linear")
    bar_ax.set_ylim(0, max(5, maximum * 1.9))
    for ax in (bar_ax, dot_ax):
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    bar_ax.yaxis.set_major_locator(NullLocator())
    bar_ax.yaxis.set_minor_locator(NullLocator())
    bar_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    dot_ax.set_ylim(1.6, -0.6)
    # The paper's numeric citations of Spinde [16] and Rodrigo-Ginés [15]; the LaTeX wrapper replaces them.
    dot_ax.set_yticks([0, 1], ["[16]", "[15]"], fontsize=FONT_SIZE * 1.4375 * (8 / 6.5))
    dot_ax.tick_params(axis="y", pad=8)
    for key, tick in zip(TAXONOMIES, dot_ax.yaxis.get_major_ticks(), strict=True):
        tick.label1.set_gid("taxonomy-citation-" + key)
    dot_ax.set_xticks(
        positions,
        [fill(row["concept"], 22) for row in rows],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=font_size * SMALL_FONT_SCALE,
        linespacing=0.9,
    )
    dot_ax.tick_params(axis="x", pad=8)
    for group, start, end in groups:
        bar_ax.plot(
            [start - 0.35, end + 0.35],
            [1.04, 1.04],
            transform=bar_ax.get_xaxis_transform(),
            color=GRAY,
            linewidth=1.5,
            clip_on=False,
        )
        bar_ax.text(
            (start + end) / 2,
            1.08,
            group,
            ha="center",
            va="bottom",
            transform=bar_ax.get_xaxis_transform(),
            weight="bold",
            fontsize=font_size * EMPHASIS_FONT_SCALE,
        )
    for row in rows:
        if row["total"] != row["seed"] + row["closure"]:
            raise ValueError("Total disagrees with source split")
    seed_bars = bar_ax.bar(
        positions, [row["seed"] for row in rows], width=0.62, color=GRAY
    )
    closure_bars = bar_ax.bar(
        positions,
        [row["closure"] for row in rows],
        bottom=[row["seed"] for row in rows],
        width=0.62,
        color=BLUE,
        alpha=BAR_ALPHA,
    )
    for row, pos in zip(rows, positions, strict=True):
        for y, key in enumerate(TAXONOMIES):
            dot_ax.plot(
                pos,
                y,
                marker="o",
                markersize=32,
                color=INK,
                markeredgewidth=2.5,
                fillstyle={"full": "full", "half": "left", "none": "none"}[row[key]],
                markerfacecolor=INK,
                markerfacecoloralt="white",
                linestyle="none",
            )
        seed, closure, total = row["seed"], row["closure"], row["total"]
        bar_ax.annotate(
            f"{total:,}\n({seed:,}, {closure:,})",
            (pos, total),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=font_size * SMALL_FONT_SCALE,
            color=INK if total else "#C62828",
            linespacing=1.0,
        )

    def dot(fillstyle: Literal["full", "left"], label):
        return Line2D(
            [],
            [],
            marker="o",
            markersize=32,
            color=INK,
            linestyle="none",
            fillstyle=fillstyle,
            markerfacecoloralt="white",
            markeredgewidth=2.5,
            label=label,
        )

    handles = [
        Patch(facecolor=BLUE, alpha=BAR_ALPHA, label="Closure papers"),
        dot("full", "Listed"),
        dot("left", "Covered under a different name"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.005),
        ncol=3,
        frameon=False,
        fontsize=font_size * SMALL_FONT_SCALE,
        handlelength=1,
        columnspacing=1.5,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # ty: ignore[unresolved-attribute]
    verify_bar_geometry(bar_ax, rows, seed_bars, closure_bars, out)
    for label in dot_ax.get_xticklabels():
        bounds = label.get_window_extent(renderer)
        if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.width:
            raise ValueError(f"Concept label exceeds figure: {label.get_text()}")
    # Crop from the real, short numeric preview labels, never from raw TeX commands.
    crop = fig.get_tightbbox(renderer).padded(0.05)
    fig.savefig(out / "taxonomy_coverage_preview.svg", dpi=300, bbox_inches=crop)
    fig.savefig(out / "taxonomy_coverage.png", dpi=300, bbox_inches=crop)
    write_latex_variant(out, fig, dot_ax, citation_keys, crop.x0)
    plt.close(fig)


def verify_bar_geometry(ax, rows, seed_bars, closure_bars, out):
    """Check the drawn bars' values, colours and on-screen seed/closure proportions; taxonomy_coverage_bar_geometry.csv."""
    from matplotlib.colors import to_rgba

    def pixel_height(patch):
        return abs(
            ax.transData.transform((0, patch.get_y() + patch.get_height()))[1]
            - ax.transData.transform((0, patch.get_y()))[1]
        )

    checks = []
    for row, gray, blue in zip(rows, seed_bars, closure_bars, strict=True):
        if (gray.get_y(), gray.get_height(), blue.get_y(), blue.get_height()) != (
            0,
            row["seed"],
            row["seed"],
            row["closure"],
        ):
            raise ValueError("Plotted bar geometry disagrees with source counts")
        if gray.get_facecolor() != to_rgba(GRAY) or blue.get_facecolor() != to_rgba(
            BLUE, BAR_ALPHA
        ):
            raise ValueError("Bar colours disagree with seed/closure legend")
        gray_pixels, blue_pixels = pixel_height(gray), pixel_height(blue)
        total_pixels = gray_pixels + blue_pixels
        observed = blue_pixels / total_pixels if total_pixels else 0
        expected = row["closure"] / row["total"] if row["total"] else 0
        if abs(observed - expected) > 1e-10:
            raise ValueError("Rendered source proportions disagree with source counts")
        checks.append(
            {
                "concept": row["concept"],
                "total": row["total"],
                "seed": row["seed"],
                "closure": row["closure"],
                "gray_height_pixels": gray_pixels,
                "blue_height_pixels": blue_pixels,
                "blue_fraction": observed,
                "expected_closure_fraction": expected,
                "scale": ax.get_yscale(),
            }
        )
    write_csv(out / "taxonomy_coverage_bar_geometry.csv", checks, list(checks[0]))


def write_latex_variant(out, fig, dot_ax, citation_keys, crop_left_inches):
    """taxonomy_coverage.svg without the citation labels, and taxonomy_coverage_include.tex, which sets them as linked \\cite at 8 pt over the scaled SVG."""
    import xml.etree.ElementTree as ET

    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(out / "taxonomy_coverage_preview.svg")
    root = tree.getroot()
    width, height = map(float, root.get("viewBox", "").split()[2:])
    overlays = []
    for taxonomy, bibkey, label in zip(
        TAXONOMIES, citation_keys, dot_ax.get_yticklabels(), strict=True
    ):
        if not re.fullmatch(r"[A-Za-z0-9_:./+\-]+", bibkey):
            raise ValueError(f"Invalid bibliography key: {bibkey}")
        group = next(
            element
            for element in tree.iter()
            if element.get("id") == "taxonomy-citation-" + taxonomy
        )
        root_defs = root.find("{" + namespace + "}defs")
        inner = group.find("{" + namespace + "}g")
        match = inner is not None and re.search(
            r"translate\([^ ]+ ([^)]+)\)", inner.get("transform", "")
        )
        if root_defs is None or not match:
            raise ValueError(
                f"The preview SVG has no drawn citation label of {taxonomy}"
            )
        for definitions in group.findall(".//{" + namespace + "}defs"):
            for glyph in definitions:
                root_defs.append(deepcopy(glyph))
        baseline = float(match.group(1))
        for child in list(group):
            group.remove(child)
        x = (
            label.get_transform().transform(label.get_position())[0] * 72 / fig.dpi
            - crop_left_inches * 72
        )
        overlays.append((x / width, (height - baseline) / width, bibkey))
    # No raw TeX in the SVG: it would distort Inkscape's drawing-area bounds.
    tree.write(out / "taxonomy_coverage.svg", encoding="utf-8", xml_declaration=True)
    lines = [
        r"% Preamble: \usepackage{svg} and \usepackage{hyperref}",
        r"% Use this wrapper instead of a bare \includesvg command.",
        r"% Inkscape and shell escape are required for SVG conversion.",
        r"\providecommand{\MBDefTaxonomyGraphic}[1]{%",
        r"  \includesvg[width=#1,inkscapelatex=false,inkscapearea=page]{taxonomy_coverage}}",
        r"\begin{figure}[t]",
        r"  \centering",
        r"  \begingroup",
        r"  \setlength{\unitlength}{\linewidth}",
        r"  \begin{picture}(1," + f"{height / width:.8f}" + ")",
        r"    \put(0,0){\MBDefTaxonomyGraphic{\linewidth}}",
    ]
    for x, y, key in overlays:
        lines.append(
            r"    \put("
            + f"{x:.8f},{y:.8f}"
            + r"){\makebox(0,0)[r]{"
            + r"\fontsize{8pt}{9.6pt}\selectfont\cite{"
            + key
            + "}}}"
        )
    lines += [
        r"  \end{picture}",
        r"  \endgroup",
        r"  % Insert caption.tex content here if desired.",
        r"  \label{fig:taxonomy-coverage}",
        r"\end{figure}",
        "",
    ]
    (out / "taxonomy_coverage_include.tex").write_text("\n".join(lines))
