"""The research disciplines of tutorial 05: which fields the MBDef papers come from, by OpenAlex's primary topic (tutorials/README.md).

MBDef's papers carry their OpenAlex id and DOI but no discipline. `enrich` asks OpenAlex for each paper's topics (by id, then by DOI) and keeps every answer in a JSON cache, so a run that stops resumes and a rerun with `offline=True` reads only the cache. `summarize` counts papers per domain, field, subfield and topic, by primary topic and by all topics; a paper OpenAlex does not classify stays `unclassified`, never inferred from its title. OpenAlex reclassifies works over time, so keep the cache with any published numbers.
"""

import csv
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from textwrap import fill
from urllib.parse import quote

from mbdef import files, postprocess
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
from mbdef.collect import openalex
from mbdef.collect.identity import normalize_doi

LEVELS = ("domain", "field", "subfield", "topic")
SELECT = "id,doi,display_name,primary_topic,topics"
BATCH = 50  # OpenAlex ids per request
INSET_SUBFIELDS = 6


def work_id(value):
    """ "W123" from an OpenAlex work URL or id, else None."""
    value = (value or "").rstrip("/").rsplit("/", 1)[-1].upper()
    return value if re.fullmatch(r"W\d+", value) else None


def doi(value):
    return normalize_doi(value) or None


def _get(url, params, api_key):
    return openalex.get_json(
        url,
        params=params,
        api_key=api_key,
        timeout_seconds=60,
        sleep_seconds=openalex.SLEEP_SECONDS,
        max_retries=openalex.MAX_RETRIES,
    )[1]


def _save(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    files.write_atomic({path: files.json_text(cache)})


def enrich(papers, cache_path):
    """Look up in OpenAlex every paper of PAPERS the cache at CACHE_PATH lacks: by OpenAlex id in batches, then by DOI for those without a found work. Saves the cache after every request; returns it."""
    files.load_env(postprocess.REPO / files.ENV_FILE)
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    cache = read_cache(cache_path)
    entries = cache["lookups"]
    url = f"{openalex.OPENALEX_URL}/works"

    def store(kind, identifiers, found):
        stamp = datetime.now(UTC).isoformat()
        for identifier in identifiers:
            entries[f"{kind}:{identifier}"] = {
                "retrieved_at": stamp,
                "work": found.get(identifier),
            }
        _save(cache_path, cache)

    ids = sorted(
        {w for p in papers if (w := work_id(p["openalex_id"]))}
        - _cached(entries, "openalex")
    )
    for start in range(0, len(ids), BATCH):
        batch = ids[start : start + BATCH]
        payload = _get(
            url,
            {
                "filter": "openalex:" + "|".join(batch),
                "per-page": 100,
                "select": SELECT,
            },
            api_key,
        )
        store(
            "openalex", batch, {work_id(w["id"]): w for w in payload.get("results", [])}
        )
        print(f"openalex: {min(start + BATCH, len(ids))}/{len(ids)} ids", flush=True)
    dois = sorted(
        {
            d
            for p in papers
            if (d := doi(p["doi"]))
            and not entries.get(f"openalex:{work_id(p['openalex_id'])}", {}).get("work")
        }
        - _cached(entries, "doi")
    )
    for number, identifier in enumerate(dois, 1):
        # One DOI per request: some released DOIs hold commas, which a filter would split.
        work = _get(
            f"{url}/doi:{quote(identifier, safe='')}", {"select": SELECT}, api_key
        )
        store("doi", [identifier], {identifier: work or None})
        if number % BATCH == 0 or number == len(dois):
            print(f"doi: {number}/{len(dois)} DOIs", flush=True)
    return cache


def _cached(entries, kind):
    return {key.split(":", 1)[1] for key in entries if key.startswith(kind + ":")}


def read_cache(path):
    path = Path(path)
    cache = (
        json.loads(path.read_text())
        if path.is_file()
        else {"version": 1, "lookups": {}}
    )
    if cache.get("version") != 1 or not isinstance(cache.get("lookups"), dict):
        raise ValueError(f"{path} is not an OpenAlex cache of this module (version 1)")
    return cache


def summarize(papers, cache):
    """(one row per paper, counts): the counts are {"primary"|"all_topics": {level: [{id, name, papers, by_corpus, ...}]}}, largest first."""
    lookups = cache["lookups"]
    rows = []
    counts = {
        mode: {level: {} for level in LEVELS} for mode in ("primary", "all_topics")
    }
    for paper in papers:
        candidates = [
            ("openalex", work_id(paper["openalex_id"])),
            ("doi", doi(paper["doi"])),
        ]
        work, entry, method = None, {}, None
        for kind, identifier in candidates:
            candidate = lookups.get(f"{kind}:{identifier}", {}) if identifier else {}
            if candidate.get("work"):
                work, entry, method = candidate["work"], candidate, kind
                break
        usable = [
            f"{kind}:{identifier}" for kind, identifier in candidates if identifier
        ]
        if work:
            status = (
                "classified"
                if work.get("primary_topic")
                else "matched_without_primary_topic"
            )
        elif not usable:
            status = "no_identifier"
        elif all(key in lookups for key in usable):
            status = "not_found"
        else:
            status = "not_queried"
        work = work or {}
        primary = work.get("primary_topic")
        row = {
            key: paper[key]
            for key in ("paper_id", "corpus", "title", "year", "doi", "openalex_id")
        }
        row |= {
            "status": status,
            "match_method": method,
            "matched_openalex_id": work.get("id"),
            "retrieved_at": entry.get("retrieved_at"),
            "primary_score": (primary or {}).get("score"),
        }
        for level in LEVELS:
            label = (primary if level == "topic" else (primary or {}).get(level)) or {}
            row[f"primary_{level}"] = label.get("display_name")
            row[f"primary_{level}_id"] = label.get("id")
        for mode, topics in (
            ("primary", [primary] if primary else []),
            ("all_topics", (work.get("topics") or []) + ([primary] if primary else [])),
        ):
            for level in LEVELS:
                labels = {}
                for topic in topics:
                    label = topic if level == "topic" else topic.get(level) or {}
                    if label.get("id"):
                        labels[label["id"]] = label.get("display_name")
                row[f"{mode}_{level}_ids"] = sorted(labels)
                if mode == "all_topics":
                    row[f"all_{level}s"] = sorted(
                        name for name in labels.values() if name
                    )
                for identifier, name in labels.items():
                    bucket = counts[mode][level].setdefault(
                        identifier,
                        {"id": identifier, "name": name, "papers": 0, "by_corpus": {}},
                    )
                    bucket["papers"] += 1
                    bucket["by_corpus"][paper["corpus"]] = (
                        bucket["by_corpus"].get(paper["corpus"], 0) + 1
                    )
        rows.append(row)
    for mode, levels in counts.items():
        for level, buckets in levels.items():
            classified = sum(bool(row[f"primary_{level}_ids"]) for row in rows)
            for bucket in buckets.values():
                bucket["percent_of_corpus"] = (
                    round(100 * bucket["papers"] / len(rows), 3) if rows else 0
                )
                if mode == "primary":
                    bucket["percent_of_classified"] = (
                        round(100 * bucket["papers"] / classified, 3)
                        if classified
                        else 0
                    )
    ranked = {
        mode: {
            level: sorted(buckets.values(), key=lambda b: (-b["papers"], b["id"]))
            for level, buckets in levels.items()
        }
        for mode, levels in counts.items()
    }
    return rows, ranked


def run(data, out, cache=None, offline=False, top_n=10, font_size=FONT_SIZE):
    """Classify every MBDef paper (DATA: `analysis.load()`) and write report.json, papers.json, papers.csv and the figure into OUT. CACHE defaults to OUT/openalex-cache.json; OFFLINE reads only the cache."""
    out = Path(out)
    cache_path = Path(cache) if cache else out / "openalex-cache.json"
    if offline and not cache_path.is_file():
        raise FileNotFoundError(
            f"{cache_path} does not exist; run once with offline=False"
        )
    papers = (
        data["papers"]
        .select("paper_id", "corpus", "title", "year", "doi", "openalex_id")
        .to_dicts()
    )
    lookups = read_cache(cache_path) if offline else enrich(papers, cache_path)
    rows, counts = summarize(papers, lookups)
    report = {
        "sources": SOURCES,
        "papers": len(papers),
        "offline": offline,
        "identifier_coverage": {
            key: sum(bool(p[key]) for p in papers) for key in ("doi", "openalex_id")
        },
        "status": dict(Counter(row["status"] for row in rows)),
        "status_by_corpus": {
            corpus: dict(Counter(r["status"] for r in rows if r["corpus"] == corpus))
            for corpus in sorted({p["corpus"] for p in papers})
        },
        "counting_unit": "One released paper_id; several paper ids can be the same OpenAlex work.",
        "method": "Primary: one category per paper at each level. All topics: each paper once per distinct category; percentages can sum above 100. Missing labels are not inferred.",
        "classification_source": "https://help.openalex.org/data/topics/",
        "counts": counts,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    (out / "papers.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False) + "\n"
    )
    with (out / "papers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                for k, v in row.items()
            }
            for row in rows
        )
    make_figure(report, rows, out, top_n, font_size)
    return report


def figure_rows(report, top_n):
    """The bars: the TOP_N fields (Medicine and Health Professions drawn as one), the remaining fields, and the unclassified papers; they add up to every paper."""
    grouped = {}
    for row in report["counts"]["primary"]["field"]:
        name = (
            "Medicine and Health Professions"
            if row["name"] in {"Medicine", "Health Professions"}
            else row["name"]
        )
        grouped.setdefault(name, Counter()).update(row["by_corpus"])
    ranked = sorted(grouped.items(), key=lambda item: (-sum(item[1].values()), item[0]))
    rows = [{"name": name, **counts} for name, counts in ranked[:top_n]]
    if ranked[top_n:]:
        rows.append(
            {
                "name": "Other classified fields",
                **sum((c for _, c in ranked[top_n:]), Counter()),
            }
        )
    missing = {
        corpus: sum(statuses.values())
        - sum(counts.get(corpus, 0) for _, counts in ranked)
        for corpus, statuses in report["status_by_corpus"].items()
    }
    if sum(missing.values()):
        rows.append({"name": "Unclassified", **missing})
    if (
        sum(row.get("seeds", 0) + row.get("closure", 0) for row in rows)
        != report["papers"]
    ):
        raise ValueError("Figure counts do not sum to the corpus size")
    return rows


def social_science_inset(ax, fields, papers, font_size):
    """Magnify the papers whose primary field is Social Sciences, by subfield; returns (inset, its bars) or None."""
    from matplotlib.patches import ConnectionPatch, FancyBboxPatch
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    social = [paper for paper in papers if paper["primary_field"] == "Social Sciences"]
    if not social:
        return None
    by_subfield = {}
    for paper in social:
        name = paper["primary_subfield"] or "Unclassified subfield"
        by_subfield.setdefault(name, Counter())[paper["corpus"]] += 1
    ranked = sorted(
        by_subfield, key=lambda name: (-sum(by_subfield[name].values()), name)
    )
    names = ranked[:INSET_SUBFIELDS]
    counts = [by_subfield[name] for name in names]
    if ranked[INSET_SUBFIELDS:]:
        names.append("Other subfields")
        counts.append(
            sum((by_subfield[name] for name in ranked[INSET_SUBFIELDS:]), Counter())
        )
    if sum(sum(count.values()) for count in counts) != len(social):
        raise ValueError("Social Sciences inset counts do not sum to its paper count")
    # A lens-like frame and two connectors point at the parent bar; the inset has its own scale.
    lens = (0.115, 0.28, 0.55, 0.67)
    for index, row in enumerate(fields):
        if row["name"] == "Social Sciences":
            total = row.get("seeds", 0) + row.get("closure", 0)
            for fraction, end_y in (
                (1.0, lens[1] + lens[3] - 0.02),
                (0.0, lens[1] + 0.02),
            ):
                ax.add_artist(
                    ConnectionPatch(
                        xyA=(index + 0.34, total * fraction),
                        coordsA=ax.transData,
                        xyB=(lens[0], end_y),
                        coordsB=ax.transAxes,
                        color=GRAY,
                        linewidth=2.5,
                        linestyle="--",
                        zorder=0.5,
                        connectionstyle="arc3",
                    )
                )
            break
    ax.add_patch(
        FancyBboxPatch(
            (lens[0], lens[1]),
            lens[2],
            lens[3],
            boxstyle="round,pad=0.012,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=GRAY,
            linewidth=3,
            zorder=2,
        )
    )
    inset = ax.inset_axes((0.300, 0.33, 0.34, 0.59), zorder=3)
    seed = [count.get("seeds", 0) for count in counts]
    closure = [count.get("closure", 0) for count in counts]
    inset.barh(range(len(names)), seed, color=GRAY, height=0.65)
    inset.barh(
        range(len(names)), closure, left=seed, color=BLUE, alpha=BAR_ALPHA, height=0.65
    )
    inset.set_yticks(
        range(len(names)),
        [fill(name, 30) for name in names],
        fontsize=font_size * SMALL_FONT_SCALE,
    )
    inset.invert_yaxis()
    inset.tick_params(axis="y", length=0, pad=8)
    inset.tick_params(axis="x", labelsize=font_size * SMALL_FONT_SCALE)
    inset.spines[["top", "right", "left"]].set_visible(False)
    inset.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    inset.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    inset.set_xlim(0, max(a + b for a, b in zip(seed, closure, strict=True)) * 1.25)
    return inset, list(zip(range(len(names)), seed, closure, strict=True))


def _fits(text, renderer, width, height):
    bounds = text.get_window_extent(renderer)
    return bounds.width + 14 <= width and bounds.height + 14 <= height


def vertical_split_labels(fig, ax, seed, closure, font_size):
    """Each column's seed and closure counts inside its segments, the total above; when a count does not fit, the total reads `total\\n(seed, closure)` instead."""
    renderer = fig.canvas.get_renderer()
    maximum = ax.get_ylim()[1]
    for x, (a, b) in enumerate(zip(seed, closure, strict=True)):
        texts, fits = [], True
        for start, value in ((0, a), (a, b)):
            if not value:
                continue
            text = ax.text(
                x,
                start + value / 2,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=font_size * SMALL_FONT_SCALE,
                color=INK,
            )
            width = abs(
                ax.transData.transform((x + 0.34, 0))[0]
                - ax.transData.transform((x - 0.34, 0))[0]
            )
            height = abs(
                ax.transData.transform((x, start + value))[1]
                - ax.transData.transform((x, start))[1]
            )
            fits = fits and _fits(text, renderer, width, height)
            texts.append(text)
        if not fits:
            for text in texts:
                text.remove()
        label = f"{a + b:,}" if fits else f"{a + b:,}\n({a:,}, {b:,})"
        ax.text(
            x,
            a + b + maximum * 0.012,
            label,
            ha="center",
            va="bottom",
            fontsize=font_size * SMALL_FONT_SCALE,
            color="black",
            linespacing=1.15,
        )


def horizontal_split_labels(fig, ax, rows, font_size):
    """The same for horizontal bars: the total right of the bar reads `total (seed, closure)` when a count does not fit."""
    renderer = fig.canvas.get_renderer()
    maximum = ax.get_xlim()[1]
    for y, seed, closure in rows:
        texts, fits = [], True
        for start, value in ((0, seed), (seed, closure)):
            if not value:
                continue
            text = ax.text(
                start + value / 2,
                y,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=font_size * SMALL_FONT_SCALE,
                color=INK,
            )
            width = (
                ax.transData.transform((start + value, y))[0]
                - ax.transData.transform((start, y))[0]
            )
            fits = fits and _fits(text, renderer, width, float("inf"))
            texts.append(text)
        if not fits:
            for text in texts:
                text.remove()
        total = seed + closure
        label = f"{total:,}" if fits else f"{total:,} ({seed:,}, {closure:,})"
        text = ax.text(
            total + maximum * 0.015,
            y,
            label,
            ha="left",
            va="center",
            fontsize=font_size * SMALL_FONT_SCALE,
            color="black",
        )
        if text.get_window_extent(renderer).x1 > ax.get_window_extent(renderer).x1:
            raise ValueError("Bar-total label exceeds plotting area")


def make_figure(report, papers, out, top_n=10, font_size=FONT_SIZE):
    """Papers per primary field, seed papers grey and closure papers blue, with the Social Sciences subfields as an inset; research-disciplines.png and .svg in OUT."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    out = Path(out)
    plt.rcParams.update(
        {"svg.fonttype": "path", "font.family": FONT_FAMILY, "font.size": font_size}
    )
    rows = figure_rows(report, top_n)
    seed = [row.get("seeds", 0) for row in rows]
    closure = [row.get("closure", 0) for row in rows]
    positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 16), layout="constrained")
    ax.bar(positions, seed, width=0.68, color=GRAY, label="_nolegend_")
    ax.bar(
        positions,
        closure,
        bottom=seed,
        width=0.68,
        color=BLUE,
        alpha=BAR_ALPHA,
        label="Closure Papers",
    )
    ax.set_xticks(
        positions,
        [fill(row["name"], width=26) for row in rows],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=font_size * SMALL_FONT_SCALE,
    )
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", labelsize=font_size * SMALL_FONT_SCALE)
    ax.set_ylabel(
        "Number of Papers", fontsize=font_size * EMPHASIS_FONT_SCALE, labelpad=12
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_ylim(0, (max(a + b for a, b in zip(seed, closure, strict=True)) or 1) * 1.15)
    ax.margins(x=0.025)
    ax.legend(loc="upper right", frameon=False, fontsize=font_size, handlelength=1.2)
    inset = social_science_inset(ax, rows, papers, font_size)
    fig.canvas.draw()
    vertical_split_labels(fig, ax, seed, closure, font_size)
    if inset:
        horizontal_split_labels(fig, *inset, font_size)
    for extension in ("png", "svg"):
        fig.savefig(out / f"research-disciplines.{extension}", dpi=180)
    plt.close(fig)
