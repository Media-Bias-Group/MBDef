"""MBDef's dataset card: an overview README.md (YAML front matter for a dataset host and the body from cards/mbdef/README.md) and the pages it links (cards/mbdef/docs/*.md, written to docs/), every number computed from the built tables, the corpus view and the recorded Step 5 numbers, or taken from the recorded numbers of the earlier steps below (`EARLIER`), and worked examples taken from real rows (the concept "fake news" when the tables hold it). `pages` fills every file of a card's template folder; gold.py uses it for MBDef-Gold's card."""

import shutil
from collections import defaultdict
from pathlib import Path

import polars as pl
import yaml

from . import BANNER_PATH, CODE_URL, EXPLORER_URL, PAPER_URL, PROJECT_URL
from .schema import (
    CORPUS_TABLES,
    EXTRA_FIELDS,
    MEMBERS,
    NAME_FIELDS,
    REDISTRIBUTABLE,
    TERM_FIELDS,
)

CARDS = (
    Path(__file__).resolve().parent / "cards"
)  # one template folder per dataset (mbdef/, gold/), laid out as the folder an upload takes
# The numbers of the earlier steps that the card quotes and this command does not rebuild: the literature search and the paper selection (Steps 1 and 2, docs/step-1-2-papers/seeds.md), the extractor's recall and precision on the ground truth's TEST papers (Step 3, the held-out rescoring of docs/step-3-extraction/evaluate.md) and the ground truth's papers from SciDef's DefExtra (docs/step-3-extraction/groundtruth.md)
DEFEXTRA_PAPERS = 73  # the ground truth's papers from SciDef's DefExtra
EARLIER = dict(
    search_documents=50_101,
    open_access_documents=38_815,
    parsed_documents=14_900,
    extractor_recall="80.5%",
    extractor_precision="30.6%",
    defextra_papers=DEFEXTRA_PAPERS,
)
LICENCE_NAMES = {
    "cc-by-4.0": "CC BY 4.0",
    "cc-by-sa-4.0": "CC BY-SA 4.0",
    "cc-by-nc-4.0": "CC BY-NC 4.0",
    "cc-by-nc-sa-4.0": "CC BY-NC-SA 4.0",
    "cc0-1.0": "CC0 1.0",
}


def fmt(x) -> str:
    return f"{x:,}" if isinstance(x, int) else str(x)


def fmt_ci(ci, sign=False, digits=4) -> str:
    lo, hi = ci
    return f"[{lo:+.{digits}f}, {hi:+.{digits}f}]" if sign else f"[{lo:.3f}, {hi:.3f}]"


def hub_url(repo_id: str) -> str:
    """A dataset's page on Hugging Face."""
    return f"https://huggingface.co/datasets/{repo_id}"


def files_url(repo_id: str) -> str:
    """The address of a dataset's files on Hugging Face, which the card's links to its pages are built on."""
    return f"{hub_url(repo_id)}/blob/main"


def links(*named: tuple[str, str]) -> str:
    """The card's line of links: the paper, the code, and the other pages NAMED gives as (text, address)."""
    pairs = (("Paper", PAPER_URL), ("Code", CODE_URL), *named)
    return " · ".join(f"[{text}]({url})" for text, url in pairs)


# The row of buttons under the banner of both cards, and under the poster of the repository's README.md (a test checks it is the same there): the MBDef Explorer and the project page. HTML, so that GitHub and Hugging Face both centre it.
BADGES = (
    '<p align="center">\n'
    f'  <a href="{EXPLORER_URL}"><img alt="Open the MBDef Explorer" '
    'src="https://img.shields.io/badge/Open%20the-MBDef%20Explorer-2f6fdb?style=for-the-badge"></a>\n'
    f'  <a href="{PROJECT_URL}"><img alt="Project page" '
    'src="https://img.shields.io/badge/Project-page-24292f?style=for-the-badge"></a>\n'
    "</p>\n"
)


def gold_badges(mbdef_url: str) -> str:
    """MBDef-Gold's row of buttons: its card is not about the Explorer, so it leads to MBDef and the project page."""
    return (
        '<p align="center">\n'
        f'  <a href="{mbdef_url}"><img alt="MBDef, the main dataset" '
        'src="https://img.shields.io/badge/The%20main%20dataset-MBDef-2f6fdb?style=for-the-badge"></a>\n'
        f'  <a href="{PROJECT_URL}"><img alt="Project page" '
        'src="https://img.shields.io/badge/Project-page-24292f?style=for-the-badge"></a>\n'
        "</p>\n"
    )


def add_banner(
    ds: Path,
    banner: Path | None,
    alt: str = "MBDef, the Media Bias Definitions Dataset",
) -> str:
    """Copy the image BANNER into the dataset folder DS at BANNER_PATH; the Markdown that shows it at the top of the card, with the text ALT for those who cannot see it ("" without a banner).
    The address is relative, so the dataset host shows the repository's own copy, also while the repository is private."""
    if banner is None:
        return ""
    target = ds / BANNER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(banner, target)
    return f"![{alt}]({BANNER_PATH})\n\n"


def size_category(n: int) -> str:
    for lo, hi, label in (
        (0, 1_000, "n<1K"),
        (1_000, 10_000, "1K<n<10K"),
        (10_000, 100_000, "10K<n<100K"),
        (100_000, 1_000_000, "100K<n<1M"),
    ):
        if lo <= n < hi:
            return label
    return "1M<n<10M"


def numbers(
    T: dict,
    dec_json: dict,
    recorded: dict,
    pick: dict,
    td_relevant: list[bool],
    year_source: dict,
    *,
    version: str,
    licence: str,
    repo_id: str,
    gold_repo_id: str,
) -> dict:
    """Every number the card states, computed from the built tables, the decisions and the recorded numbers; `td_relevant` is the label of every labelled train and dev record."""
    c = dec_json["counts"]
    t, td = recorded["test"], recorded["train_dev"]
    ens, clf, delta = t["ensemble"], t["classifier"], t["difference"]
    i, d, p, k = T["instances"], T["definitions"], T["papers"], T["concepts"]
    lic = p.group_by("open_licence").agg(
        pl.len().alias("n"), pl.col("n_definitions").sum().alias("defs")
    )
    redis = lic.filter(pl.col("open_licence").is_in(list(REDISTRIBUTABLE)))
    nc = lic.filter(pl.col("open_licence").str.starts_with("cc-by-nc"))
    weights = pick["normalised_weights"]
    n = dict(
        **EARLIER,
        version=version,
        repo_id=repo_id,
        licence=licence,
        licence_name=LICENCE_NAMES.get(licence, licence),
        gold_url=hub_url(gold_repo_id),
        repo_url=hub_url(repo_id),
        files_url=files_url(repo_id),
        explorer_url=EXPLORER_URL,
        paper_url=PAPER_URL,
        badges=BADGES,
        links=links(
            ("MBDef Explorer", EXPLORER_URL),
            ("Project page", PROJECT_URL),
            ("MBDef-Gold", hub_url(gold_repo_id)),
        ),
        concepts=k.height,
        instances=i.height,
        definitions=d.height,
        papers=p.height,
        seed_papers=p.filter(pl.col("corpus") == "seeds").height,
        closure_papers=p.filter(pl.col("corpus") == "closure").height,
        seed_instances=i.filter(pl.col("corpus") == "seeds").height,
        closure_instances=i.filter(pl.col("corpus") == "closure").height,
        all_terms=c["terms"],
        dropped_terms=c["excluded"],
        kept_share=f"{c['kept'] / c['terms']:.1%}",
        all_definitions=c["definitions"]["total"],
        all_papers=c["papers"]["total"],
        spellings=i["term"].n_unique(),
        single_concepts=k.filter(pl.col("n_instances") == 1).height,
        multi_concepts=k.filter(pl.col("n_instances") > 1).height,
        doi_papers=p.filter(pl.col("doi").is_not_null()).height,
        year_papers=p.filter(pl.col("year").is_not_null()).height,
        year_openalex=year_source.get("year from OpenAlex", 0),
        redis_papers=int(redis["n"].sum()),
        redis_defs=int(redis["defs"].sum()),
        nc_papers=int(nc["n"].sum()),
        nc_defs=int(nc["defs"].sum()),
        nolic_papers=p.filter(pl.col("open_licence").is_null()).height,
        with_citations=d.filter(pl.col("citations").list.len() > 0).height,
        extra_table="\n".join(
            f"| `{f}` | {int(i[f].list.len().sum()):,} | {i.filter(pl.col(f).list.len() > 0).height:,} |"
            for f in EXTRA_FIELDS
        ),
        extra_items=int(sum(i[f].list.len().sum() for f in EXTRA_FIELDS)),
        extra_rows=i.filter(
            pl.any_horizontal(pl.col(f).list.len() > 0 for f in EXTRA_FIELDS)
        ).height,
        cut=f"{dec_json['cut']:.4f}",
        **{f"w_{m}": f"{weights[m]:.3f}" for m in MEMBERS},
        test_records=t["records"],
        test_relevant=t["relevant"],
        test_papers=t["papers"],
        ens_auc=f"{ens['auc']:.4f}",
        clf_auc=f"{clf['auc']:.4f}",
        ens_auc_ci=fmt_ci(ens["auc_ci95"]),
        clf_auc_ci=fmt_ci(clf["auc_ci95"]),
        d_auc=f"{delta['auc']:+.4f}",
        d_auc_ci=fmt_ci(delta["auc_ci95"], sign=True),
        d_p=f"{delta['precision']:+.3f}",
        d_p_ci=fmt_ci(delta["precision_ci95"], sign=True, digits=3),
        ens_kept=ens["kept"],
        clf_kept=clf["kept"],
        test_reads=t["reads"],
        a_auc=f"{td['reference_ranks_auc']:.4f}",
        b_auc=f"{td['within_batch_auc']:.4f}",
        n_comparisons=t["comparisons"],
        op95_r=f"{td['at_cut_95']['recall']:.3f}",
        op95_p=f"{td['at_cut_95']['precision']:.3f}",
        op95_share=f"{td['at_cut_95']['kept_share']:.1%}",
        opf1_r=f"{td['at_best_f1']['recall']:.3f}",
        opf1_p=f"{td['at_best_f1']['precision']:.3f}",
        opf1_share=f"{td['at_best_f1']['kept_share']:.1%}",
        strict_cut=f"{dec_json['best_f1_cut']:.4f}",
        strict=i.filter(pl.col("strict")).height,
        strict_share=f"{i.filter(pl.col('strict')).height / c['terms']:.1%}",
        strict_concepts=k.filter(pl.col("n_strict") > 0).height,
        td_relevant_share=f"{sum(td_relevant) / len(td_relevant):.1%}",
        td_records=len(td_relevant),
        ens_r_pct=f"{ens['recall']:.1%}",
    )
    for m, r in (("ens", ens), ("clf", clf)):
        n.update(
            {
                f"{m}_{x}": f"{r[y]:.3f}"
                for x, y in (
                    ("p", "precision"),
                    ("r", "recall"),
                    ("f1", "f1"),
                    ("mcc", "mcc"),
                )
            }
        )
    return n


def terms_numbers(info: dict) -> dict:
    """The card's table of the terms view: per field, items before and after merging, merged forms, items several papers state, and items that name a concept."""
    lines = []
    for f in TERM_FIELDS:
        g = {
            k: info.get(f"{f}:{k}", 0)
            for k in ("raw", "items", "merged", "multi", "linked")
        }
        link = f"{g['linked']:,}" if f in NAME_FIELDS else "—"
        lines.append(
            f"| `{f}` | {g['raw']:,} | {g['items']:,} | {g['merged']:,} | {g['multi']:,} | {link} |"
        )
    return dict(
        terms_table="\n".join(lines),
        terms_self=sum(info.get(f"{f}:self", 0) for f in NAME_FIELDS),
    )


def pages(folder: str, values: dict, head: dict) -> dict[str, str]:
    """Every file of the card template folder `cards/<folder>/` with its `{{name}}` placeholders filled from VALUES, as {path in the dataset folder: text}; README.md gets the YAML front matter HEAD in front."""
    out = {}
    for path in sorted((CARDS / folder).rglob("*.md")):
        text = path.read_text()
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", fmt(value))
        name = path.relative_to(CARDS / folder).as_posix()
        if name == "README.md":
            text = (
                "---\n"
                + yaml.safe_dump(head, sort_keys=False, allow_unicode=True, width=120)
                + "---\n"
                + text
            )
        out[name] = text
    return out


def render(n: dict, written: dict, T: dict, pick: dict, licence: str) -> dict[str, str]:
    """The card's files: README.md (YAML front matter with the configs from the written files, and the overview) and the pages under docs/."""
    configs = [
        dict(config_name=name, data_files=[dict(split="corpus", path=written[name])])
        for name in CORPUS_TABLES
    ]
    head = dict(
        pretty_name="MBDef: media-bias definitions from the scientific literature",
        license=licence,
        language=["en"],
        task_categories=["text-classification", "text-retrieval"],
        tags=[
            "media-bias",
            "definitions",
            "terminology",
            "taxonomy",
            "scientific-literature",
        ],
        size_categories=[size_category(T["definitions"].height)],
        configs=configs,
    )
    return pages("mbdef", {**n, **examples(T, pick)}, head)


def examples(T: dict, pick: dict) -> dict:
    """Real rows for the card's worked examples: the concept "fake news" (else the largest), one of its rows, one of its definitions, its merged items in `terms`, and a traced merge."""
    k, i, d, p = T["concepts"], T["instances"], T["definitions"], T["papers"]
    top = k.row(0, named=True)
    fake = k.filter(pl.col("label").str.to_lowercase() == "fake news")
    ex = fake.row(0, named=True) if fake.height else top
    # an openly licensed paper with a year, and a definition of moderate length that cites a named reference
    named = (
        pl.col("citations")
        .list.eval(pl.element().str.contains("[A-Za-z]{3}"))
        .list.any()
    )  # "Entman, 1993", not "[2]"
    cand = (
        d.filter(
            (pl.col("concept_id") == ex["concept_id"])
            & named
            & pl.col("text").str.len_chars().is_between(60, 260)
            & pl.col("text").str.contains(r"\b(the|of|and|is|that)\b")
            & pl.col("text").str.contains(r"^[A-Z]")
        )
        .join(i.drop("term", "paper_id", "concept_id"), on="mention_id")
        .join(p, on="paper_id")
        .filter(
            pl.col("open_licence").is_in(list(REDISTRIBUTABLE))
            & pl.col("year").is_not_null()
        )
        .sort("mention_id", "def_id")
    )
    row = cand.row(0, named=True)
    out = dict(
        ex_label=ex["label"],
        ex_n_spellings=len(ex["labels"]),
        ex_spellings=", ".join(f"“{s}”" for s in ex["labels"][:6]),
        ex_instances=ex["n_instances"],
        ex_papers=ex["n_papers"],
        ex_definitions=ex["n_definitions"],
        ex_concept_id=ex["concept_id"],
        ex_key=ex["match_key"],
        ex_term=row["term"],
        ex_mention=row["mention_id"],
        ex_title=row["title"],
        ex_year=str(row["year"]),
        ex_doi=row["doi"] or "(none)",
        ex_licence=row["open_licence"],
        ex_def=row["text"],
        ex_def_id=row["def_id"],
        ex_citation_sentence=(
            "For it the paper cites "
            + "; ".join(f"“{x}”" for x in row["citations"])
            + " (`citations` column)."
        ),
        ex_score=f"{row['ensemble_score']:.3f}",
        ex_clf=f"{row['classifier_score']:.3f}",
        ex_basal=f"{row['basal_score']:.3f}",
        ex_judge=f"{row['judge_score']:.3f}",
        top_label=top["label"],
        top_instances=top["n_instances"],
    )
    r = {m: row[f"{m}_ref_rank"] for m in MEMBERS}
    w = pick["normalised_weights"]
    out.update(
        ex_rank_sentence=(
            f"For the example row above, the classifier's score {row['classifier_score']:.3f} is "
            f"higher than {r['classifier']:.1%} of the classifier's scores on those records, so it "
            f"becomes {r['classifier']:.3f}."
        ),
        ex_combo_sentence=(
            "For the example row: "
            + " + ".join(f"{w[m]:.3f} × {r[m]:.3f}" for m in MEMBERS)
            + f" = {sum(w[m] * r[m] for m in MEMBERS):.3f}, its `ensemble_score`."
        ),
    )
    # the other extracted fields: the example concept's row with the most non-empty fields (then most items)
    filled = pl.sum_horizontal(
        (pl.col(f).list.len() > 0).cast(pl.Int8) for f in EXTRA_FIELDS
    )
    items = pl.sum_horizontal(pl.col(f).list.len() for f in EXTRA_FIELDS)
    pool = i.filter(pl.col("concept_id") == ex["concept_id"])
    if not pool.filter(filled > 0).height:
        pool = i
    xr = (
        pool.with_columns(filled.alias("_f"), items.alias("_n"))
        .sort(["_f", "_n", "mention_id"], descending=[True, True, False])
        .row(0, named=True)
    )
    xp = p.filter(pl.col("paper_id") == xr["paper_id"]).row(0, named=True)

    def clip(x):
        return x if len(x) <= 90 else x[:87].rstrip() + "…"

    parts = [
        f"`{f}` "
        + ", ".join(f"“{clip(it['text'])}”" for it in xr[f][:2])
        + (f" (+{len(xr[f]) - 2} more)" if len(xr[f]) > 2 else "")
        for f in EXTRA_FIELDS
        if xr[f]
    ]
    out["ex_fields_sentence"] = (
        f"Besides definitions, a row lists what else the paper says about its term. The row "
        f"`{xr['mention_id']}` (“{xr['term']}” in *{xp['title']}*) lists: "
        + "; ".join(parts)
        + "."
    )
    # the terms view: the example concept's merged items, most stated first
    tv = T["terms"].filter(pl.col("concept_id") == ex["concept_id"]).row(0, named=True)

    def say(it):
        other = it["links_to"] and it["links_to"]["concept_id"] != ex["concept_id"]
        return (
            f"“{clip(it['text'])}” ({it['papers']} paper{'s' if it['papers'] != 1 else ''}"
            + (f"; the concept “{it['links_to']['term']}”" if other else "")
            + ")"
        )

    lines = [
        f"- `{f}`: "
        + ", ".join(say(it) for it in tv[f][:3])
        + (f", and {len(tv[f]) - 3:,} more" if len(tv[f]) > 3 else "")
        for f in ("supertypes", "subtypes", "synonyms", "related_terms")
        if tv[f]
    ]
    d0 = tv["definitions"][0]
    lines.append(
        f"- `definitions`: {len(tv['definitions']):,} after merging; the one stated by most papers "
        f"({d0['papers']}): “{clip(d0['text'])}”"
    )
    sup = tv["supertypes"][0] if tv["supertypes"] else None
    out["tv_agree"] = (
        f"“{sup['text']}” as a supertype of “{tv['term']}” is stated in {sup['papers']} papers"
        if sup
        else "an item stated in several papers is one they agree on"
    )
    per = defaultdict(dict)  # the example concept's names that sit under several fields
    for fld in NAME_FIELDS:
        for it in tv[fld]:
            per[it["text"].casefold()][fld] = it["papers"]
    name, where = max(
        ((key, v) for key, v in per.items() if len(v) > 1),
        key=lambda kv: (sum(kv[1].values()), kv[0]),
        default=(None, None),
    )
    out["tv_fields_note"] = (
        "The same name can sit under several fields: for “"
        + tv["term"]
        + "”, “"
        + name
        + "” is "
        + ", ".join(
            f"a {fld.removesuffix('s').replace('_', ' ')} in {n} papers"
            for fld, n in where.items()
        )
        + ". The view does not choose between them."
        if name is not None and where is not None
        else ""
    )
    # a real merge of a name item from different forms: the most stated one over all concepts
    best = None
    for r_ in T["terms"].select("term", *NAME_FIELDS).iter_rows(named=True):
        for fld in NAME_FIELDS:
            for it in r_[fld]:
                forms = {v["text"].casefold() for v in it["variants"]}
                if len(forms) > 1 and (
                    best is None or it["papers"] > best[2]["papers"]
                ):
                    best = (r_["term"], fld, it)
    if best:
        bt, bf, bi = best
        two = list(
            {v["text"].casefold(): v for v in reversed(bi["variants"])}.values()
        )[::-1][:2]  # first form per spelling
        out["tv_merge_example"] = (
            " and ".join(f"“{v['text']}”" for v in two)
            + f" (among the {bf.replace('_', ' ')} of “{bt}”) become one item, stated in "
            f"{bi['papers']} papers"
        )
    else:
        out["tv_merge_example"] = "no name item was written in two different forms"
    out["tv_example"] = (
        f"**Worked example.** The `terms` row of “{tv['term']}” ({tv['papers']} papers) says, most "
        "stated first:\n\n" + "\n".join(lines)
    )
    # tracing a merge: the "Refugees" row if it is kept, else the first row whose normalisation reduced a plural (prefer the row whose concept joins the most spellings, so the example shows a real merge)
    spellings = k.select("concept_id", pl.col("labels").list.len().alias("n_spellings"))
    plural = i.filter(
        pl.col("normalisation")
        .list.eval(pl.element().str.contains("'refugees' -> 'refugee'"))
        .list.any()
    )
    if not plural.height:
        plural = i.filter(
            pl.col("normalisation")
            .list.eval(pl.element().str.starts_with("plural"))
            .list.any()
        )
    tr = (
        plural.join(spellings, on="concept_id")
        .sort(["n_spellings", "mention_id"], descending=[True, False])
        .row(0, named=True)
    )
    tc = k.filter(pl.col("concept_id") == tr["concept_id"]).row(0, named=True)
    out.update(
        tr_term=tr["term"],
        tr_mention=tr["mention_id"],
        tr_ws=tr["word_set_key"],
        tr_sp=tr["spacing_key"],
        tr_steps="; ".join(f"“{s}”" for s in tr["normalisation"]),
        tr_concept=tc["label"],
        tr_cid=tc["concept_id"],
        tr_spellings=", ".join(f"“{s}”" for s in tc["labels"]),
        tr_n=f"{tc['n_instances']} row{'s' if tc['n_instances'] != 1 else ''}",
        tr_joins="; ".join(
            f"the {j['key_type']} key “{j['key']}” is shared by "
            + ", ".join(f"“{x}”" for x in j["labels"])
            for j in tc["joined_by"]
        )
        or "(one spelling only)",
        tr_flags=", ".join(f"“{x}”" for x in tc["merge_flags"])
        or "none (a plain plural or case difference is not flagged)",
    )
    return out
