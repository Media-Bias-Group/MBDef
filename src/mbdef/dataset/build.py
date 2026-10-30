"""`mbdef dataset build`: the released tables, the dataset card and a validation report.

Reads Step 5's keep or drop of every corpus term (the ensemble's decisions) and the term merging of the kept ones, and writes `<out>/taxonomy-dataset/`: README.md (the card, an overview) with its pages in docs/ and its banner image in assets/, data/{terms,concepts,instances,definitions,papers}.parquet, provenance.json and, when given, the MBDef Explorer's page data in explorer/ (checked against the tables); and `<out>/validation.{json,md}` beside it. A build that fails a check is moved to `<out>/taxonomy-dataset-FAILED/` and the command exits 1, so a failing build is never where an upload would look. What the dataset holds (docs/dataset/dataset.md): every definition text verbatim and no paragraph context; only the term-in-paper rows Step 5 keeps, with their member scores, reference ranks and ensemble score. The hand-labelled ground truth is a dataset of its own, MBDef-Gold (`dataset build-gold`, gold.py); the build reads it only for the card's count of labelled train and dev records.

Worked example (docs/dataset/dataset.md): the kept term "Fake news" of one seed paper becomes one `instances` row and, with the other spellings of the same label, one `concepts` row and one `terms` row; each of its definitions becomes one `definitions` row; its paper one `papers` row.
"""

import csv
import gzip
import hashlib
import json
import platform
import re
import shutil
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pyarrow
import yaml

from mbdef import files, postprocess
from mbdef.postprocess import label_match
from mbdef.postprocess.corpus import load_view
from mbdef.postprocess.store import loose

from . import (
    CARD_NUMBERS,
    DATASET,
    LIKELY_WRONG,
    OPENALEX,
    REPO,
    Inputs,
    card,
)
from . import store as corpus_store
from .schema import (
    CORPUS_TABLES,
    EXTRA_FIELDS,
    LICENCE_ORDER,
    MEMBERS,
    NAME_FIELDS,
    SCHEMA,
    TERM_FIELDS,
    frame,
)

LIKELY_WRONG_FLAG = (
    "likely different meanings: same words in a different order (merge audit)"
)
# An absolute or home-relative path anywhere in the release fails the build. A path starts a token (after a space, quote, bracket, "=" or ":", or in file://), so a quote such as "source/media/audience relations" is not one.
ABSOLUTE = re.compile(
    r"(?:^|[\s\"'(=:]|file://)(?:"
    + "|".join(map(re.escape, files.LOCAL_FOLDERS))
    + r")|\b[A-Za-z]:[\\/]",
    re.M,
)
PRIVATE = {
    "annotator",
    "annotator_id",
    "annotators",
    "completed_by",
    "correction_notes",
    "error_text",
    "notes",
    "email",
    "user",
    "lead_time",
    "task",
    "annotation",
}  # column names that would hold annotator data
EMAIL = files.EMAIL
ZSTD_LEVEL = 19  # parquet compression of every table
# How provenance.json names the rule that merges spellings (label_match.py), in words: its version id is internal
MERGE_RULE = (
    "two spellings are one term when they are the same after case, hyphens, spacing, quotation marks, a leading "
    "article, stop words between content words, a bracketed acronym, word order, British spelling and plurals are set "
    'aside (the card\'s "Term merging")'
)


def read_decisions(csv_path: Path, json_path: Path) -> tuple[pl.DataFrame, dict]:
    """Every decision row (keep as bool, the scores as floats; `score` renamed `ensemble_score`) and the json."""
    df = pl.read_csv(csv_path, infer_schema=False)
    df = df.with_columns(pl.col("keep").str.to_lowercase().eq("true")).rename(
        {"score": "ensemble_score"}
    )
    score_cols = [
        "ensemble_score",
        *[f"{m}_score" for m in MEMBERS],
        *[f"{m}_ref_rank" for m in MEMBERS],
    ]
    df = df.with_columns(pl.col(c).cast(pl.Float64) for c in score_cols)
    return df, json.loads(json_path.read_text())


def merge_terms(labels: list[str]) -> tuple[list[str], dict[str, str]]:
    """The Label Match rule over these labels, as `postprocess label-groups` does it: each label's concept id, and each concept's word-set key. A concept id is "g" + 12 hex of the SHA-1 of the key of its first label (input order)."""
    group = label_match.label_groups(labels)
    vocab = label_match.vocabulary(labels)
    first: dict[int, int] = {}
    for i, g in enumerate(group):
        first.setdefault(g, i)
    key = {g: label_match.label_keys(labels[i], vocab)[0] for g, i in first.items()}
    gid = {g: "g" + hashlib.sha1(k.encode()).hexdigest()[:12] for g, k in key.items()}
    if len(set(gid.values())) != len(gid):
        raise RuntimeError("two concepts got the same id")
    return [gid[g] for g in group], {gid[g]: key[g] for g in key}


def _find(parent: dict[str, str], x: str) -> str:
    """Union-find root of x (with path halving)."""
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def text_key(text: str) -> str:
    """A sentence's merge key: Unicode NFKC, case-folded, punctuation and quotes removed, spacing collapsed (the corpus store's `loose` form)."""
    return loose(text)


def terms_view(
    tables: dict[str, pl.DataFrame],
    labels: dict[str, str],
    group_of: dict[str, str],
    vocab: label_match.Vocabulary,
) -> tuple[list[dict], dict[str, int]]:
    """One row per concept with everything said about it: per field, the items of all its rows merged, each with the number of papers that state it, the forms it was written in, and its sources. A name item (NAME_FIELDS) is merged by the Label Match rule's two keys (vocabulary of all corpus terms) and links to the concept whose Label Group has one of its keys; a sentence is merged by `text_key`."""
    # Label Match key -> Label Groups, over all corpus terms
    key_group: dict[str, set[str]] = defaultdict(set)
    for m, label in labels.items():
        ws, sp = label_match.label_keys(label, vocab)
        key_group["w" + ws].add(group_of[m])
        key_group["s" + sp].add(group_of[m])
    if any(len(v) > 1 for v in key_group.values()):
        raise RuntimeError("a Label Match key belongs to two Label Groups")
    con = {r["concept_id"]: r for r in tables["concepts"].iter_rows(named=True)}
    paper = dict(
        zip(
            tables["instances"]["mention_id"],
            tables["instances"]["paper_id"],
            strict=True,
        )
    )
    # concept -> field -> sources
    raw: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in tables["definitions"].iter_rows(named=True):
        raw[r["concept_id"]]["definitions"].append(
            dict(
                paper_id=r["paper_id"],
                mention_id=r["mention_id"],
                def_id=r["def_id"],
                text=r["text"],
                citations=r["citations"],
            )
        )
    for r in (
        tables["instances"]
        .select("mention_id", "concept_id", *EXTRA_FIELDS)
        .iter_rows(named=True)
    ):
        for f in EXTRA_FIELDS:
            for it in r[f]:
                raw[r["concept_id"]][f].append(
                    dict(
                        paper_id=paper[r["mention_id"]],
                        mention_id=r["mention_id"],
                        def_id=None,
                        text=it["text"],
                        citations=it["citations"],
                    )
                )
    keys_memo: dict[str, tuple[str, ...]] = {}

    def keys(field: str, text: str) -> tuple[str, ...]:
        if field not in NAME_FIELDS:
            return ("t" + text_key(text),)
        if text not in keys_memo:
            ws, sp = label_match.label_keys(text, vocab)
            keys_memo[text] = ("w" + ws, "s" + sp)
        return keys_memo[text]

    rows: list[dict] = []
    info: Counter[str] = Counter()
    for cid, c in con.items():
        row: dict = dict(
            concept_id=cid,
            term=c["label"],
            papers=c["n_papers"],
            strict=c["n_strict"] > 0,
        )
        sp_papers: dict[str, set[str]] = defaultdict(set)
        for m in c["mention_ids"]:
            sp_papers[labels[m]].add(paper[m])
        row["spellings"] = [
            dict(text=t, papers=len(ps))
            for t, ps in sorted(sp_papers.items(), key=lambda x: (-len(x[1]), x[0]))
        ]
        for f in TERM_FIELDS:
            srcs = raw[cid][f]
            parent: dict[str, str] = {}
            for sr in srcs:
                k = keys(f, sr["text"])
                for kk in k[1:]:
                    parent[_find(parent, kk)] = _find(parent, k[0])
            groups: dict[str, list[dict]] = defaultdict(list)
            for sr in srcs:
                groups[_find(parent, keys(f, sr["text"])[0])].append(sr)
            items: list[dict] = []
            for g in groups.values():
                forms: dict[str, set[str]] = defaultdict(set)
                for sr in g:
                    forms[sr["text"]].add(sr["paper_id"])
                order = sorted(forms.items(), key=lambda x: (-len(x[1]), x[0]))
                text = order[0][0]
                link = None
                if f in NAME_FIELDS:
                    hit = {
                        gg
                        for sr in g
                        for k in keys(f, sr["text"])
                        for gg in key_group.get(k, ())
                    }
                    hit &= con.keys()
                    if len(hit) > 1:
                        raise RuntimeError(
                            f"an item of {cid} names two concepts: {hit}"
                        )
                    if hit:
                        (lc,) = hit
                        link = dict(concept_id=lc, term=con[lc]["label"])
                items.append(
                    dict(
                        text=text,
                        papers=len({sr["paper_id"] for sr in g}),
                        links_to=link,
                        variants=[dict(text=t, papers=len(ps)) for t, ps in order]
                        if len(order) > 1
                        else [],
                        sources=[
                            dict(
                                paper_id=sr["paper_id"],
                                mention_id=sr["mention_id"],
                                def_id=sr["def_id"],
                                as_written=None if sr["text"] == text else sr["text"],
                                citations=sr["citations"],
                            )
                            for sr in sorted(
                                g,
                                key=lambda x: (
                                    x["paper_id"],
                                    x["mention_id"],
                                    x["def_id"] or "",
                                ),
                            )
                        ],
                    )
                )
                info[f"{f}:items"] += 1
                info[f"{f}:merged"] += len(order) > 1
                info[f"{f}:multi"] += items[-1]["papers"] > 1
                info[f"{f}:linked"] += link is not None
                info[f"{f}:self"] += link is not None and link["concept_id"] == cid
            info[f"{f}:raw"] += len(srcs)
            row[f] = sorted(items, key=lambda it: (-it["papers"], it["text"]))
        rows.append(row)
    return rows, dict(info)


def openalex_works(path: Path) -> dict[str, dict]:
    """DOI (lower case) -> OpenAlex work, from the recorded answers."""
    works: dict[str, dict] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            for w in json.loads(line)["results"]:
                if w.get("doi"):
                    works[w["doi"].lower().removeprefix("https://doi.org/")] = w
    return works


def best_licence(work: dict | None) -> tuple[str | None, str | None]:
    """The most open licence any location of the work carries, and that location's URL."""
    if not work:
        return None, None
    found = [
        (
            (loc.get("license") or "").lower(),
            loc.get("landing_page_url") or loc.get("pdf_url"),
        )
        for loc in work.get("locations") or []
        if loc.get("license")
    ]
    if not found:
        return None, None
    rank = {lic: i for i, lic in enumerate(LICENCE_ORDER)}
    return min(found, key=lambda x: (rank.get(x[0], len(rank)), x[0]))


def paper_table(
    paper_ids: set[str],
    view_corpus: dict[str, str],
    works: dict[str, dict],
    corpus: Path,
) -> tuple[list[dict], dict[str, int]]:
    """One row per paper: DOI, title, year, OpenAlex id, open-access status and licence (without the counts)."""
    meta: dict[str, dict] = {}
    with (corpus / "seeds/papers.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            # a DOI that two seed PDFs share is unknown (release v1.1's README)
            meta[r["paper_id"]] = dict(
                doi=(r["doi"].lower() or None)
                if r["doi_shared_by_seed_papers"] == "1"
                else None,
                title=r["title"] or None,
                openalex_id=None,
            )
    with (corpus / "closure/papers.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            meta[r["paper_id"]] = dict(
                doi=r["doi"].lower() or None,
                title=r["title"] or None,
                openalex_id=r["openalex_id"] or None,
            )
    cited_year: dict[str, int] = {}
    with (corpus / "links/cited-works.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["closure_paper_id"] and r["year"]:
                cited_year[r["closure_paper_id"]] = int(r["year"])
    rows: list[dict] = []
    source: Counter[str] = Counter()
    for p in sorted(paper_ids):
        m = meta.get(p, dict(doi=None, title=None, openalex_id=None))
        w = works.get(m["doi"]) if m["doi"] else None
        lic, url = best_licence(w)
        year = (w or {}).get("publication_year") or cited_year.get(p)
        source[
            "year from OpenAlex"
            if (w or {}).get("publication_year")
            else "year from the closure's cited-works list"
            if year
            else "no year"
        ] += 1
        rows.append(
            dict(
                paper_id=p,
                corpus=view_corpus[p],
                doi=m["doi"],
                title=m["title"],
                year=year,
                openalex_id=(w or {}).get("id") or m["openalex_id"],
                oa_status=((w or {}).get("open_access") or {}).get("oa_status"),
                open_licence=lic,
                licence_url=url,
            )
        )
    return rows, dict(source)


def portable(path: Path) -> str:
    """A path without the machine's own folders: relative to the repository, `$DATA_ROOT/...` under the data root, else the file name alone."""
    p = Path(path).resolve()
    roots = [(REPO, "")]
    try:
        roots.append((files.data_root().resolve(), "$DATA_ROOT/"))
    except SystemExit:
        pass
    for root, prefix in roots:
        if p.is_relative_to(root):
            return prefix + str(p.relative_to(root))
    return p.name


def absolute_paths(ds: Path) -> list[str]:
    """Every absolute path in the release folder: text files, and every string (also nested) in the parquet files."""
    hits: list[str] = []
    for p in sorted(ds.rglob("*")):
        if p.suffix in (".md", ".json"):
            hits += [
                f"{p.relative_to(ds)}: {m.group(0)!r}"
                for m in ABSOLUTE.finditer(p.read_text())
            ]
        elif p.suffix == ".parquet":
            df = pl.read_parquet(p)
            for col in df.columns:
                text = df.select(col).write_ndjson()
                hits += [
                    f"{p.relative_to(ds)}:{col}: {m.group(0)!r}"
                    for m in ABSOLUTE.finditer(text)
                ]
    return hits


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        path, compression="zstd", compression_level=ZSTD_LEVEL, statistics=True
    )


def recorded_hashes(listing: Path, folder: str) -> dict[str, str]:
    """The `sha256sum` lines of `listing` for files under `folder/`, as {path inside folder: sha256}."""
    found: dict[str, str] = {}
    for line in listing.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            if name.startswith(folder + "/"):
                found[name.removeprefix(folder + "/")] = digest
    return found


def differing(listing: Path, folder: str, written: dict[str, str]) -> list[str]:
    """`--same-data-as`: the paths written below `folder/` whose sha256 in WRITTEN ({path: sha256}) is not the one `listing` records, or that `listing` does not hold, sorted."""
    recorded = recorded_hashes(listing, folder)
    return sorted(p for p in written if recorded.get(p) != written[p])


def not_written(listing: Path, folder: str, written: dict[str, str]) -> list[str]:
    """`--same-data-as`: the paths `listing` records below `folder/` that this build does not write (for example the ground-truth tables of a version that held them), sorted."""
    return sorted(set(recorded_hashes(listing, folder)) - set(written))


def code_record() -> dict:
    """The code that built a folder: its commit, Python and the table libraries (nothing about the machine)."""
    return dict(
        **files.git_state(Path(__file__).resolve().parent),
        python=platform.python_version(),
        polars=pl.__version__,
        pyarrow=pyarrow.__version__,
    )


def report(
    out: Path,
    folder: str,
    stem: str,
    version: str,
    checks: list[dict],
    outputs: dict[str, dict],
    t0: float,
) -> int:
    """Write `out/<stem>.{json,md}` (every check and every file written) and print the Markdown; a failing build's folder is moved to `out/<folder>-FAILED/`. 0 when no check failed, else 1."""
    failed = [x for x in checks if x["status"] == "FAILED"]
    result = dict(
        dataset=folder,
        built=datetime.now(UTC).isoformat(timespec="seconds"),
        seconds=round(time.perf_counter() - t0, 1),
        verdict="PASSED" if not failed else "FAILED",
        checks=checks,
        rows={k: v["rows"] for k, v in outputs.items()},
        bytes={k: v["bytes"] for k, v in outputs.items()},
    )
    files.write_json(out / f"{stem}.json", result, indent=1)
    ds, bad = out / folder, out / f"{folder}-FAILED"
    if bad.exists():
        shutil.rmtree(bad)
    if failed:  # never leave a failing build where an upload would look
        ds.rename(bad)
    md = [
        f"# Validation of `{folder}` {version}".rstrip(),
        "",
        *(
            [f"**FAILED: the build was moved to `{bad.name}/`; do not upload it.**", ""]
            if failed
            else []
        ),
        f"Built {result['built']} in {result['seconds']} s. Verdict: **{result['verdict']}** "
        f"({sum(x['status'] == 'PASSED' for x in checks)} passed, {len(failed)} failed, "
        f"{sum(x['status'] not in ('PASSED', 'FAILED') for x in checks)} informational).",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    md += [
        f"| {x['check']} | {x['status']} | {x['detail'].replace('|', '/')[:300]} |"
        for x in checks
    ]
    md += ["", "| File | Rows | Bytes | SHA-256 |", "|---|---:|---:|---|"]
    md += [
        f"| `{v['path']}` | {v['rows']:,} | {v['bytes']:,} | `{v['sha256']}` |"
        for v in outputs.values()
    ]
    (out / f"{stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 1 if failed else 0


def file_record(path: Path) -> dict[str, str | int]:
    return dict(
        path=portable(path), bytes=path.stat().st_size, sha256=files.file_sha256(path)
    )


def explorer_data(ds: Path, bundle: Path, check) -> str:
    """Copy the MBDef Explorer's page data BUNDLE (a folder with its SHA256SUMS, docs/dataset/explorer.md) into the dataset folder DS as explorer/, after the checks `explorer build` makes against DS's tables; the card's sentence about it."""
    from mbdef import explorer

    try:
        found = explorer.checked_bundle(bundle)
    except SystemExit as stop:
        check("explorer data: every file as its SHA256SUMS lists", False, stop)
        return ""
    check("explorer data: every file as its SHA256SUMS lists", True, len(found))
    problems, summary = explorer.disagreements(found, ds / "data")
    check(
        "explorer data agrees with the tables (exactly their rows, scores, definitions, concepts, papers)",
        not problems,
        "; ".join(problems) or summary,
    )
    history = sorted(
        name for name, data in found.items() if explorer.published(name, data) != data
    )
    check(
        "explorer data holds no run history (commits, run times, local paths)",
        not history,
        history,
    )
    target = ds / explorer.BUNDLE
    for name in [*found, explorer.SUMS]:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / name, target / name)
    return (
        f"The folder `{explorer.BUNDLE}/` holds the Explorer's page data ({len(found):,} files, listed with their SHA-256 in "
        f"`{explorer.BUNDLE}/{explorer.SUMS}`). It holds exactly the rows of these tables, and besides what they hold it gives "
        "every term-in-paper row a place on the map, a topic and its nearest rows, all computed on these rows alone.\n"
    )


def build(
    i: Inputs,
    out: Path,
    *,
    version: str,
    licence: str,
    repo_id: str,
    gold_repo_id: str = files.MBDEF_GOLD_REPO,
    same_data_as: Path | None = None,
    banner: Path | None = None,
    explorer: Path | None = None,
) -> int:
    """Build the dataset into `out` (see the module's docstring), with the image BANNER at the top of its card and the Explorer's page data EXPLORER in explorer/; 0 when every check passes, else 1."""
    from . import gold  # gold.py builds on this module

    t0 = time.perf_counter()
    ds = out / DATASET
    checks: list[dict] = []

    def check(name: str, ok: bool | str, detail: object = "") -> None:
        status = ok if isinstance(ok, str) else ("PASSED" if ok else "FAILED")
        checks.append(dict(check=name, status=status, detail=str(detail)))

    # 1. decisions: only the kept term-in-paper rows
    dec, dec_json = read_decisions(i.decisions, i.decisions_json)
    kept = dec.filter(pl.col("keep")).sort("mention_id")
    c = dec_json["counts"]
    check(
        "decision rows = the decisions json's term count",
        dec.height == c["terms"],
        f"{dec.height:,} vs {c['terms']:,}",
    )
    check(
        "kept rows = the decisions json's kept count",
        kept.height == c["kept"],
        f"{kept.height:,} vs {c['kept']:,}",
    )
    for corpus in ("seeds", "closure"):
        n = kept.filter(pl.col("corpus") == corpus).height
        check(
            f"kept {corpus} rows = the decisions json",
            n == c[corpus]["kept"],
            f"{n:,} vs {c[corpus]['kept']:,}",
        )

    # 2. the corpus view: terms, definitions, papers
    view = load_view(i.store, i.links)
    concept_of = {x.mention_id: x for x in view.concepts}
    missing = [m for m in kept["mention_id"] if m not in concept_of]
    check(
        "every kept row is a term-in-paper of the corpus view",
        not missing,
        f"{len(missing)} missing",
    )
    check(
        "the decisions cover the whole view",
        dec.height == len(view.concepts),
        f"{dec.height:,} vs {len(view.concepts):,}",
    )
    same_text = all(
        concept_of[m].label == t and concept_of[m].paper_id == p
        for m, t, p in zip(
            kept["mention_id"], kept["label"], kept["paper_id"], strict=True
        )
    )
    check("term text and paper of every kept row equal the corpus store's", same_text)

    # 3. term merging: the Label Groups over every corpus term, restricted to the kept rows
    lg_run = json.loads(postprocess.stored(i.label_groups, "run.json").read_text())
    check(
        "the rule version here = the version that built the Label Groups",
        label_match.RULE_VERSION == lg_run["rule"]["version"],
        f"{label_match.RULE_VERSION} vs {lg_run['rule']['version']}",
    )
    all_mids = [
        x.mention_id for x in view.concepts
    ]  # mention-id order, as `postprocess label-groups`
    all_cids, keys = merge_terms([x.label for x in view.concepts])
    group_of = dict(zip(all_mids, all_cids, strict=True))
    mids = kept["mention_id"].to_list()
    concept_of_mid = {m: group_of[m] for m in mids}
    with postprocess.stored(i.label_groups, "terms.jsonl").open() as f:
        stored = {r["mention_id"]: r["group_id"] for r in map(json.loads, f)}
    check(
        "the Label Match rule recomputed here over all corpus terms = the stored Label Groups",
        stored == group_of,
        f"{sum(stored.get(m) != g for m, g in group_of.items()):,} of {len(group_of):,} rows differ",
    )
    members: dict[str, list[str]] = defaultdict(list)
    for m in mids:
        members[concept_of_mid[m]].append(m)
    vocab = label_match.vocabulary([x.label for x in view.concepts])
    traced = {m: label_match.trace(concept_of[m].label, vocab) for m in mids}
    normalisation = {m: t["steps"] for m, t in traced.items()}

    def joins(ms: list[str]) -> list[dict]:
        """Every key that two or more distinct spellings of the concept share, as groups.jsonl lists them."""
        by: dict[tuple[str, str], set[str]] = defaultdict(set)
        for m in ms:
            by[("word-set", traced[m]["word_set_key"])].add(concept_of[m].label)
            by[("spacing", traced[m]["spacing_key"])].add(concept_of[m].label)
        return [
            dict(key_type=t, key=k, labels=sorted(v))
            for (t, k), v in sorted(by.items())
            if len(v) > 1
        ]

    joined_by = {cid: joins(ms) for cid, ms in members.items()}
    likely_wrong_rows = json.loads((i.recorded / LIKELY_WRONG).read_text())
    likely_wrong = {r["group_id"] for r in likely_wrong_rows}
    flags: dict[str, list[str]] = {cid: [] for cid in members}
    tms = json.loads((i.term_merging / "summary.json").read_text())
    theirs, their_labels, their_defs, their_norm, their_keys, their_joins = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    with postprocess.stored(i.term_merging, "concepts.jsonl").open() as f:
        for r in map(json.loads, f):
            their_labels[r["concept_id"]] = set(r["labels"])
            flags[r["concept_id"]] = list(r.get("merge_flags") or [])
            their_joins[r["concept_id"]] = {
                (j["key_type"], j["key"], tuple(sorted(j["labels"])))
                for j in r["joined_by"]
            }
            for x in r["instances"]:
                theirs[x["mention_id"]] = r["concept_id"]
                their_defs[x["mention_id"]] = sorted(x["definitions"])
                their_norm[x["mention_id"]] = x.get("normalisation")
                their_keys[x["mention_id"]] = (
                    x.get("word_set_key"),
                    x.get("spacing_key"),
                )
    diff = sum(theirs.get(m) != concept_of_mid[m] for m in mids)
    check(
        "concepts here = the stored term merging (concepts.jsonl)",
        not diff and set(theirs) == set(mids),
        f"{diff:,} kept rows in another concept; {len(set(theirs) - set(mids)):,} rows there that are not kept; "
        f"{len(set(mids) - set(theirs)):,} kept rows missing there",
    )
    check(
        "each concept's spellings = the stored term merging's",
        all(
            their_labels.get(cid) == {concept_of[m].label for m in ms}
            for cid, ms in members.items()
        ),
    )
    check(
        "each kept row's definition texts = the stored term merging's",
        all(
            their_defs.get(m) == sorted(d.text for d in concept_of[m].definitions)
            for m in mids
        ),
    )
    nd = sum(their_norm.get(m) != normalisation[m] for m in mids)
    check(
        "each kept row's normalisation steps (recomputed with label_match.trace) = the stored term merging's",
        not nd,
        f"{nd:,} rows differ",
    )
    kd = sum(
        their_keys.get(m) != (traced[m]["word_set_key"], traced[m]["spacing_key"])
        for m in mids
    )
    check(
        "each kept row's word-set and spacing keys = the stored term merging's",
        not kd,
        f"{kd:,} rows differ",
    )
    jd = sum(
        their_joins.get(cid)
        != {(j["key_type"], j["key"], tuple(j["labels"])) for j in js}
        for cid, js in joined_by.items()
    )
    check(
        "each concept's joined_by (recomputed here) = the stored term merging's",
        not jd,
        f"{jd:,} concepts differ",
    )
    leaked = sum(
        bool(
            {lab for j in js for lab in j["labels"]}
            - {concept_of[m].label for m in members[cid]}
        )
        for cid, js in joined_by.items()
    )
    check(
        "joined_by names only spellings of kept rows",
        not leaked,
        f"{leaked} concepts name another spelling",
    )
    mine = dict(
        concepts=len(members),
        terms=len(mids),
        papers=len({concept_of[m].paper_id for m in mids}),
        definitions=sum(len(concept_of[m].definitions) for m in mids),
    )
    theirs_n = dict(
        concepts=tms["after"]["concepts"],
        **{k: tms["kept"][k] for k in ("terms", "papers", "definitions")},
    )
    check(
        "counts = the term merging's summary", mine == theirs_n, f"{mine} vs {theirs_n}"
    )
    for cid in members:
        if cid in likely_wrong:
            flags[cid].append(LIKELY_WRONG_FLAG)
    alone, _ = merge_terms([concept_of[m].label for m in mids])
    by_alone, by_concept = defaultdict(set), defaultdict(set)
    for m, a in zip(mids, alone, strict=True):
        by_alone[a].add(concept_of_mid[m])
        by_concept[concept_of_mid[m]].add(a)
    joined = [sorted(v) for v in by_alone.values() if len(v) > 1]
    parted = [k for k, v in by_concept.items() if len(v) > 1]
    check(
        "information: the Label Match rule over the kept rows alone",
        "INFO",
        f"{len(by_alone):,} groups against {len(members):,} concepts; {len(joined)} groups there join concepts; "
        f"{len(parted)} concepts are split there "
        f"({'; '.join(', '.join(sorted({concept_of[m].label for m in members[k]})) for k in parted[:5])})",
    )

    # the Label Groups before Step 5 (all corpus terms) by what Step 5 did to them
    keep_of = dict(zip(dec["mention_id"], dec["keep"], strict=True))
    fate = defaultdict(set)
    for m, g in group_of.items():
        fate[g].add(bool(keep_of[m]))
    numbers = json.loads((i.recorded / CARD_NUMBERS).read_text())
    audit = numbers["merge_audit"]
    merge_info = dict(
        all_groups=len(fate),
        groups_all_kept=sum(v == {True} for v in fate.values()),
        groups_all_dropped=sum(v == {False} for v in fate.values()),
        groups_split=sum(len(v) == 2 for v in fate.values()),
        n_likely_wrong=len(likely_wrong_rows),
        likely_wrong_kept=sum(r["kept_terms"] for r in likely_wrong_rows),
        likely_wrong_terms=sum(r["terms"] for r in likely_wrong_rows),
        likely_wrong_concepts=sum(LIKELY_WRONG_FLAG in v for v in flags.values()),
        missed_joins=audit["missed_pairs"],
        missed_kept=audit["missed_kept_terms"],
        heads_differ=audit["word_order_heads_differ"],
        order_correct=audit["word_order_heads_differ"] - len(likely_wrong_rows),
        flagged_concepts=sum(bool(v) for v in flags.values()),
    )
    check(
        "group counts before Step 5 = the term merging's summary",
        (merge_info["all_groups"], merge_info["groups_split"])
        == (tms["before"]["groups"], tms["split_groups"]),
        f"{merge_info['all_groups']:,} / {merge_info['groups_split']:,} vs {tms['before']['groups']:,} / "
        f"{tms['split_groups']:,}",
    )
    if "label_groups" in c:
        check(
            "information: the decisions json's group counts",
            "INFO",
            f"{c['label_groups']} (counted on the Label Groups that existed when the decisions were written)",
        )

    # 4. tables
    score_cols = [
        "ensemble_score",
        *[f"{m}_score" for m in MEMBERS],
        *[f"{m}_ref_rank" for m in MEMBERS],
    ]
    strict_cut = dec_json["best_f1_cut"]  # the best-F1 cut marks `strict`
    scores = {
        r["mention_id"]: r
        for r in kept.select(["mention_id", *score_cols]).iter_rows(named=True)
    }
    inst_rows, def_rows = [], []
    all_def_ids = {d.def_id for m in mids for d in concept_of[m].definitions}
    cites = corpus_store.definition_citations(i.store, all_def_ids)
    extra = corpus_store.extracted_items(
        i.store, set(mids), ("definitions", *EXTRA_FIELDS)
    )
    moved = corpus_store.first_copy_citation_changes(i.store, extra)
    check(
        "information: items whose citations are P≥28's own and differ from the store's first-ingested copy",
        "INFO",
        f"{sum(moved.values()):,} items: "
        + ", ".join(f"{f} {n:,}" for f, n in moved.most_common()),
    )
    td = numbers["train_dev"]
    check(
        "the decisions' cut = the recorded 95%-recall cut on the reference scale",
        abs(td["cut_95"] - dec_json["cut"]) < 1e-12,
        f"{td['cut_95']:.6f} vs {dec_json['cut']:.6f}",
    )
    for m in mids:
        x = concept_of[m]
        inst_rows.append(
            dict(
                mention_id=m,
                concept_id=concept_of_mid[m],
                term=x.label,
                word_set_key=traced[m]["word_set_key"],
                spacing_key=traced[m]["spacing_key"],
                normalisation=normalisation[m],
                paper_id=x.paper_id,
                strict=scores[m]["ensemble_score"] >= strict_cut,
                corpus=x.corpus,
                n_definitions=len(x.definitions),
                **{k: scores[m][k] for k in score_cols},
                **{
                    f: [
                        dict(text=it["text"], citations=it["citations"])
                        for it in extra.get(m, {}).get(f, [])
                    ]
                    for f in EXTRA_FIELDS
                },
            )
        )
        for d in x.definitions:
            def_rows.append(
                dict(
                    def_id=d.def_id,
                    mention_id=m,
                    concept_id=concept_of_mid[m],
                    paper_id=x.paper_id,
                    term=x.label,
                    text=d.text,
                    citations=cites.get(d.def_id, []),
                )
            )
    strict_of = {r["mention_id"]: r["strict"] for r in inst_rows}
    con_rows: list[dict] = []
    for cid, ms in members.items():
        xs = [concept_of[m] for m in ms]
        spellings = [s for s, _ in Counter(x.label for x in xs).most_common()]
        con_rows.append(
            dict(
                concept_id=cid,
                label=spellings[0],
                labels=spellings,
                match_key=keys[cid],
                n_instances=len(xs),
                n_papers=len({x.paper_id for x in xs}),
                n_definitions=sum(len(x.definitions) for x in xs),
                merge_flags=flags[cid],
                joined_by=joined_by[cid],
                mention_ids=ms,
                n_strict=sum(bool(strict_of[m]) for m in ms),
            )
        )
    con_rows.sort(key=lambda r: (-r["n_definitions"], r["concept_id"]))
    works = openalex_works(i.recorded / OPENALEX)
    view_corpus = {x.paper_id: x.corpus for x in view.concepts}
    pap_rows, year_source = paper_table(
        {x["paper_id"] for x in inst_rows}, view_corpus, works, i.corpus
    )
    n_inst, n_def = (
        Counter(x["paper_id"] for x in inst_rows),
        Counter(x["paper_id"] for x in def_rows),
    )
    for r in pap_rows:
        r.update(n_instances=n_inst[r["paper_id"]], n_definitions=n_def[r["paper_id"]])
    T = dict(
        concepts=frame(con_rows, "concepts"),
        instances=frame(inst_rows, "instances"),
        definitions=frame(def_rows, "definitions"),
        papers=frame(pap_rows, "papers"),
    )
    term_rows, term_info = terms_view(
        T, {x.mention_id: x.label for x in view.concepts}, group_of, vocab
    )
    T["terms"] = frame(term_rows, "terms")

    # 5. checks on the tables
    check(
        "definitions = the decisions json's kept definitions",
        T["definitions"].height == c["definitions"]["kept"],
        f"{T['definitions'].height:,} vs {c['definitions']['kept']:,}",
    )
    check(
        "papers = the decisions json's kept papers",
        T["papers"].height == c["papers"]["kept"],
        f"{T['papers'].height:,} vs {c['papers']['kept']:,}",
    )
    check(
        "definitions = the sum of n_definitions over instances",
        T["definitions"].height == int(T["instances"]["n_definitions"].sum()),
    )
    check(
        "instances = the sum of n_instances over concepts",
        T["instances"].height == int(T["concepts"]["n_instances"].sum()),
    )
    for name, col in (
        ("concepts", "concept_id"),
        ("instances", "mention_id"),
        ("definitions", "def_id"),
        ("papers", "paper_id"),
    ):
        check(f"{col} unique in {name}", T[name][col].n_unique() == T[name].height)
    dropped = set(dec.filter(~pl.col("keep"))["mention_id"])
    leaked_rows = (
        set(T["instances"]["mention_id"])
        | set(T["definitions"]["mention_id"])
        | {m for ms in T["concepts"]["mention_ids"] for m in ms}
    ) & dropped
    check(
        "no dropped term-in-paper row appears in any table",
        not leaked_rows,
        f"{len(leaked_rows)} found",
    )
    check(
        "every definition's term-in-paper is in instances",
        set(T["definitions"]["mention_id"]) <= set(T["instances"]["mention_id"]),
    )
    check(
        "every instance's concept is in concepts",
        set(T["instances"]["concept_id"]) <= set(T["concepts"]["concept_id"]),
    )
    check(
        "every instance's paper is in papers, every paper has an instance",
        set(T["instances"]["paper_id"]) == set(T["papers"]["paper_id"]),
    )
    check(
        "no empty definition text",
        T["definitions"].filter(pl.col("text").str.strip_chars() == "").height == 0,
    )
    # the other extracted fields are read by the same rule as the definitions, so that rule must give back the definitions table, and the item counts must equal an independent count in SQL
    same_defs = sum(
        sorted(
            (it["text"], tuple(it["citations"]))
            for it in extra.get(m, {}).get("definitions", [])
        )
        != sorted(
            (d.text, tuple(cites.get(d.def_id, []))) for d in concept_of[m].definitions
        )
        for m in mids
    )
    check(
        "the extra fields' reader gives back every kept row's definitions and citations",
        not same_defs,
        f"{same_defs:,} rows differ",
    )
    sql = corpus_store.field_counts(i.store, set(mids))
    have = {f: int(T["instances"][f].list.len().sum()) for f in EXTRA_FIELDS}
    check(
        "items per extra field = an independent count in the store (P≥28 sources of the kept rows)",
        all(have[f] == sql[f] for f in EXTRA_FIELDS)
        and sql["definitions"] == T["definitions"].height,
        {f: f"{have[f]:,} vs {sql[f]:,}" for f in EXTRA_FIELDS},
    )
    empty = {
        f: int(
            T["instances"]
            .select(
                pl.col(f)
                .list.eval(pl.element().struct.field("text").str.strip_chars() == "")
                .list.any()
                .sum()
            )
            .item()
        )
        for f in EXTRA_FIELDS
    }
    check("no empty text in the extra fields", not any(empty.values()), empty)
    # the terms view: every item of every row lands in exactly one merged item of its own concept
    tv = T["terms"]
    check(
        "terms: one row per concept, the same ids",
        tv["concept_id"].to_list() == T["concepts"]["concept_id"].to_list(),
    )
    want = {
        "definitions": T["definitions"].height,
        **{f: int(T["instances"][f].list.len().sum()) for f in EXTRA_FIELDS},
    }
    got = {
        f: int(
            tv.select(
                pl.col(f)
                .list.eval(pl.element().struct.field("sources").list.len())
                .list.sum()
                .sum()
            ).item()
        )
        for f in TERM_FIELDS
    }
    check(
        "terms: every extracted item is one source of one merged item (sources = items per field)",
        got == want,
        {f: f"{got[f]:,} vs {want[f]:,}" for f in TERM_FIELDS},
    )
    own = dict(
        zip(T["instances"]["mention_id"], T["instances"]["concept_id"], strict=True)
    )
    stray, def_seen = 0, Counter()
    for r in tv.select("concept_id", *TERM_FIELDS).iter_rows(named=True):
        for f in TERM_FIELDS:
            for it in r[f]:
                for sr in it["sources"]:
                    stray += own[sr["mention_id"]] != r["concept_id"]
                    if f == "definitions":
                        def_seen[sr["def_id"]] += 1
                stray += it["papers"] != len({sr["paper_id"] for sr in it["sources"]})
    check(
        "terms: every source belongs to its concept, and each item's papers = its sources' distinct papers",
        not stray,
        f"{stray} problems",
    )
    check(
        "terms: every definition appears once",
        set(def_seen) == set(T["definitions"]["def_id"])
        and max(def_seen.values(), default=1) == 1,
    )
    linked = {
        it["links_to"]["concept_id"]
        for f in NAME_FIELDS
        for its in tv[f].to_list()
        for it in its
        if it["links_to"]
    }
    check(
        "terms: every linked concept exists",
        linked <= set(T["concepts"]["concept_id"]),
        f"{len(linked):,} concepts linked",
    )
    check(
        "terms: strict = the concept has a strict row",
        tv["strict"].to_list() == (T["concepts"]["n_strict"] > 0).to_list(),
    )
    nulls = {col: T["instances"][col].null_count() for col in score_cols}
    check("no missing score in instances", not any(nulls.values()), nulls)
    cut = dec_json["cut"]
    check(
        "every kept row's ensemble score is at or above the cut",
        bool(T["instances"]["ensemble_score"].min() >= cut),
        f"min {T['instances']['ensemble_score'].min():.5f}, cut {cut:.5f}",
    )
    inst = T["instances"]
    n_strict = inst.filter(pl.col("strict")).height
    want_strict = dec_json["comparison"]["(a) reference ranks, best-F1 cut"]["terms"]
    check(
        "strict rows = the decisions json's count at the best-F1 cut",
        n_strict == want_strict,
        f"{n_strict:,} vs {want_strict:,}",
    )
    check(
        "strict exactly when the ensemble score is at or above the best-F1 cut",
        inst.filter(pl.col("strict") != (pl.col("ensemble_score") >= strict_cut)).height
        == 0,
    )
    check(
        "the decisions' best-F1 cut = the recorded best-F1 cut on the reference scale",
        abs(td["best_f1_cut"] - strict_cut) < 1e-12,
        f"{td['best_f1_cut']:.6f} vs {strict_cut:.6f}",
    )
    check(
        "each concept's n_strict = its strict rows",
        int(T["concepts"]["n_strict"].sum()) == n_strict,
    )
    for name, df in T.items():
        check(
            f"{name}: columns and types as specified",
            dict(df.schema) == SCHEMA[name],
            f"{list(df.schema)}",
        )
        bad = [col for col in df.columns if col.lower() in PRIVATE]
        check(f"{name}: no private column", not bad, bad)
    emails = {}
    for name, df in T.items():
        n = 0
        for col, dt in df.schema.items():
            if dt == pl.String:
                n += int(df[col].drop_nulls().str.contains(EMAIL.pattern).sum())
        emails[name] = int(n)
    check(
        "e-mail-like strings in text columns (informational: quotes may contain one)",
        "INFO",
        emails,
    )

    # 6. write the tables, the card, provenance; check the card
    if ds.exists():
        shutil.rmtree(ds)
    ds.mkdir(parents=True)
    written = {}
    for name in CORPUS_TABLES:
        written[name] = f"data/{name}.parquet"
        write_parquet(T[name], ds / written[name])
    pick = json.loads(i.pick.read_text())
    n = card.numbers(
        T,
        dec_json,
        numbers,
        pick,
        gold.train_dev_relevant(i.ground_truth, i.non_relevant),
        year_source,
        version=version,
        licence=licence,
        repo_id=repo_id,
        gold_repo_id=gold_repo_id,
    )
    n["banner"] = card.add_banner(ds, banner)
    n["explorer_files"] = explorer_data(ds, explorer, check) if explorer else ""
    corpus_of = corpus_store.corpus_of(i.store)
    view_info = dict(
        view_papers=len(view.papers),
        view_seed_papers=sum(corpus_of[p] == "seeds" for p in view.papers),
        view_closure_papers=sum(corpus_of[p] == "closure" for p in view.papers),
        definition_items=corpus_store.p28_copies(
            i.store, {d.def_id for x in view.concepts for d in x.definitions}
        ),
    )
    view_info["repeated_items"] = (
        view_info["definition_items"] - c["definitions"]["total"]
    )
    pages = card.render(
        n | merge_info | view_info | card.terms_numbers(term_info),
        written,
        T,
        pick,
        licence,
    )
    for name, text in pages.items():
        (ds / name).parent.mkdir(parents=True, exist_ok=True)
        (ds / name).write_text(text)
    head = yaml.safe_load(pages["README.md"].split("---\n")[1])
    paths = [f["path"] for cfg in head["configs"] for f in cfg["data_files"]]
    check(
        "card: front matter parses and every data file it names exists",
        all((ds / p).exists() for p in paths),
        paths,
    )
    left = [x for text in pages.values() for x in re.findall(r"\{\{\w+\}\}", text)]
    check("card: no unfilled placeholder", not left, left)
    used = [
        i.decisions,
        i.decisions_json,
        i.store,
        i.links,
        i.corpus / "seeds/papers.csv",
        i.corpus / "closure/papers.csv",
        i.corpus / "links/cited-works.csv",
        i.recorded / OPENALEX,
        i.recorded / LIKELY_WRONG,
        i.recorded / CARD_NUMBERS,
        i.pick,
        postprocess.stored(i.label_groups, "terms.jsonl"),
        postprocess.stored(i.label_groups, "run.json"),
        postprocess.stored(i.term_merging, "concepts.jsonl"),
        i.term_merging / "summary.json",
        i.ground_truth,
        i.non_relevant,
        *([banner] if banner else []),
    ]
    outputs: dict[str, dict] = {
        k: {kk: vv for kk, vv in file_record(ds / v).items() if kk != "path"}
        | {"path": v, "rows": pl.scan_parquet(ds / v).select(pl.len()).collect().item()}
        for k, v in written.items()
    }
    provenance = dict(
        dataset="MBDef",
        version=version,
        command="mbdef dataset build",
        settings=dict(
            version=version,
            licence=licence,
            repo_id=repo_id,
            ground_truth_repo_id=gold_repo_id,
        ),
        code=code_record(),
        decisions=dict(
            text="every definition text word for word",
            context="no paragraph context or other text of the papers",
            rows="only the kept term-in-paper rows",
            concepts="spellings of a term merged into one term, over all corpus terms, kept rows only",
            ground_truth="a separate dataset, MBDef-Gold",
            licence=licence,
        ),
        rule=MERGE_RULE,
        inputs=[file_record(p) for p in used],
        outputs=outputs,
    )
    if explorer:
        sums = ds / "explorer/SHA256SUMS"
        provenance["explorer"] = dict(
            files=len(sums.read_text().splitlines()),
            sha256sums=files.file_sha256(sums),
        )
    files.write_json(ds / "provenance.json", provenance, indent=1)
    hits = absolute_paths(ds)
    check(
        "no absolute path anywhere in the dataset (card, provenance, every parquet column)",
        not hits,
        hits[:5],
    )
    if same_data_as:
        mine = {v: outputs[k]["sha256"] for k, v in written.items()}
        differ = differing(same_data_as, DATASET, mine)
        check(
            f"data files byte-identical to the hashes in {portable(same_data_as)}",
            not differ,
            ", ".join(differ) or "all equal",
        )
        others = not_written(same_data_as, DATASET, mine)
        if others:
            check(
                "information: files the listing records that this build does not write",
                "INFO",
                ", ".join(others),
            )
    return report(out, DATASET, "validation", version, checks, outputs, t0)
