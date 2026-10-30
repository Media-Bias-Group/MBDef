"""The release folder of the closure: seed + closure definitions in all four variants, the papers, their TEI texts, and the table of which seed definition cites which closure paper, with a plain-language README.

Every input is checked before it is copied: variant files against the SHA-256 in their DONE.json, TEI files against the SHA-256 in the extraction rows (runner-rows.json / rows-all.json). The README's numbers are computed from the copied files. MANIFEST.sha256 lists every file of the release (`finish`), and `verify` checks them all.

`closure bundle` builds the released closure (v1.1) with these functions; the swap points are the folders and the version name. The README and the other files keep the released bytes (apart from the build time, `built_utc`).
"""
# ruff: noqa: E501

import collections
import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote

from mbdef import files
from mbdef.closure.definition_citations import (
    nfkc_words,
)  # the key definition_citations.py counts n_citing_definitions with
from mbdef.extract.derive import VARIANTS


def counts(predictions: dict) -> dict:
    """Papers, definition records and definitions of one predictions.json, counted as DONE.json counts them."""
    recs = [r for p in predictions["papers"] for r in p["definition_records"]]
    return {
        "papers": len(predictions["papers"]),
        "records": len(recs),
        "definitions": sum(len(r.get("definitions") or []) for r in recs),
    }


def copy_derived(
    src: Path, dst: Path, leave_out: set[str] | frozenset[str] = frozenset()
) -> dict:
    """Copy a derived/ folder (DONE.json, one folder per variant, union) after checking every variant file.

    With `leave_out`, those papers are removed from every variant and from the union; the variant files are recounted, the counts of the unfiltered files are first checked against DONE.json, and a new DONE.json records the filtering (the run's own DONE.json is kept as DONE-unfiltered.json)."""
    done = json.loads((src / "DONE.json").read_text())
    out = {
        "papers": done["papers"],
        "created_utc": done["created_utc"],
        "variants": {},
        "left_out": len(leave_out),
    }
    for tag, name in VARIANTS.items():
        pick = done["picks"][name]
        f = src / tag / "predictions.json"
        if files.file_sha256(f) != pick["predictions_sha256"]:
            raise SystemExit(f"{f}: SHA-256 differs from DONE.json")
        if pick["missing_part_outputs"] != 0:
            raise SystemExit(f"{name}: missing part outputs")
        out["variants"][tag] = {
            "definitions": pick["definitions"],
            "records": pick["records"],
            "papers": pick["papers"],
        }
    if not leave_out:
        for part in ["DONE.json", *VARIANTS, "union"]:
            s = src / part
            if s.is_dir():
                shutil.copytree(s, dst / part, dirs_exist_ok=True)
            elif s.is_file():
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, dst / part)
        return out
    new = json.loads(json.dumps(done))
    left = sorted(leave_out)
    for tag in [*VARIANTS, "union"]:
        f = src / tag / "predictions.json"
        pred = json.loads(f.read_text())
        if tag in VARIANTS:
            c = counts(pred)
            pick = done["picks"][VARIANTS[tag]]
            assert (c["records"], c["definitions"], c["papers"]) == (
                pick["records"],
                pick["definitions"],
                pick["papers"],
            ), (tag, c)
        assert all(pid in {p["paper_id"] for p in pred["papers"]} for pid in left), (
            f"{tag}: a left-out paper is not in the set"
        )
        pred["papers"] = [p for p in pred["papers"] if p["paper_id"] not in leave_out]
        if "paper_count" in pred:
            pred["paper_count"] = len(pred["papers"])
        target = dst / tag / "predictions.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(pred, ensure_ascii=False) + "\n", encoding="utf-8")
        (dst / tag / "papers").mkdir(exist_ok=True)
        for pf in (src / tag / "papers").glob("*.json"):
            if pf.stem not in leave_out and unquote(pf.stem) not in leave_out:
                shutil.copy2(pf, dst / tag / "papers" / pf.name)
        if tag in VARIANTS:
            c = counts(pred)
            new["picks"][VARIANTS[tag]].update(
                papers=c["papers"],
                records=c["records"],
                definitions=c["definitions"],
                predictions_sha256=files.file_sha256(target),
            )
            out["variants"][tag] = {
                "definitions": c["definitions"],
                "records": c["records"],
                "papers": c["papers"],
            }
        elif isinstance(new.get("union"), dict):
            # the union's papers and records are recounted too, not only its hash
            c = counts(pred)
            new["union"].update(
                papers=c["papers"],
                records=c["records"],
                predictions_sha256=files.file_sha256(target),
            )
    new["papers"] = done["papers"] - len(left)
    new["filtered"] = {
        "from_done_sha256": files.file_sha256(src / "DONE.json"),
        "left_out": left,
        "why": "same text as a seed paper or as another closure paper (provenance/closure/exclusions.json)",
    }
    (dst / "DONE.json").write_text(
        json.dumps(new, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    shutil.copy2(src / "DONE.json", dst / "DONE-unfiltered.json")
    out["papers"] = new["papers"]
    return out


def copy_tei(rows: list[dict], find, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    for r in rows:
        src = find(r)
        if not (src and src.is_file()):
            raise SystemExit(f"no TEI for {r['paper_id']}")
        if files.file_sha256(src) != r["tei_sha256"]:
            raise SystemExit(f"{src}: SHA-256 differs from the extraction rows")
        shutil.copy2(src, dst / f"{r['paper_id']}.tei.xml")
    return len(rows)


def definition_counts(derived: Path) -> dict:
    """paper_id -> {variant: number of definitions}."""
    counts = collections.defaultdict(dict)
    for tag in VARIANTS:
        for p in json.loads((derived / tag / "predictions.json").read_text())["papers"]:
            counts[p["paper_id"]][tag] = sum(
                len(r.get("definitions") or []) for r in p["definition_records"]
            )
    return counts


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build(
    *,
    out: Path,
    closure_set: Path,
    seeds_derived: Path,
    seeds_rows: Path,
    seed_tei: Path,
    closure_tei: Path,
    links: Path,
    version: str,
    exclusions: Path | None = None,
    seeds_compare: Path | None = None,
    extra: tuple[Path, ...] = (),
    built_utc: str | None = None,
) -> dict:
    """Copy and check every part of the release into OUT (a new folder) and return its facts (not yet its README)."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; choose a new folder")
    C = Path(closure_set)
    facts = {
        "version": version,
        "built_utc": built_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # 1. Definitions (seed and closure), checked against DONE.json.
    facts["seeds"] = copy_derived(Path(seeds_derived), out / "seeds" / "derived")
    left = json.loads(Path(exclusions).read_text()) if exclusions else []
    # Only papers in the closure set are filtered; a left-out copy that was never added (its text was found in the set before its batch) stays listed in exclusions.json.
    in_set = {r["paper_id"] for r in json.loads((C / "rows-all.json").read_text())}
    leave_out = {e["paper_id"] for e in left} & in_set
    facts["never_added"] = len({e["paper_id"] for e in left} - in_set)
    facts["closure"] = copy_derived(
        C / "derived", out / "closure" / "derived", leave_out
    )

    # 2. TEI texts, checked against the extraction rows.
    seed_rows = json.loads(Path(seeds_rows).read_text())
    closure_rows = [
        r
        for r in json.loads((C / "rows-all.json").read_text())
        if r["paper_id"] not in leave_out
    ]
    assert (
        len(seed_rows) == facts["seeds"]["papers"]
        and len(closure_rows) == facts["closure"]["papers"]
    )
    tei_index = {p.name: p for p in Path(closure_tei).rglob("*.tei.xml")}
    copy_tei(
        seed_rows,
        lambda r: Path(seed_tei) / Path(r["tei_path"]).name,
        out / "seeds" / "tei",
    )
    copy_tei(
        closure_rows,
        lambda r: tei_index.get(Path(r["tei_path"]).name),
        out / "closure" / "tei",
    )

    # 3. Paper tables.
    seed_counts, closure_counts = (
        definition_counts(out / "seeds" / "derived"),
        definition_counts(out / "closure" / "derived"),
    )
    seed_meta = {
        p["paper_id"]: p.get("metadata") or {}
        for p in json.loads(
            (out / "seeds" / "derived" / "union" / "predictions.json").read_text()
        )["papers"]
    }
    # A closure paper can be the PDF of several cited works the grouping kept apart (e.g. 16 DOI-less citations of Mikolov et al. 2013), so its citing counts are taken over all of them, with the per-work definition key.
    works = collections.defaultdict(list)
    with (Path(links) / "cited-works.csv").open(newline="", encoding="utf-8") as f:
        for w in csv.DictReader(f):
            for pid in filter(None, w["closure_paper_ids"].split("; ")):
                works[pid].append(w)
    link_defs = collections.defaultdict(set)
    with (Path(links) / "definition-links.csv").open(newline="", encoding="utf-8") as f:
        for x in csv.DictReader(f):
            if x["closure_paper_id"] and x["field"] == "definitions":
                link_defs[x["closure_paper_id"]].add(
                    (
                        x["seed_paper_id"],
                        nfkc_words(x["term"]),
                        x["field"],
                        nfkc_words(x["text"]),
                    )
                )
    for pid, ws in works.items():
        if len(ws) == 1:
            assert len(link_defs[pid]) == int(ws[0]["n_citing_definitions"]), pid
    meta = json.loads((C / "paper-metadata.json").read_text())
    batch_of = {}
    for b in json.loads((C / "batches.json").read_text()):
        for shard in b["shards"]:
            batch_of[shard] = b["batch"]
    receipts, identity = {}, collections.Counter()
    for r in closure_rows:
        rec = tei_index[Path(r["tei_path"]).name].with_name(
            Path(r["tei_path"]).name[: -len(".tei.xml")] + ".tei.closure.json"
        )
        receipts[r["paper_id"]] = json.loads(rec.read_text()) if rec.is_file() else {}
    # The seed DOI comes from the seed collection's metadata; a DOI several seed papers carry is wrong for all but at most one of them (in the released seed set: 73 DOIs on 224 seed papers, e.g. BERT's on 11), so it is marked, not trusted.
    seed_doi = {
        pid: (m.get("doi") or "").strip().lower() for pid, m in seed_meta.items()
    }
    doi_count = collections.Counter(d for d in seed_doi.values() if d)
    seed_fields = [
        "paper_id",
        "doi",
        "doi_shared_by_seed_papers",
        "title",
        *[f"definitions_{t}" for t in VARIANTS],
    ]
    write_csv(
        out / "seeds" / "papers.csv",
        [
            {
                "paper_id": r["paper_id"],
                "doi": seed_meta.get(r["paper_id"], {}).get("doi", ""),
                "doi_shared_by_seed_papers": doi_count.get(
                    seed_doi.get(r["paper_id"], ""), 0
                )
                if seed_doi.get(r["paper_id"])
                else "",
                "title": r["frozen_title"],
                **{
                    f"definitions_{t}": seed_counts[r["paper_id"]].get(t, 0)
                    for t in VARIANTS
                },
            }
            for r in seed_rows
        ],
        seed_fields,
    )
    facts["seed_dois_shared"] = {
        "dois": sum(1 for n in doi_count.values() if n > 1),
        "papers": sum(n for n in doi_count.values() if n > 1),
    }
    closure_fields = [
        "paper_id",
        "doi",
        "title",
        "openalex_id",
        "cited_work_key",
        "pdf_identity_check",
        "pdf_sha256",
        "n_citing_seed_papers",
        "n_citing_seed_definitions",
        *[f"definitions_{t}" for t in VARIANTS],
    ]
    rows = []
    for r in closure_rows:
        pid, m, rc, ws = (
            r["paper_id"],
            meta.get(r["paper_id"], {}),
            receipts[r["paper_id"]],
            works.get(r["paper_id"], []),
        )
        identity[rc.get("identity_match", "")] += 1
        citing = {s for w in ws for s in w["citing_seed_papers"].split("; ") if s}
        rows.append(
            {
                "paper_id": pid,
                "doi": m.get("doi", ""),
                "title": r["frozen_title"],
                "openalex_id": m.get("openalex_id", ""),
                "cited_work_key": "; ".join(w["work_key"] for w in ws),
                "pdf_identity_check": rc.get("identity_match", ""),
                "pdf_sha256": rc.get("pdf_sha256", ""),
                "n_citing_seed_papers": len(citing) if ws else "",
                "n_citing_seed_definitions": len(link_defs[pid]) if ws else "",
                **{f"definitions_{t}": closure_counts[pid].get(t, 0) for t in VARIANTS},
            }
        )
    write_csv(out / "closure" / "papers.csv", rows, closure_fields)
    facts["closure_identity"] = dict(identity.most_common())
    facts["closure_without_citing_work"] = sum(
        1 for r in rows if not r["cited_work_key"]
    )
    facts["closure_cited_by_a_definition"] = sum(
        1 for r in rows if str(r["n_citing_seed_definitions"]) not in ("", "0")
    )
    with (Path(links) / "definition-links.csv").open(newline="", encoding="utf-8") as f:
        with_row = {
            x["closure_paper_id"] for x in csv.DictReader(f) if x["closure_paper_id"]
        }
    facts["closure_without_link_row"] = sum(
        1 for r in rows if r["paper_id"] not in with_row
    )

    # 4. Citation links and provenance records.
    (out / "links").mkdir()
    for f in (
        "definition-links.csv",
        "cited-works.csv",
        "definition-links-summary.json",
    ):
        shutil.copy2(Path(links) / f, out / "links" / f)
    facts["links"] = json.loads(
        (Path(links) / "definition-links-summary.json").read_text()
    )
    prov = out / "provenance"
    (prov / "closure").mkdir(parents=True)
    for f in ("batches.json", "set.json", "picks.json", "paper-metadata.json"):
        shutil.copy2(C / f, prov / "closure" / f)
    shutil.copytree(C / "compare", prov / "closure" / "compare")
    if exclusions:
        shutil.copy2(exclusions, prov / "closure" / "exclusions.json")
    if seeds_compare:
        shutil.copytree(Path(seeds_compare), prov / "seeds" / "compare")
    for file in extra:
        shutil.copy2(file, prov / Path(file).name)
    return facts


def readme(facts: dict) -> str:
    s, c, L = facts["seeds"], facts["closure"], facts["links"]
    total = {
        t: s["variants"][t]["definitions"] + c["variants"][t]["definitions"]
        for t in VARIANTS
    }
    st = L["work_status"]
    ident = facts["closure_identity"]
    words = {
        "doi": "the PDF carries the cited DOI",
        "title_year_author": "same title, year and a cited author",
        "title_author_year_missing": "same title and a cited author (the PDF shows no year)",
        "title_author_year_off_by_one": "same title and a cited author, year one off",
        "title_in_text": "the cited title and a cited author open the text",
        "llm_verified": "checked by a language model, confirmed by program checks",
    }
    ident_rows = "\n".join(f"| {words.get(k, k)} | {v:,} |" for k, v in ident.items())
    names = {
        "P20": "P≥20 (most complete)",
        "P24": "P≥24",
        "P28": "P≥28 (most precise)",
        "P28-single-model": "P≥28, one model only",
    }
    var_rows = "\n".join(
        f"| {names[t]} | {s['variants'][t]['definitions']:,} | {c['variants'][t]['definitions']:,} | {total[t]:,} |"
        for t in VARIANTS
    )
    return f"""# Media-bias definition corpus, {facts["version"]}

Built {facts["built_utc"]}. Everything in this folder was produced by the MBDef pipeline and checked
against the run's own records before it was copied (see "Checks" at the end).

## What is inside

Two sets of papers, each with the definitions extracted from them:

- **Seed papers** ({s["papers"]:,}): the media-bias literature collection the project started from.
- **Closure papers** ({c["papers"]:,}): works the seed papers cite in what was extracted from them (a definition, or
  its description, examples, synonyms, related terms, super- or subtypes), found and downloaded as open-access PDFs.
  {facts["closure_cited_by_a_definition"]:,} of them are cited in a definition sentence itself; the other
  {c["papers"] - facts["closure_cited_by_a_definition"]:,} only next to one. Only one level deep: the closure papers' own
  citations were not followed. {c.get("left_out", 0):,} collected papers are left out: their text is a seed paper's or another
  closure paper's, or no seed reference cites them (`provenance/closure/exclusions.json` gives the reason for each; the
  comparison of variants in `provenance/closure/compare` still counts them).

Each set comes in four **variants** of the same extraction. They differ in how strictly a candidate definition had
to pass checks, so they trade completeness against precision. The percentages below were measured on 29 held-out,
hand-annotated papers; they are somewhat optimistic because the variants were chosen on those papers.

| Variant | Seed definitions | Closure definitions | Together |
|---|---:|---:|---:|
{var_rows}

- **P≥20** reads each paper four times and keeps what a relevance filter accepts: it finds the most (about 87% of
  the hand-annotated definitions) but only about one in five of its definitions is one a human annotator listed.
- **P≥24** reads twice and lets the model re-check each of its own definitions.
- **P≥28** reads twice, filters, and has a second model (Qwen 3.6) verify every definition: fewer, more precise.
- **P≥28, one model only** is P≥28 without the second model.

The variants are nested views of the same model outputs, not independent extractions. How they were chosen is in
the repository at `docs/final-extractor-selection-and-production.md`; how the closure was built is in
`docs/citation-closure-walkthrough.md` and `docs/closure-loss-audit.md`.

## Folders and files

| Path | What it is |
|---|---|
| `seeds/derived/<variant>/predictions.json`, `closure/derived/<variant>/predictions.json` | the definitions: one entry per paper with `definition_records` (term, definitions, description, examples, synonyms, related terms, super- and subtypes, each with the citation strings the paper gives) |
| `…/derived/<variant>/papers/<paper_id>.json` | the same, one file per paper |
| `…/derived/union/predictions.json` | all four variants merged (every record any variant kept) |
| `…/derived/DONE.json` | the completion record: counts and SHA-256 of every variant file |
| `seeds/tei/`, `closure/tei/` | the full text of every paper (GROBID TEI XML) the extraction read, named by paper id |
| `seeds/papers.csv`, `closure/papers.csv` | one row per paper: id, DOI, title, definitions per variant; for closure papers also the OpenAlex id, how the PDF was confirmed to be the cited work, and how many seed papers and seed definitions cite it |
| `links/definition-links.csv` | **one row per citation**: a seed definition (paper, term, text), one of its citation strings, the reference-list entry it was linked to and how, the cited work, and the closure paper that work became (if it was collected) |
| `links/cited-works.csv` | one row per distinct cited work: what happened to it (collected, not open access, not identified, …) |
| `provenance/` | the run's own records: batches, comparison of variants, paper metadata |
| `MANIFEST.sha256` | SHA-256 of every file here |

To follow a definition to its source: find its row in `links/definition-links.csv` (join on `seed_paper_id`, `term`
and `text`), read `closure_paper_id`, then look that id up in `closure/derived/<variant>/papers/` and `closure/tei/`.
The `in_P20` … `in_P28_single_model` columns say which variants contain that seed definition.

## The citation links in numbers

- What was extracted from the seed papers carries {L["reference_occurrences"]:,} links from a citation string to a
  reference-list entry; they point to {L["works"]:,} distinct cited works.
- Of these works, {st.get("extracted", 0):,} were collected, confirmed and extracted: they are the {c["papers"]:,} closure
  papers (a few works that the citation grouping kept apart turned out to be the same PDF), {st.get("present_in_corpus", 0):,}
  were already seed papers, {st.get("resolved_no_oa_pdf", 0):,} were identified but have no open-access PDF, and
  {st.get("not_found", 0) + st.get("ambiguous", 0):,} could not be identified with certainty (books, reports, garbled or incomplete citations).
  For {st.get("oa_pdf_found_download_failed", 0):,} an open-access link was known but the download failed (mostly publisher bot
  protection); for {st.get("pdf_rejected_unconfirmed", 0) + st.get("pdf_rejected_other_work", 0) + st.get("pdf_unusable", 0):,} a PDF was
  downloaded but rejected: {st.get("pdf_rejected_unconfirmed", 0):,} not shown to be the cited work, {st.get("pdf_rejected_other_work", 0):,}
  another work (it carries that work's DOI), {st.get("pdf_unusable", 0):,} without usable text.
  {st.get("review_not_auto_acquired", 0):,} were only linked with low confidence and were deliberately not fetched.
- Every closure paper belongs to a cited work in `links/cited-works.csv` ({facts["closure_without_citing_work"]} without) and
  has at least one citing row in `links/definition-links.csv` ({facts["closure_without_link_row"]} without); the `field`
  column says whether the citation stands in a definition or in another extracted field.

How each closure PDF was confirmed to be the cited work:

| Check | Papers |
|---|---:|
{ident_rows}

## Known limits

- The DOI of a seed paper comes from the seed collection's metadata and is not always right:
  {facts["seed_dois_shared"]["papers"]:,} seed papers carry a DOI that another seed paper also carries
  (`doi_shared_by_seed_papers` above 1 in `seeds/papers.csv`); treat those DOIs as unknown.
- Works behind publisher bot protection or paywalls are missing even when cited often (e.g. Entman 1993,
  *Framing: Toward clarification of a fractured paradigm*). `links/cited-works.csv` lists every such work.
- A citation string sometimes could not be linked to a reference-list entry (footnotes, organisations, entries the
  PDF conversion lost); those citations have no row in `links/definition-links.csv`.
- When one citation string matched several entries equally well, all are kept with `link_confidence` = low.

## Checks done while building this folder

- Every `predictions.json` matches the SHA-256 in its `DONE.json`, and no variant is missing part outputs.
- Every TEI file matches the SHA-256 recorded when the extraction read it.
- The grouping of references into cited works equals the closure audit's own (checked in `links/definition-links-summary.json`,
  `grouping_check_against_audit`).
"""


def finish(out: Path, facts: dict) -> int:
    """Write the README, `provenance/release-facts.json` and MANIFEST.sha256 (a line per file but itself); return the number of files listed."""
    out = Path(out)
    (out / "README.md").write_text(readme(facts), encoding="utf-8")
    (out / "provenance" / "release-facts.json").write_text(
        json.dumps(facts, indent=1, ensure_ascii=False) + "\n"
    )
    listed_files = sorted(
        f for f in out.rglob("*") if f.is_file() and f.name != "MANIFEST.sha256"
    )
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{files.file_sha256(f)}  {f.relative_to(out)}\n" for f in listed_files)
    )
    return len(listed_files)


def verify(out: Path) -> int:
    """Check every file of the release against its MANIFEST.sha256 (`sha256sum -c`); the number of lines checked.
    A file that is missing or differs, or one the manifest does not list, stops with a refusal."""
    out = Path(out)
    listed = {}
    for line in (out / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    bad = [
        n
        for n, d in listed.items()
        if not (out / n).is_file() or files.file_sha256(out / n) != d
    ]
    present = {
        str(f.relative_to(out))
        for f in out.rglob("*")
        if f.is_file() and f.name != "MANIFEST.sha256"
    }
    if bad or present != set(listed):
        raise SystemExit(
            f"{out}: {len(bad)} listed files are missing or differ (first: {bad[:1]}), "
            f"{len(present - set(listed))} files are not listed."
        )
    return len(listed)
