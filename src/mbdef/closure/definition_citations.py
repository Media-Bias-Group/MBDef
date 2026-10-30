"""Which seed definition cites which cited work, and which closure paper that work became.

One row per citation occurrence: a seed paper's definition (or other extracted item), one of its citation strings, the reference-list entry that string was linked to, the cited work that entry belongs to after the closure audit's merges (M1 same DOI, M2 same title + year + first author, M3 same OpenAlex work), and the collected closure paper for that work, if any. An ambiguous marker keeps all its candidate entries (one row each, low confidence).

The work grouping is the closure audit's (the helpers below are the audit's own):
same inputs, same helper functions, same merge rules. As a check, the per-work totals computed here must equal the audit's cited-works.json (work keys, merged groups, citing seed papers, citing definitions, variants citing); the run fails otherwise.

Inputs are copies of the seed run's files plus the closure's GROBID ledger and its extraction list (docs/step-4-closure/closure.md lists them); `closure bundle` runs this step.
"""
# ruff: noqa: E741, E501

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

from mbdef import files
from mbdef.collect import download as collect_download
from mbdef.collect import grobid as collect_grobid
from mbdef.collect import ledger, open_access, openalex
from mbdef.collect.identity import collection_paper_id, normalize_doi
from mbdef.collect.ledger import COLLECTION_FIELDS, read_csv
from mbdef.extract.derive import VARIANTS


def nfkc_words(s: str) -> str:
    """NFKC- and case-folded word characters, one space between words ("Ｆake ﬁlter" -> "fake filter"). The frozen closure audit grouped cited works by this key (NFKC folds ligatures and full-width forms, which GROBID's reference strings hold); the grouping must equal the audit's, so it keeps the audit's rule rather than `collect.identity.unicode_words`."""
    s = unicodedata.normalize("NFKC", s or "").casefold()
    return " ".join(re.findall(r"\w+", s))


def strong_doi(value: str) -> str:
    """A DOI after stronger cleaning than the corpus check's: %-escapes, resolver prefixes, trailing punctuation."""
    v = unquote(unquote((value or "").strip()))
    v = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", v, flags=re.I)
    v = re.sub(r"^doi:\s*", "", v, flags=re.I)
    v = re.sub(r"[\s.,;:)\]}>'\"]+$", "", v)
    return normalize_doi(v) or ""


def surname(authors: list[str]) -> str:
    """The first author's surname, compared ("Smith, A." and "Ada Smith" both give "smith")."""
    for a in authors or []:
        t = nfkc_words(a.split(",")[0])
        if t:
            return t.split()[-1] if "," not in a else t.split()[0]
    return ""


class UnionFind:
    """Union-find over the reference groups; the smaller index is the representative."""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)
            return True
        return False


LINK_FIELDS = [
    "seed_paper_id",
    "term",
    "field",
    "item_index",
    "text",
    "citation",
    "marker",
    "link_method",
    "link_confidence",
    "year_check",
    "bibl_id",
    "reference_title",
    "reference_authors",
    "reference_year",
    "reference_doi",
    "work_key",
    "work_status",
    "closure_paper_id",
    *[f"in_{t.replace('-', '_')}" for t in VARIANTS],
]
WORK_FIELDS = [
    "work_key",
    "doi",
    "title",
    "year",
    "openalex_id",
    "work_status",
    "closure_paper_id",
    "closure_paper_ids",
    "n_citing_seed_papers",
    "n_citing_definitions",
    "n_citing_items",
    "outputs_citing",
    "n_reference_groups_merged",
    "citing_seed_papers",
]


def resolved_status(statuses: list[str]) -> str:
    """One label for a work's provider answers: identified beats not found (a garbled row's "not_found" must not hide another row's "not_open_access")."""
    if any(s in ("not_open_access", "open_access_but_no_pdf") for s in statuses):
        return "resolved_no_oa_pdf"
    return sorted(statuses)[0]


def rejected_pdf_status(errors: list[str]) -> str:
    """A downloaded PDF that did not become a closure paper, by why the conversion refused it."""
    if any("identity_conflict" in e for e in errors):
        return "pdf_rejected_other_work"  # the PDF carries another work's DOI
    if any("identity_unverified" in e for e in errors):
        return "pdf_rejected_unconfirmed"  # not shown to be the cited work
    if any(e for e in errors):
        return "pdf_unusable"  # no usable text
    return "downloaded"


def build(
    *,
    inputs: Path,
    provider: Path,
    downloads: Path,
    grobid_ledger: Path,
    union_dataset: Path,
    rows_all: Path,
    seeds_derived: Path,
    exclusions: Path | None = None,
    existing_added: Path | None = None,
) -> dict:
    """The cited works and the link rows (with the grouping check's inputs), from the closure run's ledgers."""
    I = Path(inputs)
    before = json.loads((I / "before.json").read_text())
    papers = before["papers"]
    n = len(papers)
    # Queue paper ids exactly as the closure audit recomputes them.
    queue = read_csv(I / "before-queue.csv", COLLECTION_FIELDS)
    nonpresent = [
        i for i, p in enumerate(papers) if p["corpus_match"]["status"] != "present"
    ]
    assert len(queue) == len(nonpresent), (len(queue), len(nonpresent))
    qid = {}
    for i, src in zip(nonpresent, queue, strict=True):
        assert (src["title"] or "") == (papers[i]["reference"].get("title") or "")
        if src["queue_status"] != "missing":
            qid[i] = None
            continue
        doi = open_access.normalized_queue_doi(src["doi"])
        row = {
            "paperid": collection_paper_id(
                doi=doi, title=src["title"], year=src["year"], authors=src["authors"]
            ),
            "queue_status": src["queue_status"],
            "title": src["title"],
            "authors": src["authors"],
            "year": src["year"],
            "doi": doi,
            "source_url": src["url"].strip(),
        }
        qid[i] = (
            row["paperid"]
            + "__"
            + files.sha256(
                json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
            )[:16]
        )
    prov = openalex.latest_status(Path(provider))
    up = open_access.latest_status_by_id(I / "unpaywall.csv")
    dl = {
        r["paperid"]: r
        for r in read_csv(
            Path(downloads),
            collect_download.FIELDNAMES,
            optional=("title", "authors", "year"),
        )
    }
    conv = collect_grobid.load_converted(Path(grobid_ledger))
    dup_of = {
        d["paperid"]: d["duplicate_of"]
        for d in json.loads((I / "download-dedup.json").read_text())[
            "duplicates_not_downloaded"
        ]
    }
    extracted = {r["paper_id"] for r in json.loads(Path(rows_all).read_text())}
    left_out = (
        {e["paper_id"]: e for e in json.loads(Path(exclusions).read_text())}
        if exclusions
        else {}
    )
    added = (
        {
            x["before_index"]: x["paper_id"]
            for x in json.loads(Path(existing_added).read_text())
        }
        if existing_added
        else {}
    )

    ev = []
    for p in papers:
        ref = p["reference"]
        years = sorted(
            set(ref.get("years") or ([ref["year"]] if ref.get("year") else []))
        )
        ev.append(
            {
                "dois": sorted({strong_doi(d) for d in ref.get("dois") or []} - {""}),
                "title": nfkc_words(ref.get("title") or ""),
                "year": years[0] if len(years) == 1 else "",
                "first_author": surname(ref.get("authors") or []),
            }
        )
    uf = UnionFind(n)
    by = defaultdict(list)
    for i, e in enumerate(ev):
        for d in e["dois"]:
            by[d].append(i)
    for members in by.values():  # M1
        for m in members[1:]:
            uf.union(members[0], m)
    by = defaultdict(list)
    for i, e in enumerate(ev):
        if e["title"] and e["year"] and e["first_author"] and len(e["title"]) >= 12:
            by[(e["title"], e["year"], e["first_author"])].append(i)
    for members in by.values():  # M2, only without conflicting DOIs
        if len(members) > 1 and len({d for m in members for d in ev[m]["dois"]}) <= 1:
            for m in members[1:]:
                uf.union(members[0], m)
    by = defaultdict(list)
    for i in range(n):
        s = prov.get(qid.get(i) or "")
        if (
            s
            and s["oa_status"] in openalex.TERMINAL_STATUSES
            and s["oa_status"] not in ("not_found", "ambiguous")
        ):
            oid = (s.get("openalex_id") or "").rsplit("/", 1)[-1]
            if oid:
                by[("openalex", oid)].append(i)
            elif s.get("doi"):
                by[("doi", strong_doi(s["doi"]))].append(i)
    for members in by.values():  # M3
        for m in members[1:]:
            uf.union(members[0], m)

    union = {
        p["paper_id"]: p for p in json.loads(Path(union_dataset).read_text())["papers"]
    }
    okeys, pick_sha = {}, {}
    derived = Path(seeds_derived)
    done = json.loads((derived / "DONE.json").read_text())
    for tag, name in VARIANTS.items():
        f = derived / tag / "predictions.json"
        pick_sha[tag] = files.file_sha256(f)
        assert pick_sha[tag] == done["picks"][name]["predictions_sha256"], (
            f"{f} does not match DONE.json"
        )
        keys = set()
        for p in json.loads(f.read_text())["papers"]:
            for r in p["definition_records"]:
                for field, items in r.items():
                    if isinstance(items, list):
                        for it in items:
                            if isinstance(it, dict):
                                for c in it.get("citation") or []:
                                    keys.add(
                                        (
                                            p["paper_id"],
                                            nfkc_words(r.get("term", "")),
                                            field,
                                            nfkc_words(it.get("text", "")),
                                            c,
                                        )
                                    )
        okeys[tag] = keys

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    works, links, unlocated = [], [], Counter()
    for root, members in sorted(groups.items()):
        refs = [papers[m]["reference"] for m in members]
        dois = sorted({d for m in members for d in ev[m]["dois"]})
        qids = [qid.get(m) for m in members if qid.get(m)]
        st = [prov[q] for q in qids if q in prov]
        oid = next(
            ((s.get("openalex_id") or "") for s in st if s.get("openalex_id")), ""
        )
        present = any(papers[m]["corpus_match"]["status"] == "present" for m in members)
        reps = [dup_of.get(q, q) for q in qids]
        # A left-out duplicate (extracted, or a copy never added because its text was already in the set) counts too.
        raw_ids = {r for r in reps if r in extracted or r in left_out} | {
            added[m]
            for m in members
            if added.get(m) in extracted or added.get(m) in left_out
        }
        closure_ids, seed_copy = set(), ""
        for cid in raw_ids:  # a left-out duplicate counts for the paper kept in its place (or for its seed paper)
            e = left_out.get(cid)
            if not e:
                closure_ids.add(cid)
            elif e.get("kept_paper_id"):
                closure_ids.add(e["kept_paper_id"])
            else:
                seed_copy = e["seed_paper_id"]
        closure_ids = sorted(closure_ids)
        closure_id = closure_ids[0] if closure_ids else ""
        status = (
            "present_in_corpus"
            if present or seed_copy
            else ("review_not_auto_acquired" if not qids else "unattempted")
        )
        if closure_id:
            status = "extracted"
        elif not present and not seed_copy and qids:
            if any(
                (c := conv.get(r)) and ledger.truthy(c.get("converted")) for r in reps
            ):
                status = "converted"
            elif any(
                (d := dl.get(r)) and ledger.truthy(d.get("downloaded")) for r in reps
            ):
                status = rejected_pdf_status(
                    [(conv.get(r) or {}).get("error") or "" for r in reps]
                )
            elif st:
                best = sorted(s["oa_status"] for s in st)
                if "has_pdf" in best or any(
                    (u := up.get(q)) and u["oa_status"] == "has_pdf" for q in qids
                ):
                    status = (
                        "oa_pdf_found_download_failed"
                        if any(dl.get(r) for r in reps)
                        else "oa_pdf_found"
                    )
                else:
                    status = resolved_status(best)
            elif any(
                dl.get(r) for r in reps
            ):  # a Semantic Scholar PDF link without a provider answer, tried and failed
                status = "oa_pdf_found_download_failed"
        key = (
            f"doi:{dois[0]}"
            if dois
            else (f"openalex:{oid.rsplit('/', 1)[-1]}" if oid else f"group:{root}")
        )
        seeds, defs, items, outs = set(), set(), set(), set()
        for m in members:
            ref = papers[m]["reference"]
            for o in papers[m]["occurrences"]:
                pid = o["paper_id"]
                seeds.add(pid)
                try:
                    rec = union[pid]["definition_records"][o["record_index"]]
                    it = rec[o["field"]][o["item_index"]]
                except KeyError, IndexError, TypeError:
                    unlocated["record or item not in the union"] += 1
                    continue
                # citation_index numbers the item's resolved markers (one citation string can hold several); the string the marker came from is o["raw"]. Indexing it["citation"] by citation_index instead would drop occurrences (2,761 on the released ledgers) and give other rows a sibling string of the same item.
                cit = o.get("raw") or ""
                if cit not in (it.get("citation") or []):
                    unlocated["citation string not in the item"] += 1
                    continue
                k = (
                    pid,
                    nfkc_words(rec.get("term", "")),
                    o["field"],
                    nfkc_words(it.get("text", "")),
                )
                items.add(k)
                if o["field"] == "definitions":
                    defs.add(k)
                inside = {tag: (*k, cit) in keys for tag, keys in okeys.items()}
                outs.update(t for t, v in inside.items() if v)
                links.append(
                    {
                        "seed_paper_id": pid,
                        "term": rec.get("term", ""),
                        "field": o["field"],
                        "item_index": o["item_index"],
                        "text": it.get("text", ""),
                        "citation": cit,
                        "marker": o.get("marker") or "",
                        "link_method": o.get("method") or "",
                        "link_confidence": o.get("confidence") or "",
                        "year_check": o.get("year_check") or "",
                        "bibl_id": o.get("bibl_id") or "",
                        "reference_title": ref.get("title") or "",
                        "reference_authors": "; ".join(ref.get("authors") or []),
                        "reference_year": "; ".join(ref.get("years") or []),
                        "reference_doi": "; ".join(ref.get("dois") or []),
                        "work_key": key,
                        "work_status": status,
                        "closure_paper_id": closure_id,
                        **{
                            f"in_{t.replace('-', '_')}": int(v)
                            for t, v in inside.items()
                        },
                    }
                )
        works.append(
            {
                "work_key": key,
                "doi": dois[0] if dois else "",
                "title": next((r.get("title") for r in refs if r.get("title")), "")
                or "",
                "year": next((ev[m]["year"] for m in members if ev[m]["year"]), ""),
                "openalex_id": oid,
                "work_status": status,
                "closure_paper_id": closure_id,
                "closure_paper_ids": closure_ids,
                "n_citing_seed_papers": len(seeds),
                "n_citing_definitions": len(defs),
                "n_citing_items": len(items),
                "outputs_citing": sorted(outs),
                "n_reference_groups_merged": len(members),
                "citing_seed_papers": sorted(seeds),
            }
        )
    relinked = relink_orphans(
        works, links, extracted, left_out, json.loads(Path(rows_all).read_text())
    )
    return {
        "works": works,
        "links": links,
        "pick_sha": pick_sha,
        "before_summary": before["summary"],
        "extracted": extracted,
        "unlocated": dict(unlocated),
        "relinked": relinked,
    }


def relink_orphans(
    works: list[dict],
    links: list[dict],
    extracted: set,
    left_out: dict,
    rows: list[dict],
) -> list[dict]:
    """A collected paper that no cited work points to any more is linked to the one work without a closure paper whose title it has (normalised titles at least 95% alike). Ignoring misplaced DOIs regroups some works under a new queue row (identified by title); the paper collected under the earlier row is still theirs."""
    import difflib

    linked = {cid for w in works for cid in w["closure_paper_ids"]}
    free = [
        w
        for w in works
        if not w["closure_paper_id"]
        and w["work_status"] != "present_in_corpus"
        and nfkc_words(w["title"])
    ]
    title = {r["paper_id"]: nfkc_words(r["frozen_title"]) for r in rows}
    done = []
    for pid in sorted(extracted - linked - set(left_out)):
        t = title.get(pid, "")
        cands = [
            w
            for w in free
            if t
            and nfkc_words(w["title"])[:6] == t[:6]
            and difflib.SequenceMatcher(None, t, nfkc_words(w["title"])).ratio() >= 0.95
        ]
        if len(cands) != 1:
            continue
        w = cands[0]
        w.update(closure_paper_id=pid, closure_paper_ids=[pid], work_status="extracted")
        free.remove(w)
        for row in links:
            if row["work_key"] == w["work_key"]:
                row.update(closure_paper_id=pid, work_status="extracted")
        done.append({"paper_id": pid, "work_key": w["work_key"], "title": w["title"]})
    return done


def check_against_audit(works: list[dict], audit_path: Path) -> dict:
    """The grouping must equal the closure audit's: same work keys, merged groups and citing seed papers. The counts that depend on locating each citation string (citing definitions, items, variants) are compared and reported: the audit located strings by marker position and so undercounted them."""
    audit = json.loads(audit_path.read_text())
    cols = ("n_reference_groups_merged", "n_citing_seed_papers", "citing_seed_papers")
    mine = Counter((w["work_key"], *(json.dumps(w[c]) for c in cols)) for w in works)
    theirs = Counter((w["work_key"], *(json.dumps(w[c]) for c in cols)) for w in audit)
    by_key = {w["work_key"]: w for w in audit}
    located = ("n_citing_definitions", "n_citing_items", "outputs_citing")
    changed = {
        c: sum(
            1
            for w in works
            if w["work_key"] in by_key
            and json.dumps(w[c]) != json.dumps(by_key[w["work_key"]][c])
        )
        for c in located
    }
    return {
        "works_here": len(works),
        "works_in_audit": len(audit),
        "identical": mine == theirs,
        "only_here": sum((mine - theirs).values()),
        "only_in_audit": sum((theirs - mine).values()),
        "works_whose_located_counts_were_corrected": changed,
    }


def write(result: dict, inputs: Path, out: Path) -> dict:
    """Write definition-links.csv, cited-works.csv and definition-links-summary.json into OUT; return the summary.

    The grouping must equal the closure audit's (`inputs/cited-works.json`); the caller decides what a difference means.
    """
    r = result
    check = check_against_audit(r["works"], Path(inputs) / "cited-works.json")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "definition-links.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LINK_FIELDS)
        w.writeheader()
        w.writerows(
            sorted(
                r["links"],
                key=lambda x: (
                    x["seed_paper_id"],
                    x["term"],
                    x["field"],
                    x["item_index"],
                    x["citation"],
                    x["work_key"],
                ),
            )
        )
    with (out / "cited-works.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=WORK_FIELDS)
        w.writeheader()
        for row in sorted(
            r["works"], key=lambda x: (-x["n_citing_seed_papers"], x["work_key"])
        ):
            w.writerow(
                {
                    **row,
                    "closure_paper_ids": "; ".join(row["closure_paper_ids"]),
                    "outputs_citing": "; ".join(row["outputs_citing"]),
                    "citing_seed_papers": "; ".join(row["citing_seed_papers"]),
                }
            )
    covered = {w["closure_paper_id"] for w in r["works"] if w["closure_paper_id"]}
    summary = {
        "schema": "closure-definition-links/1",
        "reference_occurrences": r["before_summary"]["reference_occurrences"],
        "link_rows": len(r["links"]),
        "works": len(r["works"]),
        "work_status": dict(
            Counter(w["work_status"] for w in r["works"]).most_common()
        ),
        "closure_papers_extracted": len(r["extracted"]),
        "closure_papers_linked_to_a_work": len(covered),
        "closure_papers_without_link": sorted(r["extracted"] - covered)[:50],
        "link_rows_to_extracted_papers": sum(
            1 for x in r["links"] if x["closure_paper_id"]
        ),
        "relinked_by_title": r["relinked"],
        "works_extracted_more_than_once": [
            w["closure_paper_ids"]
            for w in r["works"]
            if len(w["closure_paper_ids"]) > 1
        ],
        "closure_papers_without_link_row": sorted(
            r["extracted"]
            - {x["closure_paper_id"] for x in r["links"] if x["closure_paper_id"]}
        ),
        "occurrences_not_located": r["unlocated"],
        "grouping_check_against_audit": check,
        "seed_predictions_sha256": r["pick_sha"],
    }
    (out / "definition-links-summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n"
    )
    if not check["identical"]:
        raise SystemExit(
            "The work grouping differs from the closure audit's cited-works.json."
        )
    return summary
