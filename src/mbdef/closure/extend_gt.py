"""Extend the validated ground truth and maintain its manual citation-link ledger.

`closure resolve` (`command`) writes the dataset (the input plus TEI provenance and resolved citation rows), the report, the review CSV, and a deduplicated manual-link ledger (YAML).
Output carries no timestamps, so reruns with unchanged inputs are byte-identical.
"""

import copy
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import ParseError

from mbdef import files
from mbdef.closure import RESOLVER, resolver_fingerprint
from mbdef.closure.bibliography import parse_bibliography
from mbdef.closure.citations import (
    build_index,
    empty_index,
    resolve_citations,
)
from mbdef.closure.manual_citations import (
    dump_manual_links,
    merge_manual_links,
)
from mbdef.closure.tei_sources import Sources, load_tei
from mbdef.extract.records import FIELDS

REVIEW_COLUMNS = (
    "paper_id",
    "split",
    "paper_title",
    "term",
    "field",
    "item_text",
    "citation",
    "marker",
    "page",
    "method",
    "confidence",
    "reason",
    "year_check",
    "bibl_ids",
    "reference",
    "doi",
    "year",
    "ref_text",
    "ref_context",
    "tei_kind",
    "tei_file",
)
EMPTY_TEI: dict[str, Any] = {
    "grobid_version": None,
    "title": None,
    "entries": {},
    "refs": [],
    "back_text": "",
}


def _cut(text: str | None, size: int = 300) -> str:
    text = text or ""
    return text if len(text) <= size else text[:size] + "…"


def extend(
    ground_truth: dict, sources: Sources, refresh: bool = False
) -> tuple[dict, dict, list[dict], list[dict]]:
    from .corpus_check import _validate_dataset

    _validate_dataset(ground_truth)
    dataset = copy.deepcopy(ground_truth)
    by_method: Counter = Counter()
    by_confidence: Counter = Counter()
    by_year_check: Counter = Counter()
    by_reason: Counter = Counter()
    by_source: Counter = Counter()
    without_tei: list[str] = []
    detail: list[dict] = []
    unresolved: list[dict] = []
    review: list[dict] = []
    totals = {"items": 0, "strings": 0, "markers": 0, "linked": 0, "links": 0}
    for paper in dataset["papers"]:
        try:
            tei, provenance = load_tei(paper, sources, refresh)
            parsed = parse_bibliography(tei)
            index = build_index(parsed)
        except (OSError, ValueError, ParseError) as exc:
            parsed, index = dict(EMPTY_TEI), empty_index()
            provenance = {"kind": "missing", "file": None, "errors": [str(exc)]}
            without_tei.append(paper["paper_id"])
        by_source[provenance["kind"]] += 1
        # Provenance is a property of the paper, not of each of its citations.
        paper["tei"] = {
            "kind": provenance["kind"],
            "file": provenance["file"],
            "grobid_version": parsed["grobid_version"],
            "bibliography_entries": index["bibl_count"],
        }
        counts = {"markers": 0, "linked": 0, "unresolved": 0}
        for record in paper.get("definition_records", []):
            for field in FIELDS:
                for item in record.get(field, []):
                    if not item.get("citation"):
                        continue
                    totals["items"] += 1
                    totals["strings"] += len(item["citation"])
                    rows = resolve_citations(item, index, parsed["entries"])
                    item["resolved_citations"] = rows
                    for row in rows:
                        references = row["references"]
                        totals["markers"] += 1
                        counts["markers"] += 1
                        by_method[row["method"]] += 1
                        by_confidence[row["confidence"]] += 1
                        by_year_check[row["year_check"]] += 1
                        if row["reason"]:
                            by_reason[row["reason"]] += 1
                        if references:
                            totals["linked"] += 1
                            counts["linked"] += 1
                            totals["links"] += len(references)
                        elif row["method"] != "page_locator":
                            counts["unresolved"] += 1
                            unresolved.append(
                                {
                                    "paper_id": paper["paper_id"],
                                    "paper_title": (paper.get("metadata") or {}).get(
                                        "title"
                                    )
                                    or parsed["title"],
                                    "term": record.get("term"),
                                    "field": field,
                                    "definition": item.get("text"),
                                    "citation": row["raw"],
                                    "marker": row["marker"],
                                    "page": row["page"],
                                    "method": row["method"],
                                    "reason": row["reason"],
                                    "raw_reference": row["raw_reference"],
                                    "tei_kind": paper["tei"]["kind"],
                                    "tei_file": paper["tei"]["file"],
                                    "tei_grobid_version": paper["tei"][
                                        "grobid_version"
                                    ],
                                    "tei_bibliography_entries": index["bibl_count"],
                                }
                            )
                        review.append(
                            {
                                "paper_id": paper["paper_id"],
                                "split": paper.get("split"),
                                "paper_title": (paper.get("metadata") or {}).get(
                                    "title"
                                )
                                or parsed["title"],
                                "term": record.get("term"),
                                "field": field,
                                "item_text": _cut(item.get("text")),
                                "citation": row["raw"],
                                "marker": row["marker"] or "",
                                "page": row["page"] or "",
                                "method": row["method"],
                                "confidence": row["confidence"],
                                "reason": row["reason"] or "",
                                "year_check": row["year_check"],
                                "bibl_ids": " ".join(
                                    reference["bibl_id"] for reference in references
                                ),
                                "reference": (
                                    references[0]["formatted"]
                                    if references
                                    else (row["raw_reference"] or "")
                                ),
                                "doi": next(
                                    (r["doi"] for r in references if r["doi"]), ""
                                ),
                                "year": next(
                                    (r["year"] for r in references if r["year"]), ""
                                ),
                                "ref_text": row["ref_text"] or "",
                                "ref_context": _cut(row["ref_context"]),
                                "tei_kind": paper["tei"]["kind"],
                                "tei_file": paper["tei"]["file"] or "",
                            }
                        )
        detail.append(
            {
                "paper_id": paper["paper_id"],
                "tei_kind": paper["tei"]["kind"],
                "tei_file": paper["tei"]["file"],
                "grobid_version": paper["tei"]["grobid_version"],
                "bibliography_entries": index["bibl_count"],
                "bibr_refs": len(parsed["refs"]),
                "source_errors": provenance.get("errors", []),
                "source_doi_note": provenance.get("doi_note", ""),
                "markers": counts["markers"],
                "linked_markers": counts["linked"],
                "unresolved_markers": counts["unresolved"],
            }
        )
    report = {
        "resolver": RESOLVER,
        "resolver_sha256": resolver_fingerprint(),
        "ground_truth": None,
        "papers": len(dataset["papers"]),
        "items_with_citations": totals["items"],
        "citation_strings": totals["strings"],
        "markers": totals["markers"],
        "linked_markers": totals["linked"],
        "bibl_links": totals["links"],
        "by_method": dict(sorted(by_method.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_confidence": dict(
            sorted(by_confidence.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "by_year_check": dict(
            sorted(by_year_check.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "by_reason": dict(sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0]))),
        "papers_by_source": dict(sorted(by_source.items())),
        "papers_without_tei": without_tei,
        "papers_detail": detail,
    }
    unresolved.sort(
        key=lambda row: (
            row["paper_id"],
            row["term"] or "",
            row["field"],
            row["citation"],
        )
    )
    review.sort(
        key=lambda row: (
            row["paper_id"],
            row["term"] or "",
            row["field"],
            row["item_text"],
            row["citation"],
        )
    )
    return dataset, report, unresolved, review


def command(
    ground_truth: str,
    out: str,
    manual_links: str,
    source_zip: str | None,
    grobid_archive: str | None,
    grobid_files: str | None,
    tei_dirs: list[str],
    refresh: bool,
    min_linked: int,
) -> int:
    """`closure resolve`: resolve every citation string of GROUND_TRUTH's extracted items.

    Writes `citations.json` (the dataset with its resolved citations), `report.json` and `review.csv` into OUT, and merges the unresolved ones into the MANUAL_LINKS ledger (YAML), keeping every completed entry. The winner of each paper's TEI is cached in `OUT/tei-cache`.
    """
    folder = Path(out)
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f"{folder} is not empty; give a new or empty folder.")
    folder.mkdir(parents=True, exist_ok=True)
    nowhere = folder / "none"  # a source that was not given: this path never exists
    sources = Sources(
        archive=Path(grobid_archive) if grobid_archive else nowhere / "archive",
        grobid_files=Path(grobid_files) if grobid_files else nowhere / "grobid",
        source_zip=Path(source_zip) if source_zip else nowhere / "sources.zip",
        cache=folder / "tei-cache",
        directories=tuple(Path(d) for d in tei_dirs),
    )
    truth = json.loads(Path(ground_truth).read_text(encoding="utf-8"))
    dataset, report, rows, review = extend(truth, sources, refresh)
    report["ground_truth"] = str(ground_truth)
    if report["linked_markers"] < min_linked:
        raise SystemExit(f"linked_markers {report['linked_markers']} < {min_linked}")
    ledger = merge_manual_links(rows, sources, Path(manual_links))
    files.write_json(folder / "citations.json", dataset)
    files.write_json(folder / "report.json", report)
    dump_manual_links(Path(manual_links), ledger)
    with (folder / "review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        writer.writerows(review)
    summary = {k: v for k, v in report.items() if k != "papers_detail"}
    receipt = files.receipt(
        "mbdef closure resolve",
        {
            "resolver": RESOLVER,
            "refresh": refresh,
            "min_linked": min_linked,
            "tei_dirs": tei_dirs,
        },
        {"ground_truth": Path(ground_truth), "manual_links": Path(manual_links)},
        {"citations": folder / "citations.json", "report": folder / "report.json"},
    )
    receipt["summary"] = summary
    files.write_json(folder / files.RECEIPT, receipt)
    print(json.dumps(summary, indent=2))
    print(f"dataset: {folder / 'citations.json'}")
    print(f"manual-link entries: {len(ledger)} -> {manual_links}")
    print(f"review rows: {len(review)} -> {folder / 'review.csv'}")
    return 0
