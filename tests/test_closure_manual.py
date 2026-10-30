import csv
import json
from collections import Counter
from hashlib import sha256

import pytest
import yaml
from cli_helpers import run_cli

from mbdef.closure import resolver_fingerprint
from mbdef.closure.manual_citations import (
    dump_manual_links,
    resolve_manual_citations,
)

PAPER_ID = "paper"
TEI = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0">
<text><back><listBibl>
  <biblStruct xml:id="b0">
    <analytic><title level="a">First work</title><author><persName><forename>Ada</forename><surname>Lovelace</surname></persName></author></analytic>
    <monogr><imprint><date when="2020"/></imprint></monogr>
  </biblStruct>
  <biblStruct xml:id="b1">
    <analytic><title level="a">Second work</title><author><persName><forename>Grace</forename><surname>Hopper</surname></persName></author></analytic>
    <monogr><imprint><date when="2021"/></imprint></monogr>
  </biblStruct>
</listBibl></back></text>
</TEI>"""
REVIEW_FIELDS = [
    "paper_id",
    "citation",
    "marker",
    "method",
    "confidence",
    "reason",
    "year_check",
    "bibl_ids",
    "reference",
    "doi",
    "year",
]


def citation_row(
    raw,
    marker,
    *,
    method="unresolved",
    confidence="none",
    reason="no_year_no_surname_match",
    years=None,
):
    return {
        "raw": raw,
        "marker": marker,
        "page": None,
        "method": method,
        "confidence": confidence,
        "year_check": "none",
        "reason": reason,
        "surnames": [],
        "years": years or [],
        "ref_text": None,
        "ref_context": None,
        "ref_paragraph": None,
        "raw_reference": "reference-list snippet"
        if method == "raw_reference_text"
        else None,
        "references": [],
    }


def manual_entry(tei_path, raw, marker, links):
    return {
        "paperid": PAPER_ID,
        "tei_file_path": str(tei_path),
        "citation": raw,
        "marker": marker,
        "manual_link": links,
    }


def make_workspace(tmp_path, rows, ledger):
    tei_path = tmp_path / f"{PAPER_ID}.tei.xml"
    tei_path.write_bytes(TEI)
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "report.json"
    review_path = tmp_path / "review.csv"
    ledger_path = tmp_path / "manual-links.yaml"
    dataset = {
        "papers": [
            {
                "paper_id": PAPER_ID,
                "tei": {"kind": "archive"},
                "definition_records": [
                    {
                        "definitions": [
                            {"citation": [row["raw"]], "resolved_citations": [row]}
                            for row in rows
                        ]
                    }
                ],
            }
        ]
    }
    methods = Counter(row["method"] for row in rows)
    confidences = Counter(row["confidence"] for row in rows)
    year_checks = Counter(row["year_check"] for row in rows)
    reasons = Counter(row["reason"] for row in rows if row["reason"])
    linked = sum(bool(row["references"]) for row in rows)
    report = {
        "resolver": "taxonomy_pipeline.closure.extend_gt/5",
        "resolver_sha256": resolver_fingerprint(),
        "linked_markers": linked,
        "bibl_links": sum(len(row["references"]) for row in rows),
        "by_method": dict(methods),
        "by_confidence": dict(confidences),
        "by_year_check": dict(year_checks),
        "by_reason": dict(reasons),
        "papers_detail": [
            {
                "paper_id": PAPER_ID,
                "linked_markers": linked,
                "unresolved_markers": len(rows) - linked,
            }
        ],
    }
    review = [
        {
            "paper_id": PAPER_ID,
            "citation": row["raw"],
            "marker": row["marker"],
            "method": row["method"],
            "confidence": row["confidence"],
            "reason": row["reason"] or "",
            "year_check": row["year_check"],
            "bibl_ids": "",
            "reference": row["raw_reference"] or "",
            "doi": "",
            "year": "",
        }
        for row in rows
    ]
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review)
    dump_manual_links(ledger_path, ledger)
    return ledger_path, dataset_path, report_path, review_path, tei_path


def dataset_rows(dataset_path):
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [
        item["resolved_citations"][0]
        for item in dataset["papers"][0]["definition_records"][0]["definitions"]
    ]


def test_manual_links_fan_out_without_conflating_raw_citations_and_sync_outputs(
    tmp_path,
):
    rows = [
        citation_row("(Example 2020)", "Example 2020", years=["2020"]),
        citation_row("(Example 2020)", "Example 2020", years=["2020"]),
        citation_row("(Shared first)", "Shared"),
        citation_row("(Shared second)", "Shared"),
        citation_row(
            "(Raw source 2020)",
            "Raw source 2020",
            method="raw_reference_text",
            confidence="low",
            reason="raw_reference_only",
            years=["2020"],
        ),
        citation_row("(Bundle)", "Bundle"),
    ]
    tei_path = tmp_path / f"{PAPER_ID}.tei.xml"
    doi_url = "https://doi.org/10.1234/Example%2FThing"
    ledger = [
        manual_entry(tei_path, "(Example 2020)", "Example 2020", ["#b0"]),
        manual_entry(tei_path, "(Shared first)", "Shared", ["#b1"]),
        manual_entry(tei_path, "(Shared second)", "Shared", []),
        manual_entry(tei_path, "(Raw source 2020)", "Raw source 2020", [doi_url]),
        manual_entry(
            tei_path,
            "(Bundle)",
            "Bundle",
            ["#b1", "https://example.org/story"],
        ),
    ]
    paths = make_workspace(tmp_path, rows, ledger)

    result = resolve_manual_citations(*paths[:4])

    assert result == {
        "manual_link_entries": 4,
        "manually_linked_markers": 5,
        "changed_markers": 5,
    }
    updated = dataset_rows(paths[1])
    example_rows = [row for row in updated if row["raw"] == "(Example 2020)"]
    assert len(example_rows) == 2
    assert all(row["method"] == "manual_link" for row in example_rows)
    assert all(row["references"][0]["bibl_id"] == "b0" for row in example_rows)
    assert (
        next(row for row in updated if row["raw"] == "(Shared first)")["method"]
        == "manual_link"
    )
    assert (
        next(row for row in updated if row["raw"] == "(Shared second)")["method"]
        == "unresolved"
    )

    url_row = next(row for row in updated if row["raw"] == "(Raw source 2020)")
    expected_id = "manual:url:" + sha256(doi_url.encode()).hexdigest()[:16]
    assert url_row["method"] == "manual_link"
    assert url_row["confidence"] == "high"
    assert url_row["reason"] is None
    assert url_row["references"] == [
        {
            "bibl_id": expected_id,
            "authors": [],
            "year": None,
            "years_in_text": [],
            "title": None,
            "container": None,
            "volume": None,
            "issue": None,
            "pages": None,
            "publisher": None,
            "pub_place": None,
            "doi": "10.1234/example/thing",
            "url": doi_url,
            "formatted": doi_url,
            "raw": doi_url,
        }
    ]
    bundle = next(row for row in updated if row["raw"] == "(Bundle)")
    assert [reference["bibl_id"] for reference in bundle["references"]] == [
        "b1",
        "manual:url:" + sha256(b"https://example.org/story").hexdigest()[:16],
    ]

    report = json.loads(paths[2].read_text(encoding="utf-8"))
    assert report["resolver"] == "taxonomy_pipeline.closure.extend_gt/5+manual"
    assert report["linked_markers"] == 5
    assert report["bibl_links"] == 6
    assert report["manual_link_entries"] == 4
    assert report["manually_linked_markers"] == 5
    assert report["papers_detail"][0]["linked_markers"] == 5
    assert report["papers_detail"][0]["unresolved_markers"] == 1
    assert report["by_method"] == {"manual_link": 5, "unresolved": 1}
    assert "raw_reference_only" not in report["by_reason"]

    with paths[3].open(encoding="utf-8", newline="") as handle:
        review = list(csv.DictReader(handle))
    assert sum(row["method"] == "manual_link" for row in review) == 5
    assert all(row["bibl_ids"] == "b0" for row in review[:2])
    raw_review = next(row for row in review if row["citation"] == "(Raw source 2020)")
    assert raw_review["bibl_ids"] == expected_id
    assert raw_review["doi"] == "10.1234/example/thing"

    before = [path.read_bytes() for path in paths[1:4]]
    second = resolve_manual_citations(*paths[:4])
    assert second["changed_markers"] == 0
    assert before == [path.read_bytes() for path in paths[1:4]]


def test_blank_manual_ledger_is_a_byte_identical_noop(tmp_path):
    row = citation_row("(Pending)", "Pending")
    tei_path = tmp_path / f"{PAPER_ID}.tei.xml"
    paths = make_workspace(
        tmp_path,
        [row],
        [manual_entry(tei_path, "(Pending)", "Pending", [])],
    )
    before = [path.read_bytes() for path in paths[1:4]]

    assert resolve_manual_citations(*paths[:4])["changed_markers"] == 0

    assert before == [path.read_bytes() for path in paths[1:4]]


@pytest.mark.parametrize(
    ("links", "message"),
    [
        ([""], "cannot be empty"),
        (["#missing"], "unknown TEI target"),
        (["ftp://example.org/work"], "not an absolute HTTP URL"),
        (["https://example.org/work", "https://example.org/work"], "duplicated"),
    ],
)
def test_bad_manual_links_fail_before_outputs_are_replaced(tmp_path, links, message):
    row = citation_row("(Bad)", "Bad")
    tei_path = tmp_path / f"{PAPER_ID}.tei.xml"
    paths = make_workspace(
        tmp_path,
        [row],
        [manual_entry(tei_path, "(Bad)", "Bad", links)],
    )
    before = [path.read_bytes() for path in paths[1:4]]

    with pytest.raises(ValueError, match=message):
        resolve_manual_citations(*paths[:4])

    assert before == [path.read_bytes() for path in paths[1:4]]


def test_duplicate_stale_and_automatic_conflicts_are_rejected(tmp_path):
    row = citation_row("(Original)", "Original")
    tei_path = tmp_path / f"{PAPER_ID}.tei.xml"
    entry = manual_entry(tei_path, "(Original)", "Original", ["#b0"])
    paths = make_workspace(tmp_path, [row], [entry])
    paths[0].write_text(
        yaml.safe_dump([entry, entry], sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicates manual-link key"):
        resolve_manual_citations(*paths[:4])

    stale = manual_entry(tei_path, "(Stale)", "Original", ["#b0"])
    dump_manual_links(paths[0], [stale])
    with pytest.raises(ValueError, match="matches no dataset row"):
        resolve_manual_citations(*paths[:4])

    automatic = citation_row(
        "(Original)",
        "Original",
        method="ref_target",
        confidence="high",
        reason=None,
    )
    paths = make_workspace(tmp_path, [automatic], [entry])
    with pytest.raises(ValueError, match="conflicts with existing method 'ref_target'"):
        resolve_manual_citations(*paths[:4])


def stale_workspace(tmp_path):
    return make_workspace(
        tmp_path,
        [citation_row("(Example 2020)", "Example 2020", years=["2020"])],
        [
            manual_entry(
                tmp_path / "paper.tei.xml", "(Example 2020)", "Example 2020", ["#b0"]
            )
        ],
    )


def test_stale_resolver_report_cannot_apply_manual_links(tmp_path):
    paths = stale_workspace(tmp_path)
    report = json.loads(paths[2].read_text())
    report["resolver"] = "mbdef.closure.extend_gt/4"
    paths[2].write_text(json.dumps(report))
    before = [path.read_bytes() for path in paths[:4]]
    with pytest.raises(ValueError, match="regenerate stale outputs"):
        resolve_manual_citations(*paths[:4])
    assert [path.read_bytes() for path in paths[:4]] == before


def test_same_version_with_stale_code_fingerprint_cannot_apply_manual_links(tmp_path):
    paths = stale_workspace(tmp_path)
    report = json.loads(paths[2].read_text())
    report["resolver_sha256"] = "0" * 64
    paths[2].write_text(json.dumps(report))
    before = [path.read_bytes() for path in paths[:4]]
    with pytest.raises(ValueError, match="code changed; regenerate stale outputs"):
        resolve_manual_citations(*paths[:4])
    assert [path.read_bytes() for path in paths[:4]] == before


def test_the_manual_command_applies_a_completed_link_and_writes_a_receipt(tmp_path):
    rows = [citation_row("(Example 2020)", "Example 2020", years=["2020"])]
    tei_path = tmp_path / f"{PAPER_ID}.tei.xml"
    ledger = [manual_entry(tei_path, "(Example 2020)", "Example 2020", ["#b0"])]
    ledger_path, dataset_path, report_path, review_path, _ = make_workspace(
        tmp_path, rows, ledger
    )

    done = run_cli(
        tmp_path,
        "closure",
        "manual",
        "--manual-links",
        str(ledger_path),
        "--dataset",
        str(dataset_path),
        "--report",
        str(report_path),
        "--review",
        str(review_path),
    )

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {
        "manual_link_entries": 1,
        "manually_linked_markers": 1,
        "changed_markers": 1,
    }
    assert dataset_rows(dataset_path)[0]["method"] == "manual_link"
    receipt = json.loads((tmp_path / "manual-receipt.json").read_text())
    assert receipt["command"] == "mbdef closure manual"
    assert receipt["summary"]["changed_markers"] == 1
