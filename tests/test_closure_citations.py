import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from cli_helpers import run_cli

from mbdef.closure.bibliography import parse_bibliography
from mbdef.closure.citations import (
    build_index,
    resolve_citations,
    split_units,
)
from mbdef.closure.extend_gt import REVIEW_COLUMNS, extend
from mbdef.closure.manual_citations import (
    MANUAL_LINK_FIELDS,
    dump_manual_links,
    merge_manual_links,
)
from mbdef.closure.tei_sources import Sources

PAPER_ID = "testpaper0001"


def nowhere(folder, cache):
    """Sources with the three places a seed TEI could come from left empty."""
    return Sources(
        archive=folder / "no-archive",
        grobid_files=folder / "no-files",
        source_zip=folder / "none.zip",
        cache=cache,
    )


TEI = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
 <teiHeader>
  <encodingDesc><appInfo><application version="0.8.0" ident="GROBID"/></appInfo></encodingDesc>
  <fileDesc><titleStmt><title level="a" type="main">Sentencing and bias</title></titleStmt>
   <sourceDesc><biblStruct><analytic><title level="a" type="main">Sentencing and bias</title></analytic></biblStruct></sourceDesc>
  </fileDesc>
 </teiHeader>
 <text><body>
  <p>Selection bias appears in sentencing research <ref type="bibr" target="#b0">(Zatz 1987)</ref> and elsewhere.</p>
  <p>Racial resentment is a related construct <ref type="bibr">(Bobo et al. 1997, p. 16)</ref>.</p>
  <p>Endnote style references also occur <ref type="bibr" target="#b1">[2]</ref>.</p>
  <p>Two later works <ref type="bibr" target="#b1">(Burden 2014)</ref> and <ref type="bibr" target="#b2">(Zatz 2014)</ref> disagree.</p>
  <p>Homicide data come from <ref type="bibr" target="#b3">(Wilbanks 1987)</ref>.</p>
  <p>A synthesis followed <ref type="bibr" target="#b2">(Zatz 1995)</ref>.</p>
 </body>
 <back>
  <div type="references">
   <listBibl>
    <biblStruct xml:id="b0">
     <analytic>
      <title level="a" type="main">Race, ethnicity, and determinate sentencing</title>
      <author><persName><forename type="first">Marjorie</forename><forename type="middle">S</forename><surname>Zatz</surname></persName></author>
      <author><persName><forename type="first">John</forename><surname>Hagan</surname></persName></author>
      <idno type="DOI">10.1111/J.1745-9125.1987.TB00811.X</idno>
     </analytic>
     <monogr>
      <title level="j">Criminology</title>
      <imprint>
       <biblScope unit="volume">25</biblScope>
       <biblScope unit="issue">1</biblScope>
       <biblScope unit="page" from="69" to="92"/>
       <date type="published" when="1987"/>
      </imprint>
     </monogr>
    </biblStruct>
    <biblStruct xml:id="b1">
     <monogr>
      <title level="m">The measure of American elections</title>
      <author><persName><forename type="first">Barry</forename><surname>Burden</surname></persName></author>
      <imprint><publisher>Cambridge University Press</publisher><pubPlace>New York, NY</pubPlace><date when="2014"/></imprint>
     </monogr>
    </biblStruct>
    <biblStruct xml:id="b2">
     <analytic>
      <title level="a" type="main">The changing face of sentencing disparity</title>
      <author><persName><forename type="first">Marjorie</forename><forename type="middle">S</forename><surname>Zatz</surname></persName></author>
     </analytic>
     <monogr>
      <title level="j">Journal of Research in Crime and Delinquency</title>
      <imprint><date when="1995"/></imprint>
     </monogr>
    </biblStruct>
    <biblStruct xml:id="b3">
     <analytic>
      <title level="a" type="main">Criminal homicide offenders in the US</title>
      <author><persName><forename type="first">William</forename><surname>Wilbanks</surname></persName></author>
     </analytic>
     <monogr>
      <title level="j">Criminology</title>
      <imprint><biblScope unit="volume">23</biblScope><date when="1985"/><note>reprinted 1987</note></imprint>
     </monogr>
    </biblStruct>
    <biblStruct xml:id="b4">
     <analytic>
      <title level="a" type="main">Numeracy and judgment</title>
      <author><persName><forename type="first">Valerie</forename><surname>Reyna</surname></persName></author>
      <author><persName><forename type="first">Charles</forename><surname>Brainerd</surname></persName></author>
     </analytic>
     <monogr><title level="j">Decision Science</title><imprint><date when="2007"/></imprint></monogr>
    </biblStruct>
    <biblStruct xml:id="b5">
     <analytic>
      <title level="a" type="main">Nested sets hypothesis</title>
      <author><persName><forename type="first">Valerie</forename><surname>Reyna</surname></persName></author>
      <author><persName><forename type="first">Britain</forename><surname>Mills</surname></persName></author>
     </analytic>
     <monogr><title level="j">Behavioral Science</title><imprint><date when="2007"/></imprint></monogr>
    </biblStruct>
   </listBibl>
  </div>
  <div type="annex"><p>Bobo, L.; Kluegel, JR. Laissez faire racism and the new American dilemma. Praeger, 1997.</p></div>
 </back>
 </text>
</TEI>
"""

PARSED = parse_bibliography(TEI)
INDEX = build_index(PARSED)
ENTRIES = PARSED["entries"]


def item(*citations: str, text: str = "a systematic distortion of a sample") -> dict:
    return {"text": text, "citation": list(citations)}


def only_record(*citations: str) -> dict:
    return resolve_citations(item(*citations), INDEX, ENTRIES)[0]


def ids(record: dict) -> list[str]:
    return [reference["bibl_id"] for reference in record["references"]]


def test_exact_ref_target_hit_keeps_page_locator_and_high_confidence():
    records = resolve_citations(item("Zatz 1987, p. 75"), INDEX, ENTRIES)
    assert len(records) == 1
    assert records[0]["method"] == "ref_target"
    assert ids(records[0]) == ["b0"]
    assert records[0]["confidence"] == "high"
    assert records[0]["page"] == "p. 75"


def test_second_year_expands_to_the_matching_bibliography_entry():
    assert ids(only_record("(Zatz 1987, 1995)")) == ["b0", "b2"]


def test_standalone_page_locator_folds_into_the_preceding_record():
    records = resolve_citations(item("Zatz (1987)", "(p. 517)"), INDEX, ENTRIES)
    assert len(records) == 1
    assert records[0]["page"] == "p. 517"
    # With no marker to qualify, the locator has to keep a record of its own.
    alone = resolve_citations(item("(p. 517)"), INDEX, ENTRIES)
    assert [record["method"] for record in alone] == ["page_locator"]
    assert alone[0]["marker"] is None
    assert alone[0]["page"] == "p. 517"
    assert alone[0]["references"] == []


def test_reference_list_only_work_resolves_to_raw_text_without_a_reference():
    record = only_record("(Bobo et al. 1997)")
    assert record["method"] == "raw_reference_text"
    assert record["references"] == []
    assert "Laissez faire racism" in record["raw_reference"]


def test_numeric_markers_resolve_only_when_grobid_linked_them():
    assert only_record("2")["method"] == "ref_target"
    assert ids(only_record("2")) == ["b1"]
    missing = only_record("7")
    assert missing["method"] == "unresolved"
    assert missing["reason"] == "numeric_marker_unmatched"


def test_bare_year_marker_resolves_only_when_one_reference_carries_that_year():
    unique = only_record("1995")
    assert unique["method"] == "ref_target_partial"
    assert ids(unique) == ["b2"]
    two_refs_same_year = only_record("1987")
    assert two_refs_same_year["method"] == "unresolved"
    assert two_refs_same_year["reason"] == "year_only_marker_ambiguous"
    ambiguous = only_record("2014")
    assert ambiguous["method"] == "unresolved"
    assert ambiguous["references"] == []
    assert ambiguous["reason"] == "year_only_marker_ambiguous"


def test_marker_year_disagreeing_with_the_linked_entry_is_flagged_not_hidden():
    agreeing = only_record("Zatz 1987")
    assert agreeing["year_check"] == "match"
    assert agreeing["confidence"] == "high"
    # GROBID dated the entry 1985 but its text also carries the cited 1987.
    stale = only_record("Wilbanks 1987")
    assert ids(stale) == ["b3"]
    assert stale["year_check"] == "year_in_entry_text"
    assert stale["confidence"] == "low"
    assert "1987" in stale["references"][0]["years_in_text"]
    # GROBID linked a 2014 marker to an entry dated 1995 with no 2014 anywhere in it.
    wrong = only_record("Zatz 2014")
    assert ids(wrong) == ["b2"]
    assert wrong["year_check"] == "mismatch"
    assert wrong["confidence"] == "low"


def test_format_entry_renders_apa_like_string_with_doi():
    assert ENTRIES["b0"]["formatted"] == (
        "Zatz, M. S. & Hagan, J. (1987). Race, ethnicity, and determinate sentencing. "
        "Criminology 25(1), 69-92. https://doi.org/10.1111/j.1745-9125.1987.tb00811.x"
    )


def test_semicolon_splitting_drops_prefix_and_outer_parentheses():
    units = split_units("(e.g., Reyna & Brainerd, 1992, 1995; Reyna, 1991)")
    assert [unit["text"] for unit in units] == [
        "Reyna & Brainerd, 1992, 1995",
        "Reyna, 1991",
    ]


def test_comma_separated_works_split_without_splitting_authors_or_shared_years():
    grouped = split_units("(Goffman 1963, Hinshaw 2006, Sartorius 2007)")
    assert [unit["text"] for unit in grouped] == [
        "Goffman 1963",
        "Hinshaw 2006",
        "Sartorius 2007",
    ]
    assert len(split_units("(McNeil, Pauker, Sox, & Tversky, 1982)")) == 1
    assert len(split_units("(Reyna & Brainerd, 1992, 1995)")) == 1


def test_comma_separated_works_get_independent_resolution_records():
    records = resolve_citations(
        item("Zatz 1987, Burden 2014, Zatz 1995"), INDEX, ENTRIES
    )
    assert [record["marker"] for record in records] == [
        "Zatz 1987",
        "Burden 2014",
        "Zatz 1995",
    ]
    assert [ids(record) for record in records] == [["b0"], ["b1"], ["b2"]]


def test_author_year_fallback_requires_every_named_author():
    record = only_record("Reyna & Mills 2007")
    assert record["method"] == "bibl_author_year"
    assert ids(record) == ["b5"]


def test_record_inlines_the_marker_its_context_and_the_work_it_names():
    record = only_record("Zatz 1987, p. 75")
    assert record["raw"] == "Zatz 1987, p. 75"
    assert record["marker"] == "Zatz 1987"
    assert record["ref_text"] == "(Zatz 1987)"
    assert record["ref_context"]
    assert isinstance(record["ref_paragraph"], int)
    reference = record["references"][0]
    assert reference["bibl_id"] == "b0"
    assert reference["doi"] == ENTRIES["b0"]["doi"]
    assert reference["formatted"] == ENTRIES["b0"]["formatted"]
    assert reference["authors"] == ["Zatz, M. S.", "Hagan, J."]
    # A per-paper citation count is report material, not part of a citation record.
    assert "ref_count" not in reference


def ground_truth() -> dict:
    return {
        "schema_version": "3",
        "dataset_id": "test",
        "paper_count": 2,
        "papers": [
            {
                "paper_id": PAPER_ID,
                "split": "train",
                "metadata": {"title": "Sentencing and bias", "doi": "10.1/abc"},
                "definition_records": [
                    {
                        "term": "selection bias",
                        "definitions": [item("Zatz 1987, p. 75")],
                        "synonyms": [],
                        "description": [
                            item("(Bobo et al. 1997)", text="a distortion")
                        ],
                        "related_terms": [{"text": "sampling", "citation": []}],
                        "subtypes": [],
                        "supertypes": [],
                        "examples": [],
                    }
                ],
            },
            {
                "paper_id": "missingpaper0002",
                "split": "test",
                "metadata": {"title": "No TEI here", "doi": ""},
                "definition_records": [
                    {
                        "term": "framing",
                        "definitions": [
                            item("(Entman 1993)", text="selection and salience")
                        ],
                        "synonyms": [],
                        "description": [],
                        "related_terms": [],
                        "subtypes": [],
                        "supertypes": [],
                        "examples": [],
                    }
                ],
            },
        ],
    }


def run_extend(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / f"paper_{PAPER_ID}.grobid.tei.xml").write_bytes(TEI)
    sources = Sources(
        archive=archive,
        grobid_files=tmp_path / "files",
        source_zip=tmp_path / "none.zip",
        cache=tmp_path / "cache",
    )
    return extend(ground_truth(), sources)


def test_extend_adds_only_resolved_citations_and_tei_and_honours_source_arguments(
    tmp_path,
):
    original = ground_truth()
    dataset, report, _rows, _review = run_extend(tmp_path)
    assert set(dataset["papers"][0]) - set(original["papers"][0]) == {"tei"}
    assert dataset["papers"][0]["tei"] == {
        "kind": "archive",
        "file": f"paper_{PAPER_ID}.grobid.tei.xml",
        "grobid_version": PARSED["grobid_version"],
        "bibliography_entries": len(ENTRIES),
    }
    fields = (
        "definitions",
        "synonyms",
        "description",
        "related_terms",
        "subtypes",
        "supertypes",
        "examples",
    )
    pairs = [
        (before, after)
        for paper, extended in zip(original["papers"], dataset["papers"], strict=True)
        for record, record_extended in zip(
            paper["definition_records"], extended["definition_records"], strict=True
        )
        for field in fields
        for before, after in zip(record[field], record_extended[field], strict=True)
    ]
    assert pairs
    for before, after in pairs:
        expected = {"resolved_citations"} if before["citation"] else set()
        assert set(after) - set(before) == expected
        assert after["citation"] == before["citation"]
    assert report["papers_by_source"] == {"archive": 1, "missing": 1}


def test_missing_tei_degrades_one_paper_instead_of_failing_the_run(tmp_path):
    dataset, report, _rows, _review = run_extend(tmp_path)
    paper = dataset["papers"][1]
    assert paper["tei"] == {
        "kind": "missing",
        "file": None,
        "grobid_version": None,
        "bibliography_entries": 0,
    }
    records = paper["definition_records"][0]["definitions"][0]["resolved_citations"]
    assert [record["method"] for record in records] == ["unresolved"]
    assert [record["reason"] for record in records] == ["tei_not_found"]
    assert [record["references"] for record in records] == [[]]
    assert report["papers_without_tei"] == ["missingpaper0002"]


def test_review_rows_cover_every_citation_record_with_all_columns(tmp_path):
    dataset, report, _rows, review = run_extend(tmp_path)
    records = [
        record
        for paper in dataset["papers"]
        for definition_record in paper["definition_records"]
        for field in ("definitions", "description")
        for annotation in definition_record[field]
        for record in annotation.get("resolved_citations", [])
    ]
    assert len(review) == len(records) == report["markers"] == 3
    for row in review:
        assert tuple(row) == REVIEW_COLUMNS
        assert row["item_text"] and row["citation"] and row["marker"]
        assert row["method"] and row["confidence"]
        assert "/" not in row["tei_file"]
    resolved = next(row for row in review if row["method"] == "ref_target")
    assert resolved["bibl_ids"] == "b0"
    assert resolved["reference"] == ENTRIES["b0"]["formatted"]


def test_manual_link_ledger_is_deduplicated_sorted_and_preserves_completed_links(
    tmp_path,
):
    cache = tmp_path / "cache"
    cache.mkdir()
    for paper_id in ("a-paper", "b-paper"):
        (cache / f"{paper_id}.tei.xml").write_text("<TEI/>", encoding="utf-8")
    sources = nowhere(cache.parent, cache)
    unresolved = [
        {"paper_id": "b-paper", "citation": "(CBS01; CNN08)", "marker": "CNN08"},
        {"paper_id": "a-paper", "citation": "(CBS01; CNN08)", "marker": "CBS01"},
        {"paper_id": "a-paper", "citation": "(CBS01; CNN08)", "marker": "CBS01"},
        {"paper_id": "a-paper", "citation": "(CBS01; CNN06)", "marker": "CBS01"},
    ]
    ledger_path = tmp_path / "manual-links.yaml"

    rows = merge_manual_links(unresolved, sources, ledger_path)

    assert [(row["paperid"], row["citation"], row["marker"]) for row in rows] == [
        ("a-paper", "(CBS01; CNN06)", "CBS01"),
        ("a-paper", "(CBS01; CNN08)", "CBS01"),
        ("b-paper", "(CBS01; CNN08)", "CNN08"),
    ]
    assert all(tuple(row) == MANUAL_LINK_FIELDS for row in rows)
    assert all(isinstance(row["manual_link"], list) for row in rows)
    assert all(Path(row["tei_file_path"]).is_absolute() for row in rows)

    rows[1]["manual_link"] = ["#b12"]
    dump_manual_links(ledger_path, rows)
    first_render = ledger_path.read_text(encoding="utf-8")
    dump_manual_links(ledger_path, rows)
    assert ledger_path.read_text(encoding="utf-8") == first_render
    assert tuple(yaml.safe_load(first_render)[0]) == MANUAL_LINK_FIELDS

    regenerated = merge_manual_links(unresolved, sources, ledger_path)
    assert regenerated[1]["manual_link"] == ["#b12"]


def test_manual_link_ledger_rejects_missing_tei_and_stale_completed_work(tmp_path):
    sources = nowhere(tmp_path, tmp_path / "cache")
    missing = [{"paper_id": "paper", "citation": "(Example)", "marker": "Example"}]
    assert (
        merge_manual_links(missing, sources, tmp_path / "manual-links.yaml")[0][
            "manual_link"
        ]
        == []
    )

    sources.cache.mkdir()
    (sources.cache / "paper.tei.xml").write_text("<TEI/>", encoding="utf-8")
    ledger_path = tmp_path / "manual-links.yaml"
    rows = merge_manual_links(missing, sources, ledger_path)
    rows[0]["manual_link"] = ["https://example.org/reference"]
    dump_manual_links(ledger_path, rows)
    with pytest.raises(ValueError, match="cannot discard completed manual link"):
        merge_manual_links([], sources, ledger_path)


def test_minimum_link_gate_stops_the_run_before_it_writes_the_ledger(tmp_path):
    ground_truth_path = tmp_path / "ground-truth.json"
    ground_truth_path.write_text(json.dumps(ground_truth()), encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / f"paper_{PAPER_ID}.grobid.tei.xml").write_bytes(TEI)
    manual_links = tmp_path / "manual-links.yaml"
    out = tmp_path / "out"

    done = run_cli(
        tmp_path,
        "closure",
        "resolve",
        "--ground-truth",
        str(ground_truth_path),
        "--grobid-archive",
        str(archive),
        "--out",
        str(out),
        "--manual-links",
        str(manual_links),
        "--min-linked",
        "2",
    )

    assert done.returncode == 1
    assert "linked_markers 1 < 2" in done.stderr
    assert not manual_links.exists()
    assert not (out / "citations.json").exists()
    assert not (out / "report.json").exists()
    assert not (out / "review.csv").exists()


def test_resolve_writes_the_dataset_the_report_the_review_and_a_receipt(tmp_path):
    ground_truth_path = tmp_path / "ground-truth.json"
    ground_truth_path.write_text(json.dumps(ground_truth()), encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / f"paper_{PAPER_ID}.grobid.tei.xml").write_bytes(TEI)
    manual_links = tmp_path / "manual-links.yaml"
    out = tmp_path / "out"

    done = run_cli(
        tmp_path,
        "closure",
        "resolve",
        "--ground-truth",
        str(ground_truth_path),
        "--grobid-archive",
        str(archive),
        "--out",
        str(out),
        "--manual-links",
        str(manual_links),
    )

    assert done.returncode == 0, done.stderr
    report = json.loads((out / "report.json").read_text())
    assert report["resolver"] == "taxonomy_pipeline.closure.extend_gt/5"
    assert len(report["resolver_sha256"]) == 64
    dataset = json.loads((out / "citations.json").read_text())
    assert PAPER_ID in [p["paper_id"] for p in dataset["papers"]]
    with (out / "review.csv").open(encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == list(REVIEW_COLUMNS)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["command"] == "mbdef closure resolve"
    assert (
        receipt["inputs"]["ground_truth"]["sha256"]
        == hashlib.sha256(ground_truth_path.read_bytes()).hexdigest()
    )
    assert manual_links.exists()
