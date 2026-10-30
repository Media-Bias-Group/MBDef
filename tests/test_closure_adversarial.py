"""Adversarial identities and whole-run invariants of the closure step, independent of benchmark IDs.

The fixtures describe arbitrary publications; none encodes a benchmark paper or an observed ground-truth result. The collect step's identity rules have their own tests (tests/test_collect*.py).

Where `closure check` shows the behaviour in its outputs, the test runs the command (`closure_check` below).
The others call the functions: each checks one identity rule, or an invariant over many generated inputs (order independence, no exhaustive fuzzy matching), that no command output shows on its own.
"""

import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import pytest
from cli_helpers import run_cli

from mbdef.closure import corpus_check as corpus
from mbdef.closure.citations import build_index, resolve_unit
from mbdef.closure.corpus_check import (
    annotate_corpus_membership,
    check_citation_corpus,
    corpus_identities,
)
from mbdef.closure.extend_gt import extend
from mbdef.closure.tei_sources import Sources, load_tei
from mbdef.collect.identity import is_arxiv_doi
from mbdef.extract.records import FIELDS

"""Independent, identifier-agnostic closure regression contract.

The fixtures describe arbitrary publications; none encode a benchmark paper or an
observed GT result. These assertions specify expected behavior, so the pre-fix PR
is expected to fail them. Run with pytest from the checkout root.
"""


def _corpus_row(*, doi=None, file="paper.xml", title="A shared title", authors=None):
    return {
        "corpus_path": "corpus",
        "file": file,
        "doi": doi,
        "doi_readings": [],
        "title": title,
        "year": "2020",
        "authors": authors or ["Smith, A."],
        "raw": title,
    }


def _reference(*, doi=None, title="A shared title", year="2020", authors=None):
    return {
        "bibl_id": "b0",
        "title": title,
        "doi": doi,
        "year": year,
        "authors": authors or ["Smith, A."],
        "raw": title,
    }


def _dataset(references):
    return {
        "paper_count": 1,
        "papers": [
            {
                "paper_id": "seed",
                "metadata": {"title": "Seed paper"},
                "definition_records": [
                    {
                        "term": "term",
                        **{field: [] for field in FIELDS},
                        "definitions": [
                            {
                                "text": "Definition [1]",
                                "citation": ["[1]"],
                                "resolved_citations": [
                                    {
                                        "raw": "[1]",
                                        "marker": "1",
                                        "method": "ref_target",
                                        "confidence": "high",
                                        "references": references,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_legacy_filename_alternative_doi_is_not_verified_identity():
    existing = _corpus_row(
        doi="10.1234/a_b", title="Original paper", authors=["Alpha, A."]
    )
    existing["doi_readings"] = ["10.1234/a/b"]
    requested = _reference(
        doi="10.1234/a/b",
        title="Entirely different paper",
        year="2022",
        authors=["Beta, B."],
    )
    _, report, _ = corpus.annotate_corpus_membership(
        _dataset([requested]), [existing], []
    )
    assert report["papers"][0]["corpus_match"]["status"] != "present", (
        "A lossy underscore/slash filename reading is only a candidate, especially "
        "when independently extracted title, author and year conflict."
    )


def test_doi_less_corpus_duplicate_deduplication_is_order_independent():
    rows = [
        _corpus_row(doi="10.1234/same", file="has-doi.xml"),
        _corpus_row(file="no-doi.xml"),
    ]
    counts = [len(corpus.corpus_identities(order)) for order in (rows, rows[::-1])]
    assert counts == [1, 1], (
        "The same compatible DOI/DOI-less duplicate must not become two identities "
        "when corpus roots or filenames are reordered."
    )


def test_doi_less_citation_deduplication_is_order_independent():
    references = [_reference(doi="10.1234/same"), _reference()]
    counts = []
    for order in (references, references[::-1]):
        occurrences, _ = corpus._occurrences(_dataset(deepcopy(order)))
        counts.append(len(corpus.cited_papers(occurrences)))
    assert counts == [1, 1], (
        "Publication counts and provider work must not change with seed/citation order."
    )


def test_exact_doi_lookup_does_not_sweep_same_year_fuzzy_candidates(monkeypatch):
    rows = [
        _corpus_row(
            doi=f"10.1234/paper{i}",
            file=f"paper{i}.xml",
            title=f"A rigorous descriptive title of publication number {i:04d}",
        )
        for i in range(200)
    ]
    index = corpus._corpus_index(corpus.corpus_identities(rows))
    comparisons = 0
    original = corpus._similarity

    def counted(left, right):
        nonlocal comparisons
        comparisons += 1
        return original(left, right)

    monkeypatch.setattr(corpus, "_similarity", counted)
    for row in rows[:50]:
        paper = corpus._evidence(row["doi"], row["title"], row["year"], row["authors"])
        match, _ = corpus._match(paper, index, False)
        assert match["status"] == "present"
    assert comparisons <= 100, (
        f"{comparisons} fuzzy comparisons were performed for 50 exact DOI matches; "
        "exact identifier lookup must avoid scanning same-year corpus candidates."
    )


def test_same_exact_marker_with_different_targets_cannot_be_a_unique_high_confidence_link():
    entries = {
        f"b{i}": {
            "authors": [{"surname": "Smith"}],
            "year": "2020",
            "raw": f"Smith 2020 Different publication {i}",
        }
        for i in range(2)
    }
    parsed = {
        "entries": entries,
        "back_text": "",
        "refs": [
            {
                "text": "(Smith et al., 2020)",
                "target": f"b{i}",
                "paragraph": i,
                "context": f"Different paragraph {i}",
            }
            for i in range(2)
        ],
    }
    result = resolve_unit("(Smith et al., 2020)", build_index(parsed))
    assert len(result["bibl_ids"]) != 1 or result["confidence"] == "low", (
        "The same rendered marker points to two bibliography entries. Selecting "
        "the first target with high/medium confidence silently guesses the work."
    )


def test_malformed_seed_tei_is_isolated_and_reported(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "paper_seed.grobid.tei.xml").write_bytes(b"<TEI><broken")
    sources = Sources(
        archive=archive,
        grobid_files=tmp_path / "absent-directory",
        source_zip=tmp_path / "absent.zip",
        cache=tmp_path / "cache",
    )
    dataset = _dataset([])
    # Removing already-resolved output models the native input to extend().
    item = dataset["papers"][0]["definition_records"][0]["definitions"][0]
    item.pop("resolved_citations")
    result, report, unresolved, _ = extend(dataset, sources)
    assert report["papers"] == 1
    assert report["linked_markers"] == 0
    assert result["papers"][0]["tei"]["bibliography_entries"] == 0
    assert unresolved, (
        "Broken extraction must remain visible, not disappear from coverage."
    )


"""Desired corpus-closure invariants, independently checked against the closure's first version.

Run from the PR checkout: python -m pytest -q <this file>
Failures reproduce review findings; this is review evidence, not production code.
"""


def dataset(*, method="ref_target", confidence="high"):
    return {
        "paper_count": 1,
        "papers": [
            {
                "paper_id": "seed",
                "title": "Seed paper",
                "definition_records": [
                    {
                        "term": "bias",
                        **{field: [] for field in FIELDS},
                        "definitions": [
                            {
                                "text": "A definition",
                                "citation": ["Smith (2020)"],
                                "resolved_citations": [
                                    {
                                        "raw": "Smith (2020)",
                                        "marker": "Smith (2020)",
                                        "method": method,
                                        "confidence": confidence,
                                        "references": [
                                            {
                                                "bibl_id": "b0",
                                                "title": "Referenced work",
                                                "authors": ["Smith, A."],
                                                "year": "2020",
                                                "doi": "10.1234/example",
                                                "raw": "Referenced work",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def closure_check(
    tmp_path, payload: dict, corpus: Path
) -> tuple[dict, list[dict], dict]:
    """`closure check` of PAYLOAD against the TEI files in CORPUS: its coverage report, its queue of works to collect and its annotated dataset."""
    data = tmp_path / "citations.json"
    data.write_text(json.dumps(payload))
    # One unrelated paper, so the corpus root is not empty.
    (corpus / "unrelated.tei.xml").write_bytes(
        fulltext("10.5555/unrelated", child="10.5555/child", title="An unrelated paper")
    )
    out = tmp_path / "checked"
    done = run_cli(
        tmp_path,
        *("closure", "check", "--dataset", str(data), "--corpus", str(corpus)),
        *("--out", str(out)),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    with (out / "citation-papers-to-collect.csv").open(newline="") as handle:
        queue = list(csv.DictReader(handle))
    coverage = json.loads((out / "coverage.json").read_text())
    return coverage, queue, json.loads((out / "annotated-dataset.json").read_text())


def test_extractor_prediction_envelope_can_be_checked(tmp_path):
    payload = dataset()
    del payload["paper_count"]  # src.extract and saved predictions omit it.
    (tmp_path / "corpus").mkdir()

    coverage, _, _ = closure_check(tmp_path, payload, tmp_path / "corpus")

    assert coverage["summary"]["reference_occurrences"] == 1


def test_ambiguous_citation_candidate_requires_review_before_acquisition(tmp_path):
    payload = dataset(method="bibl_author_year_ambiguous", confidence="low")
    (tmp_path / "corpus").mkdir()

    _, queue, _ = closure_check(tmp_path, payload, tmp_path / "corpus")

    assert queue[0]["queue_status"] == "review"


def test_same_title_year_with_conflicting_authors_are_distinct_publications():
    rows = [
        {
            "corpus_path": "corpus",
            "file": f"{surname}.tei.xml",
            "title": "Introduction",
            "year": "2020",
            "doi": None,
            "authors": [f"{surname}, A."],
            "raw": "Introduction",
        }
        for surname in ("Smith", "Jones")
    ]
    assert len(corpus_identities(rows)) == 2


def test_acquired_papers_bibliographies_do_not_expand_the_seed_queue(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = corpus / "acquired.tei.xml"
    target.write_text("""<TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader><fileDesc><titleStmt><title>Referenced work</title></titleStmt>
        <sourceDesc><biblStruct><analytic><title level="a">Referenced work</title>
        <idno type="DOI">10.1234/example</idno></analytic></biblStruct></sourceDesc>
      </fileDesc></teiHeader>
      <text><body><p>First-level paper text.</p></body><back><listBibl>
        <biblStruct xml:id="b0"><analytic><title level="a">Second-level work</title>
        <idno type="DOI">10.1234/second-level</idno></analytic></biblStruct>
      </listBibl></back></text></TEI>""")

    coverage, queue, annotated = closure_check(tmp_path, dataset(), corpus)

    assert coverage["summary"]["unique_cited_papers"] == 1
    assert coverage["summary"]["present"] == 1
    assert queue == []
    assert [paper["paper_id"] for paper in annotated["papers"]] == ["seed"]


"""Whole-run invariants and adversarial identities, independent of benchmark IDs."""


def fulltext(doi, *, child="10.9876/child", title="A real paper"):
    return f"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>
    <titleStmt><title>{title}</title></titleStmt><sourceDesc><biblStruct>
    <analytic><title level="a">{title}</title><idno type="DOI">{doi}</idno>
    <author><persName><surname>Author</surname></persName></author></analytic>
    <monogr><imprint><date when="2020"/></imprint></monogr></biblStruct></sourceDesc>
    </fileDesc></teiHeader><text><body><p>A definition <ref type="bibr" target="#b0">[1]</ref>.</p></body>
    <back><listBibl><biblStruct xml:id="b0"><analytic><title level="a">Child work</title>
    <idno type="DOI">{child}</idno><author><persName><surname>Author</surname></persName></author>
    </analytic><monogr><imprint><date when="2020"/></imprint></monogr></biblStruct>
    </listBibl></back></text></TEI>""".encode()


@pytest.mark.parametrize("left,right", [("α", "β"), ("猫", "犬"), ("x+y", "x-y")])
def test_distinct_unicode_titles_do_not_become_the_same_work(left, right):
    existing = _corpus_row(title=f"An extensive analysis of {left} in systems")
    _, report, _ = annotate_corpus_membership(
        _dataset([_reference(title=f"An extensive analysis of {right} in systems")]),
        [existing],
        [],
    )
    assert report["summary"]["present"] == 0


def test_opaque_doi_suffix_does_not_override_conflicting_doi_and_year():
    assert not is_arxiv_doi("10.1000/arxiv-study")
    _, report, _ = annotate_corpus_membership(
        _dataset([_reference(doi="10.2000/different", year="2023")]),
        [_corpus_row(doi="10.1000/arxiv-study")],
        [],
    )
    assert report["summary"]["present"] == 0


def test_seed_legacy_filename_cannot_override_conflicting_header(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "doi_10.1234_a_b.grobid.tei.xml").write_bytes(fulltext("10.1234/a/b"))
    sources = Sources(
        archive=tmp_path / "archive",
        grobid_files=root,
        source_zip=tmp_path / "absent.zip",
        cache=tmp_path / "cache",
    )
    with pytest.raises(FileNotFoundError, match="source identity not verified"):
        load_tei({"paper_id": "doi:10.1234/a_b"}, sources)


# (catalogue DOI, DOI GROBID read from the PDF). All but the last are ground-truth seed papers; each DOI read from the PDF is a template or garbled, which is no evidence of another work.
GARBLED_SEED_DOIS = [
    ("10.1145/3664194", "10.1145/nnnnnnn.nnnnnnn"),  # ACM template
    ("10.48550/arxiv.2312.16148", "10.1145/1122445.1122456"),  # ACM sample DOI
    ("10.18653/v1/2020.alw-1.19", "10.18653/v1/p17"),  # ACL template
    ("10.1146/annurev-polisci-051117-073034", "10.1146/annurev-polisci-051117-"),
    ("10.1097/fch.0000000000000183", "10.1097/fch.0000000000000183."),
    ("10.1037/0022-3514.79.5.763", "10.1037//0022-3514.79.5.763"),
    (
        "10.1080/1369183x.2015.1133279",
        "10.10800/133693183x.20105.11303279.htp://www.tandfonline.com/doi/abs/10.10800/133693183x.20105.1130",
    ),
    ("10.1177/1464884915610988", "10.1177/1464884915610988jou.sagepub.com"),
]


def seed_sources(tmp_path, header_doi):
    root = tmp_path / "sources"
    root.mkdir()
    (root / "seed.tei.xml").write_bytes(fulltext(header_doi))
    return Sources(
        archive=tmp_path / "archive",
        grobid_files=tmp_path / "grobid",
        source_zip=tmp_path / "absent.zip",
        cache=tmp_path / "cache",
        directories=(root,),
    )


@pytest.mark.parametrize("catalogue,header", GARBLED_SEED_DOIS)
def test_seed_whose_pdf_doi_is_garbled_loads_with_a_note(tmp_path, catalogue, header):
    sources = seed_sources(tmp_path, header)
    paper = {"paper_id": "seed", "metadata": {"doi": catalogue}}
    for _ in range(2):  # the fresh read, then the cached copy
        data, provenance = load_tei(paper, sources)
        assert data == fulltext(header)
        assert provenance["doi_note"].startswith(f"DOI {header} ")
        assert "is no evidence" in provenance["doi_note"]


@pytest.mark.parametrize(
    "catalogue,header",
    [
        ("10.1145/3664194", "10.1145/3292500.3330701"),  # another paper
        ("10.4324/9781003163237", "10.4324/9781003163237-11"),  # a chapter of the book
    ],
)
def test_seed_whose_pdf_doi_names_another_work_is_still_refused(
    tmp_path, catalogue, header
):
    sources = seed_sources(tmp_path, header)
    with pytest.raises(FileNotFoundError, match="source identity not verified"):
        load_tei({"paper_id": "seed", "metadata": {"doi": catalogue}}, sources)


def test_seed_with_its_own_doi_has_no_note(tmp_path):
    sources = seed_sources(tmp_path, "10.9876/seed")
    paper = {"paper_id": "seed", "metadata": {"doi": "10.9876/seed"}}
    assert load_tei(paper, sources)[1]["doi_note"] == ""


def test_doi_named_seed_file_still_needs_a_readable_doi(tmp_path):
    root = tmp_path / "grobid"
    root.mkdir()
    (root / "doi_10.1145_3664194.grobid.tei.xml").write_bytes(
        fulltext("10.1145/nnnnnnn.nnnnnnn")
    )
    sources = Sources(
        archive=tmp_path / "archive",
        grobid_files=root,
        source_zip=tmp_path / "absent.zip",
        cache=tmp_path / "cache",
    )
    with pytest.raises(FileNotFoundError, match="source identity not verified"):
        load_tei({"paper_id": "doi:10.1145/3664194"}, sources)


def test_garbled_seed_doi_links_its_citations_and_is_named_in_the_report(tmp_path):
    sources = seed_sources(tmp_path, "10.1145/nnnnnnn.nnnnnnn")
    dataset = {
        "papers": [
            {
                "paper_id": "seed",
                "metadata": {"doi": "10.1145/3664194"},
                "definition_records": [
                    {
                        "term": "term",
                        "definitions": [
                            {"text": "A definition [1].", "citation": ["[1]"]}
                        ],
                    }
                ],
            }
        ]
    }
    _, report, unresolved, _ = extend(dataset, sources)
    assert report["papers_without_tei"] == [] and unresolved == []
    assert report["linked_markers"] == 1
    assert "10.1145/nnnnnnn.nnnnnnn" in report["papers_detail"][0]["source_doi_note"]


@pytest.mark.parametrize("output_index", range(3))
def test_all_membership_outputs_refuse_to_overwrite_input(tmp_path, output_index):
    dataset = tmp_path / "seed.json"
    dataset.write_text(json.dumps(_dataset([])))
    outputs = [tmp_path / f"output-{i}" for i in range(3)]
    outputs[output_index] = dataset
    before = dataset.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        check_citation_corpus(dataset, [], *outputs)
    assert dataset.read_bytes() == before


def test_order_independence_over_generated_identifiers():
    rng = random.Random(91021)
    rows = [
        _corpus_row(
            doi=f"10.9876/{rng.getrandbits(128):032x}",
            title=f"Work {i}",
            file=f"{i}.xml",
        )
        for i in range(60)
    ]
    refs = [_reference(doi=row["doi"], title=row["title"]) for row in rows]
    for _ in range(12):
        rng.shuffle(rows)
        rng.shuffle(refs)
        _, report, queue = annotate_corpus_membership(_dataset(refs), rows, [])
        assert report["summary"]["present"] == len(rows) and not queue
        assert report["summary"]["reference_occurrences"] == len(refs)


def test_dense_negative_queries_do_not_run_exhaustive_fuzzy_matching(monkeypatch):
    from mbdef.closure import corpus_check

    monkeypatch.setattr(
        corpus_check,
        "_similarity",
        lambda *a: pytest.fail("unbounded diagnostic search"),
    )
    rows = [
        _corpus_row(doi=f"10.1111/{i}", title=f"An extensive system study {i}")
        for i in range(1000)
    ]
    refs = [
        _reference(doi=f"10.2222/{i}", title=f"An extensive systems study {i}")
        for i in range(100)
    ]
    _, report, _ = annotate_corpus_membership(_dataset(refs), rows, [])
    assert report["summary"]["missing"] == 100


def test_filename_guesses_cannot_poison_a_verified_doi():
    verified = _corpus_row(doi="10.1234/a_b", title="Correct paper", file="real.xml")
    guessed = _corpus_row(doi="10.1234/a_b", title="Different paper", file="legacy.xml")
    guessed.update(doi_unverified=True, doi_readings=["10.1234/a/b"])
    _, report, _ = annotate_corpus_membership(
        _dataset([_reference(doi="10.1234/a_b", title="")]), [guessed, verified], []
    )
    match = report["papers"][0]["corpus_match"]
    assert match["status"] == "present"
    assert [row["file"] for row in match["files"]] == ["real.xml"]


def test_conflicting_extracted_doi_claims_remain_reviewable():
    rows = [
        _corpus_row(doi="10.1234/conflict", title=f"Distinct work {i}", file=f"{i}.xml")
        for i in range(2)
    ]
    _, report, _ = annotate_corpus_membership(
        _dataset([_reference(doi="10.1234/conflict", title="")]), rows, []
    )
    assert report["papers"][0]["corpus_match"]["reason"] == "ambiguous_corpus_doi"
