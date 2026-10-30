import csv
import json
import zipfile
from hashlib import sha256

import pytest
from cli_helpers import run_cli

from mbdef.closure.corpus_check import check_citation_corpus, scan_corpus
from mbdef.collect.identity import (
    is_arxiv_doi,
    normalize_doi,
    parse_paper_header,
)
from mbdef.extract.records import FIELDS

# Everything after `</teiHeader>` is deliberately unparseable: a header-only reader must never touch it, and a full-document parse of this fixture would raise.
BROKEN_BODY = "<text><body><p>unclosed\n"


def tei_payload(*, title=None, authors=(), year=None, doi=None, idno=None):
    persons = "".join(
        "<author><persName>"
        f'<forename type="first">{author.split()[0]}</forename>'
        f"<surname>{author.split()[-1]}</surname>"
        "</persName></author>"
        for author in authors
    )
    analytic = persons + (
        f'<title level="a" type="main">{title}</title>' if title else ""
    )
    imprint = f'<date type="published" when="{year}"/>' if year else ""
    idnos = f'<idno type="DOI">{doi}</idno>' if doi else ""
    if idno:
        idnos += f'<idno type="arXiv">{idno}</idno>'
    return (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
        + (
            f'<titleStmt><title level="a" type="main">{title}</title></titleStmt>'
            if title
            else "<titleStmt/>"
        )
        + f"<sourceDesc><biblStruct><analytic>{analytic}</analytic>"
        + f"<monogr><imprint>{imprint}</imprint></monogr>{idnos}</biblStruct></sourceDesc>"
        + "</fileDesc></teiHeader>"
        + BROKEN_BODY
    )


def write_tei(root, name, **header):
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(tei_payload(**header), encoding="utf-8")
    return path


def write_zip(path, members):
    """`members`: list of `(member name, header kwargs)`, as in `data/paper_sources.zip`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, header in members:
            archive.writestr(name, tei_payload(**header))
    return path


def reference(
    *,
    bibl_id="b0",
    title=None,
    authors=(),
    year=None,
    doi=None,
    url=None,
    raw=None,
    formatted=None,
):
    return {
        "bibl_id": bibl_id,
        "authors": list(authors),
        "year": year,
        "years_in_text": [year] if year else [],
        "title": title,
        "doi": doi,
        "url": url,
        "formatted": formatted or (f"{title} ({year})" if title else None),
        "raw": raw if raw is not None else (title or ""),
    }


def dataset_payload(papers):
    """`papers`: list of `(paper_id, [[reference, ...], ...])`, one marker per inner list."""
    return {
        "schema_version": "test",
        "dataset_id": "test",
        "paper_count": len(papers),
        "papers": [
            {
                "paper_id": paper_id,
                "split": "test",
                "metadata": {"title": f"Citing {paper_id}"},
                "definition_records": [
                    {
                        "term": "term",
                        **{field: [] for field in FIELDS},
                        "definitions": [
                            {
                                "citation": [f"(m{index})"],
                                "text": "definition text",
                                "resolved_citations": [
                                    {
                                        "raw": f"(m{index})",
                                        "marker": f"m{index}",
                                        "method": "ref_target",
                                        "confidence": "high",
                                        "references": references,
                                    }
                                ],
                            }
                            for index, references in enumerate(markers)
                        ],
                    }
                ],
                "tei": {"kind": "archive"},
            }
            for paper_id, markers in papers
        ],
    }


def check(tmp_path, papers, roots, **kwargs):
    kwargs.setdefault("fuzzy_suggestions", True)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset_payload(papers)), encoding="utf-8")
    closure = dataset_path.read_bytes()
    annotated_path = tmp_path / "dataset_corpus.json"
    report_path = tmp_path / "report.json"
    csv_path = tmp_path / "queue.csv"
    check_citation_corpus(
        dataset_path,
        list(roots),
        report_path,
        csv_path,
        annotated_path,
        workers=2,
        **kwargs,
    )
    assert dataset_path.read_bytes() == closure, (
        "the closure dataset is read-only input"
    )
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        json.loads(annotated_path.read_text(encoding="utf-8")),
        json.loads(report_path.read_text(encoding="utf-8")),
        rows,
    )


def annotations(dataset):
    return [
        reference["corpus_match"]
        for paper in dataset["papers"]
        for record in paper["definition_records"]
        for item in record["definitions"]
        for citation in item["resolved_citations"]
        for reference in citation["references"]
    ]


def row_for(report, title):
    found = [row for row in report["papers"] if row["reference"]["title"] == title]
    assert len(found) == 1, f"expected one report row for {title!r}"
    return found[0]


def test_parse_paper_header_reads_header_metadata_without_the_body(tmp_path):
    path = write_tei(
        tmp_path / "corpus",
        "paper_a.grobid.tei.xml",
        title="Framing Effects in Risk Communication",
        authors=["Ada Lovelace", "Grace Hopper"],
        year="1982",
        doi="https://doi.org/10.1000/Alpha",
    )
    header = parse_paper_header(path)
    assert header["title"] == "Framing Effects in Risk Communication"
    assert header["authors"] == ["Lovelace, A.", "Hopper, G."]
    assert header["year"] == "1982"
    assert header["doi"] == "10.1000/alpha"


def test_normalizers_cover_doi_prefixes_and_the_arxiv_registrant():
    assert normalize_doi(" DOI: https://doi.org/10.1000/Alpha.") == "10.1000/alpha."
    assert normalize_doi("10.1000/a%2Fb") == "10.1000/a%2fb"
    assert normalize_doi("https://doi.org/10.1000/a%2Fb") == "10.1000/a/b"
    assert normalize_doi(" 10.1000 / alpha \n") == ""
    assert normalize_doi("10.1037//1089-2680.2.2.175") == "10.1037//1089-2680.2.2.175"
    assert normalize_doi("not a doi") == ""
    assert is_arxiv_doi("10.48550/arXiv.2301.00001")
    assert not is_arxiv_doi("10.1101/2020.01.01.900001")


def test_doi_beats_title_and_duplicate_files_collapse_into_one_present_paper(tmp_path):
    archive = tmp_path / "archive"
    mirror = tmp_path / "mirror"
    write_tei(
        archive,
        "paper_alpha.grobid.tei.xml",
        title="A Header Title GROBID Mangled",
        authors=["Ada Lovelace"],
        year="1982",
        doi="10.1000/alpha",
    )
    write_tei(
        mirror,
        "doi_10.1000_alpha.grobid.tei.xml",
        title="A Header Title GROBID Mangled",
        authors=["Ada Lovelace"],
        year="1982",
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="The Published Title",
                            authors=["Lovelace, A."],
                            year="1982",
                            doi="10.1000/ALPHA",
                        )
                    ]
                ],
            )
        ],
        [archive, mirror],
    )
    match = annotations(dataset)[0]
    assert match["status"] == "present"
    assert match["in_corpus"] is True
    assert match["method"] == "doi"
    assert [entry["file"] for entry in match["files"]] == [
        "paper_alpha.grobid.tei.xml",
        "doi_10.1000_alpha.grobid.tei.xml",
    ]
    assert {entry["corpus_path"] for entry in match["files"]} == {
        str(archive),
        str(mirror),
    }
    assert report["summary"]["present"] == 1
    assert report["summary"]["unique_cited_papers"] == 1
    assert rows == []


def test_repeated_slashes_are_not_assumed_to_be_doi_aliases(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "nickerson.tei.xml",
        title="Confirmation Bias: A Ubiquitous Phenomenon in Many Guises",
        authors=["Raymond Nickerson"],
        year="1998",
        doi="10.1037/1089-2680.2.2.175",
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="Confirmation bias: A ubiquitous phenomenon in many guises.",
                            authors=["Nickerson, R. S."],
                            year="1998",
                            doi="10.1037//1089-2680.2.2.175",
                        )
                    ],
                    [
                        reference(
                            title="Confirmation bias: A ubiquitous phenomenon in many guises",
                            authors=["Nickerson, R. S."],
                            year="1998",
                            doi="10.1037/1089-2680.2.2.175",
                        )
                    ],
                ],
            )
        ],
        [corpus],
    )
    assert report["summary"]["reference_occurrences"] == 2
    assert report["summary"]["unique_cited_papers"] == 2
    assert [match["status"] for match in annotations(dataset)] == ["review", "present"]
    assert rows[0]["review_reason"] == "metadata_conflict"


def test_all_observed_titles_participate_in_matching(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "paper.tei.xml",
        title="Canonical Published Title",
        authors=["Ada Lovelace"],
        year="2001",
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="An Alternate Indexed Title",
                            authors=["Lovelace, A."],
                            year="2001",
                            doi="10.1000/shared",
                        )
                    ],
                    [
                        reference(
                            title="Canonical Published Title",
                            authors=["Lovelace, A."],
                            year="2001",
                            doi="10.1000/shared",
                        )
                    ],
                ],
            )
        ],
        [corpus],
    )
    assert report["summary"]["unique_cited_papers"] == 1
    assert [(match["status"], match["method"]) for match in annotations(dataset)] == [
        ("present", "title_year_author_exact"),
        ("present", "title_year_author_exact"),
    ]
    assert rows == []


def test_exact_title_matches_only_with_corroborating_year_or_author(tmp_path):
    corpus = tmp_path / "corpus"
    for name, title, authors, year in (
        ("one.tei.xml", "Both Year And Author", ["Ada Lovelace"], "2001"),
        ("two.tei.xml", "Only Year Agrees", ["Grace Hopper"], "2002"),
        ("three.tei.xml", "Only Author Agrees", ["Ada Lovelace"], None),
        ("four.tei.xml", "Nothing Corroborates", [], None),
    ):
        write_tei(corpus, name, title=title, authors=authors, year=year)
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="Both Year And Author",
                            authors=["Lovelace, A."],
                            year="2001",
                        )
                    ],
                    [reference(title="Only Year Agrees", authors=[], year="2002")],
                    [reference(title="Only Author Agrees", authors=["Lovelace, A."])],
                    [reference(title="Nothing Corroborates")],
                ],
            )
        ],
        [corpus],
    )
    assert [(match["status"], match["method"]) for match in annotations(dataset)] == [
        ("present", "title_year_author_exact"),
        ("present", "title_year_exact"),
        ("present", "title_author_exact"),
        ("review", None),
    ]
    assert row_for(report, "Nothing Corroborates")["corpus_match"]["reason"] == (
        "insufficient_reference_metadata"
    )
    assert [row["queue_status"] for row in rows] == ["review"]


def test_fuzzy_titles_need_exact_year_shared_author_and_no_doi_conflict(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "accepted.tei.xml",
        title="Deep Learning for Media Bias Detection",
        authors=["Ada Lovelace"],
        year="2019",
    )
    write_tei(
        corpus,
        "year.tei.xml",
        title="Automated Detection of Framing in News Coverage",
        authors=["Grace Hopper"],
        year="2015",
    )
    write_tei(
        corpus,
        "doi.tei.xml",
        title="Measuring Partisan Slant in Newspaper Headlines",
        authors=["Alan Turing"],
        year="2011",
        doi="10.1000/published",
    )
    write_tei(
        corpus,
        "author.tei.xml",
        title="Detecting Ideological Framing in Political Text",
        authors=["Grace Hopper"],
        year="2020",
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="Deep Learning for Media Bias Detections",
                            authors=["Lovelace, A."],
                            year="2019",
                        )
                    ],
                    [
                        reference(
                            title="Automated Detection of Framing in News Coverages",
                            authors=["Hopper, G."],
                            year="2016",
                        )
                    ],
                    [
                        reference(
                            title="Measuring Partisan Slant in Newspapers Headlines",
                            authors=["Turing, A."],
                            year="2011",
                            doi="10.1000/preprintless",
                        )
                    ],
                    [
                        reference(
                            title="Detecting Ideological Framings in Political Text",
                            authors=["Babbage, C."],
                            year="2020",
                        )
                    ],
                ],
            )
        ],
        [corpus],
    )
    assert [
        (match["status"], match["method"], match["reason"])
        for match in annotations(dataset)
    ] == [
        ("review", None, "unverified_corpus_candidates"),
        ("review", None, "metadata_conflict"),
        ("review", None, "metadata_conflict"),
        ("review", None, "metadata_conflict"),
    ]
    suggested = row_for(report, "Automated Detection of Framing in News Coverages")
    assert suggested["corpus_match"]["candidates"][0]["similarity"] >= 0.92
    assert {row["queue_status"] for row in rows} == {"review"}
    assert any("year.tei.xml" in row["suggested_corpus_files"] for row in rows)
    disjoint = row_for(report, "Detecting Ideological Framings in Political Text")
    assert disjoint["corpus_match"]["candidates"][0]["files"][0]["file"] == (
        "author.tei.xml"
    )


def test_conflicting_doi_year_or_author_evidence_is_never_auto_accepted(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "doi.tei.xml",
        title="Identical Title One",
        authors=["Ada Lovelace"],
        year="2001",
        doi="10.1000/corpus",
    )
    write_tei(
        corpus,
        "year.tei.xml",
        title="Identical Title Two",
        authors=["Grace Hopper"],
        year="1999",
    )
    write_tei(
        corpus,
        "author.tei.xml",
        title="Identical Title Three",
        authors=["Alan Turing"],
        year="2003",
    )
    dataset, _report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="Identical Title One",
                            authors=["Lovelace, A."],
                            year="2001",
                            doi="10.2000/reference",
                        )
                    ],
                    [
                        reference(
                            title="Identical Title Two",
                            authors=["Hopper, G."],
                            year="2004",
                        )
                    ],
                    [
                        reference(
                            title="Identical Title Three",
                            authors=["Babbage, C."],
                            year="2003",
                        )
                    ],
                ],
            )
        ],
        [corpus],
    )
    assert [
        (match["in_corpus"], match["reason"]) for match in annotations(dataset)
    ] == [
        (None, "metadata_conflict"),
        (None, "metadata_conflict"),
        (None, "metadata_conflict"),
    ]
    assert [row["queue_status"] for row in rows] == ["review"] * 3


def test_arxiv_preprint_matches_the_version_of_record_despite_doi_and_year_gap(
    tmp_path,
):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "published.tei.xml",
        title="Attention Is All You Need",
        authors=["Ada Lovelace", "Grace Hopper"],
        year="2018",
        doi="10.1000/published",
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="Attention Is All You Need",
                            authors=["Lovelace, A."],
                            year="2017",
                            doi="10.48550/arXiv.2301.00001",
                        )
                    ]
                ],
            )
        ],
        [corpus],
    )
    match = annotations(dataset)[0]
    assert (match["in_corpus"], match["status"], match["method"]) == (
        True,
        "present",
        "title_author_exact_arxiv_variant",
    )
    candidates = row_for(report, "Attention Is All You Need")["corpus_match"][
        "candidates"
    ]
    assert [(entry["doi"], entry["year"]) for entry in candidates] == [
        ("10.1000/published", "2018")
    ]
    assert candidates[0]["conflict"] is False
    assert rows == []


@pytest.mark.parametrize(
    ("title", "authors"),
    [
        ("Attention Is All You Need", ["Babbage, C."]),
        ("Attention Is All You Needs", ["Lovelace, A."]),
    ],
)
def test_arxiv_exemption_never_covers_disjoint_authors_or_fuzzy_titles(
    tmp_path, title, authors
):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "published.tei.xml",
        title="Attention Is All You Need",
        authors=["Ada Lovelace", "Grace Hopper"],
        year="2018",
        doi="10.1000/published",
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title=title,
                            authors=authors,
                            year="2018",
                            doi="10.48550/arXiv.2301.00001",
                        )
                    ]
                ],
            )
        ],
        [corpus],
    )
    match = annotations(dataset)[0]
    assert (match["in_corpus"], match["status"], match["reason"]) == (
        None,
        "review",
        "metadata_conflict",
    )
    assert match["files"] == []
    candidate = row_for(report, title)["corpus_match"]["candidates"][0]
    assert candidate["conflict"] is True
    assert [row["queue_status"] for row in rows] == ["review"]


def test_filename_dois_survive_nested_and_literal_underscore_suffixes(tmp_path):
    corpus = tmp_path / "corpus"
    # Both names come from the DOI mount, which replaced *every* `/` with `_`.
    write_tei(
        corpus,
        "doi_10.18653_v1_2023.acl-long.486.grobid.tei.xml",
        title="Conflicts, Villains, Resolutions",
        authors=["Lea Frermann"],
    )
    write_tei(
        corpus,
        "doi_10.48009_4_iis_2025_126.grobid.tei.xml",
        title="A Proceedings Paper With Underscores",
        authors=["Ada Lovelace"],
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="Conflicts, villains, resolutions",
                            authors=["Frermann, L."],
                            year="2023",
                            doi="10.18653/v1/2023.acl-long.486",
                        )
                    ],
                    [
                        reference(
                            title="A proceedings paper with underscores",
                            authors=["Lovelace, A."],
                            year="2025",
                            doi="10.48009/4_iis_2025_126",
                        )
                    ],
                ],
            )
        ],
        [corpus],
    )
    assert [(match["status"], match["method"]) for match in annotations(dataset)] == [
        ("present", "doi_filename_corroborated"),
        ("present", "doi_filename_corroborated"),
    ]
    assert [
        [entry["file"] for entry in match["files"]] for match in annotations(dataset)
    ] == [
        ["doi_10.18653_v1_2023.acl-long.486.grobid.tei.xml"],
        ["doi_10.48009_4_iis_2025_126.grobid.tei.xml"],
    ]
    assert report["summary"]["present"] == 2
    assert rows == []


def test_a_zip_root_is_a_corpus_of_tei_members(tmp_path):
    sources = write_zip(
        tmp_path / "paper_sources.zip",
        [
            (
                # `data/paper_sources.zip` names each member `quote(paper_id, safe="")`.
                "doi%3A10.1145%2F3677092.tei.xml",
                {"title": "A Header Title GROBID Mangled", "authors": ["Ada Lovelace"]},
            ),
            (
                "hashid.tei.xml",
                {
                    "title": "A Source Paper Without Any Identifier",
                    "authors": ["Grace Hopper"],
                    "year": "2011",
                },
            ),
        ],
    )
    papers, errors = scan_corpus([sources], workers=2)
    assert errors == []
    assert {row["corpus_path"] for row in papers} == {str(sources.resolve())}
    assert [(row["file"], row["doi"], row["source"]) for row in papers] == [
        ("doi%3A10.1145%2F3677092.tei.xml", "10.1145/3677092", "doi:10.1145/3677092"),
        ("hashid.tei.xml", None, "hashid"),
    ]
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [reference(title="The Published Title", doi="10.1145/3677092")],
                    [
                        reference(
                            title="A source paper without any identifier",
                            authors=["Hopper, G."],
                            year="2011",
                        )
                    ],
                ],
            )
        ],
        [sources],
    )
    assert [(match["status"], match["method"]) for match in annotations(dataset)] == [
        ("present", "doi"),
        ("present", "title_year_author_exact"),
    ]
    assert [entry["file"] for entry in annotations(dataset)[0]["files"]] == [
        "doi%3A10.1145%2F3677092.tei.xml"
    ]
    assert report["summary"]["present"] == 2
    assert rows == []


def test_a_zip_member_named_after_an_arxiv_url_is_preprint_evidence(tmp_path):
    sources = write_zip(
        tmp_path / "paper_sources.zip",
        [
            (
                "https%3A%2F%2Farxiv.org%2Fabs%2F2312.16148.tei.xml",
                {
                    "title": "The Media Bias Taxonomy",
                    "authors": ["Timo Spinde", "Bela Gipp"],
                    "year": "2024",
                    # Preprint headers carry the publisher template's sample DOI.
                    "doi": "10.1145/1122445.1122456",
                },
            )
        ],
    )
    dataset, _report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="The media bias taxonomy",
                            authors=["Spinde, T.", "Gipp, B."],
                            year="2023",
                        )
                    ]
                ],
            )
        ],
        [sources],
    )
    match = annotations(dataset)[0]
    assert match["status"] == "present"
    assert match["method"] == "title_author_exact_arxiv_variant"
    assert rows == []


def test_a_header_arxiv_idno_is_preprint_evidence_without_any_doi(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "preprint.tei.xml",
        title="Scaling Laws for Framing Detection",
        authors=["Ada Lovelace", "Grace Hopper"],
        year="2020",
        idno="2301.00001",
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="Scaling Laws for Framing Detection",
                            authors=["Lovelace, A."],
                            year="2019",
                            doi="10.1000/journal",
                        )
                    ]
                ],
            )
        ],
        [corpus],
    )
    match = annotations(dataset)[0]
    assert (match["status"], match["method"]) == (
        "present",
        "title_author_exact_arxiv_variant",
    )
    assert [entry["file"] for entry in match["files"]] == ["preprint.tei.xml"]
    candidate = row_for(report, "Scaling Laws for Framing Detection")["corpus_match"][
        "candidates"
    ][0]
    assert (candidate["doi"], candidate["conflict"]) == (None, False)
    assert rows == []


def test_preprint_and_published_citations_of_one_work_produce_one_queue_row(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "unrelated.tei.xml",
        title="An Unrelated Corpus Paper",
        authors=["Alan Turing"],
        year="2005",
        doi="10.1000/unrelated",
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing-a",
                [
                    [
                        reference(
                            title="One Work Two Manifestations",
                            authors=["Lovelace, A."],
                            year="2017",
                            doi="10.48550/arXiv.2301.00002",
                        )
                    ]
                ],
            ),
            (
                "citing-b",
                [
                    [
                        reference(
                            title="One Work Two Manifestations",
                            authors=["Lovelace, A.", "Hopper, G."],
                            year="2018",
                            doi="10.1000/journal",
                        )
                    ]
                ],
            ),
        ],
        [corpus],
    )
    assert report["summary"]["reference_occurrences"] == 2
    assert report["summary"]["unique_cited_papers"] == 1
    canonical = row_for(report, "One Work Two Manifestations")
    assert canonical["reference"]["doi"] == "10.1000/journal"
    assert canonical["reference"]["dois"] == [
        "10.1000/journal",
        "10.48550/arxiv.2301.00002",
    ]
    assert canonical["reference"]["years"] == ["2017", "2018"]
    assert len(canonical["occurrences"]) == 2
    assert len(rows) == 1
    assert rows[0]["queue_status"] == "missing"
    assert rows[0]["occurrence_count"] == "2"
    assert rows[0]["citing_paper_ids"] == "citing-a; citing-b"
    assert rows[0]["doi"] == "10.1000/journal; 10.48550/arxiv.2301.00002"
    assert rows[0]["year"] == "2017; 2018"
    assert rows[0]["review_reason"] == ""
    assert (
        annotations(dataset)
        == [
            {
                "in_corpus": False,
                "status": "missing",
                "method": None,
                "files": [],
                "reason": None,
            }
        ]
        * 2
    )


def test_every_candidate_of_a_multi_reference_marker_is_annotated(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "hit.tei.xml",
        title="Ambiguous Marker First Candidate",
        authors=["Ada Lovelace"],
        year="1990",
        doi="10.1000/first",
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            bibl_id="b1",
                            title="Ambiguous Marker First Candidate",
                            authors=["Lovelace, A."],
                            year="1990",
                            doi="10.1000/first",
                        ),
                        reference(
                            bibl_id="b2",
                            title="Ambiguous Marker Second Candidate",
                            authors=["Hopper, G."],
                            year="1991",
                            doi="10.1000/second",
                        ),
                    ]
                ],
            )
        ],
        [corpus],
    )
    assert [match["status"] for match in annotations(dataset)] == ["present", "missing"]
    assert report["summary"]["reference_occurrences"] == 2
    assert report["summary"]["unique_cited_papers"] == 2
    second = row_for(report, "Ambiguous Marker Second Candidate")
    assert second["occurrences"][0]["bibl_id"] == "b2"
    assert second["occurrences"][0]["marker"] == "m0"
    assert [row["title"] for row in rows] == ["Ambiguous Marker Second Candidate"]


def test_unresolved_markers_are_counted_and_statuses_sum_to_unique_papers(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "present.tei.xml",
        title="A Present Paper",
        authors=["Ada Lovelace"],
        year="2000",
        doi="10.1000/present",
    )
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [],
                    [
                        reference(
                            title="A Present Paper",
                            authors=["Lovelace, A."],
                            year="2000",
                            doi="10.1000/present",
                        )
                    ],
                    [
                        reference(
                            title="A Missing Paper",
                            authors=["Hopper, G."],
                            year="2001",
                            doi="10.1000/missing",
                        )
                    ],
                    [
                        reference(
                            url="https://example.org/no-title",
                            raw="untitled dataset note",
                        )
                    ],
                ],
            )
        ],
        [corpus],
    )
    summary = report["summary"]
    assert summary["resolved_markers_without_references"] == 1
    assert summary["reference_occurrences"] == 3
    assert (
        summary["present"] + summary["missing"] + summary["review"]
        == summary["unique_cited_papers"]
    )
    assert [match["in_corpus"] for match in annotations(dataset)] == [True, False, None]
    assert all(match["files"] for match in annotations(dataset) if match["in_corpus"])
    assert not any(
        match["files"] for match in annotations(dataset) if not match["in_corpus"]
    )
    assert len(rows) == summary["missing"] + summary["review"]


def test_titleless_references_without_url_or_raw_are_never_conflated(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "present.tei.xml",
        title="A Present Paper",
        authors=["Ada Lovelace"],
        year="2000",
        doi="10.1000/present",
    )
    dataset, report, rows = check(
        tmp_path,
        [("citing", [[reference(raw="")], [reference(raw="")]])],
        [corpus],
    )
    assert report["summary"]["reference_occurrences"] == 2
    assert report["summary"]["unique_cited_papers"] == 2
    assert report["summary"]["review"] == 2
    assert len(rows) == 2
    assert [match["reason"] for match in annotations(dataset)] == [
        "insufficient_reference_metadata",
        "insufficient_reference_metadata",
    ]


def test_scan_failures_downgrade_unmatched_papers_to_review_but_keep_matches(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "present.tei.xml",
        title="A Present Paper",
        authors=["Ada Lovelace"],
        year="2000",
        doi="10.1000/present",
    )
    (corpus / "broken.tei.xml").write_text("<TEI><teiHeader", encoding="utf-8")
    dataset, report, rows = check(
        tmp_path,
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="A Present Paper",
                            authors=["Lovelace, A."],
                            year="2000",
                            doi="10.1000/present",
                        )
                    ],
                    [
                        reference(
                            title="A Missing Paper",
                            authors=["Hopper, G."],
                            year="2001",
                            doi="10.1000/missing",
                        )
                    ],
                ],
            )
        ],
        [corpus],
    )
    source = report["sources"][0]
    assert (source["file_count"], source["parsed_count"], source["error_count"]) == (
        2,
        1,
        1,
    )
    assert source["errors"][0]["file"] == "broken.tei.xml"
    assert [(match["status"], match["reason"]) for match in annotations(dataset)] == [
        ("present", None),
        ("review", "corpus_scan_incomplete"),
    ]
    assert [row["review_reason"] for row in rows] == ["corpus_scan_incomplete"]


def test_an_empty_root_is_an_indexed_source_that_reviews_unless_allowed(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "present.tei.xml",
        title="A Present Paper",
        authors=["Ada Lovelace"],
        year="2000",
        doi="10.1000/present",
    )
    offline = tmp_path / "offline"
    offline.mkdir()
    papers = [
        (
            "citing",
            [
                [
                    reference(
                        title="A Missing Paper",
                        authors=["Hopper, G."],
                        year="2001",
                        doi="10.1000/missing",
                    )
                ]
            ],
        )
    ]
    dataset, report, _rows = check(tmp_path, papers, [corpus, offline])
    assert annotations(dataset)[0]["reason"] == "corpus_scan_incomplete"
    empty = [row for row in report["sources"] if row["path"] == str(offline)][0]
    assert (empty["file_count"], empty["parsed_count"], empty["error_count"]) == (
        0,
        0,
        1,
    )

    dataset, report, _rows = check(
        tmp_path, papers, [corpus, offline], allow_empty_root=True
    )
    assert annotations(dataset)[0]["status"] == "missing"
    empty = [row for row in report["sources"] if row["path"] == str(offline)][0]
    assert (empty["file_count"], empty["error_count"], empty["errors"]) == (0, 0, [])


def test_scan_without_any_tei_file_fails_before_touching_outputs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        scan_corpus([empty, tmp_path / "absent"])
    with pytest.raises(ValueError):
        scan_corpus([empty], workers=0)


def test_dataset_validation_failure_leaves_every_artifact_untouched(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "present.tei.xml",
        title="A Present Paper",
        year="2000",
        doi="10.1000/present",
    )
    dataset_path = tmp_path / "dataset.json"
    annotated_path = tmp_path / "dataset_corpus.json"
    report_path = tmp_path / "report.json"
    csv_path = tmp_path / "queue.csv"
    payload = dataset_payload(
        [
            (
                "citing",
                [
                    [
                        reference(
                            title="A Present Paper", year="2000", doi="10.1000/present"
                        )
                    ]
                ],
            )
        ]
    )
    payload["papers"][0]["definition_records"][0]["definitions"][0][
        "resolved_citations"
    ][0]["references"] = {"bibl_id": "b0"}
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    annotated_path.write_text("keep-me", encoding="utf-8")
    report_path.write_text("keep-me", encoding="utf-8")
    csv_path.write_text("keep-me", encoding="utf-8")
    before = dataset_path.read_bytes()
    with pytest.raises(
        ValueError,
        match=r"papers\[0\].definition_records\[0\].definitions\[0\].resolved_citations\[0\].references",
    ):
        check_citation_corpus(
            dataset_path, [corpus], report_path, csv_path, annotated_path, workers=2
        )
    assert dataset_path.read_bytes() == before
    assert annotated_path.read_text(encoding="utf-8") == "keep-me"
    assert report_path.read_text(encoding="utf-8") == "keep-me"
    assert csv_path.read_text(encoding="utf-8") == "keep-me"

    payload["papers"][0]["definition_records"][0]["definitions"][0][
        "resolved_citations"
    ][0]["references"] = []
    payload["paper_count"] = 7
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="paper_count"):
        check_citation_corpus(
            dataset_path, [corpus], report_path, csv_path, annotated_path, workers=2
        )
    assert report_path.read_text(encoding="utf-8") == "keep-me"


def test_annotating_the_closure_dataset_in_place_is_rejected(tmp_path):
    corpus = tmp_path / "corpus"
    write_tei(
        corpus,
        "present.tei.xml",
        title="A Present Paper",
        year="2000",
        doi="10.1000/present",
    )
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            dataset_payload(
                [
                    (
                        "citing",
                        [
                            [
                                reference(
                                    title="A Present Paper",
                                    year="2000",
                                    doi="10.1000/present",
                                )
                            ]
                        ],
                    )
                ]
            )
        ),
        encoding="utf-8",
    )
    before = dataset_path.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        check_citation_corpus(
            dataset_path,
            [corpus],
            tmp_path / "report.json",
            tmp_path / "queue.csv",
            tmp_path / "sub" / ".." / "dataset.json",
            workers=2,
        )
    assert dataset_path.read_bytes() == before


def test_check_takes_any_number_of_roots_and_reruns_byte_identically(tmp_path):
    first = tmp_path / "archive"
    second = tmp_path / "grobid"
    third = tmp_path / "extra"
    write_tei(
        first,
        "paper_one.grobid.tei.xml",
        title="First Corpus Paper",
        authors=["Ada Lovelace"],
        year="2001",
        doi="10.1000/one",
    )
    write_tei(
        second,
        "doi_10.1000_two.grobid.tei.xml",
        title="Second Corpus Paper",
        authors=["Grace Hopper"],
        year="2002",
    )
    write_tei(
        third,
        "paper_three.grobid.tei.xml",
        title="Third Corpus Paper",
        authors=["Alan Turing"],
        year="2003",
        doi="10.1000/three",
    )
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            dataset_payload(
                [
                    (
                        "citing",
                        [
                            [
                                reference(
                                    title="First Corpus Paper",
                                    authors=["Lovelace, A."],
                                    year="2001",
                                    doi="10.1000/one",
                                )
                            ],
                            [
                                reference(
                                    title="Second Corpus Paper",
                                    authors=["Hopper, G."],
                                    year="2002",
                                    doi="10.1000/two",
                                )
                            ],
                            [
                                reference(
                                    title="Third Corpus Paper",
                                    authors=["Turing, A."],
                                    year="2003",
                                    doi="10.1000/three",
                                )
                            ],
                            [
                                reference(
                                    title="Absent Paper",
                                    authors=["Babbage, C."],
                                    year="2004",
                                    doi="10.1000/absent",
                                )
                            ],
                        ],
                    )
                ]
            )
        ),
        encoding="utf-8",
    )
    closure = sha256(dataset_path.read_bytes()).hexdigest()
    outs = []
    for name in ("first", "second"):
        out = tmp_path / name
        done = run_cli(
            tmp_path,
            "closure",
            "check",
            "--dataset",
            str(dataset_path),
            "--out",
            str(out),
            "--workers",
            "3",
            "--corpus",
            str(first),
            "--corpus",
            str(second),
            "--corpus",
            str(third),
        )
        assert done.returncode == 0, done.stderr
        outs.append(out)
    annotated_path = outs[0] / "annotated-dataset.json"
    report_path = outs[0] / "coverage.json"
    printed = done.stdout.splitlines()
    assert printed[-3:] == [
        f"annotated: {outs[1] / 'annotated-dataset.json'}",
        f"coverage: {outs[1] / 'coverage.json'}",
        f"queue: {outs[1] / 'citation-papers-to-collect.csv'}",
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["checker"] == "taxonomy_pipeline.closure.corpus_check/1"
    assert report["dataset"] == "dataset.json"
    assert report["annotated_dataset"] == "annotated-dataset.json"
    assert annotations(json.loads(annotated_path.read_text(encoding="utf-8")))
    assert "corpus_match" not in dataset_path.read_text(encoding="utf-8")
    # Sources are named relative to the folder the command ran in (here tmp_path).
    assert [row["path"] for row in report["sources"]] == ["archive", "grobid", "extra"]
    # The filename DOI of `doi_<doi>.grobid.tei.xml` is load-bearing: that header has none.
    assert report["summary"] == {
        "reference_occurrences": 4,
        "resolved_markers_without_references": 0,
        "unique_cited_papers": 4,
        "present": 3,
        "missing": 1,
        "review": 0,
    }
    assert sha256(dataset_path.read_bytes()).hexdigest() == closure
    for name in (
        "annotated-dataset.json",
        "coverage.json",
        "citation-papers-to-collect.csv",
    ):
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()
    receipt = json.loads((outs[0] / "receipt.json").read_text())
    assert receipt["command"] == "mbdef closure check"
    assert receipt["settings"]["workers"] == 3
    assert receipt["inputs"]["dataset"]["sha256"] == closure


def test_check_refuses_an_output_folder_that_is_not_empty(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "kept.txt").write_text("x")
    done = run_cli(
        tmp_path,
        "closure",
        "check",
        "--dataset",
        str(tmp_path / "dataset.json"),
        "--out",
        str(tmp_path / "out"),
        "--corpus",
        str(tmp_path),
    )
    assert done.returncode == 1
    assert "is not empty" in done.stderr
    assert (tmp_path / "out" / "kept.txt").read_text() == "x"
