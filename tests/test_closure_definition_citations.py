"""The cited-work grouping of the link tables, and the closure audit's helpers it rests on.

These call the helpers directly: each is one rule of the grouping (a garbled row, a DOI's cleaning, a surname), which `closure bundle` applies only inside a whole release rebuild (test_closure_bundle.py runs the command).
"""

import pytest

from mbdef.closure import definition_citations
from mbdef.extract.derive import VARIANTS


def test_an_identified_work_is_not_labelled_not_found_because_of_a_garbled_row():
    # de Vreese 2005: 70 rows, one of them garbled
    assert (
        definition_citations.resolved_status(["not_found", "not_open_access"])
        == "resolved_no_oa_pdf"
    )
    assert (
        definition_citations.resolved_status(["ambiguous", "open_access_but_no_pdf"])
        == "resolved_no_oa_pdf"
    )
    assert (
        definition_citations.resolved_status(["not_found", "ambiguous"]) == "ambiguous"
    )


def test_a_rejected_pdf_says_why():
    assert (
        definition_citations.rejected_pdf_status(
            ["grobid_identity_conflict: DOI differs from requested work"]
        )
        == "pdf_rejected_other_work"
    )
    assert (
        definition_citations.rejected_pdf_status(
            ["", "grobid_identity_unverified: no matching DOI or title/year/author"]
        )
        == "pdf_rejected_unconfirmed"
    )
    assert (
        definition_citations.rejected_pdf_status(["invalid_tei_xml"]) == "pdf_unusable"
    )
    assert definition_citations.rejected_pdf_status([""]) == "downloaded"


# The helpers the closure audit's grouping rests on: each case is a value the audit's merges depend on.


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Fake News—A Survey", "fake news a survey"),
        ("  Ünïcode  ∑ Math ", "ünïcode math"),  # only word characters
        (
            "ﬁnding Ｆraming",
            "finding framing",
        ),  # the ligature and the full-width letter fold
        ("", ""),
        (None, ""),
    ],
)
def test_ntext_compares_words_after_nfkc_and_case_folding(text, expected):
    assert definition_citations.nfkc_words(text) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (
            "https://doi.org/10.1234/ABC.",
            "10.1234/abc",
        ),  # prefix, case and a trailing full stop
        ("doi: 10.1234/abc", "10.1234/abc"),
        ("10.1234%2Fabc", "10.1234/abc"),  # an escaped slash
        ("10.1000/xyz);", "10.1000/xyz"),  # trailing punctuation
        ("http://dx.doi.org/10.1002/(sici)1099-0771", "10.1002/(sici)1099-0771"),
        ("not a doi", ""),
        ("", ""),
    ],
)
def test_strong_doi_cleans_what_the_corpus_checks_doi_keeps(value, expected):
    assert definition_citations.strong_doi(value) == expected


def test_surname_is_the_first_authors_whatever_the_name_order():
    assert definition_citations.surname(["Smith, A.", "Jones, B."]) == "smith"
    assert definition_citations.surname(["Ada Lovelace"]) == "lovelace"
    assert definition_citations.surname(["", "Hopper, G."]) == "hopper"
    assert definition_citations.surname([]) == ""


def test_union_find_keeps_the_smallest_index_as_the_representative():
    uf = definition_citations.UnionFind(5)
    assert uf.union(3, 1) and uf.union(4, 3) and not uf.union(1, 4)
    assert [uf.find(i) for i in range(5)] == [0, 1, 2, 1, 1]


def test_the_four_variants_are_the_runs_pick_names():
    assert VARIANTS == {
        "P20": "P>=20",
        "P24": "P>=24",
        "P28": "P>=28",
        "P28-single-model": "P>=28-single-model",
    }
