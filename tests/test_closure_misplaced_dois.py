"""DOIs GROBID attached to the wrong reference-list entries are ignored (closure/bibliography.py)."""

from mbdef.closure.bibliography import parse_bibliography

TEI_NS = "http://www.tei-c.org/ns/1.0"


def bibl(bid, title, doi=None, year="2019"):
    idno = f'<idno type="DOI">{doi}</idno>' if doi else ""
    return (
        f'<biblStruct xml:id="{bid}"><analytic><title level="a" type="main">{title}</title>'
        f"<author><persName><surname>Smith</surname></persName></author>{idno}</analytic>"
        f'<monogr><imprint><date when="{year}"/></imprint></monogr></biblStruct>'
    )


def tei(entries, own_doi=None, own_title="A paper about media framing in elections"):
    idno = f'<idno type="DOI">{own_doi}</idno>' if own_doi else ""
    return (
        f'<TEI xmlns="{TEI_NS}"><teiHeader><fileDesc><titleStmt><title level="a" type="main">{own_title}</title></titleStmt>'
        f'<sourceDesc><biblStruct><analytic><title level="a" type="main">{own_title}</title></analytic>{idno}</biblStruct></sourceDesc>'
        f"</fileDesc></teiHeader><text><body><p>Text.</p></body><back><div><listBibl>{''.join(entries)}</listBibl></div></back></text></TEI>"
    ).encode()


def test_a_doi_on_two_different_works_is_ignored_on_both():
    e = parse_bibliography(
        tei(
            [
                bibl(
                    "b0",
                    "Fake News and Epistemic Vice: Combating a Uniquely Noxious Market",
                    "10.1017/apa.2021.11",
                ),
                bibl(
                    "b1",
                    "Online Misinformation and Phantom Patterns: Epistemic Exploitation",
                    "10.1017/apa.2021.11",
                ),
                bibl("b2", "The science of fake news", "10.1126/science.aao2998"),
            ]
        )
    )["entries"]
    assert e["b0"]["doi"] is None and e["b1"]["doi"] is None
    assert e["b0"]["doi_ignored"] == {
        "doi": "10.1017/apa.2021.11",
        "reason": "shared by different entries",
    }
    assert "doi.org/10.1017" not in e["b0"]["formatted"]
    assert e["b2"]["doi"] == "10.1126/science.aao2998" and "doi_ignored" not in e["b2"]


def test_a_split_entry_or_a_title_variant_keeps_its_doi():
    e = parse_bibliography(
        tei(
            [
                bibl("b0", "The science of fake news", "10.1126/science.aao2998"),
                bibl(
                    "b1", "", "10.1126/science.aao2998"
                ),  # GROBID split one reference in two
                bibl(
                    "b2",
                    "2023a. Survey of hallucination in natural language generation",
                    "10.1145/3571730",
                ),
                bibl(
                    "b3",
                    "Survey of hallucination in natural language generation",
                    "10.1145/3571730",
                ),
            ]
        )
    )["entries"]
    assert all(e[k]["doi"] for k in ("b0", "b1", "b2", "b3"))


def test_the_citing_papers_own_doi_on_a_reference_is_ignored_even_with_a_footer_suffix():
    own = "10.30958/ajmmc.7-1-3"
    e = parse_bibliography(
        tei(
            [
                bibl(
                    "b0",
                    "The changing culture of affirmative action",
                    own + "%ee%88%86",
                ),
                bibl(
                    "b1",
                    "Media discourse and public opinion on nuclear power",
                    "10.1086/229213",
                ),
            ],
            own_doi=own,
        )
    )["entries"]
    assert (
        e["b0"]["doi"] is None
        and e["b0"]["doi_ignored"]["reason"] == "the citing paper's own DOI"
    )
    assert e["b1"]["doi"] == "10.1086/229213"


def test_a_self_reference_keeps_the_papers_own_doi():
    own = "10.1000/self.1"
    e = parse_bibliography(
        tei([bibl("b0", "A paper about media framing in elections", own)], own_doi=own)
    )["entries"]
    assert e["b0"]["doi"] == own
