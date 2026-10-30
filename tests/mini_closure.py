"""Tiny inputs for the closure tests: seed papers as GROBID writes them, and their extracted definitions.

Everything is written out literally (the TEI layout, the dataset's shape), so the tests do not borrow the code's own definitions of them. The worked example is two seed papers that cite one work between them (a child nobody has collected) and one of them cites the other.
"""

import json
from pathlib import Path
from xml.sax.saxutils import escape

# The header of the collection queue that the closure step writes and `collect` reads.
QUEUE_FIELDS = [
    "queue_status",
    "title",
    "authors",
    "year",
    "doi",
    "url",
    "formatted_reference",
    "citation_methods",
    "citation_confidences",
    "citing_paper_ids",
    "occurrence_count",
    "review_reason",
    "suggested_corpus_files",
]


def reference(bibl_id: str, title: str, surname: str, year: str, doi: str = "") -> str:
    idno = f'<idno type="DOI">{escape(doi)}</idno>' if doi else ""
    return (
        f'<biblStruct xml:id="{bibl_id}"><analytic><title level="a">{escape(title)}</title>'
        f"{idno}<author><persName><surname>{escape(surname)}</surname></persName></author>"
        f'</analytic><monogr><imprint><date when="{year}"/></imprint></monogr></biblStruct>'
    )


def seed_tei(
    title: str, surname: str, year: str, doi: str, cites: list[tuple[str, str]]
) -> bytes:
    """A seed paper's TEI: its title page, a body that cites `cites` ([n] -> bibl id) and the references."""
    refs = "".join(
        f'<ref type="bibr" target="#{bibl}">[{n}]</ref> '
        for n, (_, bibl) in enumerate(cites, start=1)
    )
    listed = "".join(entry for entry, _ in cites)
    return (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
        f"<titleStmt><title>{escape(title)}</title></titleStmt><sourceDesc><biblStruct><analytic>"
        f'<title level="a" type="main">{escape(title)}</title>'
        f"<author><persName><surname>{escape(surname)}</surname></persName></author></analytic>"
        f'<monogr><imprint><date when="{year}"/></imprint></monogr>'
        f'<idno type="DOI">{escape(doi)}</idno></biblStruct></sourceDesc></fileDesc></teiHeader>'
        f"<text><body><p>A definition {refs}.</p></body>"
        f"<back><listBibl>{listed}</listBibl></back></text></TEI>"
    ).encode()


def dataset(papers: dict[str, list[str]]) -> dict:
    """Extracted definitions: each paper has one term whose definition cites `[n]` for every n it lists."""
    return {
        "papers": [
            {
                "paper_id": paper_id,
                "definition_records": [
                    {
                        "term": "term",
                        "definitions": [
                            {
                                "text": "A definition " + " ".join(markers) + ".",
                                "citation": markers,
                            }
                        ],
                    }
                ],
            }
            for paper_id, markers in papers.items()
        ]
    }


CHILD = reference("b0", "Child work", "Author", "2020", "10.9876/child")


def two_seed_papers(folder: Path) -> tuple[Path, Path]:
    """`folder/seed-tei/{seed-a,seed-b}.tei.xml` and `folder/seeds.json`.

    seed-a cites the child work ([1]); seed-b cites the child ([1]) and seed-a ([2]).
    """
    tei_folder = folder / "seed-tei"
    tei_folder.mkdir(parents=True)
    (tei_folder / "seed-a.tei.xml").write_bytes(
        seed_tei("Seed paper A", "Alpha", "2019", "10.9876/seed-a", [(CHILD, "b0")])
    )
    seed_a_entry = reference("b1", "Seed paper A", "Alpha", "2019", "10.9876/seed-a")
    (tei_folder / "seed-b.tei.xml").write_bytes(
        seed_tei(
            "Seed paper B",
            "Beta",
            "2021",
            "10.9876/seed-b",
            [(CHILD, "b0"), (seed_a_entry, "b1")],
        )
    )
    path = folder / "seeds.json"
    path.write_text(json.dumps(dataset({"seed-a": ["[1]"], "seed-b": ["[1]", "[2]"]})))
    return path, tei_folder
