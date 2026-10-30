"""The independent identity oracle at a small scale: generated references against generated corpus files.

100 seed papers cite 100 works each out of 1,000 (a quarter of them are in the corpus, a quarter conflict with one that is, the rest are missing). The expected answer for every reference comes from the integer that generated it, not from the grouping or matching code, so a linking or matching change that merges or splits works fails here. `benchmark()` with its defaults runs it at 4,000 seed papers and 20,000 works (by hand: too slow for every test run).

It calls the functions, not the command: 10,000 generated references through `closure run` and `closure check` would write and parse 100 TEI files and a corpus per run, many times slower in CI.
"""

import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from mbdef.closure.corpus_check import (
    annotate_corpus_membership,
    scan_corpus,
)
from mbdef.closure.extend_gt import extend
from mbdef.closure.tei_sources import Sources


def benchmark(seeds=4000, references_per_seed=100, unique=20000):
    def title(index):
        # Third quarter deliberately conflicts with the first quarter's title.
        mapped = (
            index - unique // 2 if unique // 2 <= index < 3 * unique // 4 else index
        )
        return f"A controlled study of publication {mapped}"

    def entry(index, local):
        return (
            f'<biblStruct xml:id="b{local}"><analytic><title level="a">{title(index)}</title>'
            f'<idno type="DOI">10.9876/verification.{index}</idno><author><persName>'
            "<surname>Author</surname></persName></author></analytic><monogr><imprint>"
            '<date when="2020"/></imprint></monogr></biblStruct>'
        )

    expected = {}
    data = {"papers": []}  # Native predictions need no paper_count field.
    with tempfile.TemporaryDirectory(prefix="closure-scale-") as work:
        root = Path(work)
        with zipfile.ZipFile(
            root / "seeds.zip", "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for seed in range(seeds):
                indices = [
                    (seed * 37 + local * 193) % unique
                    for local in range(references_per_seed)
                ]
                expected[f"seed-{seed}"] = indices
                data["papers"].append(
                    {
                        "paper_id": f"seed-{seed}",
                        "definition_records": [
                            {
                                "term": "term",
                                "definitions": [
                                    {
                                        "text": "A source-grounded definition.",
                                        "citation": [
                                            f"[{local + 1}]"
                                            for local in range(references_per_seed)
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                )
                markers = " ".join(
                    f'<ref type="bibr" target="#b{local}">[{local + 1}]</ref>'
                    for local in range(references_per_seed)
                )
                payload = (
                    '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
                    f"<titleStmt><title>Seed {seed}</title></titleStmt></fileDesc></teiHeader>"
                    f"<text><body><p>{markers}</p></body><back><listBibl>"
                    + "".join(
                        entry(index, local) for local, index in enumerate(indices)
                    )
                    + "</listBibl></back></text></TEI>"
                )
                archive.writestr(f"seed-{seed}.tei.xml", payload)
        with zipfile.ZipFile(
            root / "corpus.zip", "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for index in range(unique // 2):
                payload = (
                    '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
                    f"<titleStmt><title>{title(index)}</title></titleStmt><sourceDesc>"
                    + entry(index, 0)
                    + "</sourceDesc></fileDesc></teiHeader>"
                    "<text><body><p>Existing full text.</p></body></text></TEI>"
                )
                archive.writestr(f"corpus-{index}.tei.xml", payload)
        resolved, links, _, _ = extend(
            data,
            Sources(
                archive=root / "unused",
                grobid_files=root / "unused",
                source_zip=root / "seeds.zip",
                cache=root / "cache",
            ),
        )
        # Compare every returned DOI against the integer source generator; neither grouping nor matching implementation is used to derive the expected answer.
        for paper in resolved["papers"]:
            citations = paper["definition_records"][0]["definitions"][0][
                "resolved_citations"
            ]
            assert [row["references"][0]["doi"] for row in citations] == [
                f"10.9876/verification.{index}" for index in expected[paper["paper_id"]]
            ]
        corpus, errors = scan_corpus([root / "corpus.zip"], workers=4)
        assert not any(row["error"] for row in errors)
        _, report, _ = annotate_corpus_membership(resolved, corpus, errors)
        seen = {index for indices in expected.values() for index in indices}
        oracle = Counter(
            "present"
            if index < unique // 2
            else "review"
            if index < 3 * unique // 4
            else "missing"
            for index in seen
        )
        for paper in report["papers"]:
            index = int(paper["reference"]["doi"].rsplit(".", 1)[1])
            wanted = (
                "present"
                if index < unique // 2
                else "review"
                if index < 3 * unique // 4
                else "missing"
            )
            assert paper["corpus_match"]["status"] == wanted
        assert report["summary"]["reference_occurrences"] == seeds * references_per_seed
        assert (
            links["markers"] == links["linked_markers"] == seeds * references_per_seed
        )
        assert report["summary"]["unique_cited_papers"] == len(seen)
        for status, count in oracle.items():
            assert report["summary"][status] == count
        return report["summary"]


def test_every_reference_is_linked_and_matched_as_its_generator_says():
    summary = benchmark(seeds=100, references_per_seed=100, unique=1000)
    # Of the 1,000 works, 500 are in the corpus, 250 conflict with one of those, 250 are missing.
    assert (
        summary["present"] + summary["review"] + summary["missing"]
        == (summary["unique_cited_papers"])
    )
    assert summary["reference_occurrences"] == 100 * 100
