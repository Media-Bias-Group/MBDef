"""Which closure papers the release leaves out, and the duplicate-text check that finds them.

These call the rules directly: `closure bundle` applies them only inside a whole release rebuild (its tests, test_closure_bundle.py, run the command), and a case here (a pair of near-identical texts) would need a whole miniature release each.
"""

import json
import re

import pytest

from mbdef.closure import exclusions as rx


def test_seed_copies_are_left_out_and_the_published_version_of_a_pair_is_kept():
    info = {
        "a": {"doi": "10.48550/arxiv.1", "n_citing_seed_papers": 9},
        "b": {"doi": "10.1145/1", "n_citing_seed_papers": 1},
        "c": {"doi": "", "n_citing_seed_papers": 2},
        "d": {"doi": "", "n_citing_seed_papers": 5},
        "s": {"doi": "10.1/x"},
    }
    dups = [
        {"paper_id": "s", "duplicate_of": "pdf-1", "kind": "seed", "share": 1.0},
        {"paper_id": "a", "duplicate_of": "b", "kind": "closure", "share": 1.0},
        {"paper_id": "c", "duplicate_of": "d", "kind": "closure", "share": 0.99},
    ]
    ex = {e["paper_id"]: e for e in rx.exclusions(dups, info, [])}
    assert ex["s"]["seed_paper_id"] == "pdf-1"
    assert (
        ex["a"]["kept_paper_id"] == "b"
    )  # published DOI beats the arXiv copy, even with fewer citations
    assert ex["c"]["kept_paper_id"] == "d"  # no DOI on either: more citing seed papers
    assert set(ex) == {"s", "a", "c"}


def test_a_pair_member_that_is_a_seed_copy_points_to_the_seed():
    info = {"x": {"doi": "10.1/x"}, "y": {"doi": ""}}
    dups = [
        {"paper_id": "x", "duplicate_of": "pdf-9", "kind": "seed", "share": 1.0},
        {"paper_id": "x", "duplicate_of": "y", "kind": "closure", "share": 1.0},
    ]
    ex = {e["paper_id"]: e for e in rx.exclusions(dups, info, [])}
    assert (
        ex["x"]["seed_paper_id"] == "pdf-9" and "y" not in ex
    )  # y stays (x is the seed copy)


def test_an_already_extracted_copy_stays_over_a_new_published_one():
    info = {
        "old": {"doi": "", "n_citing_seed_papers": 3, "extracted": True},
        "new": {"doi": "10.1016/j.x", "extracted": False},
    }
    dups = [{"paper_id": "new", "duplicate_of": "old", "kind": "closure", "share": 1.0}]
    ex = {e["paper_id"]: e for e in rx.exclusions(dups, info, [])}
    assert ex["new"]["kept_paper_id"] == "old" and "old" not in ex


def test_a_chain_of_copies_points_to_the_paper_that_stays():
    info = {
        k: {"doi": "", "n_citing_seed_papers": n}
        for k, n in (("a", 1), ("b", 2), ("c", 3))
    }
    dups = [
        {"paper_id": "a", "duplicate_of": "b", "kind": "closure", "share": 1.0},
        {"paper_id": "b", "duplicate_of": "c", "kind": "closure", "share": 1.0},
    ]
    ex = {e["paper_id"]: e for e in rx.exclusions(dups, info, [])}
    assert (ex["a"]["kept_paper_id"], ex["b"]["kept_paper_id"]) == ("c", "c")


def test_a_paper_no_seed_reference_cites_is_left_out_unless_a_copy_already_is():
    info = {"x": {"doi": "10.1/x"}}
    dups = [{"paper_id": "x", "duplicate_of": "pdf-9", "kind": "seed", "share": 1.0}]
    out = rx.exclusions(dups, info, ["x", "u"])
    assert [(e["paper_id"], e["reason"]) for e in out] == [
        (
            "u",
            "not cited: collected through a DOI GROBID had put on the wrong reference entry",
        ),
        ("x", "same text as a seed paper"),
    ]
    assert rx.counts(out) == {
        "left_out": 2,
        "seed_copies": 1,
        "closure_copies": 0,
        "not_cited": 1,
    }


def paper(folder, name, body):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(f"<TEI><text><body><p>{body}</p></body></text></TEI>")


def test_a_candidate_is_the_same_document_when_most_of_the_smaller_ones_sequences_occur_in_the_other(
    tmp_path,
):
    words = " ".join(f"w{i}" for i in range(80))
    paper(tmp_path / "seeds", "s1.tei.xml", words)
    paper(tmp_path / "seeds", "s2.tei.xml", " ".join(f"z{i}" for i in range(80)))
    paper(tmp_path / "grobid", "c-copy.tei.xml", words)  # a copy of s1
    paper(tmp_path / "grobid", "c-other.tei.xml", " ".join(f"q{i}" for i in range(80)))
    paper(
        tmp_path / "moved", "c-moved.tei.xml", " ".join(f"z{i}" for i in range(80))
    )  # a copy of s2
    candidates = [
        {"paper_id": "copy", "tei": "/anywhere/c-copy.tei.xml"},
        {"paper_id": "other", "tei": "/anywhere/c-other.tei.xml"},
        {"paper_id": "moved", "tei": "/elsewhere/c-moved.tei.xml"},
    ]

    record = rx.find_duplicates(
        tmp_path / "seeds", candidates, [tmp_path / "grobid", tmp_path / "moved"]
    )

    assert record["seeds"] == 2 and record["candidates"] == 3
    assert record["share_threshold"] == 0.8
    assert [
        (d["paper_id"], d["duplicate_of"], d["kind"], d["share"])
        for d in record["duplicates"]
    ] == [
        ("copy", "s1", "seed", 1.0),
        ("moved", "s2", "seed", 1.0),
    ]


def test_a_candidate_without_a_tei_file_stops_the_check(tmp_path):
    paper(tmp_path / "seeds", "s1.tei.xml", "x " * 40)
    (tmp_path / "grobid").mkdir()
    with pytest.raises(
        SystemExit, match="No TEI file in .* for 1 candidates, for example lost"
    ):
        rx.find_duplicates(
            tmp_path / "seeds",
            [{"paper_id": "lost", "tei": "/anywhere/lost.tei.xml"}],
            [tmp_path / "grobid"],
        )


def test_the_released_byte_formats_are_kept(tmp_path):
    rx.write_duplicates(tmp_path / "d.json", {"share_threshold": 0.8, "duplicates": []})
    rx.write_exclusions(tmp_path / "e.json", [{"paper_id": "é", "reason": "r"}])
    assert (
        tmp_path / "d.json"
    ).read_text() == '{\n "share_threshold": 0.8,\n "duplicates": []\n}'
    assert (
        tmp_path / "e.json"
    ).read_text() == '[\n {\n  "paper_id": "é",\n  "reason": "r"\n }\n]\n'
    assert re.fullmatch(r"\[\n \{[^}]*\}\n\]\n", (tmp_path / "e.json").read_text())
    assert json.loads((tmp_path / "e.json").read_text())[0]["paper_id"] == "é"
