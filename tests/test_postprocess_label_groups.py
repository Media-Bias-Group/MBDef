import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from cli_helpers import package_commit

import mbdef.postprocess
from mbdef.cli import main

# A tiny store in the corpus store's schema. The view is every seed paper plus the closure papers marked in_P28 in the links file; a Definition counts when one of its sources is a P≥28 extraction ("seeds:P28" or "closure:P28").
PAPERS = [
    ("s1", "seeds"),
    ("s2", "seeds"),
    ("s3", "seeds"),
    ("c1", "closure"),
    ("c2", "closure"),
]  # s3: no terms
MENTIONS = [
    ("m1", "s1", "media bias"),
    ("m2", "s2", "bias in the media"),
    ("m3", "c1", "Framing"),
    ("m4", "c2", "media bias"),  # closure paper outside the view
    ("m5", "s1", "agenda setting"),  # only a P20 definition
    ("m6", "c1", "framing"),
    ("m7", "s2", "gatekeeping"),
]
OCCURRENCES = [  # occ_id, mention_id, field, text, sources
    ("o1", "m1", "definitions", "Media bias is slanted reporting.", ["seeds:P28"]),
    ("o2", "m1", "definitions", "Bias of the news media.", ["seeds:P20", "seeds:P28"]),
    ("o3", "m2", "definitions", "Media bias is slanted reporting.", ["seeds:P28"]),
    ("o4", "m3", "definitions", "Framing selects aspects of reality.", ["closure:P28"]),
    ("o5", "m4", "definitions", "Out of view.", ["closure:P28"]),
    ("o6", "m5", "definitions", "Only in P20.", ["seeds:P20"]),
    (
        "o7",
        "m6",
        "definitions",
        "Framing makes aspects salient.",
        ["closure:P28", "closure:P28-single-model"],
    ),
    ("o8", "m1", "description", "Not a definition.", ["seeds:P28"]),
    (
        "o9",
        "m7",
        "definitions",
        "Gatekeeping decides what becomes news.",
        ["seeds:P28"],
    ),
]
LINKS = [  # closure_paper_id, in_P28
    ("c1", "1"),
    ("c2", "0"),
    ("", "1"),
]
# ids of these groups in the frozen Label Groups (a group id is a hash of its key)
MEDIA_BIAS, FRAMING = ("g1b8640b47938", "ga67b95a680c9")


@pytest.fixture
def inputs(tmp_path):
    store = tmp_path / "store.sqlite3"
    db = sqlite3.connect(store)
    db.executescript("""
        CREATE TABLE papers(paper_id TEXT PRIMARY KEY, title TEXT, doi TEXT, corpus TEXT);
        CREATE TABLE mentions(mention_id TEXT PRIMARY KEY, paper_id TEXT, term_loose TEXT, display_term TEXT,
                              base_key TEXT, acronym TEXT, first_seen TEXT);
        CREATE TABLE occurrences(occ_id TEXT PRIMARY KEY, mention_id TEXT, paper_id TEXT, field TEXT, text TEXT,
                                 text_loose TEXT, citations TEXT);
        CREATE TABLE occurrence_sources(occ_id TEXT, source_id TEXT, record_index INT, item_index INT, raw_text TEXT,
                                        raw_citations TEXT, PRIMARY KEY(occ_id, source_id, record_index, item_index));
    """)
    db.executemany("INSERT INTO papers(paper_id, corpus) VALUES (?, ?)", PAPERS)
    db.executemany(
        "INSERT INTO mentions(mention_id, paper_id, display_term) VALUES (?, ?, ?)",
        MENTIONS,
    )
    paper = {m: p for m, p, _ in MENTIONS}
    for occ, m, field, text, sources in OCCURRENCES:
        db.execute(
            "INSERT INTO occurrences(occ_id, mention_id, paper_id, field, text) VALUES (?, ?, ?, ?, ?)",
            (occ, m, paper[m], field, text),
        )
        db.executemany(
            "INSERT INTO occurrence_sources(occ_id, source_id, record_index, item_index) VALUES (?, ?, 0, 0)",
            [(occ, s) for s in sources],
        )
    db.commit()
    db.close()
    links = tmp_path / "links.csv"
    with links.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["seed_paper_id", "term", "closure_paper_id", "in_P20", "in_P24", "in_P28"]
        )
        w.writerows([("s1", "x", c, "1", "1", p28) for c, p28 in LINKS])
    return store, links


def run(store, links, out, *extra):
    main(
        [
            "postprocess",
            "label-groups",
            "--store",
            str(store),
            "--links",
            str(links),
            "--out",
            str(out),
            *extra,
        ]
    )
    rows = lambda name: [json.loads(line) for line in (out / name).open()]  # noqa: E731
    return (
        rows("terms.jsonl"),
        rows("groups.jsonl"),
        json.loads((out / "summary.json").read_text()),
    )


def test_one_row_per_extracted_concept_of_the_view_with_its_label_group(
    inputs, tmp_path
):
    terms, _, _ = run(*inputs, tmp_path / "out")
    assert terms == [
        dict(
            mention_id="m1",
            paper_id="s1",
            corpus="seeds",
            term="media bias",
            group_id=MEDIA_BIAS,
            n_definitions=2,
        ),
        dict(
            mention_id="m2",
            paper_id="s2",
            corpus="seeds",
            term="bias in the media",
            group_id=MEDIA_BIAS,
            n_definitions=1,
        ),
        dict(
            mention_id="m3",
            paper_id="c1",
            corpus="closure",
            term="Framing",
            group_id=FRAMING,
            n_definitions=1,
        ),
        dict(
            mention_id="m6",
            paper_id="c1",
            corpus="closure",
            term="framing",
            group_id=FRAMING,
            n_definitions=1,
        ),
        dict(
            mention_id="m7",
            paper_id="s2",
            corpus="seeds",
            term="gatekeeping",
            group_id=terms[4]["group_id"],
            n_definitions=1,
        ),
    ]
    assert terms[4]["group_id"] not in (MEDIA_BIAS, FRAMING)


def test_one_row_per_label_group_with_all_its_definitions(inputs, tmp_path):
    _, groups, _ = run(*inputs, tmp_path / "out")
    assert [g["group_id"] for g in groups][:2] == [
        MEDIA_BIAS,
        FRAMING,
    ]  # most Definitions first
    media = groups[0]
    assert media["key"] == "bias media"
    assert media["labels"] == ["media bias", "bias in the media"]
    assert (
        media["terms"],
        media["papers"],
        media["n_definitions"],
        media["n_unique_texts"],
    ) == (2, 2, 3, 2)
    assert media["definitions"] == [
        dict(
            def_id="o1",
            mention_id="m1",
            paper_id="s1",
            text="Media bias is slanted reporting.",
        ),
        dict(
            def_id="o2", mention_id="m1", paper_id="s1", text="Bias of the news media."
        ),
        dict(
            def_id="o3",
            mention_id="m2",
            paper_id="s2",
            text="Media bias is slanted reporting.",
        ),
    ]
    assert groups[1]["labels"] == ["Framing", "framing"] and groups[1]["papers"] == 1
    assert len(groups) == 3


def test_summary_counts_groups_and_what_each_step_saves(inputs, tmp_path):
    _, _, summary = run(*inputs, tmp_path / "out")
    assert summary["view_papers"] == 4
    assert summary["papers_with_definitions"] == 3
    assert summary["terms"] == 5
    assert summary["definitions"] == 6
    assert summary["groups"] == 3
    assert summary["single_term_groups"] == 1
    assert summary["terms_in_multi_term_groups"] == 4
    assert summary["groups_with_2plus_definitions"] == 2
    assert summary["group_size_terms_histogram"] == {"1": 1, "2": 2}
    saved = {
        s["step"]: (s["groups_without"], s["saved"]) for s in summary["step_savings"]
    }
    assert saved["case"] == (4, 1)  # "Framing" / "framing"
    assert saved["order"] == (4, 1) and saved["stop"] == (
        4,
        1,
    )  # "bias in the media" / "media bias"
    assert saved["plural"] == (3, 0)
    assert all(s["name"] for s in summary["step_savings"])


def test_run_records_code_version_inputs_and_rule(inputs, tmp_path):
    store, links = inputs
    run(store, links, tmp_path / "out")
    record = json.loads((tmp_path / "out" / "run.json").read_text())
    assert record["rule"]["name"] == "Label Match"
    assert record["rule"]["version"]
    commit = (
        package_commit()
    )  # None outside a checkout, and then no branch or state either
    assert record["code"]["git_commit"] == commit
    assert isinstance(record["code"]["git_dirty"], bool if commit else type(None))
    assert set(record["code"]["package_sha256"]) >= {
        "src/mbdef/postprocess/label_match.py",
        "src/mbdef/postprocess/label_groups.py",
    }
    assert (
        record["inputs"]["store"]["sha256"]
        == hashlib.sha256(store.read_bytes()).hexdigest()
    )
    assert (
        record["inputs"]["links"]["sha256"]
        == hashlib.sha256(links.read_bytes()).hexdigest()
    )
    assert record["input_counts"] == dict(
        view_papers=4, papers_with_definitions=3, extracted_concepts=5, definitions=6
    )
    written = tmp_path / "out" / "terms.jsonl"
    assert (
        record["outputs"]["terms.jsonl"]["sha256"]
        == hashlib.sha256(written.read_bytes()).hexdigest()
    )
    assert record["argv"][:1] == ["label-groups"]


def test_existing_results_are_not_overwritten_unless_asked(inputs, tmp_path):
    out = tmp_path / "out"
    run(*inputs, out)
    with pytest.raises(SystemExit):
        run(*inputs, out)
    run(*inputs, out, "--overwrite")


def test_step5_code_finds_its_files_in_the_repository_and_the_data_root_only():
    """No Step 5 code looks for files in the home folder or through git's common directory; its data comes from the repository and DATA_ROOT."""
    package = Path(mbdef.postprocess.__file__).parent
    texts = [path.read_text() for path in package.rglob("*.py")]
    assert texts
    for marker in ("git-common-dir", "Path.home()"):
        assert not [t for t in texts if marker in t], marker


def test_identical_inputs_give_identical_files(inputs, tmp_path):
    run(*inputs, tmp_path / "a")
    run(*inputs, tmp_path / "b")
    for name in ("terms.jsonl", "groups.jsonl", "summary.json"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()
