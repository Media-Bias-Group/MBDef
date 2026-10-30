"""The corpus store: the four tables of release v1.1's extractions that Step 5 reads, rebuilt by `mbdef postprocess build-store`.

tests/fixtures/store-mini/ holds a miniature release folder (synthetic text) and every row, in order, of the four tables that the original ingest code made of it, which the tests compare with.
"""

import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest
from cli_helpers import run_cli

from mbdef import files
from mbdef.cli import main
from mbdef.postprocess import store
from mbdef.postprocess.corpus import default_store, load_view

MINI = Path(__file__).parent / "fixtures/store-mini"
REPO = Path(__file__).resolve().parents[1]
TABLES = ("papers", "mentions", "occurrences", "occurrence_sources")


def build_store(out, *extra, release=MINI / "release"):
    """`postprocess build-store` on the miniature (in this process: the command line starts in about 5 s)."""
    return main(
        [
            "postprocess",
            "build-store",
            "--release",
            str(release),
            "--out",
            str(out),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The miniature's store, built once: its --out folder."""
    out = tmp_path_factory.mktemp("built")
    assert build_store(out) == 0
    return out


def rows(path, table):
    db = sqlite3.connect(path)
    try:
        return [
            list(r) for r in db.execute(f"SELECT rowid, * FROM {table} ORDER BY rowid")
        ]
    finally:
        db.close()


def test_the_store_holds_the_rows_the_phase_3_ingest_made_in_its_order(built):
    expected = json.loads((MINI / "expected.json").read_text(encoding="utf-8"))
    for table in TABLES:
        assert rows(built / "library.sqlite3", table) == expected[table], table


def test_step_5_reads_the_p28_view_from_the_store(built, tmp_path):
    """The P≥28 view of the miniature: the seed papers and closure-1 (closure-2 is not in the view); a definition counts when a P≥28 variant gave it (seed-1's 'Bias in the selection of stories.' came only from P≥24)."""
    links = tmp_path / "links.csv"
    links.write_text(
        "seed_paper_id,term,closure_paper_id,in_P20,in_P24,in_P28\n"
        "seed-1,media bias,closure-1,1,1,1\nseed-1,media bias,closure-2,1,1,0\n"
    )
    view = load_view(built / "library.sqlite3", links)
    assert view.papers == {"seed-1", "seed-2", "seed-3", "closure-1"}
    assert [
        (c.paper_id, c.corpus, c.label, [d.text for d in c.definitions])
        for c in view.concepts
    ] == [
        (
            "seed-1",
            "seeds",
            "Media Bias",
            ["Partisan tilt in reporting.", "A slant in news coverage."],
        ),
        (
            "closure-1",
            "closure",
            "Gatekeeping",
            ["Gatekeeping decides what becomes news."],
        ),
        (
            "seed-2",
            "seeds",
            "the echo chambers",
            ["A closed space where beliefs are amplified."],
        ),
    ]


def test_the_store_holds_only_the_four_tables_no_paper_text(built):
    db = sqlite3.connect(built / "library.sqlite3")
    names = {
        n for (n,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    db.close()
    assert names == set(TABLES)


def test_the_receipt_lists_every_input_it_read(built):
    receipt = json.loads((built / "receipt.json").read_text())
    assert receipt["command"] == "postprocess build-store"
    assert sorted(receipt["inputs"]) == sorted(
        [f"{c}/derived/DONE.json" for c in ("seeds", "closure")]
        + [
            f"{c}:{v}"
            for c in ("seeds", "closure")
            for v in ("P20", "P24", "P28", "P28-single-model")
        ]
    )
    assert receipt["settings"]["tables"] == {
        "papers": 5,
        "mentions": 10,
        "occurrences": 22,
        "occurrence_sources": 33,
    }


def test_compare_names_the_first_different_row_and_ignores_other_tables(
    built, tmp_path, capsys
):
    stored = tmp_path / "stored.sqlite3"
    shutil.copy(built / "library.sqlite3", stored)
    db = sqlite3.connect(stored)
    db.execute("CREATE TABLE contexts(occ_id TEXT PRIMARY KEY, context TEXT)")
    db.execute("INSERT INTO contexts VALUES ('o1', 'a paragraph of the paper')")
    db.commit()
    db.close()
    capsys.readouterr()
    assert build_store(tmp_path / "same", "--compare", str(stored)) == 0
    assert f"All 4 tables equal {stored} row for row." in capsys.readouterr().out

    db = sqlite3.connect(stored)
    db.execute("UPDATE mentions SET display_term = 'media bias' WHERE rowid = 1")
    db.commit()
    db.close()
    assert build_store(tmp_path / "differs", "--compare", str(stored)) == 1
    out = capsys.readouterr().out
    assert (
        "mentions: DIFFERS, first different row: (1, 'm0634fc13aeb4eeb1', 'seed-1'"
        in out
    )
    assert f"papers: the same as {stored} row for row" in out
    assert f"1 of 4 tables differ from {stored}." in out
    compared = json.loads((tmp_path / "differs/receipt.json").read_text())["compared"]
    assert compared["tables"]["papers"] is None
    assert compared["tables"]["mentions"].startswith("first different row")


def test_a_variant_that_is_not_the_one_done_json_lists_is_refused(tmp_path):
    release = tmp_path / "release"
    shutil.copytree(MINI / "release", release)
    p28 = release / "closure/derived/P28/predictions.json"
    p28.write_text(
        p28.read_text().replace("Gatekeeping decides", "Gatekeeping chooses")
    )
    with pytest.raises(
        SystemExit, match="its sha256 is .* as DONE.json lists"
    ) as refused:
        build_store(tmp_path / "out", release=release)
    assert str(p28) in str(refused.value)
    assert list((tmp_path / "out").iterdir()) == []


def test_an_existing_store_is_replaced_only_with_overwrite(tmp_path):
    (tmp_path / "library.sqlite3").write_text("an older store")
    with pytest.raises(SystemExit, match="pass --overwrite"):
        build_store(tmp_path)
    assert build_store(tmp_path, "--overwrite") == 0
    assert rows(tmp_path / "library.sqlite3", "papers")[0] == [
        1,
        "seed-1",
        "A synthetic seed paper",
        "10.0000/mini.1",
        "seeds",
    ]


@pytest.mark.slow
def test_the_store_of_release_v1_1_comes_back_row_for_row(tmp_path):
    """The real rebuild (CPU, about a minute): release v1.1 from the data root against the stored corpus store."""
    files.load_env()
    if not os.environ.get("DATA_ROOT"):
        pytest.skip("DATA_ROOT is not set")
    stored = default_store()
    if not stored.exists() or not (files.data_root() / store.RELEASE).exists():
        pytest.skip("release v1.1 or the stored corpus store is not in the data root")
    done = run_cli(
        REPO,
        "postprocess",
        "build-store",
        "--out",
        str(tmp_path),
        "--compare",
        str(stored),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    counts = "papers 8,425 rows; mentions 147,461 rows; occurrences 524,907 rows; occurrence_sources 1,327,091 rows"
    assert counts in done.stdout
    assert f"All 4 tables equal {stored} row for row." in done.stdout
