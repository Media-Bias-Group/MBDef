"""`mbdef closure inputs` on the miniature closure of tests/mini_bundle.py (worked out by hand there).

The command rebuilds the four lists `closure bundle` reads from the closure run's own files: the collected papers (`all-candidates.json`), what was known of each when the duplicate check ran (`papers-info.json`), the collected papers no seed reference cites (`uncited.json`) and the existing-corpus works not to add (`existing-exclude.json`).
The released closure's lists are rebuilt from the data root as docs/step-4-closure/closure.md shows.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from cli_helpers import run_cli
from mini_bundle import DOIS, World, without_build_time
from test_closure_bundle import run_bundle

from mbdef.closure import bundle
from mbdef.collect import grobid


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def run_inputs(world: World, out: Path, *extra: str):
    return run_cli(
        world.work,
        "closure",
        "inputs",
        "--out",
        str(out),
        "--ledgers",
        str(world.ledgers),
        "--earlier-ledgers",
        str(world.ledgers),
        "--closure-set",
        str(world.closure_set),
        "--closure-tei",
        str(world.closure_tei),
        "--seeds",
        str(world.seeds),
        "--seed-inputs",
        str(world.seed_inputs),
        *extra,
    )


def written(path: Path):
    """The file's value; it must be in the byte format of the released lists (json.dumps defaults: one line, no final newline)."""
    text = path.read_text(encoding="utf-8")
    value = json.loads(text)
    assert text == json.dumps(value)
    return value


def test_the_candidates_are_every_collected_paper_with_its_tei_below_the_prefix(
    world, tmp_path
):
    out = tmp_path / "inputs"

    done = run_inputs(world, out, "--tei-prefix", "/elsewhere/grobid")

    assert done.returncode == 0, done.stderr
    candidates = written(out / "all-candidates.json")
    # W1 and W5 were collected and converted; W4 was written into the same folder from the existing corpus.
    expected = {
        world.ids[k]: str(
            grobid.tei_path(
                Path("/elsewhere/grobid"), paperid=world.ids[k], doi=DOIS[k]
            )
        )
        for k in ("w1", "w5", "w4")
    }
    assert sorted(candidates, key=lambda c: c["paper_id"]) == [
        {"paper_id": pid, "tei": expected[pid]} for pid in sorted(expected)
    ]


def test_by_default_the_lists_name_no_local_folder_and_follow_the_tei_paths(tmp_path):
    """With the inputs where the command looks by default, the TEI paths are written below `$DATA_ROOT/...`, not below this machine's folder; the papers come in the order of their TEI paths, whatever order the file system lists them in."""
    world = World(tmp_path, data_root=tmp_path / "data-root")
    out = tmp_path / "inputs"

    done = run_cli(
        world.work,
        *("closure", "inputs", "--out", str(out)),
        *("--earlier-ledgers", str(world.ledgers)),  # the miniature has one set
    )

    assert done.returncode == 0, done.stderr
    candidates = written(out / "all-candidates.json")
    prefix = f"$DATA_ROOT/{bundle.CLOSURE_TEI.as_posix()}/"
    assert candidates and all(c["tei"].startswith(prefix) for c in candidates)
    assert str(tmp_path) not in (out / "all-candidates.json").read_text()
    teis = [c["tei"] for c in candidates]
    assert teis == sorted(teis)
    info = written(out / "papers-info.json")
    assert list(info) == [c["paper_id"] for c in candidates]


def test_papers_info_says_what_was_known_of_each_paper_when_the_check_ran(
    world, tmp_path
):
    # The extraction ran in two batches: W1 and W5 first, then W4 (the closure set lists the rows in batch order).
    (world.closure_set / "batches.json").write_text(
        json.dumps(
            [
                {"batch": 1, "excluded": [], "papers": 2, "shards": ["cb001-000"]},
                {"batch": 2, "excluded": [], "papers": 1, "shards": ["cb002-000"]},
            ]
        )
    )
    out = tmp_path / "inputs"

    done = run_inputs(world, out, "--extracted-batches", "1")

    assert done.returncode == 0, done.stderr
    info = written(out / "papers-info.json")
    # The DOI each paper was collected under; extracted in batch 1 or not; how many seed papers cite it in the links of that time (W1: seed-a and seed-b; W5: seed-b; W4 was not a closure paper yet).
    assert info == {
        world.ids["w1"]: {
            "doi": "10.9876/child",
            "extracted": True,
            "n_citing_seed_papers": 2,
        },
        world.ids["w5"]: {
            "doi": "10.9876/preprint",
            "extracted": True,
            "n_citing_seed_papers": 1,
        },
        world.ids["w4"]: {
            "doi": "10.9876/existing",
            "extracted": False,
            "n_citing_seed_papers": 0,
        },
    }
    # The same papers in the same order as the candidates.
    candidates = written(out / "all-candidates.json")
    assert list(info) == [c["paper_id"] for c in candidates]


def test_a_collected_paper_no_seed_reference_cites_is_uncited(world, tmp_path):
    # W6 was collected and extracted through a DOI GROBID had put on the wrong reference entry; once that DOI is ignored, no seed reference cites it.
    stray = "doi_10.9876_stray__0123456789abcdef"
    path = grobid.tei_path(world.closure_tei, paperid=stray, doi="10.9876/stray")
    text = f"<TEI><text><body><p>{' '.join(f'zeta{i}' for i in range(40))}</p></body></text></TEI>"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_suffix(".closure.json").write_text(
        json.dumps({"paperid": stray, "requested_doi": "10.9876/stray"})
    )
    rows = json.loads((world.closure_set / "rows-all.json").read_text())
    rows.append(
        {
            "paper_id": stray,
            "tei_path": f"/cluster/run/grobid/{path.parent.name}/{path.name}",
            "tei_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "tei_size_bytes": len(text.encode()),
            "frozen_title": "A stray paper",
        }
    )
    (world.closure_set / "rows-all.json").write_text(json.dumps(rows))
    (world.closure_set / "batches.json").write_text(
        json.dumps([{"batch": 1, "excluded": [], "papers": 4, "shards": ["cb001-000"]}])
    )
    out = tmp_path / "inputs"

    done = run_inputs(world, out)

    assert done.returncode == 0, done.stderr
    # W5 has no link row either, but it is left out already: its text is seed-b's.
    assert written(out / "uncited.json") == [stray]
    assert stray in written(out / "papers-info.json")


def test_the_bundle_built_on_the_rebuilt_lists_is_the_same_release(world, tmp_path):
    rebuilt = tmp_path / "inputs"
    assert run_inputs(world, rebuilt).returncode == 0

    builds = {}
    for name, audit_inputs in (("carried", world.v11), ("rebuilt", rebuilt)):
        out = tmp_path / name
        done = run_bundle(world, out, "--audit-inputs", str(audit_inputs))
        assert done.returncode == 0, done.stderr
        builds[name] = out

    for name in ("duplicate-texts.json", "exclusions.json"):
        assert (builds["carried"] / "audit" / name).read_bytes() == (
            builds["rebuilt"] / "audit" / name
        ).read_bytes()
    assert without_build_time(builds["carried"] / "release") == without_build_time(
        builds["rebuilt"] / "release"
    )


def test_the_receipt_names_the_inputs_the_settings_and_the_counts(world, tmp_path):
    out = tmp_path / "inputs"

    done = run_inputs(world, out, "--extracted-batches", "1")

    assert done.returncode == 0, done.stderr
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["command"] == "mbdef closure inputs"
    assert receipt["settings"]["extracted_batches"] == 1
    assert receipt["settings"]["tei_prefix"] == str(world.closure_tei)
    # three papers, each a TEI and a receipt
    assert receipt["inputs"]["closure_tei"]["files"] == 6
    assert set(receipt["outputs"]) == {
        "all-candidates.json",
        "papers-info.json",
        "uncited.json",
        "existing-exclude.json",
    }
    # One batch only, of three papers: all three were extracted when the check ran.
    assert receipt["summary"] == {
        "candidates": 3,
        "extracted": 3,
        "copies": 1,
        "uncited": 0,
        "existing_only_works": 1,
        "existing_not_added": 0,
        "seconds": receipt["summary"]["seconds"],
    }


def test_an_input_that_is_missing_stops_with_where_to_get_it(world, tmp_path):
    done = run_inputs(
        world, tmp_path / "inputs", "--earlier-ledgers", str(tmp_path / "nowhere")
    )

    assert done.returncode == 1
    assert f"{tmp_path / 'nowhere'} does not exist (earlier_ledgers)" in done.stderr
    assert "docs/step-4-closure/closure.md" in done.stderr
    assert not (tmp_path / "inputs").exists()


def cite_a_copy_of_the_existing_work(world: World, citing: list[str]) -> str:
    """A second cited work found only in the existing corpus, whose TEI there is W4's text again (the corpus holds the paper twice, under another DOI); the seed papers CITING cite it. Returns its paper id."""
    from mbdef.closure import existing

    member = "doi_10.9876_existing-copy.grobid.tei.xml"
    data = zipfile.ZipFile(world.seed_inputs / "existing-corpus-tei.zip").read(
        "doi_10.9876_existing.grobid.tei.xml"
    )
    with zipfile.ZipFile(world.seed_inputs / "existing-corpus-tei.zip", "a") as zf:
        zf.writestr(member, data)
    listed = world.seed_inputs / "existing-corpus-tei.sha256"
    listed.write_text(
        listed.read_text() + f"{hashlib.sha256(data).hexdigest()}  {member}\n"
    )
    before = json.loads((world.ledgers / "before.json").read_text())
    w4 = next(
        p
        for p in before["papers"]
        if p["reference"]["title"] == "A work only in the existing corpus"
    )
    copy = copy_of(w4)
    copy["reference"].update(
        title="The existing work, a second version",
        doi="10.9876/existing-copy",
        dois=["10.9876/existing-copy"],
    )
    copy["occurrences"] = [
        {**w4["occurrences"][0], "paper_id": seed} for seed in citing
    ]
    copy["corpus_match"]["files"][0]["file"] = member
    before["papers"].append(copy)
    (world.ledgers / "before.json").write_text(json.dumps(before))
    return existing.paper_id(existing.queue_row(copy["reference"]))


def copy_of(value):
    return json.loads(json.dumps(value))


@pytest.mark.parametrize(
    ("citing", "dropped"),
    [
        # cited by two seed papers, W4 by one: the copy stays, W4 is not added
        (["seed-a", "seed-b"], "w4"),
        # one each: the larger paper id stays (doi_10.9876_existing__… > doi_10.9876_existing-copy__…)
        (["seed-b"], "copy"),
    ],
)
def test_of_two_existing_corpus_works_with_one_text_the_one_fewer_seeds_cite_is_not_added(
    world, tmp_path, citing, dropped
):
    copy = cite_a_copy_of_the_existing_work(world, citing)
    out = tmp_path / "inputs"

    done = run_inputs(world, out)

    assert done.returncode == 0, done.stderr
    assert copy < world.ids["w4"]
    expected = world.ids["w4"] if dropped == "w4" else copy
    assert written(out / "existing-exclude.json") == [expected]


def test_a_missing_tei_folder_is_named_as_paper_text(world, tmp_path):
    done = run_inputs(
        world, tmp_path / "inputs", "--closure-tei", str(tmp_path / "nowhere")
    )

    assert done.returncode == 1
    assert f"{tmp_path / 'nowhere'} does not exist (closure_tei)" in done.stderr
    assert "the closure papers' TEI files" in done.stderr
    assert "not public: they hold the papers' text" in done.stderr
    assert "--closure-tei" in done.stderr
    assert "development" not in done.stderr
    assert "extraction" not in done.stderr
