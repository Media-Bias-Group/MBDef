"""A corpus run's two commands of its own: `extract plan` and `extract derive` (tests/mini_derive.py).

`extract plan` cuts a corpus into shards balanced by their windows and lists the jobs (`extract run`, `verify`, `filter`) that make every stage the variants need; it is checked on five synthetic TEIs whose window counts are worked out by hand. `extract derive` reads the saved stage records of every part, from the part tarballs under the data root or from a new run's per-paper outputs, pools each variant's parts per paper, writes the per-paper files, `predictions.json` and the union, and compares the variants without gold. The expected numbers are worked out by hand from the records in tests/mini_derive.py (the worked example of docs/step-3-extraction/extract.md).
"""

import hashlib
import json
import shlex
import shutil
from pathlib import Path
from urllib.parse import unquote

import pytest
from cli_helpers import run_cli, write_env
from mini_derive import (
    FRAMING,
    GATEKEEPING,
    NEWS_VALUES,
    SEEDS,
    SELECTION,
    SLANTED,
    make_corpus,
    make_per_paper_run,
    make_seed_run,
    tei,
)

REPO = Path(__file__).resolve().parents[1]
PICKS = REPO / "data/extractors/picks.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def seed_run(tmp_path):
    """A work folder whose `.env` points at a data root holding the miniature run."""
    work, root = tmp_path / "work", tmp_path / "data-root"
    work.mkdir()
    make_seed_run(root)
    write_env(work, DATA_ROOT=str(root))
    return work, root


def derive(work: Path, out: Path, *extra: str):
    return run_cli(work, "extract", "derive", "--out", str(out), *extra)


def definitions(records: list[dict]) -> list[tuple[str, str]]:
    return [(r["term"], d["text"]) for r in records for d in r["definitions"]]


def test_each_variant_pools_its_parts_per_paper(seed_run, tmp_path):
    work, _root = seed_run
    out = tmp_path / "out"

    done = derive(work, out)

    assert done.returncode == 0, done.stderr
    derived = out / "derived"
    p20 = read(derived / "P20" / "papers" / "pdf-aaa.json")
    # Gemma's w8k record and Qwen's are one term ignoring case: the first spelling wins, the second definition is added, and Qwen's "Framing" has no definition, so it is dropped.
    assert definitions(p20["definition_records"]) == [
        ("Media bias", SLANTED),
        ("Media bias", SELECTION),
    ]
    assert p20["definition_records"][0]["definitions"][0]["citation"] == ["[1]"]
    assert p20["part_record_counts"] == {
        "gemma12b/TR-bootstrap-8k/chat-default-think": 1,
        "qwen/A-minimal-8k/chat-default": 2,
    }
    assert p20["title"] == "Media bias in the news"
    assert p20["stage"] == "filtered-ordinary-noverify"
    # P>=24 has one part, so its records are the stage's records as they are.
    p24 = read(derived / "P24" / "papers" / "pdf-aaa.json")
    assert definitions(p24["definition_records"]) == [("Media bias", SLANTED)]
    # P>=28 reads the Qwen verifier's stage from the .qv tarballs.
    p28 = read(derived / "P28" / "papers" / "pdf-aaa.json")
    assert definitions(p28["definition_records"]) == [
        ("Media bias", SLANTED),
        ("News values", NEWS_VALUES),
    ]
    single = read(derived / "P28-single-model" / "papers" / "pdf-aaa.json")
    assert definitions(single["definition_records"]) == [
        ("Media bias", SLANTED),
        ("News values", NEWS_VALUES),
    ]
    # A paper id is quoted into a file name.
    assert (derived / "P20" / "papers" / "doi%3A10.1%2Fb.json").is_file()
    ccc = read(derived / "P28" / "papers" / "pdf-ccc.json")
    assert definitions(ccc["definition_records"]) == [("Gatekeeping", GATEKEEPING)]


def test_predictions_the_union_and_the_summary(seed_run, tmp_path):
    work, _root = seed_run
    out = tmp_path / "out"

    done = derive(work, out)

    assert done.returncode == 0, done.stderr
    derived = out / "derived"
    p20 = read(derived / "P20" / "predictions.json")
    assert p20["paper_count"] == 3
    assert [p["paper_id"] for p in p20["papers"]] == [
        "doi:10.1/b",
        "pdf-aaa",
        "pdf-ccc",
    ]
    # The DOI from the TEI header wins over another one; a paper without a DOI has none.
    assert [p["metadata"] for p in p20["papers"]] == [
        {"title": "Framing effects", "doi": "10.1/b"},
        {"title": "Media bias in the news", "doi": "10.9/aaa"},
        {"title": "Gatekeeping"},
    ]
    assert p20["missing_part_outputs"] == []
    # The union pools all four variants per paper: pdf-aaa's two Media bias definitions (P>=20) and News values (P>=28).
    union = read(derived / "union" / "predictions.json")
    assert {
        p["paper_id"]: definitions(p["definition_records"]) for p in union["papers"]
    } == {
        "doi:10.1/b": [("Framing effect", FRAMING)],
        "pdf-aaa": [
            ("Media bias", SLANTED),
            ("Media bias", SELECTION),
            ("News values", NEWS_VALUES),
        ],
        "pdf-ccc": [("Gatekeeping", GATEKEEPING)],
    }
    summary = read(derived / "DONE.json")
    assert {
        name: (v["records"], v["definitions"]) for name, v in summary["picks"].items()
    } == {
        "P>=20": (3, 4),
        "P>=24": (2, 2),
        "P>=28": (3, 3),
        "P>=28-single-model": (3, 3),
    }
    assert summary["papers"] == 3
    assert summary["union"]["records"] == 4
    assert summary["problems"] == []
    assert summary["picks"]["P>=28"]["predictions"] == str(
        (derived / "P28" / "predictions.json").resolve()
    )


def test_a_missing_tarball_is_reported_and_its_papers_count_as_missing(
    seed_run, tmp_path
):
    work, root = seed_run
    qv = "parts/gemma12b/TR-bootstrap-16k32k/chat-default-think/shard-001.qv.tar.zst"
    (root / SEEDS / qv).unlink()
    out = tmp_path / "out"

    done = derive(work, out)

    assert done.returncode == 0, done.stderr
    assert f"PROBLEM: missing archive {qv}" in done.stdout.splitlines()
    summary = read(out / "derived" / "DONE.json")
    assert summary["problems"] == [f"missing archive {qv}"]
    assert summary["picks"]["P>=28"]["missing_part_outputs"] == 1
    assert summary["picks"]["P>=28-single-model"]["missing_part_outputs"] == 0
    p28 = read(out / "derived" / "P28" / "predictions.json")
    assert p28["missing_part_outputs"] == [
        "pdf-ccc gemma12b/TR-bootstrap-16k32k/chat-default-think"
    ]
    # The other part's records of that paper are still there.
    ccc = read(out / "derived" / "P28" / "papers" / "pdf-ccc.json")
    assert definitions(ccc["definition_records"]) == [("Gatekeeping", GATEKEEPING)]
    assert ccc["part_record_counts"] == {
        "gemma12b/TR-bootstrap-8k/chat-default-think": 1,
        "gemma12b/TR-bootstrap-16k32k/chat-default-think": 0,
    }


def test_the_variants_are_compared_without_gold(seed_run, tmp_path):
    work, _root = seed_run
    out = tmp_path / "out"

    done = derive(work, out)

    assert done.returncode == 0, done.stderr
    report = read(out / "compare" / "compare.json")
    assert report["papers"] == 3
    # Definitions per paper after folding case: P>=20 has 4 (pdf-aaa's two Media bias ones, the framing effect, gatekeeping), P>=28 has 3 (one Media bias, News values, gatekeeping).
    # They share 2: Jaccard 2 / (2 + 2 + 1) = 0.4, and 1 P>=28 definition is not in P>=20.
    assert report["pairwise_definition_keys"]["P>=20 vs P>=28"] == {
        "shared": 2,
        "only_P>=20": 2,
        "only_P>=28": 1,
        "jaccard": 0.4,
    }
    assert {k: v["jaccard"] for k, v in report["pairwise_definition_keys"].items()} == {
        "P>=20 vs P>=24": 0.5,
        "P>=20 vs P>=28": 0.4,
        "P>=20 vs P>=28-single-model": 0.4,
        "P>=24 vs P>=28": 0.6667,
        "P>=24 vs P>=28-single-model": 0.6667,
        "P>=28 vs P>=28-single-model": 1.0,
    }
    assert report["totals"]["P>=24"] == {
        "records": 2,
        "definitions": 2,
        "unique_definition_keys": 2,
        "papers_with_zero_definitions": 1,
    }
    assert report["union_definition_keys"] == 5
    # Counted only for exactly three variants, as the released comparison did.
    assert report["shared_by_all_three"] is None
    # P>=28's definitions by part: Media bias and gatekeeping from w8k, News values from w16k+w32k, each from that part only.
    assert report["part_contribution"]["P>=28"] == {
        "gemma12b/TR-bootstrap-8k/chat-default-think": {
            "definitions_in_part": 2,
            "only_from_this_part": 2,
        },
        "gemma12b/TR-bootstrap-16k32k/chat-default-think": {
            "definitions_in_part": 1,
            "only_from_this_part": 1,
        },
    }
    per_paper = read(out / "compare" / "per-paper.json")
    assert per_paper[1] == {
        "paper_id": "pdf-aaa",
        "P>=20": {"records": 1, "definitions": 2, "keys": 2},
        "P>=24": {"records": 1, "definitions": 1, "keys": 1},
        "P>=28": {"records": 2, "definitions": 2, "keys": 2},
        "P>=28-single-model": {"records": 2, "definitions": 2, "keys": 2},
    }
    table = (out / "compare" / "compare.md").read_text(encoding="utf-8")
    assert "| P>=20 vs P>=28 | 2 | 2 | 1 | 0.4 |" in table.splitlines()


def files(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def assert_same_but_the_time_and_the_output_folder(first: Path, second: Path) -> None:
    """Every file of two `extract derive` outputs is the same byte for byte, except the receipt and the two files that hold the time and the output folder (DONE.json, compare.json)."""
    a, b = files(first), files(second)
    assert a.keys() == b.keys()
    for name in a.keys() - {"receipt.json"}:
        if name not in ("compare/compare.json", "derived/DONE.json"):
            assert a[name] == b[name], name
            continue
        old, new = json.loads(a[name]), json.loads(b[name])
        old.pop("created_utc"), new.pop("created_utc")
        text = json.dumps(new).replace(str(second.resolve()), str(first.resolve()))
        assert json.loads(text) == old


def test_only_the_time_and_the_output_paths_change_between_two_runs(seed_run, tmp_path):
    work, _root = seed_run
    first, second = tmp_path / "first", tmp_path / "second"

    assert derive(work, first).returncode == 0
    assert derive(work, second).returncode == 0

    assert_same_but_the_time_and_the_output_folder(first, second)


def test_per_paper_outputs_give_the_same_files_as_the_tarballs(seed_run, tmp_path):
    """A new run keeps each part's stages as the per-paper files of `extract run`, `verify` and `filter` (tests/mini_derive.py: make_per_paper_run); the same records give the same files."""
    work, _root = seed_run
    per_paper = make_per_paper_run(tmp_path / "new-run")
    from_tarballs, from_folders = tmp_path / "tarballs", tmp_path / "folders"

    tarballs = derive(work, from_tarballs)
    folders = derive(work, from_folders, "--set", str(per_paper))

    assert folders.returncode == 0, folders.stderr
    assert_same_but_the_time_and_the_output_folder(from_tarballs, from_folders)
    assert folders.stdout.replace(str(from_folders), str(from_tarballs)) == (
        tarballs.stdout
    )
    receipt = read(from_folders / "receipt.json")
    assert set(receipt["inputs"]) == {
        "picks",
        "plans",
        "shards",
        "runs",
        "paper-metadata",
    }


def test_a_missing_stage_folder_is_reported_and_its_papers_count_as_missing(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write_env(work, DATA_ROOT=str(tmp_path / "data-root"))
    per_paper = make_per_paper_run(tmp_path / "new-run")
    folder = "runs/gemma12b/TR-bootstrap-16k32k/chat-default-think/shard-001/filtered-ordinary-qv"
    shutil.rmtree(per_paper / folder)
    out = tmp_path / "out"

    done = derive(work, out, "--set", str(per_paper))

    assert done.returncode == 0, done.stderr
    assert f"PROBLEM: missing folder {folder}" in done.stdout.splitlines()
    summary = read(out / "derived" / "DONE.json")
    assert summary["problems"] == [f"missing folder {folder}"]
    assert summary["picks"]["P>=28"]["missing_part_outputs"] == 1
    ccc = read(out / "derived" / "P28" / "papers" / "pdf-ccc.json")
    assert definitions(ccc["definition_records"]) == [("Gatekeeping", GATEKEEPING)]


def test_the_receipt_lists_the_inputs_settings_and_commit(seed_run, tmp_path):
    work, root = seed_run
    out = tmp_path / "out"

    assert derive(work, out).returncode == 0

    receipt = read(out / "receipt.json")
    assert receipt["command"] == "mbdef extract derive"
    seeds = root / SEEDS
    assert (
        receipt["inputs"]["picks"]["sha256"]
        == hashlib.sha256((seeds / "picks.json").read_bytes()).hexdigest()
    )
    # 3 parts x 2 shards, and a .qv tarball per shard of the 2 Gemma parts
    assert receipt["inputs"]["parts"]["files"] == 10
    assert set(receipt["inputs"]) == {
        "picks",
        "plans",
        "shards",
        "parts",
        "seed-manifest",
    }
    assert receipt["settings"]["set"] == str(seeds)
    assert receipt["settings"]["json_format"] == {"indent": 1, "sort_keys": True}
    # per variant 3 papers and predictions.json; the union and DONE.json
    assert receipt["outputs"]["derived"]["files"] == 4 * (3 + 1) + 2
    assert receipt["summary"]["P>=28"] == {"records": 3, "definitions": 3}
    assert "commit" in receipt["code"]


def plan(work: Path, teis: Path, out: Path, *extra: str):
    return run_cli(
        work,
        *("extract", "plan", "--teis", str(teis), "--picks", str(PICKS)),
        *("--out", str(out), *extra),
    )


def test_plan_cuts_the_corpus_into_shards_balanced_by_their_windows(tmp_path):
    """The five synthetic TEIs of tests/mini_derive.py (CORPUS: its table works out the shards)."""
    corpus = make_corpus(tmp_path / "teis")
    out = tmp_path / "set"

    done = plan(tmp_path, corpus, out, "--shard-papers", "2")

    assert done.returncode == 0, done.stderr
    shards = {p.stem: read(p) for p in sorted((out / "shards").glob("*.json"))}
    assert {k: [r["paper_id"] for r in v] for k, v in shards.items()} == {
        "shard-000": ["p1"],
        "shard-001": ["doi:10.1/p2", "p4"],
        "shard-002": ["p3", "p5"],
    }
    # A row: the paper's id, its TEI title and its TEI's sha256, in the recorded run's format.
    data = tei("Short paper", 5000, "10.5/p4")
    assert shards["shard-001"][1] == {
        "frozen_title": "Short paper",
        "paper_id": "p4",
        "tei_sha256": hashlib.sha256(data).hexdigest(),
    }
    manifest = out / "shards" / "shard-001.json"
    assert manifest.read_text(encoding="utf-8") == (
        json.dumps(shards["shard-001"], indent=1, sort_keys=True) + "\n"
    )
    inputs = read(out / "plans" / "grid.json")["inputs"]
    assert inputs["shard-001"] == {
        "manifest": "shards/shard-001.json",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "papers": 2,
        "windows": {"8000/1000": 4, "16000/2000": 3, "32000/4000": 2},
    }
    assert inputs["shard-000"]["windows"] == {
        "8000/1000": 6,
        "16000/2000": 3,
        "32000/4000": 2,
    }
    # The DOIs the TEIs name, which `extract derive` writes into each variant.
    assert read(out / "paper-metadata.json") == {
        "p1": {"tei_header_doi": "10.5/p1"},
        "p4": {"tei_header_doi": "10.5/p4"},
    }
    assert (out / "picks.json").read_bytes() == PICKS.read_bytes()
    assert "5 papers in 3 shards of 1 to 2 papers" in done.stdout


def test_plan_lists_the_jobs_that_make_every_stage_the_variants_need(tmp_path):
    """The released variants (data/extractors/picks.json) need, for each chat part, Gemma's own verification, the filter on the Qwen verifier's records and the filter on the union; for each JSON part only the filter on the union. Each stage reads the folder of the one before."""
    corpus = make_corpus(tmp_path / "teis")
    out = tmp_path / "set"

    assert plan(tmp_path, corpus, out, "--shard-papers", "2").returncode == 0

    jobs = read(out / "plans" / "grid.json")["jobs"]
    chat = "gemma12b/TR-bootstrap-16k32k/chat-default-think"
    assert [
        (j["stage"], j["input"], j["model"]) for j in jobs if j["part"] == chat
    ] == [
        ("combined", None, "gemma12b"),
        ("verified-self", "combined", "gemma12b"),
        ("filtered-ordinary-noverify", "combined", None),
        ("verified-qwen", "combined", "qwen"),
        ("filtered-ordinary-qv", "verified-qwen", None),
    ]
    json_part = "gemma12b/TR-bootstrap-8k/json-default-think"
    assert [j["stage"] for j in jobs if j["part"] == json_part] == [
        "combined",
        "filtered-ordinary-noverify",
    ]
    assert len(jobs) == 2 * 5 + 2 * 2
    folders = {(j["part"], j["stage"]): j["folder"] for j in jobs}
    assert folders[chat, "combined"] == f"runs/{chat}/{{shard}}/extract/w16k+w32k"
    assert folders[chat, "verified-qwen"] == f"runs/{chat}/{{shard}}/verified-qwen"
    # jobs.tsv: every job of every shard, a shard's jobs in the order to run them.
    lines = (out / "plans" / "jobs.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "shard\tpart\tstage\tmodel\tcommand"
    rows = [line.split("\t") for line in lines[1:]]
    assert len(rows) == 3 * 14
    commands = {(r[0], r[1], r[2]): shlex.split(r[4]) for r in rows}
    runs = f"{out}/runs/{chat}/shard-001"
    assert commands["shard-001", chat, "combined"] == [
        *("mbdef", "extract", "run", "--teis", str(corpus)),
        *("--shard", f"{out}/shards/shard-001.json", "--part", "w16k+w32k"),
        *("--format", "chat", "--out", f"{runs}/extract"),
    ]
    assert commands["shard-001", chat, "verified-self"] == [
        *("mbdef", "extract", "verify"),
        *(
            "--extraction",
            f"{runs}/extract/w16k+w32k",
            "--out",
            f"{runs}/verified-self",
        ),
        *("--model", "google/gemma-4-12B-it-qat-w4a16-ct"),
        *("--revision", "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee"),
    ]
    assert commands["shard-001", chat, "filtered-ordinary-noverify"] == [
        *("mbdef", "extract", "filter"),
        *("--run", f"{runs}/extract/w16k+w32k"),
        *("--out", f"{runs}/filtered-ordinary-noverify"),
        *("--filter", "ordinary", "--records", "union"),
    ]
    assert commands["shard-001", chat, "filtered-ordinary-qv"] == [
        *("mbdef", "extract", "filter", "--run", f"{runs}/verified-qwen"),
        *("--out", f"{runs}/filtered-ordinary-qv", "--filter", "ordinary"),
    ]
    assert [r[3] for r in rows[:5]] == ["gemma12b", "gemma12b", "-", "qwen", "-"]


def test_plan_keeps_given_rows_and_refuses_a_changed_tei(tmp_path):
    """--rows: the papers and their titles as a file of rows; every key is kept as given."""
    corpus = make_corpus(tmp_path / "teis")
    rows = [
        {"paper_id": "p4", "frozen_title": "A title from elsewhere", "source": "x"},
        {"paper_id": "p1", "frozen_title": "Long", "tei_sha256": "0" * 64},
    ]
    given = tmp_path / "rows.json"
    given.write_text(json.dumps(rows))

    refused = plan(tmp_path, corpus, tmp_path / "refused", "--rows", str(given))

    assert refused.returncode == 1
    assert "p1: its TEI's sha256 is not the row's" in refused.stderr
    assert not (tmp_path / "refused").exists()
    rows[1]["tei_sha256"] = hashlib.sha256(
        (corpus / "p1.tei.xml").read_bytes()
    ).hexdigest()
    given.write_text(json.dumps(rows))
    out = tmp_path / "set"
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"p1": {"doi": "10.9/from-elsewhere"}}')
    done = plan(
        tmp_path,
        corpus,
        out,
        *("--rows", str(given), "--prefix", "closure", "--metadata", str(metadata)),
    )
    assert done.returncode == 0, done.stderr
    # --metadata replaces the TEI headers' DOIs (a closure run's DOIs come from its acquisition).
    assert (out / "paper-metadata.json").read_bytes() == metadata.read_bytes()
    # One shard, its rows as given (p4 gets no tei_sha256), sorted by paper id.
    assert read(out / "shards" / "closure-000.json") == [rows[1], rows[0]]


def test_plan_reads_a_grobid_folder_by_its_conversion_receipts(tmp_path):
    """A closure run's TEIs are named by the acquisition, below sub-folders, each beside a conversion receipt that names its paper (`*.grobid.tei.closure.json`, `paperid`): `--teis` takes that folder as it is."""
    corpus = make_corpus(tmp_path / "teis")
    grobid = tmp_path / "grobid"
    for n, path in enumerate(sorted(corpus.glob("*.tei.xml"))):
        paper = path.name.removesuffix(".tei.xml")
        stem = grobid / f"{n:02d}" / f"work_{n:04d}.grobid.tei"
        stem.parent.mkdir(parents=True)
        stem.with_name(stem.name + ".xml").write_bytes(path.read_bytes())
        stem.with_name(stem.name + ".closure.json").write_text(
            json.dumps({"paperid": unquote(paper)})
        )
    by_id = plan(tmp_path, corpus, tmp_path / "by-id", "--shard-papers", "2")
    by_receipt = plan(tmp_path, grobid, tmp_path / "by-receipt", "--shard-papers", "2")

    assert by_id.returncode == by_receipt.returncode == 0, by_receipt.stderr
    for name in ("shard-000", "shard-001", "shard-002"):
        assert (tmp_path / f"by-receipt/shards/{name}.json").read_bytes() == (
            tmp_path / f"by-id/shards/{name}.json"
        ).read_bytes()


def test_without_the_seed_runs_saved_parts_derive_says_they_are_not_public(tmp_path):
    """The data root has no plan or parts of the seed run (they hold paper text): the default run stops with a sentence that names them and what to run instead, not a traceback."""
    write_env(tmp_path, DATA_ROOT=str(tmp_path / "data-root"))

    done = run_cli(tmp_path, "extract", "derive", "--out", str(tmp_path / "out"))

    assert done.returncode == 1, done.stdout + done.stderr
    assert "Traceback" not in done.stderr
    assert "frozen/runs/production-20260925/corpus/seeds/plans/grid.json" in done.stderr
    assert "are not public: the parts hold paper text" in done.stderr
    assert "extract plan" in done.stderr
    assert not (tmp_path / "out").exists()


def test_a_set_folder_without_a_plan_is_refused_with_a_pointer(tmp_path):
    (tmp_path / "run").mkdir()
    write_env(tmp_path, DATA_ROOT=str(tmp_path / "data-root"))

    done = run_cli(
        tmp_path,
        *("extract", "derive", "--set", str(tmp_path / "run")),
        *("--out", str(tmp_path / "out")),
    )

    assert done.returncode == 1, done.stdout + done.stderr
    assert "Traceback" not in done.stderr
    assert f"{tmp_path / 'run'}/plans/grid.json does not exist" in done.stderr
    assert "extract plan" in done.stderr


def test_two_conversion_receipts_of_one_paper_are_refused(tmp_path):
    """A GROBID folder whose receipts name one paper twice is refused (which TEI to read would otherwise depend on the order the file system lists the folders)."""
    corpus = make_corpus(tmp_path / "teis")
    grobid = tmp_path / "grobid"
    source = sorted(corpus.glob("*.tei.xml"))[0]
    paper = unquote(source.name.removesuffix(".tei.xml"))
    for n in range(2):
        stem = grobid / f"{n:02d}" / f"work_{n:04d}.grobid.tei"
        stem.parent.mkdir(parents=True)
        stem.with_name(stem.name + ".xml").write_bytes(source.read_bytes())
        stem.with_name(stem.name + ".closure.json").write_text(
            json.dumps({"paperid": paper})
        )

    done = plan(tmp_path, grobid, tmp_path / "set", "--shard-papers", "2")

    assert done.returncode == 1, done.stdout + done.stderr
    assert f"{paper}: two conversion receipts" in done.stderr
    assert "00/work_0000.grobid.tei.closure.json" in done.stderr
    assert "01/work_0001.grobid.tei.closure.json" in done.stderr
    assert not (tmp_path / "set").exists()
