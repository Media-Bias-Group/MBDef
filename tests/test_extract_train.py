"""`mbdef extract train` and `extract check-programs`, served from recorded responses.

No network, no GPU and no model. Training is tested on the synthetic papers of `tests/synthetic_papers.py`: `tests/fixtures/synthetic-training-replay.zip` holds a w8k run on them, recorded with a stand-in model and stand-in pair scores (`tests/fixtures/make_synthetic_replays.py` made it), with the synthetic papers' training dataset and split.
The released training files (the dataset, the split and the plans in `data/frozen/results/training-20260925/`) are checked against the released programs' manifests, and the released dataset and split refuse a TEST paper before any TEI is read.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import synthetic_papers as synthetic
from cli_helpers import run_cli, write_env

REPO = Path(__file__).resolve().parents[1]
RELEASED = REPO / "data/extractors/gemma12b-bootstrap"
FROZEN = REPO / "data/frozen/results/training-20260925"
TEIS = REPO / "data/paper_sources.zip"  # paper text, not part of this repository
SYNTHETIC = Path(__file__).with_name("fixtures") / "synthetic-training-replay.zip"
TRAIN = (
    "f5c282326822c5c1cca05a49940024480948e09c",  # Racism and Health: 9 / 5 / 2 windows
    "cbd5e98add83dafeff4bb3b6a538deaf1c3db0ea",
    "23eb7b57f821ec3b579605540b494d7c079a8467",
    "e049ae6d7d3e07536d096ebcd4ddf82e7c54a0ec",
)
DEV = (
    "92b4c8deecee703569b9e909dfb88aa70e691219",
    "79ddbeff892277592aa387045692e0a738ef9edf",
)
WINDOWS = ("w8k", "w16k", "w32k")
PREDICT = "chunk_extractor.extract_v3.predict"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def inputs() -> list[str]:
    """The repository's training dataset, split and TEIs (the commands run in a work folder)."""
    return [
        "--dataset",
        str(FROZEN / "runner/train-dataset.json"),
        "--split",
        str(FROZEN / "runner/split.json"),
        "--teis",
        str(TEIS),
    ]


def train(work: Path, recorded: Path, out: Path, window: str, *extra: str, seed="0"):
    args = ["extract", "train", "--window", window, "--out", str(out), *inputs()]
    for paper in (*TRAIN, *DEV):
        args += ["--paper", paper]
    args += ["--replay", str(recorded / window / "requests-train")]
    args += ["--replay-scores", str(recorded / window / "pair-scores.json"), *extra]
    return run_cli(work, *args, env={"PYTHONHASHSEED": seed})


def test_training_refuses_a_test_paper(tmp_path):
    """The released dataset and split are in this repository; the refusal comes before any TEI is read."""
    work, recorded = tmp_path / "work", tmp_path / "recorded"
    work.mkdir()
    test_paper = (
        "4243aa9bacd51bc02d4240378e7528912962b53e"  # a TEST paper of the ground truth
    )

    done = train(work, recorded, tmp_path / "out", "w8k", "--paper", test_paper)

    assert done.returncode == 1
    assert f"Paper {test_paper} is a TEST paper; training never reads TEST papers." in (
        done.stderr
    )
    assert not (tmp_path / "out").exists()


def test_the_frozen_training_files_are_the_ones_the_released_programs_name():
    """The plan files are the ones the released program manifests record, and the dataset and split are the ones the released runner's job plan pinned (data/frozen/README.md)."""
    sha = {
        name: hashlib.sha256((FROZEN / name).read_bytes()).hexdigest()
        for name in (
            "plan.json",
            "plan-amendment-1.json",
            "runner/train-dataset.json",
            "runner/split.json",
        )
    }
    job = read(FROZEN / "runner/job-plan.json")
    assert (sha["runner/train-dataset.json"], sha["runner/split.json"]) == (
        job["dataset_sha256"],
        job["split_sha256"],
    )
    for window in WINDOWS:
        manifest = read(RELEASED / window / "manifest.json")
        assert (sha["plan.json"], sha["plan-amendment-1.json"]) == (
            manifest["plan_sha256"],
            manifest["amendment_sha256"],
        )


# Training on the synthetic papers ----------------------------------------------------------------


def synthetic_setup(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """(a work folder with a `.env`, the synthetic recording unpacked, the inputs' arguments)."""
    work, recorded = tmp_path / "work", tmp_path / "recorded"
    work.mkdir(parents=True)
    write_env(work, DATA_ROOT=str(tmp_path / "data-root"))
    with zipfile.ZipFile(SYNTHETIC) as bundle:
        bundle.extractall(recorded)
    teis = synthetic.write_teis(tmp_path / "teis", (*synthetic.TRAIN, *synthetic.DEV))
    papers = [
        *("--dataset", str(recorded / "dataset.json")),
        *("--split", str(recorded / "split.json"), "--teis", str(teis)),
    ]
    return work, recorded, papers


def synthetic_train(tmp_path: Path, out: Path, *extra: str):
    work, recorded, papers = synthetic_setup(tmp_path)
    return run_cli(
        work,
        *("extract", "train", "--window", "w8k", "--out", str(out), *papers),
        *("--replay", str(recorded / "w8k/requests-train")),
        *("--replay-scores", str(recorded / "w8k/pair-scores.json"), *extra),
        env={"PYTHONHASHSEED": "0"},
    )


@pytest.fixture(scope="module")
def synthetic_trained(tmp_path_factory):
    """One replayed w8k training run on the synthetic papers: (finished process, out, recording)."""
    tmp_path = tmp_path_factory.mktemp("synthetic-train")
    out = tmp_path / "out"
    return synthetic_train(tmp_path, out), out, tmp_path / "recorded"


def without_versions(program: dict) -> dict:
    """A saved program without the versions of the environment that saved it (DSPy writes the Python and DSPy versions into a program)."""
    return {**program, "metadata": {}}


def test_on_synthetic_papers_a_replay_trains_the_recorded_program(synthetic_trained):
    done, out, recorded = synthetic_trained

    assert done.returncode == 0, done.stderr
    assert without_versions(read(out / "program.json")) == without_versions(
        read(recorded / "w8k/program.json")
    )
    assert {p.name for p in (out / "requests-train").iterdir()} == {
        p.name for p in (recorded / "w8k/requests-train").iterdir()
    }


def test_on_synthetic_papers_bootstrap_goes_on_until_a_paper_scores_and_keeps_its_window(
    synthetic_trained,
):
    """s1, the first TRAIN paper of the seed-9 order, has references that paraphrase its definitions, so the stand-in's quotes score 0 there; s2's match its references and score 1.
    So Bootstrap stops after two calls, and the program's one demo is s2's only window (a paper shorter than 8,000 characters is one window), with the title and the stand-in's answer."""
    done, out, _recorded = synthetic_trained

    assert done.returncode == 0, done.stderr
    summary = read(out / "receipt.json")["summary"]
    assert summary["papers_tried"] == [
        {"paper_id": "s1", "score": 0.0},
        {"paper_id": "s2", "score": 1.0},
    ]
    assert summary["demo"] == {"paper_id": "s2", "window": 0, "windows": 1}
    assert (summary["model_calls"], summary["failed_calls"]) == (2, 0)
    assert summary["papers"] == {"train": 4, "dev": 2}
    (demo,) = read(out / "program.json")[PREDICT]["demos"]
    assert demo["paper_text"] == synthetic.text_of("s2")
    assert demo["title"] == synthetic.PAPERS["s2"][0]
    assert [
        (r["term"], [d["text"] for d in r["definitions"]])
        for r in demo["definition_records"]
    ] == [(t, [s]) for t, s in synthetic.terms_of("s2").items()]
    # The program keeps the student's signature: the released programs' instruction.
    released = read(RELEASED / "w8k/program.json")[PREDICT]["signature"]
    assert read(out / "program.json")[PREDICT]["signature"] == released
    manifest = read(out / "manifest.json")
    assert (manifest["window_chars"], manifest["overlap_chars"]) == (8000, 1000)


def test_on_synthetic_papers_a_call_that_was_not_recorded_stops_the_run(tmp_path):
    out = tmp_path / "out"

    # Another output budget is another request: none of the recorded calls answers it.
    done = synthetic_train(tmp_path, out, "--max-tokens", "1000")

    assert done.returncode == 1
    assert "was not recorded" in done.stderr
    assert "No program was written" in done.stderr
    assert not (out / "program.json").exists()


def test_on_synthetic_papers_a_pair_score_that_was_not_recorded_stops_the_run(
    tmp_path,
):
    work, recorded, papers = synthetic_setup(tmp_path / "setup")
    scores = read(recorded / "w8k/pair-scores.json")
    empty = {**scores, "texts": [], "similarity": [], "entailment": []}
    (recorded / "w8k/pair-scores.json").write_text(json.dumps(empty))
    out = tmp_path / "out"

    done = run_cli(
        work,
        *("extract", "train", "--window", "w8k", "--out", str(out), *papers),
        *("--replay", str(recorded / "w8k/requests-train")),
        *("--replay-scores", str(recorded / "w8k/pair-scores.json")),
    )

    assert done.returncode == 1
    assert "no recorded" in done.stderr and "No program was written" in done.stderr
    assert not (out / "program.json").exists()


def test_on_synthetic_papers_training_refuses_a_test_paper(tmp_path):
    out = tmp_path / "out"

    done = synthetic_train(tmp_path, out, "--paper", "t1")

    assert done.returncode == 1
    assert "Paper t1 is a TEST paper; training never reads TEST papers." in done.stderr
    assert not out.exists()


def test_on_synthetic_papers_check_programs_finds_the_demo_and_refuses_a_changed_one(
    synthetic_trained, tmp_path
):
    done, out, recorded = synthetic_trained
    assert done.returncode == 0, done.stderr
    work, _recorded, papers = synthetic_setup(tmp_path / "setup")

    checked = run_cli(
        work, "extract", "check-programs", "--programs", str(out), *papers
    )

    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert (
        "demo 1: window 0 of 1 (numbered from 0; 8,000/1,000 characters) of TRAIN paper s2 "
        '"Who Gets the Last Word", word for word'
    ) in checked.stdout
    # The stand-in wrote the terms in lower case: the window has "Last-word bias".
    assert "1 in another case or spacing (term 'last-word bias')" in checked.stdout

    changed = tmp_path / "changed"
    changed.mkdir()
    program = read(out / "program.json")
    demo = program[PREDICT]["demos"][0]
    demo["paper_text"] = demo["paper_text"].replace("dispute", "quarrel")
    data = json.dumps(program).encode()
    (changed / "program.json").write_bytes(data)
    manifest = read(out / "manifest.json") | {
        "program_sha256": hashlib.sha256(data).hexdigest()
    }
    (changed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    refused = run_cli(
        work, "extract", "check-programs", "--programs", str(changed), *papers
    )

    assert refused.returncode == 1
    assert "demo 1 is not a window of any TRAIN paper" in refused.stdout
