"""Step 5.4, the adopted ensemble: its cuts and corpus decisions equal those of the original implementation.

tests/fixtures/ensemble-mini/ holds a miniature input and, in `expected/`, the outputs of the original implementation on it, which the tests compare with byte for byte. The miniature has its own 1,200 train + dev records, so these tests call `decide_ensemble` with them; the command (`postprocess ensemble`) always reads the ground truth's 1,700, and the tests that run it need the frozen inputs in the data root. The slow test at the end checks the real frozen decisions.
"""

import csv
import json
import shutil
from pathlib import Path

import pytest
from cli_helpers import run_cli

from mbdef import files, postprocess
from mbdef.postprocess import ensemble
from mbdef.postprocess.groundtruth import Record

MINI = Path(__file__).parent / "fixtures/ensemble-mini"
TERM_MINI = Path(__file__).parent / "fixtures/term-merging-mini"


REPO = Path(__file__).resolve().parents[1]


def frozen_or_skip(rel):
    """A frozen Step 5 file `<date>/...`, in git or under DATA_ROOT/frozen (from .env); skip when it is not there."""
    files.load_env()
    try:
        path = postprocess.frozen_file(rel)
    except SystemExit:
        pytest.skip("DATA_ROOT is not set")
    if not path.exists():
        pytest.skip(f"{path} is not there yet")
    return path


def mini_records():
    rows = json.loads((MINI / "inputs/records.json").read_text())
    return [
        Record(
            id=r["id"],
            paper_id=r["paper_id"],
            split=r["split"],
            term="",
            label=r["label"],
            definitions=(),
        )
        for r in rows
    ]


def mini_inputs(**changes):
    i = MINI / "inputs"
    inputs = dict(
        pick=i / "pick.json",
        weight_tables={m: i / f"train-dev/{m}.csv" for m in ensemble.MEMBERS},
        references={
            "classifier": i / "train-dev/classifier.csv",
            "basal": i / "train-dev/basal-h200.csv",
            "judge": i / "train-dev/judge.csv",
        },
        corpus_scores={
            "classifier": i / "classifier-decisions.csv",
            "basal": i / "corpus/basal.csv",
            "judge": i / "corpus/judge.csv",
        },
        terms=i / "terms.jsonl",
        classifier_decisions=i / "classifier-decisions.csv",
        records=mini_records(),
    )
    return {**inputs, **changes}


def test_the_cut_on_the_reference_scale_equals_the_original_implementations(tmp_path):
    expected = json.loads((MINI / "expected/reference-scale.json").read_text())
    report = ensemble.decide_ensemble(**mini_inputs(), out=tmp_path)
    assert report["cut"] == expected["cuts_reference_scale"]["cut_95"]
    assert report["best_f1_cut"] == expected["cuts_reference_scale"]["best_f1"]


def test_the_corpus_decisions_equal_the_original_implementations_byte_for_byte(
    tmp_path,
):
    ensemble.decide_ensemble(**mini_inputs(), out=tmp_path)
    assert (tmp_path / "ensemble.csv").read_bytes() == (
        MINI / "expected/ensemble.csv"
    ).read_bytes()
    got = json.loads((tmp_path / "ensemble.json").read_text())
    for m in got["inputs"].values():
        m.pop("path")  # a folder of this machine; the sha256 and row count stay
    expected = json.loads((MINI / "expected/ensemble.json").read_text())
    assert {k: got[k] for k in expected} == expected


def test_the_frozen_weights_come_back_when_refitted_on_the_members_tables(tmp_path):
    report = ensemble.decide_ensemble(**mini_inputs(), out=tmp_path)
    assert report["weights_refit"]["max_abs_difference"] < 1e-9


def test_a_corpus_term_without_a_score_from_a_member_is_refused(tmp_path):
    judge = tmp_path / "judge.csv"
    lines = (MINI / "inputs/corpus/judge.csv").read_text().splitlines(keepends=True)
    judge.write_text("".join(lines[:-1]))  # the last term loses its judge score
    inputs = mini_inputs()
    inputs["corpus_scores"] = {**inputs["corpus_scores"], "judge": judge}
    with pytest.raises(ValueError, match="judge: 1 of 500 corpus terms have no score"):
        ensemble.decide_ensemble(**inputs, out=tmp_path / "out")
    assert not (tmp_path / "out/ensemble.csv").exists()


def test_no_decisions_file_when_the_kept_share_leaves_the_sanity_range(tmp_path):
    # every corpus score of every member at its lowest: (a) keeps nothing, far below 70%
    low = {}
    for m, path in mini_inputs()["corpus_scores"].items():
        rows = list(csv.DictReader(path.open(newline="")))
        for r in rows:
            r["score"] = "0.0"
        low[m] = tmp_path / f"{m}.csv"
        with low[m].open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with pytest.raises(SystemExit, match="sanity rule failed"):
        ensemble.decide_ensemble(**mini_inputs(corpus_scores=low), out=tmp_path / "out")
    assert not (tmp_path / "out/ensemble.csv").exists()


@pytest.mark.slow
def test_the_frozen_corpus_decisions_come_back_byte_for_byte(tmp_path):
    """The real run: the frozen member tables, the Label Groups' terms from the data root, the frozen pick."""
    frozen = frozen_or_skip("20261021/decisions/ensemble.csv")
    shutil.rmtree(tmp_path)
    done = run_cli(REPO, "postprocess", "ensemble", "--out", str(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "ensemble.csv").read_bytes() == frozen.read_bytes()
    stored = json.loads(frozen_or_skip("20261021/decisions/ensemble.json").read_text())
    assert json.loads((tmp_path / "ensemble.json").read_text())["cut"] == stored["cut"]


def test_the_command_takes_a_members_scores_from_a_flag_and_refuses_a_term_without_one(
    tmp_path,
):
    """`--corpus-scores judge=FILE` replaces the judge's frozen corpus scores; a corpus term without a score is refused and nothing is written (the real frozen inputs, CPU only)."""
    judge = frozen_or_skip(ensemble.CORPUS_SCORES["judge"])
    frozen_or_skip(ensemble.TERMS)
    lines = judge.read_text().splitlines(keepends=True)
    short = tmp_path / "judge.csv"
    short.write_text("".join(lines[:-1]))

    done = run_cli(
        REPO,
        *("postprocess", "ensemble", "--corpus-scores", f"judge={short}"),
        *("--out", str(tmp_path / "out")),
    )

    assert done.returncode == 1
    assert "judge: 1 of 67394 corpus terms have no score" in done.stderr
    assert not (tmp_path / "out/ensemble.csv").exists()
