"""`mbdef evaluate heldout`: the held-out TEST rescoring.

It reads the frozen TEST plan, reference and readiness record from the repository's data folder (`data/frozen/results/test-eval-20260924/`), the cohort (test-sources.json) and the saved rows from the data root, scores every row with the general scorer and applies the plan's frozen eligibility and pick rules. The tests build small rows roots: rows without predictions need no pair score at all, and two rows hold the two papers of tests/fixtures/score-sample/ with its recorded scores.
The original TEST scorer was run on the same rows once, and what it wrote is stored with its answers to the rule cases below (tests/fixtures/released-outputs.zip). Both results must be equal apart from the run details (times, paths, the scoring code's name and hashes).
"""

import copy
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote

import pytest
from cli_helpers import run_cli, write_env
from released_outputs import read_json

REPO = Path(__file__).resolve().parents[1]
TEST_EVAL = REPO / "data/frozen/results/test-eval-20260924"
SAMPLE = Path(__file__).resolve().parent / "fixtures/score-sample"
SOURCES = (
    "frozen/artifacts/experiments/test-eval-20260924/final/inputs/test-sources.json"
)
ROWS = "frozen/artifacts/experiments/test-eval-20260924/collected/rows"
COMPLETE = {
    "complete": True,
    "papers": 29,
    "failed_windows": 0,
    "total_windows": 90,
    "failed_batches": 1,
    "total_batches": 40,
    "requests": 130,
    "cost_usd": 0,
    "notes": "made up for the test",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cohort() -> list[str]:
    return [
        p["paper_id"] for p in read(TEST_EVAL / "inputs/test-reference.json")["papers"]
    ]


def write_row(rows: Path, row: str, status: dict | None, papers: Mapping[str, object]):
    folder = rows / row
    (folder / "papers").mkdir(parents=True)
    if status is not None:
        (folder / "status.json").write_text(json.dumps({"row": row, **status}))
    for pid, records in papers.items():
        (folder / "papers" / f"{quote(pid, safe='')}.json").write_text(
            json.dumps(
                {
                    "paper_id": pid,
                    "row": row,
                    "stage": "final",
                    "definition_records": records,
                }
            )
        )


def heldout_setup(tmp_path, with_predictions: bool = False):
    """A work folder whose data folder is the repository's, a data root with the cohort, and a rows root: an eligible row with empty outputs, an incomplete one, one with two papers, one with a broken paper file, and a row the plan does not know."""
    work, root = tmp_path / "work", tmp_path / "data-root"
    work.mkdir()
    (work / "data").symlink_to(REPO / "data")
    write_env(work, DATA_ROOT=str(root))
    ids = cohort()
    (root / SOURCES).parent.mkdir(parents=True)
    (root / SOURCES).write_text(json.dumps({"papers": [{"paper_id": p} for p in ids]}))
    rows = root / ROWS
    empty = {pid: [] for pid in ids}
    write_row(rows, "R0bea6bc", COMPLETE, empty)
    write_row(rows, "Rde76462", {**COMPLETE, "complete": False}, empty)
    write_row(rows, "R13f6cd1", COMPLETE, {**empty, ids[0]: "not a list"})
    write_row(rows, "EXTRA", COMPLETE, empty)
    sample = tmp_path / "sample"
    shutil.copytree(SAMPLE, sample)
    for row in ("Rdb54680", "QWEN-POOL3"):
        papers = {}
        if with_predictions:
            for path in sorted((sample / f"rows/{row}/papers").glob("*.json")):
                papers[path.stem] = read(path)["definition_records"]
        write_row(rows, row, {**COMPLETE, "papers": 2}, papers)
    return work, root, sample


@pytest.fixture(scope="module")
def released() -> dict:
    """What the original TEST scorer gave: its results on the rows of `heldout_setup` (without and with predictions) and its answers to the pick-rule and eligibility cases below."""
    return read_json("heldout/rescoring.json")


def without_run_details(result: dict) -> dict:
    """RESULT without its run details: the times, the rows roots, the Slurm job id, the reference paths up to `inputs/`, each row's folder and scoring time, and the scoring code's description and hashes."""
    result = copy.deepcopy(result)
    for key in ("created_utc", "rows_roots", "matcher", "scoring_code_sha256"):
        del result[key]
    del result["environment"]["slurm_job_id"]
    for ref in (result["reference"], result["reference"]["ready"]):
        ref["path"] = "inputs/" + ref["path"].rsplit("/inputs/", 1)[1]
    for row in result["rows"]:
        row.pop("directory", None)
        row.pop("scoring_seconds", None)
    return result


# The plan the original scorer read and its neutral copy, with only its `authorization` field reworded; the two results differ in this one value.
ORIGINAL_PLAN = "7cb89ade3465b54a6ffcb9ad4981483fbff105102fed3741bed61d186bbb552f"
NEUTRAL_PLAN = "bd80acc8da32d6a97815358327076a2fd715f1656ebc6100783f34ca70767e9d"


def plan_apart(new: dict, before: dict) -> tuple[dict, dict]:
    """NEW and BEFORE without their `plan_sha256`, which must be the neutral copy's and the original plan's."""
    new, before = copy.deepcopy(new), copy.deepcopy(before)
    assert (new.pop("plan_sha256"), before.pop("plan_sha256")) == (
        NEUTRAL_PLAN,
        ORIGINAL_PLAN,
    )
    return new, before


def heldout(work: Path, *extra: str):
    return run_cli(work, "evaluate", "heldout", "--out", "out", *extra)


def test_the_rescoring_equals_the_original_scorer_on_the_same_rows(tmp_path, released):
    work, _, sample = heldout_setup(tmp_path)

    done = heldout(work, "--replay-scores", str(sample / "pair-scores.json"))

    assert done.returncode == 0, done.stderr
    new = read(work / "out/results.json")
    now, before = plan_apart(new, released["without_predictions"])
    assert without_run_details(now) == without_run_details(before)
    rows = {r["row"]: r for r in new["rows"]}
    assert rows["R0bea6bc"]["eligible"] is True
    assert rows["Rde76462"]["ineligible_reasons"] == ["status.complete is not true"]
    assert rows["Rdb54680"]["ineligible_reasons"] == [
        "only 0/29 papers have output files",
        "status reports 2/29 papers with output",
    ]
    assert rows["R13f6cd1"]["errors"][0].startswith("invalid paper file")
    assert rows["Ra1da1ae"]["state"] == "missing"
    assert new["unplanned_rows"] == ["EXTRA"]
    assert new["decision"]["branch"] == "no eligible row reaches 80% recall"
    assert new["decision"]["pick"] is None
    assert (work / "out/results.md").read_text(encoding="utf-8").splitlines()[0] == (
        "# TEST evaluation results (29 held-out papers)"
    )


def test_rows_with_predictions_are_scored_like_the_original_scorer(tmp_path, released):
    work, _, sample = heldout_setup(tmp_path, with_predictions=True)

    done = heldout(work, "--replay-scores", str(sample / "pair-scores.json"))

    assert done.returncode == 0, done.stderr
    new = read(work / "out/results.json")
    now, before = plan_apart(new, released["with_predictions"])
    assert without_run_details(now) == without_run_details(before)
    rows = {
        r["row"]: r["score"]["binary_one_to_one"] for r in new["rows"] if "score" in r
    }
    # tests/test_evaluate_score.py: 7 and 10 of the two papers' 10 definitions; the recall denominator is all 365 TEST definitions.
    assert (rows["Rdb54680"]["tp"], rows["Rdb54680"]["predicted"]) == (7, 27)
    assert (rows["QWEN-POOL3"]["tp"], rows["QWEN-POOL3"]["gold"]) == (10, 365)
    assert done.stdout.splitlines()[-2:] == [
        "Pick: none. No eligible row reaches 80% recall and none has precision >= 15%; "
        "no row can be picked.",
        "wrote out",
    ]
    receipt = read(work / "out/receipt.json")
    assert receipt["command"] == "evaluate heldout"
    assert receipt["settings"]["plan_sha256"] == NEUTRAL_PLAN
    assert receipt["inputs"]["reference"]["sha256"] == (
        "1d99ec81f04a95206e08a9704d316e9452d2e808fef0645184effd1d1ca519de"
    )
    assert receipt["inputs"]["rows"]["files"] > 0


def test_a_row_that_cannot_be_scored_fails_the_rescoring(tmp_path):
    """Here the pair scores are missing, so the first row with predictions in the plan's order (QWEN-POOL3) cannot be scored: the command stops and writes nothing."""
    work, _, sample = heldout_setup(tmp_path, with_predictions=True)
    empty = tmp_path / "empty-scores.json"
    data = read(sample / "pair-scores.json")
    empty.write_text(
        json.dumps({**data, "texts": [], "similarity": [], "entailment": []})
    )

    done = heldout(work, "--replay-scores", str(empty))

    assert done.returncode == 1
    assert (
        done.stderr.strip()
        .splitlines()[-1]
        .startswith(
            "Scoring QWEN-POOL3 failed: NotRecorded: no recorded similarity for ("
        )
    )
    assert not (work / "out/results.json").exists()


def test_a_changed_plan_is_refused(tmp_path):
    work, _, _ = heldout_setup(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text((TEST_EVAL / "plan.json").read_text(encoding="utf-8") + " ")

    done = heldout(work, "--plan", str(plan))

    assert done.returncode == 1
    assert (
        done.stderr.strip() == f"{plan} is not the frozen TEST plan (sha256 bd80acc8…)."
    )


def test_a_readiness_record_that_names_no_hash_of_the_reference_is_refused(tmp_path):
    work, _, sample = heldout_setup(tmp_path)
    ready = tmp_path / "READY.json"
    marker = read(TEST_EVAL / "inputs/READY.json")
    del marker["files_sha256"]["test-reference.json"]  # it still lists the other inputs
    ready.write_text(json.dumps(marker))

    done = heldout(
        work, "--ready", str(ready), "--replay-scores", str(sample / "pair-scores.json")
    )

    assert done.returncode == 1
    assert done.stderr.strip().splitlines()[-1] == (
        "ValueError: The READY record names no sha256 of test-reference.json, so the "
        "reference cannot be checked against it"
    )
    assert not (work / "out/results.json").exists()


def pick_row(rid, tp, predicted, qwen=False, eligible=True):
    return {
        "row": rid,
        "eligible": eligible,
        "self_hosted_qwen": qwen,
        "score": {"binary_one_to_one": {"tp": tp, "predicted": predicted}},
    }


# The original scorer's own checks of its rules (`selftest`, written before any TEST score), on made-up rows with 100 reference definitions: (case, rows, expected pick). The scorer's whole decision for each case is stored.
PICK_CASES = [
    ("leader non-Qwen, Qwen within 2 at >=80%", [pick_row("A", 85, 200), pick_row("Q", 83, 300, True)], "Q"),
    ("leader non-Qwen, Qwen 3 behind", [pick_row("A", 85, 200), pick_row("Q", 82, 300, True)], "A"),
    ("Qwen more matches but lower precision", [pick_row("A", 81, 200), pick_row("Q", 90, 400, True)], "Q"),
    ("Qwen within 2 but below 80%", [pick_row("A", 80, 200), pick_row("Q", 79, 250, True)], "A"),
    ("leader is Qwen", [pick_row("Q", 85, 150, True), pick_row("A", 90, 300)], "Q"),
    ("ineligible leader ignored", [pick_row("A", 90, 100, eligible=False), pick_row("B", 82, 300)], "B"),
    ("no 80%: highest recall with precision >=15%", [pick_row("A", 70, 400), pick_row("B", 75, 600), pick_row("C", 60, 100)], "A"),
    ("no 80%: none with precision >=15%", [pick_row("A", 70, 1000)], None),
    ("exact 80% boundary counts", [pick_row("A", 80, 400), pick_row("B", 79, 100)], "A"),
    ("exact 15% boundary counts", [pick_row("A", 30, 200), pick_row("B", 20, 50)], "A"),
]  # fmt: skip
STATUS = {"complete": True, "papers": 29, "failed_windows": 1, "total_windows": 20, "failed_batches": 5, "total_batches": 100}  # fmt: skip
ELIGIBILITY_CASES = [
    ("5% exactly is allowed", {**STATUS}, 29, True),
    ("windows above 5%", {**STATUS, "failed_windows": 2}, 29, False),
    ("batches above 5%", {**STATUS, "failed_batches": 6}, 29, False),
    ("no verifier batches is n/a", {**STATUS, "failed_batches": 0, "total_batches": 0}, 29, True),
    ("window counts missing", {**STATUS, "total_windows": None}, 29, False),
    ("incomplete", {**STATUS, "complete": False}, 29, False),
    ("failed paper listed", {**STATUS, "failed_paper_ids": ["x"]}, 29, False),
    ("missing paper file", {**STATUS}, 28, False),
]  # fmt: skip


@pytest.mark.parametrize(
    "case, rows, expected", PICK_CASES, ids=[c[0] for c in PICK_CASES]
)
def test_the_pick_rule_decides_as_the_original_one(released, case, rows, expected):
    """The rule's branches that the released TEST rows never reached (no row had 80%)."""
    from mbdef.evaluate import heldout

    decision = heldout.apply_pick_rule(copy.deepcopy(rows), 100)

    assert decision["pick"] == expected
    assert decision == released["pick_rule"][case]


@pytest.mark.parametrize(
    "case, status, papers, expected",
    ELIGIBILITY_CASES,
    ids=[c[0] for c in ELIGIBILITY_CASES],
)
def test_eligibility_decides_as_the_original_one(
    released, case, status, papers, expected
):
    from mbdef.evaluate import heldout

    row = {
        "status": status,
        "papers": {str(i): [] for i in range(papers)},
        "errors": [],
    }

    ok, reasons = heldout.eligibility(copy.deepcopy(row), 29)

    assert ok is expected
    assert [ok, reasons] == released["eligibility"][case]
