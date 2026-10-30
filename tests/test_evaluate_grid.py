"""`mbdef evaluate grid` and `evaluate select` give what the original grid scorer gave.

The original scorer (its `evaluate`, which scores and picks in one go, and `add_replicate`) was run once on a miniature grid (tests/mini_grid.py) with a made-up pair scorer, and its results are stored (tests/released_grid.py reads them from tests/fixtures/released-outputs.zip); the package runs on the same grid with the same scorer, so no model is needed. The results must be identical apart from the run's time, the seconds each scoring took, and the name of the merge function in the recipes.
"""

import json
import shutil
from pathlib import Path

import pytest
from cli_helpers import run_cli, write_env
from mini_grid import make_grid, write_audit
from released_grid import (
    OPUS_CALLS,
    TEST_RESULTS,
    our_grid,
    released_environment,
    released_grid_results,
)

from mbdef.evaluate import selection

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    folder = tmp_path_factory.mktemp("grid")
    roots = make_grid(folder)
    audit = write_audit(folder / "audit.json")
    results = folder / "test-results.json"
    results.write_text(json.dumps(TEST_RESULTS))
    return folder, roots, audit, results


@pytest.fixture(scope="module")
def released(mini):
    """The original scorer's results on the miniature grid, and the environment it set."""
    folder = mini[0]
    return released_grid_results(folder), released_environment()


@pytest.fixture(scope="module")
def ours(mini, released, tmp_path_factory):
    """`evaluate grid` (in this process, with the made-up scorer) and `evaluate select`."""
    folder, roots, audit, results = mini
    _, environment = released
    out = tmp_path_factory.mktemp("ours")
    our_grid(
        out / "grid",
        roots,
        REPO / "data",
        results,
        out / "pair-scores",
        environment["SCIDEF_NLI_DEVICE"],
    )
    selection.select_command(
        str(out / "grid/scores.json"),
        str(out / "select"),
        str(REPO / "data"),
        str(audit),
        None,
        None,
        None,
    )
    return out


def comparable(value, key=""):
    """Without the run's time and the seconds of each scoring; the merge function by one name."""
    if isinstance(value, dict):
        return {
            k: comparable(v, k)
            for k, v in value.items()
            if k not in ("seconds", "created_utc")
        }
    if isinstance(value, list):
        return [comparable(v) for v in value]
    if isinstance(value, str):
        return value.replace("mbdef.extract.records.merge", "src.extractor.merge")
    return value


def test_grid_and_select_give_the_original_results(ours, released):
    before, _ = released
    after = json.loads((ours / "select/results.json").read_text())

    assert comparable(after) == comparable(before)
    # The miniature grid exercises what it should: matches, pools, trained pools, the audit.
    configs = {c["id"]: c for c in after["configs"]}
    assert configs["test/qwen/A-minimal-8k/chat+schema8192/combined"]["matches"] > 0
    assert "test/gemma12b/TR-bootstrap-8k+16k32k/chat-default-think/combined" in configs
    assert not configs["test/qwen/A-minimal-8k/chat+schema8192/combined"][
        "passes_audit"
    ]
    assert not configs["test/qwen/A-minimal-8k/chat/combined"]["eligible"]


def test_the_scores_carry_no_audit_or_pick(ours):
    scores = json.loads((ours / "grid/scores.json").read_text())

    assert scores["schema"] == "taxonomy_pipeline/grid-scores-1"
    assert scores["opus_test_calls"] == OPUS_CALLS
    assert not {"audit", "final_picks", "decision"} & set(scores)
    assert not any("passes_audit" in c for c in scores["configs"])


def test_replayed_scores_equal_the_recorded_run(ours, mini, tmp_path):
    folder, roots, _, results = mini
    write_env(folder, SCIDEF_NLI_DEVICE="cuda")
    done = run_cli(
        folder,
        "evaluate",
        "grid",
        "--grid",
        str(roots["grid"]),
        "--trained",
        str(roots["trained"]),
        "--data",
        str(REPO / "data"),
        "--test-results",
        str(results),
        "--replay-scores",
        str(ours / "pair-scores"),
        "--workers",
        "2",
        "--out",
        str(tmp_path / "replayed"),
    )

    assert done.returncode == 0, done.stdout + done.stderr
    replayed = json.loads((tmp_path / "replayed/scores.json").read_text())
    recorded = json.loads((ours / "grid/scores.json").read_text())
    for value in (replayed, recorded):
        value.pop("workers")
        value.pop("environment")
    assert comparable(replayed) == comparable(recorded)
    assert sorted(p.name for p in (ours / "pair-scores").iterdir()) == [
        "dev-gemma12b.json",
        "dev-qwen.json",
        "test-gemma12b.json",
        "test-qwen.json",
    ]


def test_a_pair_that_was_not_recorded_stops_the_scoring(ours, mini, tmp_path):
    folder, roots, _, results = mini
    recording = tmp_path / "recording"
    recording.mkdir()
    for path in (ours / "pair-scores").iterdir():
        data = json.loads(path.read_text())
        data["entailment"] = data["entailment"][1:]
        (recording / path.name).write_text(json.dumps(data))

    done = run_cli(
        folder,
        "evaluate",
        "grid",
        "--grid",
        str(roots["grid"]),
        "--trained",
        str(roots["trained"]),
        "--data",
        str(REPO / "data"),
        "--test-results",
        str(results),
        "--replay-scores",
        str(recording),
        "--family",
        "test/qwen",
        "--out",
        str(tmp_path / "replayed"),
    )

    assert done.returncode == 1
    assert "no recorded entailment" in done.stderr
    assert "No results were written" in done.stderr
    assert not (tmp_path / "replayed").exists()


def test_the_general_scorer_scores_the_written_scorings_as_the_grid_does(
    ours, mini, tmp_path
):
    """The cluster job runs `evaluate score` on the scorings `--write-scorings` writes. Fed the grid's recorded pair scores, it asks for no other pair and gives every record set the grid's numbers, so a recording of the job serves `evaluate grid --replay-scores`."""
    folder, roots, _, results = mini
    common = ["--grid", str(roots["grid"]), "--trained", str(roots["trained"])]
    common += ["--data", str(REPO / "data"), "--test-results", str(results)]
    done = run_cli(
        folder,
        "evaluate",
        "grid",
        *common,
        "--write-scorings",
        "--out",
        str(tmp_path / "scorings"),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    grid_scores = json.loads((ours / "grid/scores.json").read_text())
    by_records = {
        c["records_sha256"]: c
        for c in grid_scores["configs"] + grid_scores["cohort_scores"]
        if c.get("state") == "scored"
    }
    for family in ("test-qwen", "test-gemma12b", "dev-qwen", "dev-gemma12b"):
        listing = tmp_path / "scorings" / f"{family}.json"
        assert {e["id"] for e in json.loads(listing.read_text())} <= set(by_records)
        done = run_cli(
            folder,
            "evaluate",
            "score",
            "--scorings",
            str(listing),
            "--replay-scores",
            str(ours / "pair-scores" / f"{family}.json"),
            "--out",
            str(tmp_path / "scored" / family),
        )
        assert done.returncode == 0, done.stdout + done.stderr
        scored = json.loads((tmp_path / "scored" / family / "scores.json").read_text())
        assert scored["scorings"]
        for entry in scored["scorings"]:
            b = entry["score"]["binary_one_to_one"]
            setup = by_records[entry["id"]]
            assert (b["tp"], b["predicted"]) == (setup["matches"], setup["predicted"])
            assert entry["score"]["full_record_macro_score"] == setup["full_record"]


def test_an_unreadable_request_receipt_of_a_thinking_cohort_stops_the_scoring(
    ours, mini, tmp_path
):
    """The reasoning-text count reads every request receipt of a thinking cohort; a receipt it cannot read stops the run instead of being skipped (it used to undercount silently)."""
    folder, roots, _, results = mini
    trained = tmp_path / "trained"  # the miniature's thinking cohorts are trained ones
    shutil.copytree(roots["trained"], trained)
    broken = sorted(
        trained.glob("test/gemma12b/*/*-think/run/requests-extract/*.json")
    )[0]
    broken.write_text("{not json")

    done = run_cli(
        folder,
        "evaluate",
        "grid",
        *("--grid", str(roots["grid"]), "--trained", str(trained)),
        *("--data", str(REPO / "data"), "--test-results", str(results)),
        *("--replay-scores", str(ours / "pair-scores"), "--family", "test/gemma12b"),
        *("--out", str(tmp_path / "replayed")),
    )

    assert done.returncode == 1
    assert f"Cannot read the request receipt {broken}" in done.stderr
    assert not (tmp_path / "replayed").exists()
