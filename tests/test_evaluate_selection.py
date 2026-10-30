"""`mbdef evaluate select`: the pick rule of the extractor selection (plan amendment 8).

Run through the command line on a miniature table of scored setups (tests/mini_selection.py), whose numbers are chosen so that each part of the rule decides one pick.
"""

import json

from cli_helpers import run_cli, write_env
from mini_selection import TABLE, audit_file, plan_folder, scores_file


def select(tmp_path, table=TABLE, *extra: str):
    repo = tmp_path / "repo"
    repo.mkdir()
    plan_folder(repo)
    write_env(repo, DATA_ROOT=str(tmp_path / "data-root"))
    scores = scores_file(tmp_path / "scores.json", table)
    audit = audit_file(tmp_path / "audit.json", table)
    done = run_cli(
        repo,
        "evaluate",
        "select",
        "--scores",
        str(scores),
        "--audit",
        str(audit),
        "--out",
        str(tmp_path / "out"),
        *extra,
    )
    return done


def test_each_precision_floor_picks_by_the_frozen_rule(tmp_path):
    done = select(tmp_path)

    assert done.returncode == 0, done.stdout + done.stderr
    results = json.loads((tmp_path / "out/results.json").read_text())
    picks = {name: pick["pick"] for name, pick in results["final_picks"].items()}
    # P>=20: 315 and 314 matches are within 2 of each other; 314/1250 has the higher precision.
    # P>=24: only 314/1250 (25.1%) reaches 24% with 292+ matches.
    # P>=28: 294/1029 and 292/1022 tie on matches (within 2) and precision (2/7): fewer calls.
    # 330/1100 would win every floor, but the audit excludes its cohort; 291/700 is below 292.
    assert picks == {
        "P>=20": "test/gemma12b/A-minimal-8k/chat/verified-self",
        "P>=24": "test/gemma12b/A-minimal-8k/chat/verified-self",
        "P>=28": "test/gemma12b/B-minimal-16k32k/all3/filtered-title",
    }
    assert results["final_picks"]["P>=28"]["selection"]["tied"] == [
        "test/gemma12b/B-minimal-16k32k/chat/filtered-title",
        "test/gemma12b/B-minimal-16k32k/all3/filtered-title",
    ]
    excluded = results["decision"]["audit_excluded_test_configs"]
    assert [x["id"] for x in excluded] == ["test/gemma12b/C-gepa-16k32k/chat/combined"]
    assert "P>=28: gemma12b/B-minimal-16k32k/all3/filtered-title" in done.stdout


def test_no_setup_below_292_matches_is_picked(tmp_path):
    only_below = [("gemma12b/B-minimal-16k32k/chat/combined", 291, 700, 1)]

    done = select(tmp_path, only_below)

    assert done.returncode == 0, done.stdout + done.stderr
    results = json.loads((tmp_path / "out/results.json").read_text())
    assert {p["pick"] for p in results["final_picks"].values()} == {"none"}
    assert "P>=20: none" in done.stdout


def test_the_rules_numbers_are_flags_whose_defaults_are_the_plans(tmp_path):
    only_below = [("gemma12b/B-minimal-16k32k/chat/combined", 291, 700, 1)]

    done = select(
        tmp_path,
        only_below,
        *("--match-floor", "291", "--precision-floors", "20", "--tie-matches", "0"),
    )

    assert done.returncode == 0, done.stdout + done.stderr
    results = json.loads((tmp_path / "out/results.json").read_text())
    assert {name: p["pick"] for name, p in results["final_picks"].items()} == {
        "P>=20": "test/gemma12b/B-minimal-16k32k/chat/combined"
    }
    settings = json.loads((tmp_path / "out/receipt.json").read_text())["settings"]
    assert settings["final_match_floor"] == 291
    assert settings["final_precision_floors"] == {"P>=20": 20}
    assert settings["final_tie_matches"] == 0
