"""`mbdef postprocess` through the command line: the TEST split's guard, and no run writing into a frozen folder.

Every command reads Step 5's frozen files (data/frozen/step5-post-processing/) and writes only to the folder its --out names. A read of the TEST split, by `test --read` or `extractor-gate --read-test`, warns, needs a reason and a person typing the confirmation at a terminal, and is logged; without a terminal it is refused before anything of TEST is read.

The saved TEST result (test-eval.json) and the TEST read log (test-reads.json) are not part of this repository, so the commands that need them stop with a message that says so, before anything of TEST is read (the last tests of this file).
"""

from pathlib import Path

import pytest
from cli_helpers import run_cli

REPO = Path(__file__).resolve().parents[1]
NOT_HERE = "is not part of this repository"


def test_no_command_writes_into_a_frozen_folder(tmp_path):
    """A frozen folder of a throwaway data root, so that a broken refusal could not write into `data/frozen/`."""
    root = tmp_path / "data-root"
    inside = root / "frozen/step5-post-processing/20261015/new-folder"
    for action in (
        ["ensemble"],
        ["merge-terms"],
        ["decide", "--path", "classifier"],
        ["evaluate", "--paths", "keyword"],
        ["freeze"],
        ["extractor-gate"],
    ):
        done = run_cli(
            REPO,
            "postprocess",
            *action,
            "--out",
            str(inside),
            env={"DATA_ROOT": str(root)},
        )
        assert done.returncode == 1, action
        assert "is inside a frozen folder" in done.stderr, action
    assert not inside.exists()


def test_every_writing_command_needs_an_out_folder():
    for action in (
        ["ensemble"],
        ["merge-terms"],
        ["decide", "--path", "classifier"],
        ["label-groups"],
        ["evaluate"],
        ["freeze"],
        ["extractor-gate"],
    ):
        done = run_cli(REPO, "postprocess", *action)
        assert done.returncode == 2, action
        assert "the following arguments are required: --out" in done.stderr, action


def test_a_read_without_the_read_log_is_refused_before_anything_is_written(tmp_path):
    """Without test-reads.json nobody can tell how often the TEST split was read, so the read is refused and says which log is missing."""
    from mbdef.postprocess import testrun

    inputs = tmp_path / "test-evaluation"
    inputs.mkdir()
    (inputs / "frozen-paths.json").write_text("{}")
    out = tmp_path / "read"

    with pytest.raises(SystemExit) as refused:
        testrun.start_from_frozen(out, inputs)

    assert f"{inputs / 'test-reads.json'}) {NOT_HERE}" in str(refused.value)
    assert "does not read the TEST split" in str(refused.value)
    assert not out.exists()


def test_the_hash_comparison_says_the_saved_result_is_missing(tmp_path, monkeypatch):
    from mbdef.postprocess import testrun

    monkeypatch.setattr(testrun, "TEST_EVAL", tmp_path / "test-eval.json")

    with pytest.raises(SystemExit) as refused:
        testrun.compare_test()

    assert NOT_HERE in str(refused.value)
    assert "nothing to compare here" in str(refused.value)


def test_the_commands_say_that_the_test_files_are_not_here(tmp_path):
    compare = run_cli(REPO, "postprocess", "test")
    read = run_cli(
        REPO,
        "postprocess",
        "test",
        "--read",
        "--reason",
        "a test",
        "--out",
        str(tmp_path / "r"),
    )
    gate = run_cli(
        REPO,
        "postprocess",
        "extractor-gate",
        "--read-test",
        "--reason",
        "a test",
        "--out",
        str(tmp_path / "g"),
    )

    assert (compare.returncode, read.returncode, gate.returncode) == (1, 1, 1)
    for done in (compare, read, gate):
        assert NOT_HERE in done.stderr
    for done in (compare, read, gate):
        assert "Traceback" not in done.stderr
    assert not (tmp_path / "r").exists() and not (tmp_path / "g").exists()
