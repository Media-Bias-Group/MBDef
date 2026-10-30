"""The commands that read MBDef-Gold, the ground truth, find it in the data root, where `data fetch mbdef-gold/` puts it from Hugging Face (docs/setup/data-fetch.md); git no longer holds it."""

from pathlib import Path

import pytest
from cli_helpers import mini_gold, run_cli, write_env

from mbdef import files
from mbdef.postprocess import groundtruth

REPO = Path(__file__).resolve().parents[1]
GOLD = f"$DATA_ROOT/{files.MBDEF_GOLD}"


def test_the_step_5_reader_loads_mbdef_gold_from_the_data_root(tmp_path, monkeypatch):
    mini_gold(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    records = groundtruth.load(votes=None)

    assert [(r.id, r.term, r.label) for r in records] == [
        ("p1:1:0", "framing bias", 1),
        ("p1:0:0", "LSTM", 0),
    ]


def test_without_mbdef_gold_in_the_data_root_a_reader_says_how_to_fetch_it(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as stop:
        groundtruth.load(votes=None)

    assert str(stop.value) == (
        f"{tmp_path}/{files.MBDEF_GOLD}/validated_gt.json is missing. It is a file of MBDef-Gold, the ground truth, "
        f"on Hugging Face ({files.MBDEF_GOLD_REPO}): `mbdef data fetch {files.MBDEF_GOLD}/` downloads it "
        "into the data root (docs/setup/data-fetch.md)."
    )


@pytest.mark.parametrize(
    "command, flag",
    [
        (("extract", "train-dataset"), "--ground-truth"),
        (("extract", "train-dataset"), "--non-relevant"),
        (("extract", "train-filter"), "--ground-truth"),
        (("extract", "train-filter"), "--non-relevant"),
        (("dataset", "build"), "--ground-truth"),
        (("dataset", "build"), "--non-relevant"),
        (("embed", "build"), "--ground-truth"),
        (("postprocess", "judge", "package"), "--ground-truth-folder"),
    ],
    ids=lambda value: " ".join(value) if isinstance(value, tuple) else value,
)
def test_each_flag_for_mbdef_gold_defaults_to_the_data_root(command, flag):
    done = run_cli(
        REPO, *command, "--help", env={"COLUMNS": "1000"}
    )  # no wrapped lines

    assert done.returncode == 0, done.stderr
    text = " ".join(done.stdout.split())
    help_of_flag = text.split(f" {flag} ", 1)[1].split(" --", 1)[0]
    assert f"default: {GOLD}" in help_of_flag, help_of_flag
    assert "data/validated_gt" not in text


def test_a_command_reads_mbdef_gold_from_the_data_root_when_no_flag_names_it(
    tmp_path,
):
    """`extract train-dataset` without --ground-truth stops at the missing file in the data root, and names it."""
    work = tmp_path / "work"
    work.mkdir()
    write_env(work, DATA_ROOT=str(tmp_path / "data-root"))

    done = run_cli(
        work, "extract", "train-dataset", "--out", str(tmp_path / "out"), "--teis", "x"
    )

    assert done.returncode == 1
    assert (
        f"{tmp_path}/data-root/{files.MBDEF_GOLD}/validated_gt.json is missing"
        in done.stderr
    )
