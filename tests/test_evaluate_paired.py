"""`mbdef evaluate paired`: our P≥28 extractor against Opus, paper by paper, from saved per-paper counts.

The original comparison resampled whole papers with Python's `random.Random(20261013)` (20,000 draws) and swapped the two systems within each paper with `random.Random(20261014)` (20,000 swaps). The command must print exactly what it printed for the same inputs: the same draws in the same order, the same interval indices. The inputs (tests/mini_scoring.py) are made up and small, so the first lines can be checked by hand. What the original comparison printed for these inputs is stored (tests/fixtures/released-outputs.zip).
"""

import json
from pathlib import Path

from cli_helpers import run_cli, write_env
from mini_scoring import OURS, released_paired
from mini_scoring import paired_inputs as write_inputs


def paired(work: Path, inputs: list[Path]):
    selection, test_results, opus_dev = inputs
    return run_cli(
        work,
        "evaluate",
        "paired",
        "--selection",
        str(selection),
        "--test-results",
        str(test_results),
        "--opus-dev",
        str(opus_dev),
        "--out",
        "out",
    )


def test_the_comparison_prints_exactly_what_the_original_comparison_printed(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write_env(work)
    inputs = write_inputs(tmp_path / "inputs")

    done = paired(work, inputs)

    assert done.returncode == 0, done.stderr
    output = (work / "out/output.txt").read_bytes()
    assert output == released_paired()
    lines = output.decode().splitlines()
    # By hand: DEV ours finds 3 + 2 + 0 = 5 of 7, Opus 4 + 1 + 1 = 6; -1 is -14.3 points of recall; ours finds more in b, fewer in a and c.
    assert lines[0] == "dev: papers=3 ours=(5, 10, 7) opus=(6, 11, 7)"
    assert lines[1].startswith("  found diff (ours-opus) -1  95% CI (")
    assert lines[2].startswith("  recall diff pp -14.3  95% CI (")
    assert lines[4] == "  papers where ours finds more / fewer / same: 1 / 2 / 0"
    assert lines[5] == "test: papers=4 ours=(10, 21, 15) opus=(10, 21, 15)"
    assert lines[6].startswith("  found diff (ours-opus) +0  95% CI (")
    assert lines[9] == "  papers where ours finds more / fewer / same: 1 / 1 / 2"
    assert lines[10].startswith("dev: permutation p (two-sided, 20000 swaps): recall ")
    assert lines[11].endswith(
        "sign test on papers (1 more / 1 fewer, 2 ties): p = 1.000"
    )
    assert len(lines) == 12


def test_the_numbers_and_the_pinned_settings_are_recorded(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write_env(work)
    inputs = write_inputs(tmp_path / "inputs")

    done = paired(work, inputs)

    assert done.returncode == 0, done.stderr
    numbers = json.loads((work / "out/paired.json").read_text(encoding="utf-8"))
    assert numbers["test"]["papers"] == 4
    assert numbers["test"]["ours"] == {"found": 10, "predicted": 21, "reference": 15}
    assert numbers["test"]["opus"] == {"found": 10, "predicted": 21, "reference": 15}
    assert numbers["dev"]["recall_diff_pp"]["observed"] == -100 / 7
    assert numbers["dev"]["papers_more_fewer_same"] == [1, 2, 0]
    receipt = json.loads((work / "out/receipt.json").read_text(encoding="utf-8"))
    assert receipt["command"] == "evaluate paired"
    assert receipt["settings"] == {
        "ours": OURS,
        "opus_test_row": "Rdb54680",
        "bootstrap": {
            "rng": "Python random.Random",
            "seed": 20261013,
            "replicates": 20000,
            "unit": "paper",
            "interval": "sorted draws at index int(0.025 n) and int(0.975 n) - 1",
        },
        "permutation": {
            "rng": "Python random.Random",
            "seed": 20261014,
            "swaps": 20000,
            "unit": "paper",
        },
        "splits": ["dev", "test"],
    }
    assert set(receipt["inputs"]) == {"selection", "test_results", "opus_dev"}


def test_papers_that_differ_between_the_two_systems_are_refused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write_env(work)
    inputs = write_inputs(tmp_path / "inputs")
    data = json.loads(inputs[2].read_text())
    data["biasdef"]["per_paper"].pop()
    inputs[2].write_text(json.dumps(data))

    done = paired(work, inputs)

    assert done.returncode == 1
    assert done.stderr.strip() == "dev: the two systems have different papers: c."
    assert not (work / "out/output.txt").exists()
