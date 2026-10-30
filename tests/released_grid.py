"""The original grid scorer's outputs on the miniature grid, and ours with the same made-up scorer.

What the original scorer wrote on the miniature grid (tests/mini_grid.py) with the made-up pair scorer (`mini_grid.made_up_scorers()`) is stored (tests/fixtures/released-outputs.zip, read by tests/released_outputs.py): its results on the grid. `our_grid()` runs `evaluate grid` in this process with the same scorer, recording its pair scores.
"""

from pathlib import Path

import pytest
from mini_grid import made_up_scorers
from released_outputs import read_json

REPO = Path(__file__).resolve().parents[1]

# The TEST evaluation's results, as far as the grid reads them: Opus's calls and the replicate.
OPUS_CALLS = 494
TEST_RESULTS = {
    "rows": [
        {"row": "Rdb54680", "status": {"requests": OPUS_CALLS}},
        {
            "row": "QWEN-POOL3",
            "score": {"binary_one_to_one": {"tp": 3, "predicted": 7, "gold": 365}},
        },
    ]
}


def released_grid_results(folder: Path) -> dict:
    """The original scorer's results on the miniature grid made under FOLDER."""
    return read_json("grid/mini-results.json", MINI=str(folder), REPO=str(REPO))


def released_environment() -> dict:
    """The environment values that importing the original scorer set (the device, the batch size)."""
    return read_json("grid/environment.json")


def our_grid(
    out: Path, roots: dict, data: Path, results: Path, record: Path, device: str
) -> None:
    """`evaluate grid` in this process, with the made-up scorer, recording its pair scores."""
    from mbdef.evaluate import grid, metric

    patch = pytest.MonkeyPatch()
    patch.setattr(metric, "models", lambda device, batch: made_up_scorers())
    patch.setenv("SCIDEF_NLI_DEVICE", device)
    patch.delenv("SLURM_JOB_ID", raising=False)
    try:
        grid.grid_command(
            str(out),
            str(roots["grid"]),
            str(roots["trained"]),
            str(data),
            str(results),
            None,
            1,
            1,
            None,
            str(record),
            metric.NLI_BATCH_SIZE,
        )
    finally:
        patch.undo()
