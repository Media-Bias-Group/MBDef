"""The real scoring models on the scorer tests' sample (tests/fixtures/score-sample/): run by hand on a GPU, never in CI.

    uv run pytest -m slow tests/test_evaluate_score_gpu.py   # on a shared machine, under flock

It needs the two models in the Hugging Face cache at their pinned revisions (BAAI/bge-base-en-v1.5 a5beb1e3…, tasksource/ModernBERT-large-nli ca476cb9…) and a CUDA GPU. `evaluate score` runs the sample's two scorings with the models instead of the recorded scores; every pair score it computes must be within ±1e-5 of the recorded one (the tolerance for GPU probabilities), and the counts must equal the released run's (tests/test_evaluate_score.py).
"""

import json
import shutil
from pathlib import Path

import pytest
from cli_helpers import run_cli, write_env
from test_evaluate_score import RELEASED, SAMPLE

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


def pair_scores(path: Path) -> dict[str, dict[tuple[str, str], float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    texts = data["texts"]
    return {
        kind: {(texts[i], texts[j]): value for i, j, value in data[kind]}
        for kind in ("similarity", "entailment")
    }


def test_the_real_models_give_the_recorded_scores_and_the_released_counts(tmp_path):
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    work = tmp_path / "work"
    work.mkdir()
    write_env(work, SCIDEF_NLI_DEVICE="cuda")
    shutil.copytree(SAMPLE, work / "sample")

    done = run_cli(
        work,
        "evaluate",
        "score",
        "--scorings",
        "sample/scorings.json",
        "--record-scores",
        "computed.json",
        "--out",
        "out",
        env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )

    assert done.returncode == 0, done.stderr
    computed = pair_scores(work / "computed.json")
    recorded = pair_scores(work / "sample/pair-scores.json")
    for kind in ("similarity", "entailment"):
        assert computed[kind].keys() == recorded[kind].keys()
        worst = max(abs(computed[kind][p] - recorded[kind][p]) for p in recorded[kind])
        assert worst <= 1e-5, (kind, worst)
    scored = json.loads((work / "out/scores.json").read_text(encoding="utf-8"))
    for scoring in scored["scorings"]:
        for paper in scoring["score"]["per_paper"]:
            tp, fp, fn, _ = RELEASED[scoring["id"]][paper["paper_id"]]
            counts = paper["binary_one_to_one"]
            assert (counts["tp"], counts["fp"], counts["fn"]) == (tp, fp, fn)
    receipt = json.loads((work / "out/receipt.json").read_text(encoding="utf-8"))
    assert receipt["settings"]["device"] == "cuda"


def test_step_5_can_still_call_the_entailment_scorer_as_before():
    """Step 5 (src/step5/nli.py on its branches) calls `NLIScorer().directional(pairs)`."""
    pytest.importorskip("transformers")
    from mbdef.evaluate.metric import NLIScorer

    scorer = NLIScorer(device="cuda")
    values = scorer.directional(
        [
            ("Media bias is a slanted portrayal of events.", "Media bias is slant."),
            ("Media bias is slant.", "Media bias is a slanted portrayal of events."),
        ]
    )
    assert len(values) == 2 and all(0 <= v <= 1 for v in values)
    assert values[0] > 0.5
