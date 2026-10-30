"""Miniature inputs of `mbdef evaluate paired`.

`paired_inputs` writes the three saved outputs that the paired comparison with Opus reads, made up and small; `released_paired` is what the original comparison printed for them (tests/fixtures/released-outputs.zip).
"""

import json
from pathlib import Path

from released_outputs import read


def released_paired() -> bytes:
    """What the original comparison printed for `paired_inputs`."""
    return read("paired/output.txt")


OURS = "gemma12b/TR-bootstrap-8k+16k32k/chat-default-think/filtered-ordinary-qv"
# (found, predicted, reference definitions) per paper
PAIRED = {
    "dev": {
        "ours": {"a": (3, 5, 4), "b": (2, 4, 2), "c": (0, 1, 1)},
        "opus": {"a": (4, 6, 4), "b": (1, 3, 2), "c": (1, 2, 1)},
    },
    "test": {
        "ours": {"p": (5, 9, 6), "q": (3, 3, 4), "r": (2, 8, 2), "s": (0, 1, 3)},
        "opus": {"p": (4, 10, 6), "q": (3, 5, 4), "r": (2, 4, 2), "s": (1, 2, 3)},
    },
}


def per_paper(counts: dict) -> list[dict]:
    return [
        {"paper_id": pid, "tp": tp, "predicted": predicted, "gold": gold}
        for pid, (tp, predicted, gold) in counts.items()
    ]


def paired_inputs(folder: Path) -> list[Path]:
    """The selection's results.json (our config on DEV and TEST), the TEST rescoring's results.json (the Opus row) and the Opus DEV evaluation.json, in their stored shapes, as selection.json, test_results.json and opus_dev.json under FOLDER."""
    selection = {
        "configs": [
            {"id": f"{s}/{OURS}", "per_paper": per_paper(PAIRED[s]["ours"])}
            for s in ("dev", "test")
        ]
        + [{"id": "test/other", "per_paper": []}]
    }
    opus_test = [
        {"paper_id": p["paper_id"], "binary_one_to_one": p}
        for p in per_paper(PAIRED["test"]["opus"])
    ]
    test_rows = {
        "rows": [
            {"row": "R0bea6bc", "score": {"per_paper": []}},
            {"row": "Rdb54680", "score": {"per_paper": opus_test}},
        ]
    }
    opus_dev = {
        "biasdef": {
            "per_paper": [
                {"paper_id": p["paper_id"], "M3": {"binary_one_to_one": p}}
                for p in per_paper(PAIRED["dev"]["opus"])
            ]
        }
    }
    written = []
    for name, data in zip(
        ("selection", "test_results", "opus_dev"),
        (selection, test_rows, opus_dev),
        strict=True,
    ):
        path = folder / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        written.append(path)
    return written
