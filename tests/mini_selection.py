"""A miniature selection table: a few scored setups on TEST and DEV, an audit, and the frozen plan.

`scores_file` writes what `mbdef evaluate grid` writes (the scored grid, before any audit or pick), but for a handful of made-up setups whose numbers exercise the pick rule of plan amendment 8. `audit_file` writes an audit in the auditor's format: one verdict per cohort.
`plan_folder` copies the frozen plan and its amendments (their sha256 is checked by the command).
"""

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLAN = "data/frozen/results/grid-20260925"


def entry(split: str, cid: str, matches: int, predicted: int, calls: int) -> dict:
    """One scored setup (config) as the grid scorer writes it. `cid` is model/arm/passes/stage, for example gemma12b/A-minimal-8k/chat/combined."""
    model, arm, passes, stage = cid.split("/")
    formats = {"chat": ["chat8192"], "all3": ["chat8192", "schema8192", "schema16384"]}
    gold = 365 if split == "test" else 115
    precision = matches / predicted
    recall = matches / gold
    labels = [f"{split}/{model}/{arm}/{f}" for f in formats[passes]]
    return {
        "id": f"{split}/{cid}",
        "split": split,
        "model": model,
        "arm": arm,
        "arms": [arm],
        "passes": passes,
        "formats": formats[passes],
        "stage": stage,
        "candidate": True,
        "trained": False,
        "cohort_labels": labels,
        "state": "scored",
        "matches": matches,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "full_record": 0.2,
        "per_paper": [],
        "seconds": 1.0,
        "records_sha256": f"{split}-{cid}",
        "model_calls": calls,
        "model_calls_by_part": {label: calls // len(labels) for label in labels},
        "model_calls_note": None,
        "ineligible_reasons": [],
        "eligible": True,
        "part_status_reasons": {label: [] for label in labels},
        "compute": {},
        "reasoning": "off",
    }


# TEST: (setup, matches, predicted, LLM calls). DEV gets 90/500 for every setup.
TABLE = [
    # Its cohort has a defect in the audit: it would win every floor, but is excluded.
    ("gemma12b/C-gepa-16k32k/chat/combined", 330, 1100, 1),
    # Precision floor 20%: the most matches are 315; within 2 of it (313+) the tie goes to the higher precision, 314/1250 = 25.1% over 315/1500 = 21.0%.
    ("gemma12b/A-minimal-8k/chat/combined", 315, 1500, 10),
    ("gemma12b/A-minimal-8k/chat/verified-self", 314, 1250, 400),
    # Precision floor 28%: two setups of the same precision, 2/7 = 28.6%, within 2 matches of each other: the tie goes to fewer LLM calls.
    ("gemma12b/B-minimal-16k32k/chat/filtered-title", 294, 1029, 300),
    ("gemma12b/B-minimal-16k32k/all3/filtered-title", 292, 1022, 100),
    # Below the 292-match floor (80% recall), whatever its precision.
    ("gemma12b/B-minimal-16k32k/chat/combined", 291, 700, 1),
]
DEFECT = "test/gemma12b/C-gepa-16k32k/chat8192"


def scores_file(path: Path, table=TABLE) -> Path:
    configs = []
    for split in ("dev", "test"):
        for cid, matches, predicted, calls in table:
            if split == "test":
                configs.append(entry(split, cid, matches, predicted, calls))
            else:
                configs.append(entry(split, cid, 90, 500, calls))
    scores = {
        "schema": "taxonomy_pipeline/grid-scores-1",
        "created_utc": "2026-10-20T00:00:00+00:00",
        "grid_root": "$DATA_ROOT/grid",
        "references": {},
        "cohorts": {},
        "missing_cohorts": [],
        "missing_trained_cohorts": [],
        "missing_stages": [],
        "cohort_scores": [],
        "configs": configs,
        "workers": 1,
        "environment": {},
        "trained_roots": [],
        "trained_root_conflicts": [],
        "trained_arms_found": {},
        "opus_test_calls": 494,
    }
    path.write_text(json.dumps(scores, indent=2) + "\n")
    return path


def audit_file(path: Path, table=TABLE) -> Path:
    labels = sorted(
        {
            label
            for split in ("dev", "test")
            for cid, *_ in table
            for label in entry(split, cid, 1, 1, 1)["cohort_labels"]
        }
    )
    cohorts = {
        label: {"verdict": "defect", "reasons": ["made-up defect"]}
        if label == DEFECT
        else {"verdict": "ok", "reasons": []}
        for label in labels
    }
    path.write_text(json.dumps({"schema": "mini-audit", "cohorts": cohorts}) + "\n")
    return path


def plan_folder(repo: Path) -> None:
    shutil.copytree(REPO / PLAN, repo / PLAN, ignore=shutil.ignore_patterns("audit"))
