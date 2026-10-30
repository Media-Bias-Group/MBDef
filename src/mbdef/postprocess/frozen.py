"""The record of frozen paths: a path's settings and the train+dev evidence for them, written before the test split is read. The test run only accepts paths listed here and builds each one from its record (`settings["kwargs"]` are the constructor arguments).

mbdef postprocess freeze --out FOLDER [--paths clustering,zero-shot,...]
"""

import datetime as dt
import json
from pathlib import Path
from typing import Any

from . import FROZEN, REPO

FILE = "frozen-paths.json"
# The record the TEST run froze; commands that build paths take their settings from it
RECORD = REPO / FROZEN / "20261015/test-evaluation"


def load_frozen(out_dir):
    f = Path(out_dir) / FILE
    return json.loads(f.read_text()) if f.exists() else {}


def frozen_kwargs(name, out_dir=RECORD):
    """A frozen path's constructor arguments from the record, or None for a variant that was never frozen (it then keeps its class defaults, as exploratory variants do)."""
    entry = load_frozen(out_dir).get(name)
    return None if entry is None else dict(entry["settings"].get("kwargs", {}))


def freeze(out_dir, name, settings, evidence):
    from .compare import provenance

    frozen = load_frozen(out_dir)
    frozen[name] = dict(
        settings=settings,
        evidence=evidence,
        frozen_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        provenance=provenance(),
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / FILE).write_text(json.dumps(frozen, indent=1, ensure_ascii=False))
    return frozen[name]


# Settings chosen on train + dev (the clustering sweep; evaluation/metrics.json). The evidence texts are those of the frozen record, frozen-paths.json.
FROZEN_SETTINGS: dict[str, dict[str, Any]] = {
    "clustering": dict(
        kwargs=dict(dims=20, k=100, a=2.0, neighbours=15),
        evidence="clustering sweep 15 Oct (nested CV, train+dev): PCA-20 + k-means k=100 chosen by every fold; a=2 in 4 of 5",
    ),
    "clustering@group": dict(
        kwargs=dict(dims=20, k=100, a=2.0, neighbours=15),
        member="all",
        evidence="Label Group setting of the frozen clustering path",
    ),
    "representative": dict(
        kwargs=dict(direction="representative_to_definition"),
        evidence="best of four direction combinations on train+dev (AUC 0.664)",
    ),
    "representative@group": dict(
        kwargs=dict(direction="representative_to_definition"),
        member="representative",
        evidence="Label Group setting: each group scored by its Representative Definition",
    ),
    "zero-shot": dict(
        kwargs=dict(hypothesis="This text is about media bias."),
        evidence="best of five predeclared hypotheses on train+dev (AUC 0.781)",
    ),
    "zero-shot@group": dict(
        kwargs=dict(hypothesis="This text is about media bias."),
        member="representative",
        evidence="Label Group setting: each group scored by its Representative Definition",
    ),
    "keyword": dict(kwargs={}, evidence="comparison row"),
    "neighbours": dict(kwargs=dict(neighbours=15), evidence="comparison row"),
    "classifier": dict(
        kwargs={},
        evidence="comparison row (C chosen by inner CV at fit time); promoted to a candidate path, "
        "settings unchanged",
    ),
}


def freeze_paths(out, names):
    """Record the paths `names` with their frozen settings (FROZEN_SETTINGS) in `out`/frozen-paths.json."""
    for name in names:
        entry = dict(FROZEN_SETTINGS[name])
        evidence = entry.pop("evidence")
        freeze(out, name, settings=entry, evidence=evidence)
    print(f"frozen: {', '.join(names)} -> {Path(out) / FILE}")
    return 0


def freeze_outside(out, name, settings, evidence):
    """Record an outside method (for example another group's judge) with its fixed `settings` before it sees the TEST split."""
    freeze(out, name, settings=dict(outside=True, **settings), evidence=evidence)
    print(f"frozen: outside method {name} -> {Path(out) / FILE}")
    return 0
