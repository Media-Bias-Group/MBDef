"""The postprocess step (paper Step 5): which corpus Extracted Concepts are about media bias (docs/step-5-post-processing/postprocess.md).

Label Groups, the relevance paths and their shared evaluation, the frozen paths' one TEST read, the keep or drop decision for every Extracted Concept of the corpus, and the merging of the kept ones. Each command is a plain function that `mbdef postprocess <action>` calls (its parser is in cli.py). It finds the data that git does not hold through DATA_ROOT (data/data-root.tsv).

Step 5's frozen inputs and stored outputs are in one folder, FROZEN, one sub-folder per date (`20261015/`, `20261020/`, `20261021/`). A file over 10 MB is not in git but at the same path under `$DATA_ROOT/frozen/` (`stored`). Every command writes only to the folder its `--out` names, never into a frozen folder (`refuse_frozen`).
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Step 5's frozen inputs and stored outputs, one sub-folder per run date
FROZEN = "data/frozen/step5-post-processing"
SEED = 20261015  # the seed of every Step 5 fold split, inner fit, bootstrap and sample


def stored(base, rel) -> Path:
    """A frozen Step 5 file `rel` of the folder `base`: `base/rel` when it is there; a file of the frozen folder that is over 10 MB is at the same path under `$DATA_ROOT/frozen/` (data/frozen/X in git, frozen/X in the data root)."""
    from mbdef import files

    here = Path(base) / rel
    if here.exists():
        return here
    try:
        inside = here.resolve().relative_to((REPO / "data").resolve())
    except ValueError:
        return here
    return files.data_root() / inside


def frozen_file(rel="") -> Path:
    """The frozen Step 5 file `<date>/...` (see `stored`)."""
    return stored(REPO / FROZEN, rel)


def refuse_frozen(out) -> Path:
    """`out` as given, after refusing a folder inside the frozen folders (in git or in the data root): a run never overwrites the recorded files that the tests compare with or that later steps read."""
    out = Path(out)
    where = [REPO / "data/frozen"]
    if os.environ.get("DATA_ROOT"):
        where.append(Path(os.environ["DATA_ROOT"]).expanduser() / "frozen")
    if any(out.resolve().is_relative_to(w.resolve()) for w in where):
        raise SystemExit(
            f"Refused: {out} is inside a frozen folder ({', '.join(map(str, where))}); "
            "name a new folder with --out."
        )
    return out
